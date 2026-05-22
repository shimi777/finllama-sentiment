"""Drive the NER matrix (local + cheap APIs) on FiNER-ORD.

Reads configs/ner.yaml. For each enabled (model, template, shots) tuple it:
  1) creates results/predictions/{run_id}/
  2) writes meta.json + progress.json
  3) appends predictions.jsonl incrementally (every 25 samples)
  4) honors the budget cap (BudgetExceeded -> stop that model, continue others)

Run-id schema (mirrors §14 of project_plan.md):
    {model}__FiNER-ORD__{template}__{shots}shot__seed{seed}

Usage:
    py scripts/run_ner.py                       # local + cheap APIs (with keys)
    py scripts/run_ner.py --models gliner-large # whitelist
    py scripts/run_ner.py --include-mid         # adds gemini-flash + claude-haiku
    py scripts/run_ner.py --include-frontier    # adds claude-sonnet on 50-subset
    py scripts/run_ner.py --dry-run             # print plan + est cost, no calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ner.cost import (  # noqa: E402
    BudgetExceeded, CostTracker, MODEL_PRICING,
    estimate_chars_to_tokens, have_key, is_local, usd_cost,
)
from src.ner.data_loader import load_finer_ord  # noqa: E402
from src.ner.prompts import SYSTEM_PROMPT, build_prompt  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402

logger = get_logger("run_ner")


def _select_models(cfg: dict, include_mid: bool, include_frontier: bool, whitelist: list[str] | None) -> list[str]:
    out = []
    for name, m in cfg["models"].items():
        if whitelist and name not in whitelist:
            continue
        optional = m.get("optional", False)
        frontier = m.get("frontier", False)
        if frontier and not include_frontier:
            continue
        if optional and not (include_mid or include_frontier):
            continue
        out.append(name)
    return out


def _est_total_cost(models: list[str], samples_per_model: dict[str, int], avg_in: int, avg_out: int) -> dict:
    """Compute a pre-flight cost estimate broken down by model."""
    rows = []
    total = 0.0
    for m in models:
        n = samples_per_model.get(m, 0)
        if is_local(m):
            rows.append({"model": m, "n": n, "est_usd": 0.0})
            continue
        c = n * usd_cost(m, avg_in, avg_out)
        rows.append({"model": m, "n": n, "est_usd": round(c, 4)})
        total += c
    return {"per_model": rows, "total_est_usd": round(total, 4)}


def _write_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _append_pred(run_dir: Path, pred: dict) -> None:
    with open(run_dir / "predictions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(pred) + "\n")


def _write_progress(run_dir: Path, idx: int, n_total: int) -> None:
    (run_dir / "progress.json").write_text(json.dumps({
        "last_completed_idx": idx,
        "n_total": n_total,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def _completed_ids(run_dir: Path) -> set[str]:
    out: set[str] = set()
    p = run_dir / "predictions.jsonl"
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _run_local(model_name: str, model_cfg: dict, samples: list, run_id: str, run_dir: Path, started_at: str) -> dict:
    from src.ner.runners.gliner_runner import GLiNERRunner
    threshold = float(model_cfg.get("threshold", 0.5))
    hf_id = model_cfg.get("hf_id", "urchade/gliner_large-v2.1")
    runner = GLiNERRunner(model_name=hf_id, threshold=threshold)

    done_ids = _completed_ids(run_dir)
    n_total = len(samples)
    t0 = time.perf_counter()
    n_ok = 0
    for i, s in enumerate(samples):
        if s["id"] in done_ids:
            continue
        pred = runner.predict_one(s)
        _append_pred(run_dir, pred)
        if pred["parse_ok"]:
            n_ok += 1
        if (i + 1) % 25 == 0:
            _write_progress(run_dir, i + 1, n_total)
            logger.info("[%s] %d/%d", run_id, i + 1, n_total)
    runtime_s = time.perf_counter() - t0
    _write_progress(run_dir, n_total, n_total)
    return {
        "run_id": run_id, "model_hf_id": hf_id, "dataset": "FiNER-ORD",
        "template": None, "shots": None, "seed": 42, "n_total": n_total,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(runtime_s, 2),
        "backend": "local",
    }


def _run_api(model_name: str, template: str, shots: int, samples: list, run_id: str,
             run_dir: Path, started_at: str, tracker: CostTracker, max_out: int) -> dict:
    from src.ner.runners.api_runner import APINERRunner
    runner = APINERRunner(
        model=model_name, template=template, n_shots=shots,
        cost_tracker=tracker, max_output_tokens=max_out,
    )

    done_ids = _completed_ids(run_dir)
    n_total = len(samples)
    t0 = time.perf_counter()
    stopped_early = False
    for i, s in enumerate(samples):
        if s["id"] in done_ids:
            continue
        try:
            pred = runner.predict_one(s, run_id=run_id)
        except BudgetExceeded as e:
            logger.warning("Budget hit on %s after %d samples: %s", run_id, i, e)
            stopped_early = True
            break
        _append_pred(run_dir, pred)
        if (i + 1) % 25 == 0:
            _write_progress(run_dir, i + 1, n_total)
            logger.info("[%s] %d/%d  spent=$%.4f", run_id, i + 1, n_total, tracker.cumulative_usd)
    runtime_s = time.perf_counter() - t0
    tracker.save()
    return {
        "run_id": run_id, "model_hf_id": model_name, "dataset": "FiNER-ORD",
        "template": template, "shots": shots, "seed": 42, "n_total": n_total,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(runtime_s, 2),
        "backend": "api",
        "stopped_early": stopped_early,
        "spend_after_run_usd": round(tracker.cumulative_usd, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ner.yaml")
    ap.add_argument("--models", nargs="*", default=None,
                    help="whitelist of model names from configs/ner.yaml")
    ap.add_argument("--include-mid", action="store_true",
                    help="include optional mid-tier models (gemini-flash, claude-haiku)")
    ap.add_argument("--include-frontier", action="store_true",
                    help="include claude-sonnet on the frontier subset")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan + cost estimate, do not call APIs")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="override dataset.max_samples")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / args.config))
    set_seed(cfg["seed"])

    max_samples = args.max_samples if args.max_samples is not None else cfg["dataset"]["max_samples"]
    samples = load_finer_ord(
        split=cfg["dataset"]["split"], max_samples=max_samples, seed=cfg["seed"],
    )
    samples_frontier = samples[: cfg["budget"]["frontier_subset"]]

    models = _select_models(cfg, args.include_mid, args.include_frontier, args.models)
    # Sanity-skip models whose API key is missing.
    runnable = []
    for m in models:
        if cfg["models"][m].get("backend") == "api" and not have_key(m):
            logger.warning("Skipping %s — no API key in env", m)
            continue
        runnable.append(m)
    models = runnable

    samples_per_model = {
        m: (len(samples_frontier) if cfg["models"][m].get("frontier") else len(samples))
        for m in models
    }
    # Pre-flight cost estimate using one prompt as a proxy.
    avg_prompt = build_prompt("A", samples[0]["text"] if samples else "x", n_shots=0)
    avg_in = estimate_chars_to_tokens(SYSTEM_PROMPT + "\n" + avg_prompt)
    avg_out = cfg["inference"]["max_output_tokens"]
    plan = _est_total_cost(models, samples_per_model, avg_in, avg_out)

    print("=" * 60)
    print(f"NER plan — dataset: FiNER-ORD test ({len(samples)} samples)")
    print(f"Budget cap: ${cfg['budget']['cap_usd']:.2f}")
    print(f"Pre-flight estimate (avg_in={avg_in}, avg_out={avg_out}):")
    for row in plan["per_model"]:
        print(f"  {row['model']:30s}  n={row['n']:4d}  est ${row['est_usd']:.4f}")
    print(f"  TOTAL est: ${plan['total_est_usd']:.4f}")
    print("=" * 60)

    if args.dry_run:
        print("Dry-run: no API calls dispatched.")
        return

    tracker = CostTracker(cap_usd=cfg["budget"]["cap_usd"],
                          state_path=ROOT / "results" / "_ner_spend.json")
    pred_dir = ROOT / cfg["paths"]["predictions_dir"]
    pred_dir.mkdir(parents=True, exist_ok=True)

    templates = cfg["prompts"]["templates"]
    shots_list = cfg["prompts"]["shots"]

    run_summary = []
    for m in models:
        backend = cfg["models"][m].get("backend")
        is_frontier = cfg["models"][m].get("frontier", False)
        samples_for_model = samples_frontier if is_frontier else samples

        if backend == "local":
            run_id = f"{m}__FiNER-ORD__seed{cfg['seed']}"
            run_dir = pred_dir / run_id
            run_dir.mkdir(exist_ok=True)
            started_at = datetime.now(timezone.utc).isoformat()
            meta = _run_local(m, cfg["models"][m], samples_for_model, run_id, run_dir, started_at)
            _write_meta(run_dir, meta)
            run_summary.append(meta)
            logger.info("Done %s in %.1fs", run_id, meta["runtime_s"])
            continue

        for template in templates:
            for shots in shots_list:
                run_id = f"{m}__FiNER-ORD__{template}__{shots}shot__seed{cfg['seed']}"
                run_dir = pred_dir / run_id
                run_dir.mkdir(exist_ok=True)
                started_at = datetime.now(timezone.utc).isoformat()
                try:
                    meta = _run_api(
                        m, template, shots, samples_for_model, run_id, run_dir,
                        started_at, tracker, cfg["inference"]["max_output_tokens"],
                    )
                except BudgetExceeded as e:
                    logger.error("Top-level budget abort: %s", e)
                    tracker.save()
                    break
                _write_meta(run_dir, meta)
                run_summary.append(meta)
                logger.info("Done %s in %.1fs ($%.4f cum)",
                            run_id, meta["runtime_s"], tracker.cumulative_usd)

    print("\nFinal spend summary:")
    print(json.dumps(tracker.summary(), indent=2))


if __name__ == "__main__":
    main()
