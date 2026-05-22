"""Drive the LLM NER matrix on Modal: article models + modern models on FiNER-ORD.

Uses the same persistent LLMRunner container as the sentiment matrix
(scripts/modal_app.py), reusing its model-load/swap + chat-template logic.
Output of each call is parsed by src.ner.parser.parse_json_to_bio into BIO
tags aligned to FiNER-ORD gold tokens.

Run-id schema (mirrors §14):
    {model}__FiNER-ORD__{template}__{shots}shot__seed{seed}

Cost: cumulative spend is tracked by scripts/modal_budget (T4 = $0.000164/s).
Hard cap defaults to $3.00 — adjust with --cap-usd.

Usage:
    py scripts/run_ner_modal.py                            # full default lineup
    py scripts/run_ner_modal.py --models qwen3_8b plutus8b # whitelist
    py scripts/run_ner_modal.py --max-samples 50           # smoke run
    py scripts/run_ner_modal.py --dry-run                  # plan only, no Modal
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

# Lazy-import Modal only when we actually need it — keeps --dry-run usable
# without Modal installed.
from scripts.modal_budget import (  # noqa: E402
    can_afford, record, remaining_usd, summary as budget_summary,
)
from src.ner.data_loader import load_finer_ord  # noqa: E402
from src.ner.evaluation import compute_ner_metrics  # noqa: E402
from src.ner.parser import parse_json_to_bio  # noqa: E402
from src.ner.prompts import SYSTEM_PROMPT, build_prompt  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402

logger = get_logger("run_ner_modal")

SEED = 42
GPU = "T4"

# (short_name, hf_id, flags). 'chat': use tokenizer chat template (required for
# instruct models trained with a strict format). 'article': model from the
# original FinLLaMA paper lineup. 'modern': new entrants (Qwen3, Gemma).
MODELS: list[tuple[str, str, dict]] = [
    # ---- Article-replication lineup (matches scripts/run_llm_matrix.py) ----
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct",            {"chat": True,  "tier": "article"}),
    ("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3",  {"chat": True,  "tier": "article"}),
    ("plutus8b",  "TheFinAI/plutus-8B-instruct",         {"chat": True,  "tier": "article"}),
    # ---- Modern models (lecturer's ask) ----
    ("qwen3_8b",  "Qwen/Qwen3-8B",                       {"chat": True,  "tier": "modern"}),
    # Smaller Qwen3 — fast/cheap intra-family size comparison.
    ("qwen3_4b",  "Qwen/Qwen3-4B-Instruct-2507",         {"chat": True,  "tier": "modern"}),
    # Gemma weights are gated on HF — need to be approved on the model page first.
    ("gemma2_9b", "google/gemma-2-9b-it",                {"chat": True,  "tier": "modern", "gated": True}),
    ("gemma2_2b", "google/gemma-2-2b-it",                {"chat": True,  "tier": "modern", "gated": True}),
    # Optional / gated — only run if --include-gated.
    ("llama31_8b","meta-llama/Llama-3.1-8B-Instruct",    {"chat": True,  "tier": "article", "gated": True}),
    ("finma7b",   "TheFinAI/finma-7b-full",              {"chat": True,  "tier": "article", "gated": True}),
]
DATASETS = ["FiNER-ORD"]


def run_id_for(model_short: str, tpl: str, shots: int) -> str:
    return f"{model_short}__FiNER-ORD__{tpl}__{shots}shot__seed{SEED}"


def already_done(run_dir: Path, n_total: int) -> bool:
    pf = run_dir / "progress.json"
    if not pf.exists():
        return False
    try:
        p = json.loads(pf.read_text())
    except json.JSONDecodeError:
        return False
    return int(p.get("last_completed_idx", 0)) >= n_total


def _existing_preds(run_dir: Path) -> dict:
    """Return id -> existing prediction dict, so an incremental Modal run only
    pays for the new samples that aren't in predictions.jsonl yet."""
    out: dict = {}
    p = run_dir / "predictions.jsonl"
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[d["id"]] = d
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def write_run(run_dir: Path, run_id: str, hf_id: str, tpl: str, shots: int,
              samples: list, preds: list, runtime_s: float, started_at: str,
              max_samples_cfg: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    completed_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "run_id": run_id, "model_hf_id": hf_id, "dataset": "FiNER-ORD",
        "template": tpl, "shots": shots, "seed": SEED,
        "n_total": len(samples), "started_at": started_at,
        "completed_at": completed_at, "runtime_s": round(runtime_s, 2),
        "subsample_per_dataset": max_samples_cfg,
        "backend": "modal",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "progress.json").write_text(json.dumps({
        "last_completed_idx": len(samples),
        "n_total": len(samples),
        "updated_at": completed_at,
    }, indent=2))


def _select_models(whitelist: list[str] | None, include_gated: bool, include_modern: bool, include_article: bool):
    out = []
    for entry in MODELS:
        short, hf_id, flags = entry
        if whitelist and short not in whitelist:
            continue
        if flags.get("gated") and not include_gated:
            continue
        tier = flags.get("tier")
        if tier == "modern" and not include_modern:
            continue
        if tier == "article" and not include_article:
            continue
        out.append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="whitelist of model short names")
    ap.add_argument("--templates", nargs="*", default=["A"],
                    help="prompt templates to run (A and/or B). Default: A only.")
    ap.add_argument("--shots", nargs="*", type=int, default=[0],
                    help="few-shot counts to run. Default: 0 only.")
    ap.add_argument("--max-samples", type=int, default=200,
                    help="cap on FiNER-ORD test samples. Default 200.")
    ap.add_argument("--cap-usd", type=float, default=3.0,
                    help="hard budget cap in USD. Default $3.")
    ap.add_argument("--est-seconds-per-run", type=float, default=400.0,
                    help="estimated wall time per run for pre-flight budget check.")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="batch size for LLM generation (NER prompts are longer, keep small).")
    ap.add_argument("--max-new-tokens", type=int, default=180,
                    help="JSON for one sentence rarely needs more.")
    ap.add_argument("--include-gated", action="store_true",
                    help="include gated models (llama-3.1, finma)")
    ap.add_argument("--no-article", action="store_true",
                    help="skip the original-article lineup")
    ap.add_argument("--no-modern", action="store_true",
                    help="skip the modern models (qwen3, gemma)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan + cost estimate, no Modal dispatch")
    args = ap.parse_args()

    set_seed(SEED)
    selected = _select_models(
        whitelist=args.models,
        include_gated=args.include_gated,
        include_modern=not args.no_modern,
        include_article=not args.no_article,
    )

    samples = load_finer_ord(split="test", max_samples=args.max_samples, seed=SEED)
    n_total = len(samples)

    # ---- pre-flight ----
    plan_rows = []
    for entry in selected:
        for tpl in args.templates:
            for shots in args.shots:
                plan_rows.append((entry, tpl, shots))

    print("=" * 64)
    print(f"NER Modal plan — dataset: FiNER-ORD test ({n_total} samples)")
    print(f"GPU: {GPU} @ ${0.000164:.6f}/sec  | cap: ${args.cap_usd:.2f}")
    print(f"Estimated wall: {args.est_seconds_per_run:.0f}s per run "
          f"~ ${args.est_seconds_per_run * 0.000164:.3f}")
    print(f"Currently spent: {budget_summary()}")
    est_total = len(plan_rows) * args.est_seconds_per_run * 0.000164
    print(f"This batch est total: ${est_total:.3f}")
    print()
    for (entry, tpl, shots) in plan_rows:
        short, hf_id, flags = entry
        tier = flags.get("tier", "")
        print(f"  {short:12s} ({tier:7s}) {hf_id:42s} tpl={tpl} shots={shots}")
    print("=" * 64)

    if args.dry_run:
        print("Dry-run — no Modal dispatch.")
        return

    # ---- Modal dispatch ----
    import modal  # noqa: E402
    modal.enable_output()
    from scripts.modal_app import app, LLMRunner  # noqa: E402  (lazy)

    pred_root = ROOT / "results" / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    ran, skipped, aborted = 0, 0, 0
    with app.run():
        runner = LLMRunner()

        for (entry, tpl, shots) in plan_rows:
            short, hf_id, flags = entry
            rid = run_id_for(short, tpl, shots)
            run_dir = pred_root / rid

            if already_done(run_dir, n_total):
                logger.info("[skip] %s already complete", rid)
                skipped += 1
                continue

            if not can_afford(args.est_seconds_per_run, GPU, args.cap_usd):
                logger.warning("[abort] would exceed cap; %s  remaining=$%.4f",
                               budget_summary(), remaining_usd(args.cap_usd))
                aborted = len(plan_rows) - skipped - ran
                break

            # Incremental: skip samples we already have predictions for.
            existing = _existing_preds(run_dir)
            todo_samples = [s for s in samples if s["id"] not in existing]
            if not todo_samples:
                logger.info("[skip] %s — all %d samples already predicted", rid, len(samples))
                skipped += 1
                continue

            # Build NER prompts only for the new ones. We prepend SYSTEM_PROMPT
            # into the user turn because LLMRunner.generate() takes a single
            # prompt list; chat template wraps it as a user message.
            prompts = [
                SYSTEM_PROMPT + "\n\n" + build_prompt(tpl, s["text"], n_shots=shots)
                for s in todo_samples
            ]

            logger.info("[run ] %s  (%d new prompts, %d kept, ~$%.3f budgeted)",
                        rid, len(prompts), len(existing),
                        args.est_seconds_per_run * 0.000164)
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()
            try:
                raw_outputs = runner.generate.remote(
                    hf_id=hf_id, prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    use_chat_template=flags.get("chat", True),
                )
            except Exception as e:
                logger.error("[fail] %s: %s", rid, e)
                aborted += 1
                continue
            wall = time.perf_counter() - t0
            record(rid, GPU, wall)

            # Merge: keep existing preds (by id), add new preds for todo_samples.
            new_preds = {}
            for s, raw in zip(todo_samples, raw_outputs):
                bio, pred_entities = parse_json_to_bio(raw, s["tokens"])
                new_preds[s["id"]] = {
                    "id": s["id"],
                    "pred_tags": bio,
                    "pred_entities": pred_entities,
                    "raw_output": raw,
                    "parse_ok": bio is not None,
                    "latency_ms": (wall * 1000.0) / max(len(raw_outputs), 1),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": (wall * 0.000164) / max(len(raw_outputs), 1),
                }
            preds = [
                new_preds.get(s["id"]) or existing[s["id"]]
                for s in samples
            ]
            write_run(run_dir, rid, hf_id, tpl, shots, samples, preds,
                      runtime_s=wall, started_at=started_at,
                      max_samples_cfg=args.max_samples)

            m = compute_ner_metrics(samples, preds)
            logger.info(
                "[done] %s  strict_f1=%.3f  partial_f1=%.3f  cov=%.2f  (%.1fs, $%.4f cum spend)",
                rid, m["strict"]["f1"], m["partial"]["f1"], m["coverage"], wall,
                args.cap_usd - remaining_usd(args.cap_usd),
            )
            ran += 1

    logger.info("NER Modal matrix done. ran=%d skipped=%d aborted=%d", ran, skipped, aborted)
    logger.info("Final %s", budget_summary())


if __name__ == "__main__":
    main()
