"""Drive the NER matrix on Modal: reproduce Open-FinLLMs Table 7 NER numbers.

Reuses the existing `LLMRunner` Modal class (it's task-agnostic — takes prompts,
returns raw text). Differences vs. `run_llm_matrix.py`:
  - Loads the Alvarado-2015 FIN dataset via `src/ner_loader.py`.
  - Uses NER prompts (`paper` / `strict`) and entity-list parser.
  - Reports entity-level F1 (micro, restricted to {PER, ORG, LOC} per the paper).
  - Higher max_new_tokens (NER outputs are longer than a single label).

Run-id schema (parallel to sentiment side, distinct namespace to avoid clobber):
    ner__{model_short}__{ds}__{tpl}__{shots}shot__seed{SEED}

Idempotent: skips runs whose progress.json reports completion.

Usage:
    .venv/Scripts/python.exe scripts/run_ner_matrix.py
    .venv/Scripts/python.exe scripts/run_ner_matrix.py --only mistral7b qwen25_7b
    .venv/Scripts/python.exe scripts/run_ner_matrix.py --dry-run    # builds prompts only, no Modal
    .venv/Scripts/python.exe scripts/run_ner_matrix.py --limit 30   # cap eval set for smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.modal_app import app, LLMRunner  # noqa: E402
from scripts.modal_budget import (  # noqa: E402
    can_afford, record, remaining_usd, summary as budget_summary,
)

from src.ner_loader import load_fin_alvarado  # noqa: E402
from src.ner_prompts import build_ner_prompt, sample_fewshot  # noqa: E402
from src.ner_parser import parse_ner  # noqa: E402
from src.ner_evaluation import compute_ner_metrics  # noqa: E402
from src.utils import set_seed, get_logger  # noqa: E402

logger = get_logger("run_ner_matrix")

SEED = 42
CAP_USD = 7.0
GPU = "T4"
ESTIMATED_SECONDS_PER_RUN = 600  # NER prompts are longer; assume ~2x sentiment

# Same family as the sentiment matrix so we can directly compare "does the
# financial-tuned model win on its native task too?".
MODELS = [
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct",          {"chat": True}),
    ("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", {"chat": True}),
    ("plutus8b",  "TheFinAI/plutus-8B-instruct",        {"chat": True}),
]

TEMPLATES = ["paper", "strict"]
SHOTS = [0, 3]

# Try the PIXIU FLARE wrapping first; fall back to alternative HF IDs the user
# can supply via env if this 404s. The Alvarado FIN dataset has been mirrored
# under several names.
FIN_HF_CANDIDATES = [
    "TheFinAI/flare-ner",
    "ChanceFocus/flare-ner",
    "tner/fin",
]


def load_fin_with_fallback(split: str = "test"):
    last_err = None
    override = os.environ.get("FIN_HF_ID")
    candidates = [override] if override else FIN_HF_CANDIDATES
    for hf_id in candidates:
        if not hf_id:
            continue
        try:
            return load_fin_alvarado(hf_id=hf_id, split=split), hf_id
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("FIN load via %s failed: %s", hf_id, e)
            continue
    raise RuntimeError(
        "All FIN dataset candidates failed. Set FIN_HF_ID env to override. "
        f"Last error: {last_err}"
    )


def run_id_for(model_short: str, ds: str, tpl: str, shots: int, dry: bool = False) -> str:
    base = f"ner__{model_short}__{ds}__{tpl}__{shots}shot__seed{SEED}"
    return ("dryrun__" + base) if dry else base


def already_done(run_dir: str, n_total: int) -> bool:
    pf = os.path.join(run_dir, "progress.json")
    if not os.path.exists(pf):
        return False
    with open(pf) as f:
        p = json.load(f)
    return p.get("last_completed_idx", 0) >= n_total


def write_run(run_dir, run_id, hf_id, ds, tpl, shots, samples, preds, runtime_s, started_at, metrics):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    completed_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "run_id": run_id, "task": "ner", "model_hf_id": hf_id, "dataset": ds,
        "template": tpl, "shots": shots, "seed": SEED,
        "n_total": len(samples), "started_at": started_at,
        "completed_at": completed_at, "runtime_s": round(runtime_s, 2),
        "metrics": metrics,
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(run_dir, "progress.json"), "w") as f:
        json.dump({
            "last_completed_idx": len(samples),
            "n_total": len(samples),
            "updated_at": completed_at,
        }, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="Only these model short names")
    ap.add_argument("--templates", nargs="+", default=TEMPLATES)
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--limit", type=int, default=0, help="Cap eval set (0=all)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build prompts and run parser/metrics on synthetic outputs; no Modal calls.")
    args = ap.parse_args()

    global MODELS
    if args.only:
        keep = set(args.only)
        MODELS = [m for m in MODELS if m[0] in keep]

    set_seed(SEED)
    logger.info("Budget at start: %s", budget_summary())
    logger.info("Cap: $%.2f  remaining: $%.4f", CAP_USD, remaining_usd(CAP_USD))

    test_pool, fin_hf_id = load_fin_with_fallback("test")
    train_pool, _ = load_fin_with_fallback("train") if any(s > 0 for s in args.shots) else ([], fin_hf_id)

    eval_set = list(test_pool)
    if args.limit and len(eval_set) > args.limit:
        rng = random.Random(SEED)
        eval_set = rng.sample(eval_set, args.limit)
    logger.info("FIN eval set: %d  | source: %s  | train pool: %d",
                len(eval_set), fin_hf_id, len(train_pool))

    pred_dir = os.path.join(ROOT, "results", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    plan = []
    for entry in MODELS:
        model_short, hf_id, flags = entry[0], entry[1], entry[2]
        for tpl in args.templates:
            for shots in args.shots:
                plan.append((model_short, hf_id, "FIN", tpl, shots, flags))

    skipped = ran = aborted = 0

    def execute_run(model_short, hf_id, ds, tpl, shots, flags, runner_call):
        nonlocal skipped, ran, aborted
        rid = run_id_for(model_short, ds, tpl, shots, dry=args.dry_run)
        run_dir = os.path.join(pred_dir, rid)
        if already_done(run_dir, len(eval_set)):
            logger.info("[skip] %s already complete", rid)
            skipped += 1
            return
        if not args.dry_run and not can_afford(ESTIMATED_SECONDS_PER_RUN, GPU, CAP_USD):
            logger.warning("[abort] would exceed cap; %s", budget_summary())
            aborted += 1
            return

        few_shot = sample_fewshot(train_pool, shots, seed=SEED) if shots > 0 else []
        prompts = [build_ner_prompt(tpl, s["text"], few_shot) for s in eval_set]
        logger.info("[run ] %s  (%d prompts)", rid, len(prompts))

        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        try:
            raw_outputs = runner_call(hf_id, prompts, flags.get("chat", True))
        except Exception as e:  # noqa: BLE001
            logger.error("[fail] %s: %s", rid, e)
            aborted += 1
            return
        wall = time.perf_counter() - t0

        if not args.dry_run:
            record(rid, GPU, wall)

        preds = []
        for s, raw in zip(eval_set, raw_outputs):
            ents = parse_ner(raw)
            preds.append({
                "id": s["id"],
                "pred_entities": ents,
                "raw_output": raw,
                "parse_ok": ents is not None,
                "latency_ms": (wall * 1000.0) / max(len(raw_outputs), 1),
            })

        metrics = compute_ner_metrics(eval_set, preds, only={"PER", "ORG", "LOC"})
        write_run(run_dir, rid, hf_id, ds, tpl, shots, eval_set, preds,
                  runtime_s=wall, started_at=started_at, metrics=metrics)

        logger.info(
            "[done] %s  micro_f1=%.3f  cov=%.2f  parse_fail=%d  (%.1fs)",
            rid, metrics["micro_f1"], metrics["coverage"],
            metrics["n_parse_failures"], wall,
        )
        ran += 1

    if args.dry_run:
        # Return synthetic outputs so we can exercise parser + metrics end-to-end
        # without touching Modal. We echo the gold (as if a perfect oracle) to
        # verify F1 should be 1.0 across the board; useful for unit-of-work check.
        def runner_call(hf_id, prompts, use_chat):  # noqa: ARG001
            gold_strs = []
            for s in eval_set:
                if s["entities"]:
                    gold_strs.append("; ".join(f"{e['text']}, {e['type']}" for e in s["entities"]))
                else:
                    gold_strs.append("NONE")
            return gold_strs
        for (model_short, hf_id, ds, tpl, shots, flags) in plan:
            execute_run(model_short, hf_id, ds, tpl, shots, flags, runner_call)
    else:
        with app.run():
            modal_runner = LLMRunner()

            def runner_call(hf_id, prompts, use_chat):
                return modal_runner.generate.remote(
                    hf_id=hf_id, prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    use_chat_template=use_chat,
                )

            for (model_short, hf_id, ds, tpl, shots, flags) in plan:
                execute_run(model_short, hf_id, ds, tpl, shots, flags, runner_call)

    logger.info("NER matrix done. ran=%d skipped=%d aborted=%d", ran, skipped, aborted)
    logger.info("Final %s", budget_summary())


if __name__ == "__main__":
    main()
