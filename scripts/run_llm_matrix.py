"""Drive the LLM matrix on Modal: models x 2 datasets x templates A/B x {0,3}-shot,
plus template C x 0-shot (the prompt-ensemble's third member for E3/E5, project_plan §16).

Subsamples 300 from each dataset's test split for LLM runs (baselines used full test sets).
Writes per-run dirs under results/predictions/{run_id}/ per project_plan.md §14.
Idempotent: existing complete runs are skipped via progress.json.

Hard budget guard: aborts before launching a run if cumulative Modal spend would exceed CAP_USD.

Usage: .venv/Scripts/python.exe scripts/run_llm_matrix.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Modal app + class
from scripts.modal_app import app, LLMRunner  # noqa: E402
from scripts.modal_budget import (  # noqa: E402
    can_afford, record, remaining_usd, summary as budget_summary,
)

from src.data_loader import load_fpb, load_fiqa  # noqa: E402
from src.prompts import build_prompt, sample_fewshot  # noqa: E402
from src.parser import parse  # noqa: E402
from src.evaluation import compute_metrics  # noqa: E402
from src.utils import set_seed, get_logger  # noqa: E402

logger = get_logger("run_llm_matrix")

SEED = 42
SUBSAMPLE_PER_DATASET = 300         # LLM matrix size per dataset per run
CAP_USD = 7.0                       # hard stop; well under $30 free tier
ESTIMATED_SECONDS_PER_RUN = 280     # 300 prompts × ~0.7s/sample + model-switch overhead
GPU = "T4"

# Order matters: same-model runs are batched on the same warm container.
# Plutus replaces the now-unpublished TheFinAI/FinLLaMA-instruct as the focal
# financial-instruction-tuned 8B model from the same research group.
# Per-model flags. `chat`: wrap prompts using the tokenizer's chat template before
# generation. Required for instruct models trained with a strict chat format
# (e.g. plutus-8B-instruct emits EOS immediately on raw classification prompts).
MODELS = [
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct",          {"chat": False}),
    ("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", {"chat": False}),
    ("plutus8b",  "TheFinAI/plutus-8B-instruct",        {"chat": True}),
    # Newer-generation 8B from the Qwen family (April 2025) — added at lecturer's
    # recommendation to test whether a more recent base catches up with FinBERT.
    ("qwen3_8b",  "Qwen/Qwen3-8B",                      {"chat": True}),
    # Predecessor of FinLLaMA from TheFinAI (gated; included for reference but
    # blocked unless TheFinAI grants access).
    ("finma_7b",  "TheFinAI/finma-7b-full",             {"chat": False}),
]
DATASETS = ["FPB", "FiQA"]
TEMPLATES = ["A", "B", "C"]
SHOTS = [0, 3]
# Per-template shot grids. A/B keep the full {0,3} grid; C is the ensemble's third
# 0-shot member (E3/E5) — run it 0-shot only, so it costs just 1 extra run per
# (model, dataset). To run only the C pass cheaply for the ensemble models:
#   python scripts/run_llm_matrix.py --only qwen25_7b mistral7b plutus8b
# (existing A/B runs are skipped via progress.json; only the new C0 runs execute).
TEMPLATE_SHOTS = {"A": [0, 3], "B": [0, 3], "C": [0]}


def subsample(samples: list, n: int, seed: int) -> list:
    if len(samples) <= n:
        return list(samples)
    rng = random.Random(seed)
    return rng.sample(samples, n)


def run_id_for(model_short: str, ds: str, tpl: str, shots: int) -> str:
    return f"{model_short}__{ds}__{tpl}__{shots}shot__seed{SEED}"


def already_done(run_dir: str, n_total: int) -> bool:
    pf = os.path.join(run_dir, "progress.json")
    if not os.path.exists(pf):
        return False
    with open(pf) as f:
        p = json.load(f)
    return p.get("last_completed_idx", 0) >= n_total


def write_run(run_dir: str, run_id: str, hf_id: str, ds: str, tpl: str, shots: int,
              samples: list, preds: list, runtime_s: float, started_at: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    completed_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "run_id": run_id, "model_hf_id": hf_id, "dataset": ds,
        "template": tpl, "shots": shots, "seed": SEED,
        "n_total": len(samples), "started_at": started_at,
        "completed_at": completed_at, "runtime_s": round(runtime_s, 2),
        "subsample_per_dataset": SUBSAMPLE_PER_DATASET,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only run these model short names (e.g. --only plutus8b). "
             "Useful for running the rest of the matrix in parallel with a long-running first job.",
    )
    args = parser.parse_args()

    global MODELS
    if args.only:
        keep = set(args.only)
        MODELS = [m for m in MODELS if m[0] in keep]
        logger.info("--only filter: running %s", [m[0] for m in MODELS])

    set_seed(SEED)
    logger.info("Budget at start: %s", budget_summary())
    logger.info("Cap: $%.2f  | remaining: $%.4f", CAP_USD, remaining_usd(CAP_USD))

    # --- Load + subsample datasets once ---
    fpb_train, fpb_test = load_fpb(seed=SEED)
    fiqa_test = load_fiqa()

    fpb_eval = subsample(fpb_test, SUBSAMPLE_PER_DATASET, SEED)
    fiqa_eval = subsample(fiqa_test, SUBSAMPLE_PER_DATASET, SEED)
    test_by_ds = {"FPB": fpb_eval, "FiQA": fiqa_eval}
    logger.info("Eval set sizes: FPB=%d  FiQA=%d  (FPB train pool=%d for few-shot)",
                len(fpb_eval), len(fiqa_eval), len(fpb_train))

    pred_dir = os.path.join(ROOT, "results", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # --- Plan ordering: group by model so the Modal container can keep weights warm ---
    plan = []
    for entry in MODELS:
        model_short, hf_id = entry[0], entry[1]
        flags = entry[2] if len(entry) > 2 else {}
        for ds in DATASETS:
            for tpl in TEMPLATES:
                for shots in TEMPLATE_SHOTS.get(tpl, SHOTS):
                    plan.append((model_short, hf_id, ds, tpl, shots, flags))

    skipped, ran, aborted = 0, 0, 0

    with app.run():
        runner = LLMRunner()  # one persistent container shared across all runs

        for (model_short, hf_id, ds, tpl, shots, flags) in plan:
            rid = run_id_for(model_short, ds, tpl, shots)
            run_dir = os.path.join(pred_dir, rid)
            samples = test_by_ds[ds]

            if already_done(run_dir, len(samples)):
                logger.info("[skip] %s already complete", rid)
                skipped += 1
                continue

            # Budget pre-flight
            if not can_afford(ESTIMATED_SECONDS_PER_RUN, GPU, CAP_USD):
                logger.warning("[abort] would exceed cap; budget=%s remaining=$%.4f",
                               budget_summary(), remaining_usd(CAP_USD))
                aborted = len(plan) - skipped - ran
                break

            # Build prompts
            few_shot = sample_fewshot(fpb_train, shots, seed=SEED) if shots > 0 else []
            prompts = [build_prompt(tpl, s["text"], few_shot) for s in samples]

            logger.info("[run ] %s  (%d prompts, ~$%.3f budgeted)",
                        rid, len(prompts), ESTIMATED_SECONDS_PER_RUN * 0.000164)

            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()
            try:
                raw_outputs = runner.generate.remote(
                    hf_id=hf_id, prompts=prompts, max_new_tokens=20, batch_size=8,
                    use_chat_template=flags.get("chat", False),
                )
            except Exception as e:
                logger.error("[fail] %s: %s", rid, e)
                aborted += 1
                continue
            wall = time.perf_counter() - t0

            # Record budget. Wall time on the local end overcounts (network), but it's a safe upper bound.
            record(rid, GPU, wall)

            # Parse + persist
            preds = []
            for s, raw in zip(samples, raw_outputs):
                lbl = parse(raw)
                preds.append({
                    "id": s["id"],
                    "pred_label": lbl,
                    "raw_output": raw,
                    "parse_ok": lbl is not None,
                    "latency_ms": (wall * 1000.0) / max(len(raw_outputs), 1),
                })
            write_run(run_dir, rid, hf_id, ds, tpl, shots, samples, preds,
                      runtime_s=wall, started_at=started_at)

            m = compute_metrics(samples, preds)
            logger.info(
                "[done] %s  acc=%.3f  f1m=%.3f  cov=%.2f  (%.1fs, $%.4f cum)",
                rid, m["accuracy"], m["f1_macro"], m["coverage"], wall,
                CAP_USD - remaining_usd(CAP_USD),
            )
            ran += 1

    logger.info("Matrix done. ran=%d skipped=%d aborted=%d", ran, skipped, aborted)
    logger.info("Final %s", budget_summary())


if __name__ == "__main__":
    main()
