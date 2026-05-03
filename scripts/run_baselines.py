"""Run FinBERT + VADER baselines on FPB + FiQA test splits.

Writes per-run directories under results/predictions/{run_id}/ following
project_plan.md §14, plus an aggregate results/summary/baselines.csv.

Usage: py scripts/run_baselines.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

# Make `src` importable when run as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import load_fpb, load_fiqa  # noqa: E402
from src.evaluation import compute_metrics  # noqa: E402
from src.utils import load_config, set_seed, get_logger  # noqa: E402

logger = get_logger("run_baselines")


def write_run(run_id: str, run_dir: str, model_hf_id: str, dataset: str,
              samples: list, preds: list, runtime_s: float, started_at: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    meta = {
        "run_id": run_id,
        "model_hf_id": model_hf_id,
        "dataset": dataset,
        "template": None,
        "shots": None,
        "seed": SEED,
        "n_total": len(samples),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(runtime_s, 2),
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    progress = {"last_completed_idx": len(samples), "n_total": len(samples),
                "updated_at": meta["completed_at"]}
    with open(os.path.join(run_dir, "progress.json"), "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def main() -> None:
    cfg = load_config(os.path.join(ROOT, "configs", "experiment.yaml"))
    global SEED
    SEED = cfg["seed"]
    set_seed(SEED)

    pred_dir = os.path.join(ROOT, cfg["paths"]["predictions_dir"])
    summary_dir = os.path.join(ROOT, cfg["paths"]["summary_dir"])
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    logger.info("Loading datasets…")
    _, fpb_test = load_fpb(
        config=cfg["datasets"]["fpb"]["config"],
        test_fraction=cfg["datasets"]["fpb"]["test_fraction"],
        seed=SEED,
    )
    fiqa_test = load_fiqa(neutral_band=cfg["datasets"]["fiqa"]["neutral_band"])
    datasets = {"FPB": fpb_test, "FiQA": fiqa_test}
    logger.info("Loaded: %s", {k: len(v) for k, v in datasets.items()})

    rows: list[dict] = []

    # ---- VADER ----
    from src.models.vader_runner import VADERRunner
    vader = VADERRunner()
    for ds_name, samples in datasets.items():
        texts = [s["text"] for s in samples]
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        labels = vader.predict(texts)
        runtime_s = time.perf_counter() - t0
        preds = [
            {"id": s["id"], "pred_label": lbl, "raw_output": "",
             "parse_ok": True, "latency_ms": 0.0}
            for s, lbl in zip(samples, labels)
        ]
        run_id = f"vader__{ds_name}__seed{SEED}"
        write_run(run_id, os.path.join(pred_dir, run_id),
                  model_hf_id="vaderSentiment", dataset=ds_name,
                  samples=samples, preds=preds, runtime_s=runtime_s,
                  started_at=started_at)
        m = compute_metrics(samples, preds)
        rows.append({
            "model": "VADER", "dataset": ds_name, "template": "-", "shots": 0,
            "seed": SEED, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"], "coverage": m["coverage"],
            "n_samples": m["n_samples"], "runtime_s": round(runtime_s, 2),
        })
        logger.info("VADER %s: acc=%.3f  f1_macro=%.3f  (%.1fs)",
                    ds_name, m["accuracy"], m["f1_macro"], runtime_s)

    # ---- FinBERT (CPU; switch device=0 if CUDA available) ----
    from src.models.finbert_runner import FinBERTRunner
    import torch
    device = 0 if torch.cuda.is_available() else -1
    logger.info("FinBERT device=%d (CUDA=%s)", device, torch.cuda.is_available())
    finbert = FinBERTRunner(hf_id=cfg["models"]["finbert"]["hf_id"], device=device)
    for ds_name, samples in datasets.items():
        texts = [s["text"] for s in samples]
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        labels = finbert.predict(texts)
        runtime_s = time.perf_counter() - t0
        preds = [
            {"id": s["id"], "pred_label": lbl, "raw_output": "",
             "parse_ok": True, "latency_ms": 0.0}
            for s, lbl in zip(samples, labels)
        ]
        run_id = f"finbert__{ds_name}__seed{SEED}"
        write_run(run_id, os.path.join(pred_dir, run_id),
                  model_hf_id=cfg["models"]["finbert"]["hf_id"], dataset=ds_name,
                  samples=samples, preds=preds, runtime_s=runtime_s,
                  started_at=started_at)
        m = compute_metrics(samples, preds)
        rows.append({
            "model": "FinBERT", "dataset": ds_name, "template": "-", "shots": 0,
            "seed": SEED, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"], "coverage": m["coverage"],
            "n_samples": m["n_samples"], "runtime_s": round(runtime_s, 2),
        })
        logger.info("FinBERT %s: acc=%.3f  f1_macro=%.3f  (%.1fs)",
                    ds_name, m["accuracy"], m["f1_macro"], runtime_s)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(summary_dir, "baselines.csv")
    df.to_csv(out_csv, index=False)
    logger.info("Saved → %s", out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
