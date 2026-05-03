"""Walk results/predictions/*/ and produce results/summary/final_table.csv + per-run confusions.

Reads each run's meta.json + predictions.jsonl, joins predictions back to the appropriate
dataset's gold labels via id, computes metrics, writes:
  - results/summary/final_table.csv
  - results/summary/confusions/{run_id}.json

Usage: .venv/Scripts/python.exe scripts/aggregate.py
"""

from __future__ import annotations

import json
import os
import sys
from glob import glob

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import load_fpb, load_fiqa  # noqa: E402
from src.evaluation import compute_metrics  # noqa: E402
from src.utils import set_seed, get_logger  # noqa: E402

logger = get_logger("aggregate")


def main() -> None:
    set_seed(42)

    # Load gold labels once, indexed by id
    _, fpb_test = load_fpb(seed=42)
    fiqa_test = load_fiqa()
    gold = {s["id"]: s for s in (fpb_test + fiqa_test)}

    pred_root = os.path.join(ROOT, "results", "predictions")
    summary_dir = os.path.join(ROOT, "results", "summary")
    conf_dir = os.path.join(summary_dir, "confusions")
    os.makedirs(conf_dir, exist_ok=True)

    rows = []
    for run_dir in sorted(glob(os.path.join(pred_root, "*"))):
        meta_path = os.path.join(run_dir, "meta.json")
        pred_path = os.path.join(run_dir, "predictions.jsonl")
        if not (os.path.exists(meta_path) and os.path.exists(pred_path)):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        preds: list[dict] = []
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    preds.append(json.loads(line))

        # Build aligned samples list from gold (in same order as preds)
        samples = []
        kept_preds = []
        for p in preds:
            s = gold.get(p["id"])
            if s is None:
                continue  # ids not in gold — skip rather than fail
            samples.append(s)
            kept_preds.append(p)

        if not samples:
            logger.warning("No matching gold for %s; skipping", meta["run_id"])
            continue

        m = compute_metrics(samples, kept_preds)

        # Persist confusion matrix per run
        with open(os.path.join(conf_dir, f"{meta['run_id']}.json"), "w") as f:
            json.dump({
                "run_id": meta["run_id"],
                "labels": m.get("confusion_labels", []),
                "matrix": m["confusion"],
                "per_class": m["per_class"],
                "n_samples": m["n_samples"],
                "coverage": m["coverage"],
            }, f, indent=2)

        # Friendly model short-name (first segment of run_id)
        model_short = meta["run_id"].split("__")[0]

        rows.append({
            "model": model_short,
            "model_hf_id": meta.get("model_hf_id", ""),
            "dataset": meta["dataset"],
            "template": meta.get("template") or "-",
            "shots": meta.get("shots") if meta.get("shots") is not None else 0,
            "seed": meta["seed"],
            "n_samples": m["n_samples"],
            "accuracy": round(m["accuracy"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "f1_weighted": round(m["f1_weighted"], 4),
            "coverage": round(m["coverage"], 4),
            "runtime_s": meta.get("runtime_s", 0.0),
        })

    if not rows:
        logger.warning("No runs found under %s", pred_root)
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "model", "template", "shots"]).reset_index(drop=True)
    out_csv = os.path.join(summary_dir, "final_table.csv")
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(df))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
