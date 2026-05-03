"""Pull 30 FinLLaMA misses for hand-categorization.

Picks misses across both datasets, biased toward the best FinLLaMA config (highest f1_macro
in final_table.csv). Writes a CSV with text, gold, predicted, raw_output — leaves a `category`
column blank for hand-tagging.

Usage: .venv/Scripts/python.exe scripts/error_analysis.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from glob import glob

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import load_fpb, load_fiqa  # noqa: E402

SEED = 42
N_MISSES = 30


FOCAL_MODELS = ("plutus8b", "finllama")  # whichever made it into the matrix


def _best_focal_runs() -> list[str]:
    table = os.path.join(ROOT, "results", "summary", "final_table.csv")
    if not os.path.exists(table):
        return []
    df = pd.read_csv(table)
    df = df[df["model"].isin(FOCAL_MODELS)].sort_values("f1_macro", ascending=False)
    return df.head(2).apply(  # top-2 (typically one per dataset)
        lambda r: f"{r['model']}__{r['dataset']}__{r['template']}__{int(r['shots'])}shot__seed{int(r['seed'])}",
        axis=1,
    ).tolist()


def main() -> None:
    rng = random.Random(SEED)
    _, fpb_test = load_fpb(seed=SEED)
    fiqa_test = load_fiqa()
    gold = {s["id"]: s for s in (fpb_test + fiqa_test)}

    runs = _best_focal_runs()
    if not runs:
        # Fallback: any focal-model run we can find
        candidates = []
        for short in FOCAL_MODELS:
            candidates += [os.path.basename(p) for p in glob(os.path.join(
                ROOT, "results", "predictions", f"{short}__*"))]
        runs = candidates

    if not runs:
        print("no focal-model predictions found; run the matrix first")
        return

    misses: list[dict] = []
    for rid in runs:
        pred_file = os.path.join(ROOT, "results", "predictions", rid, "predictions.jsonl")
        if not os.path.exists(pred_file):
            continue
        with open(pred_file) as f:
            for line in f:
                p = json.loads(line)
                s = gold.get(p["id"])
                if s is None or not p["parse_ok"]:
                    continue
                if p["pred_label"] != s["label"]:
                    misses.append({
                        "run_id": rid,
                        "id": p["id"],
                        "dataset": s["dataset"],
                        "text": s["text"],
                        "gold": s["label"],
                        "pred": p["pred_label"],
                        "raw_output": p["raw_output"],
                        "category": "",  # to fill in by hand
                    })

    if not misses:
        print("no FinLLaMA misses found in available runs")
        return

    rng.shuffle(misses)
    sample = misses[:N_MISSES]

    out_dir = os.path.join(ROOT, "results", "summary")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "focal_error_sample.csv")
    pd.DataFrame(sample).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(sample)} misses for hand-tagging)")
    print("Suggested categories: negation | numerical_reasoning | domain_jargon | "
          "ambiguous | sarcasm | factual_neutral_misclassed")


if __name__ == "__main__":
    main()
