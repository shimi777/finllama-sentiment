"""Bootstrap 95% confidence intervals for the headline sentiment macro-F1s.

Resamples the committed per-example predictions (with replacement, B=2000) for
the best config of each model on FPB (75%-agree), FiQA, and FPB AllAgree (100%),
and reports the 2.5/97.5 percentile CI. Shows that sub-1-point gaps are noise.

No GPU needed — reads results/predictions/<run_id>/predictions.jsonl + gold.
Writes results/summary/bootstrap_ci.csv.
"""
from __future__ import annotations
import csv, json, pathlib, sys
import numpy as np
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_fpb, load_fiqa  # noqa: E402

PRED = ROOT / "results" / "predictions"
B = 2000
RNG = np.random.default_rng(42)

# best config per model (verified from final_table.csv / allagree_table.csv)
JOBS = [
    # (dataset_label, gold_key, model_label, run_id)
    ("FPB", "fpb",   "FinBERT",      "finbert__FPB__seed42"),
    ("FPB", "fpb",   "FinBERT-tone", "finbert_tone__FPB__seed42"),
    ("FPB", "fpb",   "Mistral-7B",   "mistral7b__FPB__A__3shot__seed42"),
    ("FPB", "fpb",   "Qwen2.5-7B",   "qwen25_7b__FPB__A__3shot__seed42"),
    ("FPB", "fpb",   "plutus-8B",    "plutus8b__FPB__A__3shot__seed42"),
    ("FPB", "fpb",   "VADER",        "vader__FPB__seed42"),
    ("FiQA", "fiqa", "FinBERT",      "finbert__FiQA__seed42"),
    ("FiQA", "fiqa", "Mistral-7B",   "mistral7b__FiQA__A__0shot__seed42"),
    ("FiQA", "fiqa", "Qwen2.5-7B",   "qwen25_7b__FiQA__B__0shot__seed42"),
    ("FiQA", "fiqa", "plutus-8B",    "plutus8b__FiQA__A__0shot__seed42"),
    ("FiQA", "fiqa", "VADER",        "vader__FiQA__seed42"),
    ("FPB-AllAgree", "fpball", "FinBERT",    "finbert__FPBall__seed42"),
    ("FPB-AllAgree", "fpball", "Mistral-7B", "mistral7b__FPBall__A__3shot__seed42"),
    ("FPB-AllAgree", "fpball", "Qwen2.5-7B", "qwen25_7b__FPBall__A__3shot__seed42"),
    ("FPB-AllAgree", "fpball", "plutus-8B",  "plutus8b__FPBall__A__3shot__seed42"),
    ("FPB-AllAgree", "fpball", "VADER",      "vader__FPBall__seed42"),
]


def gold_maps():
    _, fpb = load_fpb(config="sentences_75agree", seed=42)
    fiqa = load_fiqa()
    _, fpball = load_fpb(config="sentences_allagree", seed=42)
    return {"fpb": {s["id"]: s["label"] for s in fpb},
            "fiqa": {s["id"]: s["label"] for s in fiqa},
            "fpball": {s["id"]: s["label"] for s in fpball}}


def load_preds(run_id):
    p = PRED / run_id / "predictions.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def boot_ci(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    n = len(y_true)
    base = f1_score(y_true, y_pred, average="macro", zero_division=0)
    stats = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        stats[b] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return base, lo, hi, n


def main():
    gold = gold_maps()
    rows = []
    for ds, gkey, model, rid in JOBS:
        preds = load_preds(rid)
        if preds is None:
            print(f"  [missing] {rid}"); continue
        g = gold[gkey]
        yt, yp = [], []
        for p in preds:
            if not p.get("parse_ok") or p.get("pred_label") is None:
                continue
            if p["id"] not in g:
                continue
            yt.append(g[p["id"]]); yp.append(p["pred_label"])
        base, lo, hi, n = boot_ci(yt, yp)
        rows.append({"dataset": ds, "model": model, "n_eval": n,
                     "f1_macro": round(base, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "ci_halfwidth": round((hi - lo) / 2, 4)})
        print(f"  {ds:13s} {model:13s} n={n:>4} F1={base:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  ±{(hi-lo)/2:.3f}")

    out = ROOT / "results" / "summary" / "bootstrap_ci.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
