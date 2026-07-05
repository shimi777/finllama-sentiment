"""Baseline subsample-parity check (review Stage 3b).

final_table.csv scores FinBERT/VADER on the FULL test sets (FPB 690 / FiQA 1173)
while all LLM cells use a 300-sample subset. This script re-scores every baseline
run on exactly the 300 ids the LLMs saw (review/evidence/subset_ids_{ds}.json,
extracted from the cached LLM predictions), so all models can be compared on a
common footing.

Reads:  results/predictions/<baseline runs>/predictions.jsonl, gold via src.data_loader
Writes: review/evidence/parity_table.csv
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_fpb, load_fiqa  # noqa: E402
from src.evaluation import compute_metrics  # noqa: E402

BASELINE_RUNS = {
    "FPB": ["finbert__FPB__seed42", "finbert_tone__FPB__seed42", "vader__FPB__seed42"],
    "FiQA": ["finbert__FiQA__seed42", "finbert_tone__FiQA__seed42", "vader__FiQA__seed42"],
}


def load_gold(dataset: str) -> dict[str, dict]:
    if dataset == "FPB":
        _, test = load_fpb()
        return {s["id"]: s for s in test}
    test = load_fiqa()
    return {s["id"]: s for s in test}


def load_preds(run_id: str) -> dict[str, dict]:
    path = ROOT / "results" / "predictions" / run_id / "predictions.jsonl"
    if not path.exists():
        return {}
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            preds[p["id"]] = p
    return preds


def score(gold_by_id: dict, preds_by_id: dict, ids: list[str]) -> dict:
    samples = [gold_by_id[i] for i in ids]
    preds = [preds_by_id[i] for i in ids]
    return compute_metrics(samples, preds)


def main() -> None:
    rows = []
    for dataset, runs in BASELINE_RUNS.items():
        gold = load_gold(dataset)
        subset_ids = json.loads(
            (ROOT / "review" / "evidence" / f"subset_ids_{dataset}.json").read_text()
        )
        if isinstance(subset_ids, dict):  # tolerate {"ids": [...]} shape
            subset_ids = subset_ids.get("ids", list(subset_ids.values())[0])
        for run_id in runs:
            preds = load_preds(run_id)
            if not preds:
                print(f"skip (no local predictions): {run_id}")
                continue
            full_ids = [i for i in preds if i in gold]
            missing_subset = [i for i in subset_ids if i not in preds]
            if missing_subset:
                print(f"WARNING {run_id}: {len(missing_subset)} subset ids missing")
                continue
            m_full = score(gold, preds, full_ids)
            m_sub = score(gold, preds, subset_ids)
            rows.append({
                "run_id": run_id,
                "dataset": dataset,
                "n_full": len(full_ids),
                "acc_full": round(m_full["accuracy"], 4),
                "f1_macro_full": round(m_full["f1_macro"], 4),
                "n_subset": len(subset_ids),
                "acc_subset": round(m_sub["accuracy"], 4),
                "f1_macro_subset": round(m_sub["f1_macro"], 4),
                "d_acc_subset_minus_full": round(m_sub["accuracy"] - m_full["accuracy"], 4),
                "d_f1_subset_minus_full": round(m_sub["f1_macro"] - m_full["f1_macro"], 4),
            })
            print(f"{run_id}: full f1={m_full['f1_macro']:.4f} -> subset f1={m_sub['f1_macro']:.4f}")

    out = ROOT / "review" / "evidence" / "parity_table.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
