"""Independent recomputation of accuracy / macro-F1 for all FPB / FiQA sentiment runs.

This script is a from-scratch, independent check of the numbers reported in
results/summary/final_table.csv. It deliberately does NOT import
scripts/aggregate.py or src/evaluation.py — it only reuses src/data_loader.py
to obtain gold labels (same data source, but all metric math below is
reimplemented independently in plain Python).

For every run directory under results/predictions/ whose meta.json declares
dataset "FPB" or "FiQA" (runs on FPBall, FIN, ner_*, finbert_ner* are skipped),
this script:

  1. Loads predictions.jsonl (one prediction per line: id, pred_label,
     raw_output, parse_ok, latency_ms).
  2. Loads the matching gold-label split via src.data_loader (load_fpb()[1]
     for FPB test split, load_fiqa() for FiQA) and builds an id -> gold label
     lookup.
  3. Aligns predictions to gold by id (inner join on id; a prediction id not
     found in gold, or vice versa, is reported as a hard error rather than
     silently dropped).
  4. Excludes parse_ok == false rows from accuracy / macro-F1, but counts
     them in `coverage` = n_parsed / n_total.
  5. Computes accuracy and an unweighted macro-F1 over the three canonical
     labels {positive, neutral, negative} using hand-rolled per-class
     precision/recall/F1 (no sklearn).
  6. Writes review/evidence/recomputed_metrics.csv.

Run with the project's venv interpreter, e.g.:
  C:\\python_projects\\finllama-sentiment\\.venv\\Scripts\\python.exe review\\scripts\\recompute_metrics.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Make the repo root importable as `src...` regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_fpb, load_fiqa  # noqa: E402

PRED_ROOT = REPO_ROOT / "results" / "predictions"
OUT_CSV = REPO_ROOT / "review" / "evidence" / "recomputed_metrics.csv"

CANONICAL_LABELS = ("positive", "neutral", "negative")

# Datasets we actually score. Everything else (FPBall, FIN, FiNER-ORD, ner_*,
# finbert_ner*) is out of scope for this sentiment-metrics recheck.
SCORED_DATASETS = {"FPB", "FiQA"}


def discover_run_dirs() -> list[Path]:
    """Return run directories under results/predictions/ that are in-scope.

    In-scope = meta.json exists, is readable JSON, and its "dataset" field is
    exactly "FPB" or "FiQA" (this naturally excludes FPBall, FIN, FiNER-ORD,
    and any ner_*/finbert_ner* dirs, whose dataset field is something else).
    """
    run_dirs = []
    for child in sorted(PRED_ROOT.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        pred_path = child / "predictions.jsonl"
        if not meta_path.exists() or not pred_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        dataset = meta.get("dataset")
        if dataset not in SCORED_DATASETS:
            continue
        run_dirs.append(child)
    return run_dirs


def load_gold_lookup(dataset: str) -> dict[str, str]:
    """Build {id: gold_label} for the given dataset name."""
    if dataset == "FPB":
        _train, test = load_fpb()
        samples = test
    elif dataset == "FiQA":
        samples = load_fiqa()
    else:
        raise ValueError(f"Unsupported dataset for gold lookup: {dataset}")

    lookup: dict[str, str] = {}
    for s in samples:
        lookup[s["id"]] = s["label"]
    return lookup


def load_predictions(pred_path: Path) -> list[dict]:
    rows = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_macro_f1(pairs: list[tuple[str, str]]) -> float:
    """pairs: list of (gold_label, pred_label) for parsed rows only.

    Computes per-class precision/recall/F1 for each of the 3 canonical
    labels using plain-Python counting, then returns the unweighted mean
    of the per-class F1 scores. A class with zero predicted and zero gold
    instances in this run contributes F1 = 0.0 to the mean (matches the
    standard "undefined precision/recall -> 0" convention used when a class
    never appears).
    """
    f1_scores = []
    for label in CANONICAL_LABELS:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def score_run(run_dir: Path) -> dict:
    with (run_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    dataset = meta["dataset"]
    predictions = load_predictions(run_dir / "predictions.jsonl")
    gold_lookup = load_gold_lookup(dataset)

    n_total = len(predictions)
    n_parsed = 0
    pairs: list[tuple[str, str]] = []
    missing_gold_ids = []

    for row in predictions:
        pid = row["id"]
        if pid not in gold_lookup:
            missing_gold_ids.append(pid)
            continue

        parse_ok = row.get("parse_ok", True)
        if not parse_ok:
            continue

        n_parsed += 1
        gold_label = gold_lookup[pid]
        pred_label = row.get("pred_label")
        pairs.append((gold_label, pred_label))

    if missing_gold_ids:
        raise RuntimeError(
            f"{run_dir.name}: {len(missing_gold_ids)} prediction id(s) not found "
            f"in gold labels for dataset {dataset}, e.g. {missing_gold_ids[:5]}"
        )

    n_correct = sum(1 for g, p in pairs if g == p)
    accuracy = n_correct / n_parsed if n_parsed > 0 else 0.0
    f1_macro = compute_macro_f1(pairs)
    coverage = n_parsed / n_total if n_total > 0 else 0.0

    return {
        "run_id": meta["run_id"],
        "model": meta.get("model_hf_id", ""),
        "dataset": dataset,
        "template": meta.get("template") if meta.get("template") is not None else "-",
        "shots": meta.get("shots") if meta.get("shots") is not None else 0,
        "n": n_total,
        "coverage": round(coverage, 4),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
    }


def main() -> None:
    run_dirs = discover_run_dirs()
    if not run_dirs:
        raise SystemExit(f"No in-scope run directories found under {PRED_ROOT}")

    rows = []
    for run_dir in run_dirs:
        print(f"Scoring {run_dir.name} ...", file=sys.stderr)
        rows.append(score_run(run_dir))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "model", "dataset", "template", "shots", "n", "coverage", "accuracy", "f1_macro"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
