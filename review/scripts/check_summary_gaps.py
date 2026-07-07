"""
Cross-reference results/predictions/* run directories against
results/summary/final_table.csv and results/summary_ner/final_table_ner.csv.

Writes review/evidence/summary_vs_runs_gaps.md

Read-only w.r.t. results/. Only writes into review/evidence/.
"""
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "results" / "predictions"
FINAL_TABLE = REPO_ROOT / "results" / "summary" / "final_table.csv"
FINAL_TABLE_NER = REPO_ROOT / "results" / "summary_ner" / "final_table_ner.csv"
OUT_MD = REPO_ROOT / "review" / "evidence" / "summary_vs_runs_gaps.md"

# model_hf_id -> short model name used in run_id, from meta.json inspection
MODEL_NAME_MAP = {
    "ProsusAI/finbert": "finbert",
    "vaderSentiment": "vader",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral7b",
    "TheFinAI/plutus-8B-instruct": "plutus8b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen25_7b",
    "urchade/gliner_large-v2.1": "gliner-large",
    "urchade/gliner_small-v2.1": "gliner-small",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3_4b",
    "Qwen/Qwen3-8B": "qwen3_8b",
    "dslim/bert-base-NER": "finbert_ner",
    "yiyanghkust/finbert-tone": "finbert_tone",
}


def load_meta(run_dir):
    p = run_dir / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_id_to_expected_csv_key(run_dir_name, meta):
    """Build a (model, dataset, template, shots) tuple to match against CSV rows."""
    model_hf = meta.get("model_hf_id", "")
    model_short = MODEL_NAME_MAP.get(model_hf, model_hf)
    dataset = meta.get("dataset", "")
    template = meta.get("template")
    shots = meta.get("shots")
    template_norm = template if template else "-"
    shots_norm = shots if shots is not None else 0
    return (model_short, dataset, template_norm, shots_norm)


def load_csv_keys(csv_path, template_col="template", shots_col="shots"):
    keys = []
    rows = []
    if not csv_path.exists():
        return keys, rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", "")
            dataset = row.get("dataset", "")
            template = row.get(template_col, "-")
            shots_raw = row.get(shots_col, "0")
            try:
                shots = int(float(shots_raw))
            except (ValueError, TypeError):
                shots = shots_raw
            keys.append((model, dataset, template, shots))
            rows.append(row)
    return keys, rows


def main():
    run_dirs = sorted([p for p in PRED_DIR.iterdir() if p.is_dir()])

    # Build run-side records
    run_records = []
    for run_dir in run_dirs:
        meta = load_meta(run_dir)
        key = run_id_to_expected_csv_key(run_dir.name, meta)
        task = meta.get("task", "")
        is_ner_track = run_dir.name.startswith("ner__") or task == "ner" or "NER" in meta.get("dataset", "") or meta.get("dataset") == "FiNER-ORD"
        run_records.append({
            "run_id": run_dir.name,
            "key": key,
            "dataset": meta.get("dataset", ""),
            "is_ner_track": is_ner_track,
        })

    sentiment_table_keys, sentiment_rows = load_csv_keys(FINAL_TABLE)
    ner_table_keys, ner_rows = load_csv_keys(FINAL_TABLE_NER)

    sentiment_keys_set = set(sentiment_table_keys)
    ner_keys_set = set(ner_table_keys)

    lines = []
    lines.append("# Summary vs. run-directory gap check\n")
    lines.append(f"Total run directories under `results/predictions/`: {len(run_dirs)}\n")
    lines.append(f"Rows in `results/summary/final_table.csv`: {len(sentiment_rows)}\n")
    lines.append(f"Rows in `results/summary_ner/final_table_ner.csv`: {len(ner_rows)}\n")

    # ---- Runs present but missing from summaries ----
    lines.append("\n## Runs present in results/predictions/ but MISSING from a summary CSV\n")

    missing_from_any = []
    for rec in run_records:
        run_id = rec["run_id"]
        dataset = rec["dataset"]
        key = rec["key"]

        in_sentiment = key in sentiment_keys_set
        in_ner = key in ner_keys_set

        # Special-case runs with no meaningful key match expected (exploratory NER, FIN NER)
        if run_id.startswith("ner__"):
            # ner__{model}__FIN__paper__0shot__seed42 -- FIN NER benchmark; not in either summary CSV (FIN not a col there)
            note = "FIN NER benchmark run (98-doc CoNLL-style set) -- not represented in either final_table CSV"
            missing_from_any.append((run_id, dataset, "BOTH", note))
            continue
        if run_id.startswith("finbert_ner__") or run_id.startswith("finbert_tone__"):
            note = "exploratory NER/tone track (entities.jsonl or tone-classifier) -- not part of the FiNER-ORD NER benchmark or sentiment final_table"
            missing_from_any.append((run_id, dataset, "BOTH", note))
            continue
        if "FPBall" in run_id:
            note = "FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA)"
            missing_from_any.append((run_id, dataset, "final_table.csv", note))
            continue

        if dataset == "FiNER-ORD":
            if not in_ner:
                missing_from_any.append((run_id, dataset, "final_table_ner.csv", f"key {key} not found in final_table_ner.csv"))
            continue

        if dataset in ("FPB", "FiQA"):
            if not in_sentiment:
                missing_from_any.append((run_id, dataset, "final_table.csv", f"key {key} not found in final_table.csv"))
            continue

        # anything else (FIN etc already handled above)
        if not in_sentiment and not in_ner:
            missing_from_any.append((run_id, dataset, "BOTH", f"key {key} not found in either summary CSV"))

    if missing_from_any:
        lines.append("| run_id | dataset | missing_from | note |")
        lines.append("|---|---|---|---|")
        for run_id, dataset, missing_from, note in missing_from_any:
            lines.append(f"| {run_id} | {dataset} | {missing_from} | {note} |")
    else:
        lines.append("_None._\n")

    # ---- CSV rows without a matching run directory ----
    lines.append("\n## Rows in summary CSVs with NO matching prediction directory\n")

    run_keys_set = set(rec["key"] for rec in run_records)

    lines.append("\n### final_table.csv\n")
    orphan_sentiment = [row for key, row in zip(sentiment_table_keys, sentiment_rows) if key not in run_keys_set]
    if orphan_sentiment:
        lines.append("| model | dataset | template | shots | n_samples |")
        lines.append("|---|---|---|---|---|")
        for row in orphan_sentiment:
            lines.append(f"| {row.get('model')} | {row.get('dataset')} | {row.get('template')} | {row.get('shots')} | {row.get('n_samples')} |")
    else:
        lines.append("_None -- every row has a matching run directory._\n")

    lines.append("\n### final_table_ner.csv\n")
    orphan_ner = [row for key, row in zip(ner_table_keys, ner_rows) if key not in run_keys_set]
    if orphan_ner:
        lines.append("| model | dataset | template | shots | n_samples |")
        lines.append("|---|---|---|---|---|")
        for row in orphan_ner:
            lines.append(f"| {row.get('model')} | {row.get('dataset')} | {row.get('template')} | {row.get('shots')} | {row.get('n_samples')} |")
    else:
        lines.append("_None -- every row has a matching run directory._\n")

    # ---- Extra note: n_samples discrepancy spotted during census ----
    lines.append("\n## Additional discrepancy noted during census\n")
    lines.append(
        "- `final_table_ner.csv` reports `n_samples=200` for `mistral7b`, `plutus8b`, and "
        "`qwen25_7b` on `FiNER-ORD`, but the corresponding `results/predictions/{model}__FiNER-ORD__A__0shot__seed42/` "
        "directories contain 300 lines in `predictions.jsonl` and `meta.json` / `progress.json` both say `n_total=300`. "
        "This suggests the NER summary aggregation step may have restricted to a 200-sample subset (or dropped rows) "
        "for the LLM rows while gliner-large/gliner-small kept 300. Flagging for the next review stage -- not resolved here.\n"
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Runs missing from summaries: {len(missing_from_any)}")
    print(f"Orphan rows in final_table.csv: {len(orphan_sentiment)}")
    print(f"Orphan rows in final_table_ner.csv: {len(orphan_ner)}")


if __name__ == "__main__":
    main()
