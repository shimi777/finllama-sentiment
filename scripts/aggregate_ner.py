"""Walk results/predictions/*FiNER-ORD*/ and write summary_ner/final_table_ner.csv.

For each NER run dir it joins predictions back to FiNER-ORD gold by id,
computes strict/partial/type-only F1 (src.ner.evaluation.compute_ner_metrics),
and writes:

    results/summary_ner/final_table_ner.csv
    results/summary_ner/confusions/{run_id}.json

Scope note: this script is authoritative for the FiNER-ORD benchmark track
(GLiNER vs LLMs, glob "*FiNER-ORD*") and reproduces the committed
final_table_ner.csv byte-identically. It does NOT touch the separate,
exploratory "FIN/paper" NER track (run dirs named "ner__*__FIN__paper__*"),
which is re-scored by scripts/reeval_ner.py instead.

Usage: py scripts/aggregate_ner.py
"""

from __future__ import annotations

import json
import sys
from glob import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ner.cost import MODEL_PRICING  # noqa: E402
from src.ner.data_loader import load_finer_ord  # noqa: E402
from src.ner.evaluation import compute_ner_metrics  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402

logger = get_logger("aggregate_ner")


def _load_meta(run_dir: Path) -> dict | None:
    p = run_dir / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_preds(run_dir: Path) -> list[dict]:
    p = run_dir / "predictions.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> None:
    set_seed(42)

    # Load gold full FiNER-ORD test once — runs may have sampled a subset, that's fine.
    gold_list = load_finer_ord(split="test", max_samples=None, seed=42)
    gold = {s["id"]: s for s in gold_list}

    pred_root = ROOT / "results" / "predictions"
    summary_dir = ROOT / "results" / "summary_ner"
    conf_dir = summary_dir / "confusions"
    summary_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir_str in sorted(glob(str(pred_root / "*FiNER-ORD*"))):
        run_dir = Path(run_dir_str)
        meta = _load_meta(run_dir)
        if meta is None:
            continue
        preds = _load_preds(run_dir)
        if not preds:
            continue

        samples, kept_preds = [], []
        for p in preds:
            s = gold.get(p["id"])
            if s is None:
                continue
            samples.append(s)
            kept_preds.append(p)
        if not samples:
            continue

        m = compute_ner_metrics(samples, kept_preds)

        # Persist per-run confusion + per-type breakdown.
        (conf_dir / f"{meta['run_id']}.json").write_text(json.dumps({
            "run_id": meta["run_id"],
            "n_samples": m["n_samples"],
            "coverage": m["coverage"],
            "strict": m["strict"],
            "partial": m["partial"],
            "type_only": m["type_only"],
            "per_type": m["per_type"],
            "confusion_by_type": m["confusion_by_type"],
        }, indent=2))

        # Friendly short model name (first segment of run_id).
        model_short = meta["run_id"].split("__")[0]
        price_in, price_out = MODEL_PRICING.get(model_short, (0.0, 0.0))

        # Aggregate cost / tokens / latency from predictions.
        tot_cost = sum(p.get("cost_usd", 0.0) for p in kept_preds)
        tot_in = sum(p.get("input_tokens", 0) for p in kept_preds)
        tot_out = sum(p.get("output_tokens", 0) for p in kept_preds)
        avg_latency = (
            sum(p.get("latency_ms", 0.0) for p in kept_preds) / len(kept_preds)
            if kept_preds else 0.0
        )

        rows.append({
            "model": model_short,
            "model_hf_id": meta.get("model_hf_id", ""),
            "dataset": meta.get("dataset", "FiNER-ORD"),
            "template": meta.get("template") or "-",
            "shots": meta.get("shots") if meta.get("shots") is not None else 0,
            "seed": meta.get("seed", 42),
            "backend": meta.get("backend", ""),
            "n_samples": m["n_samples"],
            "coverage": round(m["coverage"], 4),
            "strict_p": round(m["strict"]["precision"], 4),
            "strict_r": round(m["strict"]["recall"], 4),
            "strict_f1": round(m["strict"]["f1"], 4),
            "partial_f1": round(m["partial"]["f1"], 4),
            "type_only_f1": round(m["type_only"]["f1"], 4),
            "f1_PER": round(m["per_type"]["PER"]["strict"]["f1"], 4),
            "f1_LOC": round(m["per_type"]["LOC"]["strict"]["f1"], 4),
            "f1_ORG": round(m["per_type"]["ORG"]["strict"]["f1"], 4),
            "runtime_s": meta.get("runtime_s", 0.0),
            "avg_latency_ms": round(avg_latency, 2),
            "input_tokens": tot_in,
            "output_tokens": tot_out,
            "cost_usd": round(tot_cost, 6),
            "price_in_per_M": price_in,
            "price_out_per_M": price_out,
        })

    if not rows:
        logger.warning("No NER runs found under %s", pred_root)
        return

    df = pd.DataFrame(rows).sort_values(
        ["dataset", "model", "template", "shots"]
    ).reset_index(drop=True)
    out_csv = summary_dir / "final_table_ner.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(df))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
