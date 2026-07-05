"""Re-parse and re-score existing NER runs without re-calling the LLM.

Useful when the parser is improved after the fact. Reads raw_output from
each `ner__*/predictions.jsonl`, re-runs `parse_ner`, recomputes metrics,
and updates `meta.json` + `predictions.jsonl` in place.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ner_parser import parse_ner            # noqa: E402
from src.ner_evaluation import compute_ner_metrics  # noqa: E402
from src.ner_loader import load_fin_alvarado    # noqa: E402


def reeval_run(run_dir: str, samples_by_id: dict) -> dict | None:
    meta_path = os.path.join(run_dir, "meta.json")
    pred_path = os.path.join(run_dir, "predictions.jsonl")
    if not (os.path.exists(meta_path) and os.path.exists(pred_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    with open(pred_path) as f:
        preds = [json.loads(l) for l in f]

    new_preds = []
    for p in preds:
        ents = parse_ner(p.get("raw_output", ""))
        new_preds.append({
            **p,
            "pred_entities": ents,
            "parse_ok": ents is not None,
        })

    samples = [samples_by_id[p["id"]] for p in new_preds if p["id"] in samples_by_id]
    metrics = compute_ner_metrics(samples, new_preds, only={"PER", "ORG", "LOC"})

    meta["metrics"] = metrics
    meta["reparsed_at"] = datetime.now(timezone.utc).isoformat()

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in new_preds:
            f.write(json.dumps(p) + "\n")
    return metrics


def main():
    samples = load_fin_alvarado("TheFinAI/flare-ner", "test")
    by_id = {s["id"]: s for s in samples}

    pred_dir = os.path.join(ROOT, "results", "predictions")
    for name in sorted(os.listdir(pred_dir)):
        if not name.startswith("ner__"):
            continue
        run_dir = os.path.join(pred_dir, name)
        m = reeval_run(run_dir, by_id)
        if m is None:
            print(f"[skip] {name}")
            continue
        print(f"[reeval] {name}: micro_f1={m['micro_f1']:.3f} "
              f"cov={m['coverage']:.2f} parse_fail={m['n_parse_failures']}")


if __name__ == "__main__":
    main()
