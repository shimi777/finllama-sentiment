"""
Census script for pre-submission review.
Reads results/predictions/*/{meta.json, predictions.jsonl or entities.jsonl, progress.json}
and writes review/evidence/run_inventory.csv.

Read-only: does not modify anything under results/.
"""
import json
import csv
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "results" / "predictions"
OUT_CSV = REPO_ROOT / "review" / "evidence" / "run_inventory.csv"


def sha1_of_sorted_ids(ids):
    joined = "\n".join(sorted(ids))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e)}


def load_jsonl_ids_and_lines(path):
    """Returns (n_lines, first_id, id_list, parse_error_note)"""
    if not path.exists():
        return 0, None, [], "file missing"
    ids = []
    n_lines = 0
    note = ""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
                _id = obj.get("id")
                ids.append(_id if _id is not None else f"__NOID_line{i}")
            except Exception as e:
                note = f"json parse error at line {i}: {e}"
    first_id = ids[0] if ids else None
    return n_lines, first_id, ids, note


def main():
    rows = []
    run_dirs = sorted([p for p in PRED_DIR.iterdir() if p.is_dir()])
    print(f"Found {len(run_dirs)} run directories under {PRED_DIR}", file=sys.stderr)

    for run_dir in run_dirs:
        run_id = run_dir.name
        meta = load_json(run_dir / "meta.json") or {}
        progress = load_json(run_dir / "progress.json")

        # Determine which predictions file this run uses
        pred_file = run_dir / "predictions.jsonl"
        entities_file = run_dir / "entities.jsonl"
        notes = []

        if pred_file.exists():
            n_lines, first_id, ids, parse_note = load_jsonl_ids_and_lines(pred_file)
            pred_filename_used = "predictions.jsonl"
        elif entities_file.exists():
            n_lines, first_id, ids, parse_note = load_jsonl_ids_and_lines(entities_file)
            pred_filename_used = "entities.jsonl"
            notes.append("exploratory NER track: uses entities.jsonl, no progress.json")
        else:
            n_lines, first_id, ids, parse_note = 0, None, [], "no predictions.jsonl or entities.jsonl found"
            pred_filename_used = "NONE"

        if parse_note:
            notes.append(parse_note)

        id_hash = sha1_of_sorted_ids([str(x) for x in ids]) if ids else ""

        # complete-ness from progress.json
        if progress is None:
            complete = "" if pred_filename_used == "entities.jsonl" else "FALSE(no progress.json)"
            if pred_filename_used == "entities.jsonl":
                complete = "N/A"
        elif "__error__" in progress:
            complete = f"ERROR:{progress['__error__']}"
        else:
            last_idx = progress.get("last_completed_idx")
            n_total_p = progress.get("n_total")
            if last_idx is not None and n_total_p is not None:
                complete = str(last_idx >= n_total_p)
                if last_idx != n_total_p:
                    notes.append(f"progress last_completed_idx={last_idx} != n_total={n_total_p}")
            else:
                complete = "UNKNOWN"

        if meta is None:
            notes.append("meta.json MISSING")
            meta = {}
        elif "__error__" in meta:
            notes.append(f"meta.json parse error: {meta['__error__']}")

        n_samples_meta = meta.get("n_total", meta.get("n_samples", ""))

        # cross-check n_lines vs meta n_total
        if isinstance(n_samples_meta, int) and n_lines != n_samples_meta:
            notes.append(f"n_lines({n_lines}) != meta.n_total({n_samples_meta})")

        model = meta.get("model_hf_id", "")
        dataset = meta.get("dataset", "")
        template = meta.get("template", "")
        shots = meta.get("shots", "")
        seed = meta.get("seed", "")
        started_at = meta.get("started_at", "")
        completed_at = meta.get("completed_at", "")
        timestamps = ""
        if started_at or completed_at:
            timestamps = f"started={started_at};completed={completed_at}"
        reparsed_at = meta.get("reparsed_at")
        if reparsed_at:
            timestamps += f";reparsed={reparsed_at}"

        task = meta.get("task", "")
        backend = meta.get("backend", "")
        if task:
            notes.append(f"task={task}")
        if backend:
            notes.append(f"backend={backend}")

        rows.append({
            "run_id": run_id,
            "model": model,
            "dataset": dataset,
            "template": template if template is not None else "",
            "shots": shots if shots is not None else "",
            "seed": seed if seed is not None else "",
            "n_samples_meta": n_samples_meta,
            "n_predictions_lines": n_lines,
            "complete": complete,
            "first_pred_id": first_id if first_id is not None else "",
            "sha1_of_sorted_id_list": id_hash,
            "timestamps": timestamps,
            "notes": "; ".join(notes),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "model", "dataset", "template", "shots", "seed",
        "n_samples_meta", "n_predictions_lines", "complete",
        "first_pred_id", "sha1_of_sorted_id_list", "timestamps", "notes",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
