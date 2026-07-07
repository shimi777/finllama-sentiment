"""
Subset-parity check for pre-submission review.

For each sentiment dataset (FPB, FiQA):
  - collect the id-set from every LLM run's predictions.jsonl
    (models: mistral7b, plutus8b, qwen25_7b; all template/shot cells)
  - verify all runs on the same dataset used the IDENTICAL 300-id set
  - write review/evidence/subset_ids_check.json
  - write review/evidence/subset_ids_FPB.json and subset_ids_FiQA.json
    (canonical sorted 300-id arrays)

Also checks that finbert/vader full-set runs (FPB 690, FiQA 1173) contain
ALL of the 300 subset ids (superset check).

Read-only w.r.t. results/. Only writes into review/evidence/.
"""
import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "results" / "predictions"
EVID_DIR = REPO_ROOT / "review" / "evidence"

LLM_MODELS = ["mistral7b", "plutus8b", "qwen25_7b"]
DATASETS = ["FPB", "FiQA"]
BASELINE_MODELS = ["finbert", "vader"]


def sha1_of_sorted_ids(ids):
    joined = "\n".join(sorted(str(x) for x in ids))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def load_ids(pred_path):
    ids = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.append(obj["id"])
    return ids


def find_llm_runs_for_dataset(dataset):
    """Return dict run_id -> Path(predictions.jsonl) for LLM runs matching
    model__DATASET__template__shots__seedNN exactly (not FPBall, not FiNER-ORD)."""
    runs = {}
    pattern = re.compile(
        r"^(?P<model>[a-zA-Z0-9_]+)__(?P<dataset>[A-Za-z-]+)__(?P<template>[A-Z])__(?P<shots>\d+)shot__seed(?P<seed>\d+)$"
    )
    for d in PRED_DIR.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        if m.group("model") not in LLM_MODELS:
            continue
        if m.group("dataset") != dataset:
            continue
        pred_file = d / "predictions.jsonl"
        if pred_file.exists():
            runs[d.name] = pred_file
    return runs


def find_baseline_run(model, dataset):
    run_dir = PRED_DIR / f"{model}__{dataset}__seed42"
    pred_file = run_dir / "predictions.jsonl"
    if pred_file.exists():
        return run_dir.name, pred_file
    return None, None


def main():
    result = {}
    canonical_sets = {}

    for dataset in DATASETS:
        runs = find_llm_runs_for_dataset(dataset)
        run_ids_sorted = sorted(runs.keys())
        print(f"[{dataset}] found {len(run_ids_sorted)} LLM runs: {run_ids_sorted}")

        per_run_idsets = {}
        for run_id, pred_file in runs.items():
            ids = load_ids(pred_file)
            per_run_idsets[run_id] = set(ids)
            if len(ids) != len(set(ids)):
                print(f"WARNING: {run_id} has duplicate ids in predictions.jsonl")

        # canonical = the most common id-set (by hash), or just first one if all same
        hash_to_runs = defaultdict(list)
        hash_to_set = {}
        for run_id, idset in per_run_idsets.items():
            h = sha1_of_sorted_ids(idset)
            hash_to_runs[h].append(run_id)
            hash_to_set[h] = idset

        # pick majority hash as canonical
        canonical_hash = max(hash_to_runs.keys(), key=lambda h: len(hash_to_runs[h])) if hash_to_runs else None
        canonical_set = hash_to_set[canonical_hash] if canonical_hash else set()
        all_identical = len(hash_to_runs) == 1

        deviations = []
        for h, run_ids in hash_to_runs.items():
            if h == canonical_hash:
                continue
            for run_id in run_ids:
                idset = per_run_idsets[run_id]
                sym_diff = idset.symmetric_difference(canonical_set)
                deviations.append({
                    "run_id": run_id,
                    "symmetric_difference_size": len(sym_diff),
                    "example_ids": sorted(sym_diff)[:10],
                })

        result[dataset] = {
            "n_runs_checked": len(run_ids_sorted),
            "run_ids_checked": run_ids_sorted,
            "all_identical": all_identical,
            "canonical_id_set_size": len(canonical_set),
            "canonical_id_set_sha1": canonical_hash,
            "deviating_runs": deviations,
        }

        canonical_sets[dataset] = sorted(canonical_set)

        # write canonical id list file
        out_path = EVID_DIR / f"subset_ids_{dataset}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sorted(canonical_set), f, indent=2)
        print(f"[{dataset}] canonical set size={len(canonical_set)}, all_identical={all_identical}, wrote {out_path}")

        # baseline superset check
        baseline_checks = {}
        for bmodel in BASELINE_MODELS:
            brun_id, bpred_file = find_baseline_run(bmodel, dataset)
            if bpred_file is None:
                baseline_checks[bmodel] = {"found": False}
                continue
            bids = set(load_ids(bpred_file))
            missing = canonical_set - bids
            baseline_checks[bmodel] = {
                "found": True,
                "run_id": brun_id,
                "n_baseline_ids": len(bids),
                "is_superset_of_canonical_300": len(missing) == 0,
                "n_missing_from_baseline": len(missing),
                "example_missing_ids": sorted(missing)[:10],
            }
        result[dataset]["baseline_superset_check"] = baseline_checks

    out_check = EVID_DIR / "subset_ids_check.json"
    with open(out_check, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_check}")


if __name__ == "__main__":
    main()
