"""Prompt-ensemble PoC: vote over the existing A/B x 0/3-shot cells per model.

For every (model, dataset) we already have four single-prompt runs:
    {A,B} x {0-shot, 3-shot}
This script aggregates them example-by-example (src.ensemble.aggregate) and compares
the ensemble against the *mean*, *worst*, and *best* single prompt — the last being
the leakage-requiring upper bound you could only reach by peeking at test. No new
model inference is needed: it reads results/predictions/ only.

Three aggregation methods are evaluated:
  - unweighted     : plain majority vote (with a tie-break ablation: abstain/order/neutral).
  - cv_weighted    : LEAKAGE-FREE soft vote. Per-member weights are estimated by k-fold
                     CV — each example is scored with weights learned only on the *other*
                     folds, so a member's weight never sees the example it judges.
  - oracle_weighted: soft vote with weights = member accuracy on the FULL eval set.
                     This USES TEST LABELS and is reported only as an upper-bound ceiling,
                     never as a deployable result.

Outputs:
  - results/summary/ensemble_table.csv         (one row per model x dataset x method x tie_break)
  - results/summary/confusions_ensemble/*.json (unweighted + cv_weighted, abstain)

Usage: .venv/Scripts/python.exe scripts/aggregate_ensemble.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import load_fpb, load_fiqa  # noqa: E402
from src.ensemble import aggregate  # noqa: E402
from src.evaluation import compute_metrics  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402

logger = get_logger("aggregate_ensemble")

MODELS = ["mistral7b", "plutus8b", "qwen25_7b"]
DATASETS = ["FPB", "FiQA"]
# Named ensembles, each a list of members (template, shots). An ensemble is computed
# only when ALL its members have prediction runs on disk — so E3/E5 (which need the
# 0-shot template C) appear automatically once that Colab/Modal pass has run.
#   E4 — the free baseline: A/B x {0,3}-shot (always available).
#   E3 — the clean headline: diverse 0-shot triad {A,B,C} (needs template C runs).
#   E5 — extended: 0-shot triad + the few-shot members.
ENSEMBLES: dict[str, list[tuple[str, int]]] = {
    "E4": [("A", 0), ("A", 3), ("B", 0), ("B", 3)],
    "E3": [("A", 0), ("B", 0), ("C", 0)],
    "E5": [("A", 0), ("B", 0), ("C", 0), ("A", 3), ("B", 3)],
}
HEADLINE_ENSEMBLE = "E4"  # the one always present; used for the detailed printouts
SEED = 42
# Tie-break policies to compare. "abstain" is the honest primary (parse-failure
# semantics); "order"/"neutral" recover the coverage abstain throws away.
TIE_BREAKS = ["abstain", "order", "neutral"]
PRIMARY_TIE_BREAK = "abstain"
N_FOLDS = 5


def load_preds_by_id(run_id: str) -> dict[str, dict] | None:
    path = os.path.join(ROOT, "results", "predictions", run_id, "predictions.jsonl")
    if not os.path.exists(path):
        return None
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                out[p["id"]] = p
    return out


def member_accuracy(ids_subset: list[str], preds_k: dict[str, dict], gold: dict) -> float:
    """Fraction of ids_subset the member gets right (parse failure counts as wrong)."""
    if not ids_subset:
        return 0.0
    correct = sum(
        1 for i in ids_subset
        if preds_k[i].get("parse_ok") and preds_k[i].get("pred_label") == gold[i]["label"]
    )
    return correct / len(ids_subset)


def cv_weight_map(ids: list[str], keys: list[str], member_preds: dict, gold: dict,
                  k: int = N_FOLDS) -> dict[str, list[float]]:
    """Leakage-free per-example weights via positional k-fold (ids are pre-sorted).

    For fold f, weights come from member accuracy on the *training* ids (all folds
    but f); those weights are assigned to the held-out fold's ids.
    """
    id_weights: dict[str, list[float]] = {}
    n = len(ids)
    for f in range(k):
        test_pos = set(range(f, n, k))
        train_ids = [ids[p] for p in range(n) if p not in test_pos]
        w = [member_accuracy(train_ids, member_preds[key], gold) for key in keys]
        for p in range(f, n, k):
            id_weights[ids[p]] = w
    return id_weights


def build_ensemble(ids, keys, member_preds, tie_break, id_weights=None):
    out = []
    for i in ids:
        votes = [member_preds[k][i] for k in keys]
        w = id_weights[i] if id_weights is not None else None
        out.append(aggregate(votes, tie_break=tie_break, weights=w))
    return out


def main() -> None:
    set_seed(SEED)

    _, fpb_test = load_fpb(seed=SEED)
    fiqa_test = load_fiqa()
    gold = {s["id"]: s for s in (fpb_test + fiqa_test)}

    conf_dir = os.path.join(ROOT, "results", "summary", "confusions_ensemble")
    os.makedirs(conf_dir, exist_ok=True)

    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            for ens_name, members in ENSEMBLES.items():
                run_ids = {
                    f"{tmpl}{sh}": f"{model}__{dataset}__{tmpl}__{sh}shot__seed{SEED}"
                    for (tmpl, sh) in members
                }
                keys = list(run_ids)
                member_preds = {k: load_preds_by_id(rid) for k, rid in run_ids.items()}
                missing = [k for k, v in member_preds.items() if v is None]
                if missing:
                    # Quietly skip ensembles whose members aren't on disk yet (e.g. E3/E5
                    # before the template-C pass); warn only for the always-present headline.
                    if ens_name == HEADLINE_ENSEMBLE:
                        logger.warning("Skipping %s %s/%s — missing members %s",
                                       ens_name, model, dataset, missing)
                    continue

                # Common ids across all members, that also have gold (deterministic order)
                common = set.intersection(*(set(v) for v in member_preds.values())) & set(gold)
                ids = sorted(common)
                if not ids:
                    logger.warning("Skipping %s %s/%s — no common ids", ens_name, model, dataset)
                    continue
                samples = [gold[i] for i in ids]

                # Per-member single-prompt metrics (independent of aggregation)
                single_acc, single_f1 = {}, {}
                for k, preds in member_preds.items():
                    m = compute_metrics(samples, [preds[i] for i in ids])
                    single_acc[k] = m["accuracy"]
                    single_f1[k] = m["f1_macro"]
                accs = list(single_acc.values())
                f1s = list(single_f1.values())
                mean_acc, mean_f1, best_f1 = statistics.mean(accs), statistics.mean(f1s), max(f1s)

                # Weight maps
                cv_w = cv_weight_map(ids, keys, member_preds, gold)
                oracle_w_vec = [member_accuracy(ids, member_preds[k], gold) for k in keys]
                oracle_w = {i: oracle_w_vec for i in ids}

                # (method, tie_break, id_weights)
                plans = [("unweighted", tb, None) for tb in TIE_BREAKS]
                plans.append(("cv_weighted", PRIMARY_TIE_BREAK, cv_w))
                plans.append(("oracle_weighted", PRIMARY_TIE_BREAK, oracle_w))

                for method, tie_break, id_weights in plans:
                    ens_preds = build_ensemble(ids, keys, member_preds, tie_break, id_weights)
                    em = compute_metrics(samples, ens_preds)

                    # Persist confusion for the two deployable headline variants
                    if tie_break == PRIMARY_TIE_BREAK and method in ("unweighted", "cv_weighted"):
                        tag = ens_name if method == "unweighted" else f"{ens_name}_cvw"
                        with open(os.path.join(conf_dir, f"{model}__{dataset}__{tag}.json"), "w") as f:
                            json.dump({
                                "run_id": f"{model}__{dataset}__{tag}",
                                "ensemble": ens_name,
                                "method": method,
                                "members": list(run_ids.values()),
                                "tie_break": tie_break,
                                "labels": em.get("confusion_labels", []),
                                "matrix": em["confusion"],
                                "per_class": em["per_class"],
                                "n_samples": em["n_samples"],
                                "coverage": em["coverage"],
                            }, f, indent=2)

                    rows.append({
                        "ensemble": ens_name,
                        "model": model,
                        "dataset": dataset,
                        "method": method,
                        "tie_break": tie_break,
                        "n_members": len(keys),
                        "n_samples": em["n_samples"],
                        "ens_acc": round(em["accuracy"], 4),
                        "ens_f1": round(em["f1_macro"], 4),
                        "ens_cov": round(em["coverage"], 4),
                        "single_mean_acc": round(mean_acc, 4),
                        "single_worst_acc": round(min(accs), 4),
                        "single_best_acc": round(max(accs), 4),
                        "single_std_acc": round(statistics.pstdev(accs), 4),
                        "single_mean_f1": round(mean_f1, 4),
                        "single_worst_f1": round(min(f1s), 4),
                        "single_best_f1": round(best_f1, 4),
                        "d_acc_vs_mean": round(em["accuracy"] - mean_acc, 4),
                        "d_acc_vs_best": round(em["accuracy"] - max(accs), 4),
                        "d_f1_vs_mean": round(em["f1_macro"] - mean_f1, 4),
                        "d_f1_vs_best": round(em["f1_macro"] - best_f1, 4),
                    })

    if not rows:
        logger.warning("No ensemble rows produced")
        return

    df = pd.DataFrame(rows).sort_values(
        ["dataset", "model", "ensemble", "method", "tie_break"]
    ).reset_index(drop=True)
    out_csv = os.path.join(ROOT, "results", "summary", "ensemble_table.csv")
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(df))

    present = [e for e in ENSEMBLES if (df["ensemble"] == e).any()]
    head = df[df["ensemble"] == HEADLINE_ENSEMBLE]

    # --- Primary view: headline ensemble, unweighted majority, abstain ---
    prim = head[(head["method"] == "unweighted") & (head["tie_break"] == PRIMARY_TIE_BREAK)]
    print(f"\n=== {HEADLINE_ENSEMBLE} unweighted majority vote, tie->{PRIMARY_TIE_BREAK} ===\n")
    print(prim[[
        "model", "dataset", "ens_f1", "single_mean_f1", "single_best_f1",
        "d_f1_vs_mean", "d_f1_vs_best", "ens_acc", "single_std_acc", "ens_cov",
    ]].to_string(index=False))
    n = len(prim)
    print(f"\nCells where ensemble F1 > mean single prompt: {(prim['d_f1_vs_mean'] > 0).sum()}/{n}")
    print(f"Cells where ensemble F1 >= best single prompt: {(prim['d_f1_vs_best'] >= 0).sum()}/{n}")
    print(f"Mean cross-prompt accuracy std (single): {prim['single_std_acc'].mean():.4f}")
    print(f"Min ensemble coverage: {prim['ens_cov'].min():.4f}")

    # --- Method comparison at the abstain tie-break (headline ensemble) ---
    print("\n=== Method comparison (tie->abstain): does weighting reach the oracle? ===\n")
    mc = head[head["tie_break"] == PRIMARY_TIE_BREAK]
    pivot = mc.pivot_table(index=["dataset", "model"], columns="method",
                           values="ens_f1").reset_index()
    pivot["best_single_f1"] = mc.groupby(["dataset", "model"])["single_best_f1"].first().values
    cols = ["dataset", "model", "unweighted", "cv_weighted", "oracle_weighted", "best_single_f1"]
    print(pivot[cols].to_string(index=False))
    for method in ["unweighted", "cv_weighted", "oracle_weighted"]:
        sub = mc[mc["method"] == method]
        ge_best = int((sub["d_f1_vs_best"] >= -1e-9).sum())
        gt_mean = int((sub["d_f1_vs_mean"] > 0).sum())
        print(f"  {method:16s}: F1>=best in {ge_best}/{len(sub)} cells, "
              f"F1>mean in {gt_mean}/{len(sub)}, mean F1={sub['ens_f1'].mean():.4f}, "
              f"mean cov={sub['ens_cov'].mean():.4f}")

    # --- Tie-break ablation (headline ensemble, unweighted) ---
    print(f"\n=== Tie-break ablation ({HEADLINE_ENSEMBLE} unweighted, averaged over cells) ===\n")
    ab = head[head["method"] == "unweighted"].groupby("tie_break").agg(
        mean_ens_f1=("ens_f1", "mean"),
        mean_ens_acc=("ens_acc", "mean"),
        mean_ens_cov=("ens_cov", "mean"),
        cells_ge_best=("d_f1_vs_best", lambda s: int((s >= -1e-9).sum())),
    ).reindex(TIE_BREAKS).round(4)
    print(ab.to_string())

    # --- Cross-ensemble comparison (unweighted abstain + cv_weighted) ---
    print(f"\n=== Cross-ensemble comparison (members present: {', '.join(present)}) ===\n")
    for ens_name in present:
        for method in ("unweighted", "cv_weighted"):
            sub = df[(df["ensemble"] == ens_name) & (df["method"] == method)
                     & (df["tie_break"] == PRIMARY_TIE_BREAK)]
            if sub.empty:
                continue
            ge_best = int((sub["d_f1_vs_best"] >= -1e-9).sum())
            gt_mean = int((sub["d_f1_vs_mean"] > 0).sum())
            print(f"  {ens_name:3s} {method:12s}: mean F1={sub['ens_f1'].mean():.4f}, "
                  f"mean cov={sub['ens_cov'].mean():.4f}, "
                  f"F1>=best {ge_best}/{len(sub)}, F1>mean {gt_mean}/{len(sub)}")
    if present == [HEADLINE_ENSEMBLE]:
        print(f"\n  (Only {HEADLINE_ENSEMBLE} available. Run template C on the matrix "
              f"to unlock E3/E5 — see scripts/run_llm_matrix.py.)")


if __name__ == "__main__":
    main()
