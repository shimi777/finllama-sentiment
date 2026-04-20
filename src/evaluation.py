"""Compute metrics (accuracy, F1-macro/weighted) and confusion matrices."""

from __future__ import annotations

LABEL_ORDER = ["negative", "neutral", "positive"]


def compute_metrics(samples: list[dict], preds: list[dict]) -> dict:
    """Return a metrics dict from parallel lists of Sample and Prediction dicts.

    Parse failures (pred_label=None) are excluded from accuracy/F1 but counted
    in coverage. The caller must guarantee len(samples) == len(preds) and that
    samples[i].id == preds[i].id.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    if len(samples) != len(preds):
        raise ValueError("samples and preds must be the same length")

    true_labels: list[str] = []
    pred_labels: list[str] = []
    n_parse_ok = 0

    for s, p in zip(samples, preds):
        if s["id"] != p["id"]:
            raise ValueError(f"ID mismatch: {s['id']} vs {p['id']}")
        if p["parse_ok"] and p["pred_label"] is not None:
            true_labels.append(s["label"])
            pred_labels.append(p["pred_label"])
            n_parse_ok += 1

    coverage = n_parse_ok / len(samples) if samples else 0.0

    if not true_labels:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "per_class": {},
            "confusion": [],
            "coverage": coverage,
            "n_samples": len(samples),
        }

    accuracy = float(accuracy_score(true_labels, pred_labels))
    f1_macro = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
    f1_weighted = float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0))

    present_labels = sorted(set(true_labels) | set(pred_labels))
    p_arr, r_arr, f_arr, sup_arr = precision_recall_fscore_support(
        true_labels, pred_labels, labels=present_labels, zero_division=0
    )
    per_class = {
        lbl: {
            "precision": float(p_arr[i]),
            "recall": float(r_arr[i]),
            "f1": float(f_arr[i]),
            "support": int(sup_arr[i]),
        }
        for i, lbl in enumerate(present_labels)
    }

    # Confusion matrix ordered neg / neu / pos (only labels that appear)
    cm = confusion_matrix(true_labels, pred_labels, labels=present_labels)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": present_labels,
        "coverage": coverage,
        "n_samples": len(samples),
    }
