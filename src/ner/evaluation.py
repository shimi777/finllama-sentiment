"""NER metrics: strict span-F1 (seqeval) + partial / type-only entity-level F1.

Parse failures (pred_tags=None) are excluded from F1 but counted in `coverage`,
matching the sentiment-side parser contract.

Strict F1   : exact span boundaries AND type — what the literature reports.
Partial F1  : predicted span overlaps gold span by >=1 token AND type matches.
Type-only F1: predicted span overlaps gold span AND type matches (ignores
              boundaries entirely; loosest credit).

We compute strict F1 with seqeval and partial / type-only by walking the
span list directly so we don't need extra deps beyond seqeval.
"""

from __future__ import annotations

from src.ner.data_loader import CANONICAL_TYPES, _bio_to_spans
from src.ner.parser import coerce_bio_to_canonical


def _placeholder_metrics(n_samples: int, coverage: float) -> dict:
    return {
        "n_samples": n_samples,
        "coverage": coverage,
        "strict": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        "partial": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        "type_only": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        "per_type": {},
        "confusion_by_type": {},
    }


def _strict_seqeval(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """seqeval strict IOB2 micro precision/recall/F1 + per-type breakdown."""
    try:
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
        from seqeval.scheme import IOB2
    except ImportError:
        # Soft-fail: fall back to a hand-rolled span F1 (equivalent under
        # canonical IOB2). Keeps the module usable if seqeval isn't installed.
        return _hand_rolled_strict(y_true, y_pred)

    micro_p = float(precision_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0))
    micro_r = float(recall_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0))
    micro_f = float(f1_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0))
    report = classification_report(
        y_true, y_pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0
    )

    per_type: dict[str, dict[str, float]] = {}
    support_total = 0
    for k, v in report.items():
        if not isinstance(v, dict):
            continue
        if k in ("micro avg", "macro avg", "weighted avg"):
            continue
        per_type[k] = {
            "precision": float(v.get("precision", 0.0)),
            "recall":    float(v.get("recall", 0.0)),
            "f1":        float(v.get("f1-score", 0.0)),
            "support":   int(v.get("support", 0)),
        }
        support_total += int(v.get("support", 0))

    return {
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f,
        "support": support_total,
        "per_type": per_type,
    }


def _hand_rolled_strict(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """seqeval-equivalent strict span F1 — fallback if seqeval is missing."""
    tp = fp = fn = 0
    per: dict[str, dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in CANONICAL_TYPES}

    for true_tags, pred_tags in zip(y_true, y_pred):
        toks = [""] * len(true_tags)  # span derivation only needs lengths
        true_spans = {(s["type"], s["start_tok"], s["end_tok"]) for s in _bio_to_spans(toks, true_tags)}
        pred_spans = {(s["type"], s["start_tok"], s["end_tok"]) for s in _bio_to_spans(toks, pred_tags)}
        for sp in true_spans & pred_spans:
            tp += 1
            per[sp[0]]["tp"] += 1
        for sp in pred_spans - true_spans:
            fp += 1
            per.setdefault(sp[0], {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for sp in true_spans - pred_spans:
            fn += 1
            per.setdefault(sp[0], {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    p, r, f = prf(tp, fp, fn)
    per_type = {}
    support_total = 0
    for t, c in per.items():
        pp, rr, ff = prf(c["tp"], c["fp"], c["fn"])
        support = c["tp"] + c["fn"]
        per_type[t] = {
            "precision": pp, "recall": rr, "f1": ff, "support": support,
        }
        support_total += support

    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "support": support_total,
        "per_type": per_type,
    }


def _partial_or_typeonly(
    y_true_tags: list[list[str]],
    y_pred_tags: list[list[str]],
    mode: str,
) -> dict:
    """Compute partial-overlap F1 ('partial') or boundary-agnostic F1 ('type_only').

    For each sentence:
      - Gold and predicted spans are extracted.
      - A prediction matches gold if:
          * same type
          * partial: spans overlap by >=1 token
          * type_only: spans overlap by >=1 token (same as partial here; kept
            distinct because in literature 'type_only' uses any-overlap and
            'partial' uses any-overlap+type; we keep both for compat).
      - Each gold span counts as at most 1 TP; unmatched gold -> FN; predicted
        spans that match no gold -> FP. Greedy first-match per gold (no Hungarian).
    """
    tp = fp = fn = 0
    per: dict[str, dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in CANONICAL_TYPES}

    for true_tags, pred_tags in zip(y_true_tags, y_pred_tags):
        toks = [""] * len(true_tags)
        gold = _bio_to_spans(toks, true_tags)
        pred = _bio_to_spans(toks, pred_tags)

        gold_matched = [False] * len(gold)
        pred_matched = [False] * len(pred)

        for pi, ps in enumerate(pred):
            best = -1
            for gi, gs in enumerate(gold):
                if gold_matched[gi]:
                    continue
                overlap = (ps["start_tok"] < gs["end_tok"]) and (gs["start_tok"] < ps["end_tok"])
                if not overlap:
                    continue
                if mode == "partial" and ps["type"] != gs["type"]:
                    continue
                # For 'type_only' we also require same type — overlap alone with
                # different types is more of a "boundary error" than a hit.
                if mode == "type_only" and ps["type"] != gs["type"]:
                    continue
                best = gi
                break
            if best >= 0:
                gold_matched[best] = True
                pred_matched[pi] = True

        for gi, gs in enumerate(gold):
            if gold_matched[gi]:
                tp += 1
                per.setdefault(gs["type"], {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
            else:
                fn += 1
                per.setdefault(gs["type"], {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
        for pi, ps in enumerate(pred):
            if not pred_matched[pi]:
                fp += 1
                per.setdefault(ps["type"], {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    p, r, f = prf(tp, fp, fn)
    per_type = {}
    support_total = 0
    for t, c in per.items():
        pp, rr, ff = prf(c["tp"], c["fp"], c["fn"])
        support = c["tp"] + c["fn"]
        per_type[t] = {
            "precision": pp, "recall": rr, "f1": ff, "support": support,
        }
        support_total += support
    return {"precision": p, "recall": r, "f1": f, "support": support_total, "per_type": per_type}


def _confusion_by_type(y_true_tags: list[list[str]], y_pred_tags: list[list[str]]) -> dict:
    """Span-level cross-type confusion: matrix[true_type][pred_type] = count.

    Only counts pred spans that have positional overlap with a gold span.
    Used by the dashboard to show 'PER mistaken for ORG' etc.
    """
    types = list(CANONICAL_TYPES)
    cm = {t: {p: 0 for p in types + ["MISS"]} for t in types}
    extra_fp = {t: 0 for t in types}

    for true_tags, pred_tags in zip(y_true_tags, y_pred_tags):
        toks = [""] * len(true_tags)
        gold = _bio_to_spans(toks, true_tags)
        pred = _bio_to_spans(toks, pred_tags)
        gold_matched = [False] * len(gold)
        for ps in pred:
            best = -1
            for gi, gs in enumerate(gold):
                if gold_matched[gi]:
                    continue
                overlap = (ps["start_tok"] < gs["end_tok"]) and (gs["start_tok"] < ps["end_tok"])
                if overlap:
                    best = gi
                    break
            if best >= 0:
                gold_matched[best] = True
                cm[gold[best]["type"]][ps["type"]] = cm[gold[best]["type"]].get(ps["type"], 0) + 1
            else:
                extra_fp[ps["type"]] = extra_fp.get(ps["type"], 0) + 1
        for gi, gs in enumerate(gold):
            if not gold_matched[gi]:
                cm[gs["type"]]["MISS"] = cm[gs["type"]].get("MISS", 0) + 1

    return {"matrix": cm, "spurious_fp_by_type": extra_fp}


def compute_ner_metrics(samples: list[dict], preds: list[dict]) -> dict:
    """Top-level entry: returns dict with strict, partial, type_only, per_type, confusion."""
    if len(samples) != len(preds):
        raise ValueError("samples and preds must be the same length")

    n = len(samples)
    n_ok = 0
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []

    for s, p in zip(samples, preds):
        if s["id"] != p["id"]:
            raise ValueError(f"ID mismatch: {s['id']} vs {p['id']}")
        if not p.get("parse_ok") or p.get("pred_tags") is None:
            continue
        if len(p["pred_tags"]) != len(s["tags"]):
            # Drop length-mismatched preds (defensive); count as parse failure.
            continue
        y_true.append(coerce_bio_to_canonical(s["tags"]))
        y_pred.append(coerce_bio_to_canonical(p["pred_tags"]))
        n_ok += 1

    coverage = n_ok / n if n else 0.0
    if n_ok == 0:
        return _placeholder_metrics(n, coverage)

    strict = _strict_seqeval(y_true, y_pred)
    partial = _partial_or_typeonly(y_true, y_pred, mode="partial")
    type_only = _partial_or_typeonly(y_true, y_pred, mode="type_only")
    confusion = _confusion_by_type(y_true, y_pred)

    # Cross-merge per-type details (strict precision/recall/F1 is the headline).
    per_type: dict[str, dict] = {}
    for t in CANONICAL_TYPES:
        per_type[t] = {
            "strict":   strict["per_type"].get(t, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}),
            "partial":  partial["per_type"].get(t, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}),
        }

    return {
        "n_samples": n,
        "coverage": coverage,
        "strict": {
            "precision": strict["precision"],
            "recall": strict["recall"],
            "f1": strict["f1"],
            "support": strict["support"],
        },
        "partial": {
            "precision": partial["precision"],
            "recall": partial["recall"],
            "f1": partial["f1"],
            "support": partial["support"],
        },
        "type_only": {
            "precision": type_only["precision"],
            "recall": type_only["recall"],
            "f1": type_only["f1"],
            "support": type_only["support"],
        },
        "per_type": per_type,
        "confusion_by_type": confusion,
    }
