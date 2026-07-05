"""Entity-level F1 for NER, matching the Open-FinLLMs Table 7 metric.

Conventions:
- Strict match: a predicted entity counts as TP iff (lowercased text, type)
  appears in the gold set. Both case- and whitespace-normalised.
- Per-type P/R/F1 and a micro-F1 across all types are reported. The paper's
  headline "NER" score (0.57 for FinLLaMA, 0.80 for GPT-4) corresponds to
  micro-F1 restricted to {PER, ORG, LOC} (the types the prompt asks for).
- Coverage = fraction of predictions with parse_ok=True. Parse failures are
  excluded from the F1 computation, in line with the sentiment evaluator.
"""

from __future__ import annotations

from collections import defaultdict


def _norm(e: dict) -> tuple[str, str]:
    return (" ".join(e["text"].split()).lower(), e["type"].upper())


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def compute_ner_metrics(
    samples: list[dict],
    predictions: list[dict],
    only: set[str] | None = None,
) -> dict:
    """Compute strict entity-level P/R/F1 over aligned (sample, prediction) pairs.

    `predictions` items must carry:
        {"id": str, "pred_entities": list[Entity] | None, "parse_ok": bool}

    `only`: if given, only entity types in this set count (paper uses {PER,ORG,LOC}).
    """
    by_id = {p["id"]: p for p in predictions}
    total_tp = total_fp = total_fn = 0
    per_type_tp: dict[str, int] = defaultdict(int)
    per_type_fp: dict[str, int] = defaultdict(int)
    per_type_fn: dict[str, int] = defaultdict(int)

    n_eval = 0
    n_parse_fail = 0

    for s in samples:
        p = by_id.get(s["id"])
        if p is None or not p.get("parse_ok") or p.get("pred_entities") is None:
            n_parse_fail += 1
            continue
        n_eval += 1

        gold = {_norm(e) for e in s.get("entities", []) if (only is None or e["type"].upper() in only)}
        pred = {_norm(e) for e in (p["pred_entities"] or []) if (only is None or e["type"].upper() in only)}

        tp = gold & pred
        fp = pred - gold
        fn = gold - pred

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)
        for _, t in tp:
            per_type_tp[t] += 1
        for _, t in fp:
            per_type_fp[t] += 1
        for _, t in fn:
            per_type_fn[t] += 1

    micro_p, micro_r, micro_f1 = _prf(total_tp, total_fp, total_fn)

    per_type = {}
    macro_f1_sum = 0.0
    macro_n = 0
    for typ in sorted(set(per_type_tp) | set(per_type_fp) | set(per_type_fn)):
        p_, r_, f_ = _prf(per_type_tp[typ], per_type_fp[typ], per_type_fn[typ])
        per_type[typ] = {"precision": round(p_, 4), "recall": round(r_, 4), "f1": round(f_, 4),
                          "support": per_type_tp[typ] + per_type_fn[typ]}
        macro_f1_sum += f_
        macro_n += 1
    macro_f1 = macro_f1_sum / macro_n if macro_n else 0.0

    return {
        "n_samples": len(samples),
        "n_evaluated": n_eval,
        "n_parse_failures": n_parse_fail,
        "coverage": round(n_eval / max(len(samples), 1), 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_type": per_type,
        "restricted_to": sorted(only) if only else None,
    }
