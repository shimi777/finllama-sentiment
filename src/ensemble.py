"""Prompt-ensemble voting: combine several per-prompt predictions into one label.

Motivation: single-prompt results in this project swing by up to ~26 accuracy
points on the same model purely from prompt choice (see results/summary/final_table.csv).
Picking the "best" prompt requires peeking at the test labels — that is leakage.
A majority vote over several fixed, diverse prompts removes the prompt-selection
gamble without ever looking at the test set.

A *vote set* is the list of per-prompt Prediction dicts for ONE example (same id),
one per ensemble member. ``aggregate`` reduces it to a single Prediction dict that is
drop-in compatible with ``src.evaluation.compute_metrics`` — same id / pred_label /
parse_ok contract, including the invariant that a parse/abstain failure
(``parse_ok=False``, ``pred_label=None``) is excluded from accuracy and only counted
in coverage. We never force an abstention to a label.
"""

from __future__ import annotations

from collections import Counter, defaultdict

# Deterministic priority used only when tie_break="order".
_ORDER = ["negative", "neutral", "positive"]
# Float tolerance for declaring a weighted-score tie.
_TIE_TOL = 1e-9


def aggregate(
    votes: list[dict],
    tie_break: str = "abstain",
    weights: list[float] | None = None,
) -> dict:
    """Reduce per-prompt votes for one example to a single (weighted) majority Prediction.

    Args:
        votes: Prediction dicts, all sharing the same ``id``. Members with
            ``parse_ok=False`` / ``pred_label=None`` abstain (do not vote), which
            preserves the project's "never force a parse failure to a label" rule.
        tie_break: what to do when the top label is not unique:
            ``"abstain"`` (default) -> ``parse_ok=False`` (counts against coverage);
            ``"neutral"``           -> resolve to "neutral" (majority-class prior);
            ``"order"``             -> first tied label by [negative, neutral, positive].
        weights: optional per-member weights, parallel to ``votes`` (same length).
            ``None`` (default) means equal weight 1.0 each — plain majority vote.
            With weights, the winner is the label with the largest summed weight
            over its (non-abstaining) members; this is "soft"/confidence voting and
            lets weak prompts be down-weighted instead of dragging the vote to the mean.

    Returns:
        A Prediction dict with the standard keys
        ``{id, pred_label, raw_output, parse_ok, latency_ms}`` plus ensemble
        diagnostics ``{n_votes, n_valid, vote_dist}``. ``latency_ms`` is the sum
        across members (total compute spent on this example); ``vote_dist`` maps
        each label to how many members voted for it (raw counts, weight-agnostic).
    """
    if not votes:
        raise ValueError("aggregate() requires at least one vote")
    if weights is None:
        weights = [1.0] * len(votes)
    if len(weights) != len(votes):
        raise ValueError("weights must be parallel to votes (same length)")

    ids = {v["id"] for v in votes}
    if len(ids) != 1:
        raise ValueError(f"all votes must share one id, got {sorted(ids)}")
    ex_id = next(iter(ids))

    valid = [
        (v, w) for v, w in zip(votes, weights)
        if v.get("parse_ok") and v.get("pred_label")
    ]
    total_latency = float(sum(v.get("latency_ms") or 0.0 for v in votes))

    base = {
        "id": ex_id,
        "raw_output": "",
        "latency_ms": total_latency,
        "n_votes": len(votes),
        "n_valid": len(valid),
    }

    if not valid:
        return {**base, "pred_label": None, "parse_ok": False, "vote_dist": {}}

    counts = Counter(v["pred_label"] for v, _ in valid)
    vote_dist = dict(counts)
    scores: dict[str, float] = defaultdict(float)
    for v, w in valid:
        scores[v["pred_label"]] += w
    top = max(scores.values())
    winners = sorted(lbl for lbl, s in scores.items() if abs(s - top) <= _TIE_TOL)

    if len(winners) == 1:
        label = winners[0]
    elif tie_break == "neutral":
        label = "neutral"
    elif tie_break == "order":
        label = next(lbl for lbl in _ORDER if lbl in winners)
    else:  # "abstain"
        return {
            **base,
            "pred_label": None,
            "parse_ok": False,
            "raw_output": f"tie:{vote_dist}",
            "vote_dist": vote_dist,
        }

    return {
        **base,
        "pred_label": label,
        "parse_ok": True,
        "raw_output": f"vote:{label}:{vote_dist}",
        "vote_dist": vote_dist,
    }
