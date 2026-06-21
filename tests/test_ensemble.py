"""Tests for src/ensemble.py."""

import pytest
from src.ensemble import aggregate


def _vote(id_, label, parse_ok=True, latency_ms=1.0):
    return {
        "id": id_,
        "pred_label": label if parse_ok else None,
        "raw_output": label or "",
        "parse_ok": parse_ok,
        "latency_ms": latency_ms,
    }


# --- clear majority ---

def test_unanimous():
    out = aggregate([_vote("a", "positive")] * 3)
    assert out["pred_label"] == "positive"
    assert out["parse_ok"] is True
    assert out["n_valid"] == 3
    assert out["vote_dist"] == {"positive": 3}


def test_plurality_wins():
    out = aggregate([
        _vote("a", "positive"),
        _vote("a", "positive"),
        _vote("a", "negative"),
        _vote("a", "neutral"),
    ])
    assert out["pred_label"] == "positive"
    assert out["parse_ok"] is True


def test_abstainers_do_not_vote():
    # 2 valid 'negative' vs 2 abstain -> negative wins (abstainers excluded)
    out = aggregate([
        _vote("a", "negative"),
        _vote("a", "negative"),
        _vote("a", None, parse_ok=False),
        _vote("a", None, parse_ok=False),
    ])
    assert out["pred_label"] == "negative"
    assert out["n_votes"] == 4
    assert out["n_valid"] == 2


# --- ties ---

def test_tie_abstains_by_default():
    out = aggregate([_vote("a", "positive"), _vote("a", "negative")])
    assert out["pred_label"] is None
    assert out["parse_ok"] is False
    assert out["vote_dist"] == {"positive": 1, "negative": 1}


def test_tie_break_neutral():
    out = aggregate(
        [_vote("a", "positive"), _vote("a", "negative")], tie_break="neutral"
    )
    assert out["pred_label"] == "neutral"
    assert out["parse_ok"] is True


def test_tie_break_order():
    out = aggregate(
        [_vote("a", "positive"), _vote("a", "negative")], tie_break="order"
    )
    # "negative" precedes "positive" in canonical order
    assert out["pred_label"] == "negative"


def test_all_abstain():
    out = aggregate([
        _vote("a", None, parse_ok=False),
        _vote("a", None, parse_ok=False),
    ])
    assert out["pred_label"] is None
    assert out["parse_ok"] is False
    assert out["n_valid"] == 0


# --- weighted / soft voting ---

def test_weights_flip_majority():
    # Count says 'positive' (2 vs 1), but weights favor the single 'negative'.
    votes = [_vote("a", "positive"), _vote("a", "positive"), _vote("a", "negative")]
    out = aggregate(votes, weights=[0.1, 0.1, 5.0])
    assert out["pred_label"] == "negative"
    assert out["parse_ok"] is True
    # vote_dist stays raw counts, weight-agnostic
    assert out["vote_dist"] == {"positive": 2, "negative": 1}


def test_weights_none_equals_unweighted():
    votes = [_vote("a", "positive"), _vote("a", "positive"), _vote("a", "negative")]
    assert aggregate(votes)["pred_label"] == aggregate(votes, weights=[1, 1, 1])["pred_label"]


def test_weighted_tie_abstains():
    votes = [_vote("a", "positive"), _vote("a", "negative")]
    out = aggregate(votes, weights=[2.0, 2.0])
    assert out["pred_label"] is None
    assert out["parse_ok"] is False


def test_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        aggregate([_vote("a", "positive")], weights=[1.0, 2.0])


def test_abstainer_weight_ignored():
    # The heavy weight is on an abstaining member -> it must not count.
    votes = [_vote("a", "positive"), _vote("a", None, parse_ok=False)]
    out = aggregate(votes, weights=[1.0, 100.0])
    assert out["pred_label"] == "positive"
    assert out["n_valid"] == 1


# --- bookkeeping ---

def test_latency_is_summed():
    out = aggregate([
        _vote("a", "positive", latency_ms=10.0),
        _vote("a", "positive", latency_ms=15.0),
    ])
    assert out["latency_ms"] == pytest.approx(25.0)


def test_id_propagated():
    out = aggregate([_vote("FPB_00042", "neutral")])
    assert out["id"] == "FPB_00042"


# --- error conditions ---

def test_empty_raises():
    with pytest.raises(ValueError):
        aggregate([])


def test_mixed_ids_raise():
    with pytest.raises(ValueError):
        aggregate([_vote("a", "positive"), _vote("b", "positive")])
