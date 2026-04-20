"""Tests for src/evaluation.py."""

import pytest
from src.evaluation import compute_metrics


def _make_pair(id_, true, pred, parse_ok=True):
    s = {"id": id_, "text": "x", "label": true, "dataset": "FPB", "split": "test"}
    p = {"id": id_, "pred_label": pred if parse_ok else None,
         "raw_output": pred or "", "parse_ok": parse_ok, "latency_ms": 1.0}
    return s, p


def _build(pairs):
    samples, preds = zip(*pairs)
    return list(samples), list(preds)


# --- basic correctness ---

def test_perfect_accuracy():
    s, p = _build([_make_pair(f"x{i}", lbl, lbl) for i, lbl in enumerate(
        ["positive", "negative", "neutral"])])
    m = compute_metrics(s, p)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["coverage"] == pytest.approx(1.0)


def test_zero_accuracy():
    s, p = _build([
        _make_pair("a", "positive", "negative"),
        _make_pair("b", "negative", "neutral"),
        _make_pair("c", "neutral", "positive"),
    ])
    m = compute_metrics(s, p)
    assert m["accuracy"] == pytest.approx(0.0)


def test_partial_accuracy():
    s, p = _build([
        _make_pair("a", "positive", "positive"),
        _make_pair("b", "negative", "positive"),
    ])
    m = compute_metrics(s, p)
    assert m["accuracy"] == pytest.approx(0.5)


# --- coverage ---

def test_full_coverage():
    s, p = _build([_make_pair("a", "positive", "positive")])
    assert compute_metrics(s, p)["coverage"] == pytest.approx(1.0)


def test_partial_coverage():
    s, p = _build([
        _make_pair("a", "positive", "positive", parse_ok=True),
        _make_pair("b", "negative", None, parse_ok=False),
    ])
    m = compute_metrics(s, p)
    assert m["coverage"] == pytest.approx(0.5)
    # only 1 parseable prediction, accuracy computed on that 1
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["n_samples"] == 2


def test_zero_coverage_returns_zeros():
    s, p = _build([_make_pair("a", "positive", None, parse_ok=False)])
    m = compute_metrics(s, p)
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["f1_macro"] == pytest.approx(0.0)
    assert m["coverage"] == pytest.approx(0.0)


# --- output structure ---

def test_per_class_keys():
    s, p = _build([
        _make_pair("a", "positive", "positive"),
        _make_pair("b", "negative", "negative"),
        _make_pair("c", "neutral", "neutral"),
    ])
    m = compute_metrics(s, p)
    for lbl in ["positive", "negative", "neutral"]:
        assert lbl in m["per_class"]
        assert {"precision", "recall", "f1", "support"} <= m["per_class"][lbl].keys()


def test_confusion_matrix_shape():
    s, p = _build([
        _make_pair("a", "positive", "positive"),
        _make_pair("b", "negative", "negative"),
        _make_pair("c", "neutral", "neutral"),
    ])
    m = compute_metrics(s, p)
    cm = m["confusion"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


def test_n_samples_includes_failures():
    s, p = _build([
        _make_pair("a", "positive", "positive"),
        _make_pair("b", "negative", None, parse_ok=False),
    ])
    assert compute_metrics(s, p)["n_samples"] == 2


# --- error conditions ---

def test_length_mismatch_raises():
    s = [{"id": "a", "label": "positive", "text": "", "dataset": "", "split": ""}]
    p = []
    with pytest.raises(ValueError):
        compute_metrics(s, p)


def test_id_mismatch_raises():
    s, pr = _build([_make_pair("a", "positive", "positive")])
    pr[0]["id"] = "WRONG"
    with pytest.raises(ValueError):
        compute_metrics(s, pr)
