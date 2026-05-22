"""Tests for src.ner.evaluation — strict / partial / type-only span F1."""

from src.ner.evaluation import compute_ner_metrics


def _sample(id_: str, tokens: list[str], tags: list[str]) -> dict:
    return {
        "id": id_, "tokens": tokens, "tags": tags,
        "text": " ".join(tokens), "entities": [],
        "dataset": "FiNER-ORD", "split": "test",
    }


def _pred(id_: str, tags: list[str] | None) -> dict:
    return {
        "id": id_, "pred_tags": tags, "pred_entities": [],
        "raw_output": "", "parse_ok": tags is not None,
        "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0,
        "cost_usd": 0.0,
    }


def test_perfect_predictions_give_f1_1():
    samples = [
        _sample("a", ["Apple", "rose"], ["B-ORG", "O"]),
        _sample("b", ["Tim", "Cook", "spoke"], ["B-PER", "I-PER", "O"]),
    ]
    preds = [
        _pred("a", ["B-ORG", "O"]),
        _pred("b", ["B-PER", "I-PER", "O"]),
    ]
    m = compute_ner_metrics(samples, preds)
    assert m["coverage"] == 1.0
    assert m["strict"]["f1"] == 1.0
    assert m["partial"]["f1"] == 1.0
    assert m["per_type"]["ORG"]["strict"]["f1"] == 1.0
    assert m["per_type"]["PER"]["strict"]["f1"] == 1.0


def test_parse_failure_excluded_from_f1_but_in_coverage():
    samples = [
        _sample("a", ["Apple", "rose"], ["B-ORG", "O"]),
        _sample("b", ["nothing"], ["O"]),
    ]
    preds = [
        _pred("a", ["B-ORG", "O"]),
        _pred("b", None),
    ]
    m = compute_ner_metrics(samples, preds)
    assert m["coverage"] == 0.5
    assert m["strict"]["f1"] == 1.0


def test_boundary_error_hurts_strict_but_partial_credits():
    # gold: B-ORG I-ORG (Apple Inc.)  pred: B-ORG (just Apple)
    samples = [_sample("a", ["Apple", "Inc.", "rose"], ["B-ORG", "I-ORG", "O"])]
    preds = [_pred("a", ["B-ORG", "O", "O"])]
    m = compute_ner_metrics(samples, preds)
    assert m["strict"]["f1"] == 0.0  # boundary mismatch
    assert m["partial"]["f1"] == 1.0  # same type, overlap


def test_type_swap_zero_credit():
    # gold ORG, predicted PER on same span -> partial AND strict should be 0
    samples = [_sample("a", ["Apple", "rose"], ["B-ORG", "O"])]
    preds = [_pred("a", ["B-PER", "O"])]
    m = compute_ner_metrics(samples, preds)
    assert m["strict"]["f1"] == 0.0
    assert m["partial"]["f1"] == 0.0


def test_zero_predictions_zero_f1():
    samples = [_sample("a", ["Apple", "rose"], ["B-ORG", "O"])]
    preds = [_pred("a", ["O", "O"])]
    m = compute_ner_metrics(samples, preds)
    assert m["strict"]["f1"] == 0.0
    assert m["strict"]["recall"] == 0.0


def test_all_parse_failures_returns_placeholder():
    samples = [_sample("a", ["x"], ["O"])]
    preds = [_pred("a", None)]
    m = compute_ner_metrics(samples, preds)
    assert m["coverage"] == 0.0
    assert m["strict"]["f1"] == 0.0
    assert m["n_samples"] == 1


def test_id_mismatch_raises():
    samples = [_sample("a", ["x"], ["O"])]
    preds = [_pred("b", ["O"])]
    import pytest
    with pytest.raises(ValueError):
        compute_ner_metrics(samples, preds)
