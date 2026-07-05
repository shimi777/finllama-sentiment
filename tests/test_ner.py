"""Smoke tests for the NER pipeline (parser, prompts, evaluation).

Loader hits HF and is therefore skipped here — covered by the dry-run path of
`scripts/run_ner_matrix.py --dry-run --limit 5`.
"""

from src.ner_parser import parse_ner
from src.ner_prompts import build_ner_prompt
from src.ner_evaluation import compute_ner_metrics


# ── parser ─────────────────────────────────────────────────────────────────

def test_parse_paper_format():
    out = parse_ner("Apple, ORG; Tim Cook, PER; Cupertino, LOC")
    assert out == [
        {"text": "Apple", "type": "ORG"},
        {"text": "Tim Cook", "type": "PER"},
        {"text": "Cupertino", "type": "LOC"},
    ]


def test_parse_newline_separated():
    out = parse_ner("Apple, ORG\nTim Cook, PER")
    assert {("Apple", "ORG"), ("Tim Cook", "PER")} == {(e["text"], e["type"]) for e in out}


def test_parse_parenthetical():
    out = parse_ner("Apple (Organization), Tim Cook (Person)")
    assert out and {("Apple", "ORG"), ("Tim Cook", "PER")} == {(e["text"], e["type"]) for e in out}


def test_parse_leading_label():
    out = parse_ner("Entities: Apple, ORG; Tim Cook, PER")
    assert {("Apple", "ORG"), ("Tim Cook", "PER")} == {(e["text"], e["type"]) for e in out}


def test_parse_none_means_empty_list():
    assert parse_ner("NONE") == []
    assert parse_ner("No entities") == []
    assert parse_ner("") == []
    assert parse_ner("   ") == []


def test_parse_malformed_returns_none():
    # No comma, no parens, no recognised types — caller should mark parse_ok=False.
    assert parse_ner("the model was confused and rambled at length") is None


def test_parse_dedupe():
    out = parse_ner("Apple, ORG; apple, organization")
    assert len(out) == 1


def test_parse_stops_at_next_example():
    out = parse_ner("Apple, ORG; Tim Cook, PER\nSentence: foo\nEntities: ignored, ORG")
    assert {(e["text"], e["type"]) for e in out} == {("Apple", "ORG"), ("Tim Cook", "PER")}


# ── prompts ────────────────────────────────────────────────────────────────

def test_prompt_zero_shot_contains_paper_text():
    p = build_ner_prompt("paper", "Apple sued Samsung.", [])
    assert "U.S. SEC filings" in p
    assert p.rstrip().endswith("Entities:")


def test_prompt_few_shot_renders_examples():
    fs = [{"text": "Tim Cook leads Apple.", "entities": [
        {"text": "Tim Cook", "type": "PER"}, {"text": "Apple", "type": "ORG"},
    ]}]
    p = build_ner_prompt("strict", "Microsoft is in Redmond.", fs)
    assert "Tim Cook, PER" in p and "Apple, ORG" in p
    assert "Microsoft is in Redmond." in p
    assert "Reply with ONLY" in p


# ── metrics ────────────────────────────────────────────────────────────────

def _sample(sid, ents):
    return {"id": sid, "text": "", "entities": ents, "dataset": "FIN", "split": "test",
            "tokens": [], "tags": []}


def _pred(sid, ents, parse_ok=True):
    return {"id": sid, "pred_entities": ents, "parse_ok": parse_ok}


def test_metrics_perfect_match():
    samples = [_sample("a", [{"text": "Apple", "type": "ORG"}])]
    preds = [_pred("a", [{"text": "Apple", "type": "ORG"}])]
    m = compute_ner_metrics(samples, preds, only={"PER", "ORG", "LOC"})
    assert m["micro_f1"] == 1.0
    assert m["coverage"] == 1.0


def test_metrics_partial_match():
    samples = [
        _sample("a", [{"text": "Apple", "type": "ORG"}, {"text": "Tim Cook", "type": "PER"}]),
        _sample("b", [{"text": "Microsoft", "type": "ORG"}]),
    ]
    preds = [
        _pred("a", [{"text": "Apple", "type": "ORG"}]),               # 1 TP, 1 FN
        _pred("b", [{"text": "Microsoft", "type": "ORG"},
                     {"text": "Bill Gates", "type": "PER"}]),         # 1 TP, 1 FP
    ]
    m = compute_ner_metrics(samples, preds, only={"PER", "ORG", "LOC"})
    # TP=2, FP=1, FN=1 → P=2/3, R=2/3, F1=2/3
    assert abs(m["micro_f1"] - (2 / 3)) < 1e-3


def test_metrics_parse_failures_excluded():
    samples = [_sample("a", [{"text": "Apple", "type": "ORG"}]),
               _sample("b", [{"text": "MSFT", "type": "ORG"}])]
    preds = [_pred("a", [{"text": "Apple", "type": "ORG"}]),
             _pred("b", None, parse_ok=False)]
    m = compute_ner_metrics(samples, preds, only={"PER", "ORG", "LOC"})
    assert m["n_parse_failures"] == 1
    assert m["n_evaluated"] == 1
    assert m["micro_f1"] == 1.0  # the 1 evaluated sample was perfect
    assert m["coverage"] == 0.5


def test_metrics_restriction_to_three_types():
    # MISC golds and preds should be ignored when only={PER,ORG,LOC}.
    samples = [_sample("a", [{"text": "Apple", "type": "ORG"},
                              {"text": "iPhone", "type": "MISC"}])]
    preds = [_pred("a", [{"text": "Apple", "type": "ORG"},
                          {"text": "iPhone", "type": "MISC"}])]
    m = compute_ner_metrics(samples, preds, only={"PER", "ORG", "LOC"})
    # Only Apple/ORG counted on both sides → perfect.
    assert m["micro_f1"] == 1.0
