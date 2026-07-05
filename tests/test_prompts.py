"""Tests for src/prompts.py."""

import pytest
from src.prompts import build_prompt, sample_fewshot, TEMPLATES

_POOL = [
    {"id": f"FPB_{i:05d}", "text": f"sentence {i}", "label": lbl, "dataset": "FPB", "split": "train"}
    for i, lbl in enumerate(["positive", "negative", "neutral"] * 5)
]


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_template_a_contains_text():
    p = build_prompt("A", "earnings beat estimates")
    assert "earnings beat estimates" in p

def test_template_b_contains_text():
    p = build_prompt("B", "earnings beat estimates")
    assert "earnings beat estimates" in p

def test_template_a_has_sentiment_colon():
    p = build_prompt("A", "some text")
    assert "Sentiment:" in p

def test_template_b_has_answer_line():
    p = build_prompt("B", "some text")
    assert "Answer with one word only" in p

def test_template_c_contains_text():
    p = build_prompt("C", "earnings beat estimates")
    assert "earnings beat estimates" in p

def test_template_c_one_word_instruction():
    p = build_prompt("C", "some text")
    assert "exactly one word" in p

def test_template_c_fewshot_block():
    p = build_prompt("C", "query", few_shot=_POOL[:3])
    # 3 examples + 1 query
    assert p.count("Text:") == 4

def test_template_d_contains_text():
    p = build_prompt("D", "earnings beat estimates")
    assert "earnings beat estimates" in p

def test_template_d_has_label_colon():
    p = build_prompt("D", "some text")
    assert "Label:" in p

def test_template_d_fewshot_block():
    p = build_prompt("D", "query", few_shot=_POOL[:3])
    assert p.count("Text:") == 4

def test_template_f_contains_text():
    p = build_prompt("F", "earnings beat estimates")
    assert "earnings beat estimates" in p

def test_template_f_has_sentiment_colon():
    p = build_prompt("F", "some text")
    assert "Sentiment:" in p

def test_template_f_has_neutral_tiebreak_rule():
    p = build_prompt("F", "some text")
    assert "only choose neutral" in p

def test_template_f_fewshot_block():
    p = build_prompt("F", "query", few_shot=_POOL[:3])
    assert p.count("Text:") == 4

def test_template_h_contains_text():
    p = build_prompt("H", "earnings beat estimates")
    assert "earnings beat estimates" in p

def test_template_h_has_answer_colon():
    p = build_prompt("H", "some text")
    assert "Answer:" in p

def test_template_h_neutral_last_resort():
    p = build_prompt("H", "some text")
    assert "last resort" in p

def test_template_h_fewshot_block():
    p = build_prompt("H", "query", few_shot=_POOL[:3])
    assert p.count("Text:") == 4

def test_templates_keys_are_a_b_c_d_f_h():
    assert set(TEMPLATES.keys()) == {"A", "B", "C", "D", "F", "H"}

def test_unknown_template_raises():
    with pytest.raises(ValueError):
        build_prompt("Z", "text")

def test_zero_shot_no_fewshot_block():
    p = build_prompt("A", "text", few_shot=[])
    # Only one "Text:" occurrence (the query itself)
    assert p.count("Text:") == 1

def test_fewshot_block_inserted():
    examples = _POOL[:3]
    p = build_prompt("A", "query text", few_shot=examples)
    # 3 examples + 1 query = 4 "Text:" occurrences
    assert p.count("Text:") == 4

def test_fewshot_labels_present():
    examples = [_POOL[0]]  # positive
    p = build_prompt("A", "q", few_shot=examples)
    assert "positive" in p

def test_none_fewshot_same_as_empty():
    assert build_prompt("A", "x", few_shot=None) == build_prompt("A", "x", few_shot=[])


# ---------------------------------------------------------------------------
# sample_fewshot
# ---------------------------------------------------------------------------

def test_zero_shots_returns_empty():
    assert sample_fewshot(_POOL, 0) == []

def test_three_shot_count():
    result = sample_fewshot(_POOL, 3, seed=42)
    assert len(result) == 3

def test_three_shot_balanced():
    result = sample_fewshot(_POOL, 3, seed=42)
    labels = [s["label"] for s in result]
    assert labels.count("positive") == 1
    assert labels.count("negative") == 1
    assert labels.count("neutral") == 1

def test_five_shot_count():
    result = sample_fewshot(_POOL, 5, seed=42)
    assert len(result) == 5

def test_five_shot_balance():
    result = sample_fewshot(_POOL, 5, seed=42)
    labels = [s["label"] for s in result]
    # 2/2/1 distribution
    counts = {l: labels.count(l) for l in ["positive", "negative", "neutral"]}
    assert max(counts.values()) == 2
    assert min(counts.values()) == 1

def test_seed_reproducible():
    a = sample_fewshot(_POOL, 3, seed=7)
    b = sample_fewshot(_POOL, 3, seed=7)
    assert [s["id"] for s in a] == [s["id"] for s in b]

def test_different_seeds_differ():
    a = sample_fewshot(_POOL, 3, seed=1)
    b = sample_fewshot(_POOL, 3, seed=999)
    # Very unlikely to be identical with 15-element pool
    assert [s["id"] for s in a] != [s["id"] for s in b]
