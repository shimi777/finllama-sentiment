"""Tests for src.ner.parser — JSON extraction + BIO alignment."""

from src.ner.parser import (
    _extract_json,
    _normalize_entity_dict,
    coerce_bio_to_canonical,
    parse_json_to_bio,
)


# ---------- _extract_json ----------

def test_extract_json_plain():
    assert _extract_json('{"PER": ["Tim Cook"]}') == {"PER": ["Tim Cook"]}


def test_extract_json_with_code_fence():
    raw = '```json\n{"ORG": ["Apple"]}\n```'
    assert _extract_json(raw) == {"ORG": ["Apple"]}


def test_extract_json_with_prose_after():
    raw = '{"LOC": ["Tokyo"]}\nThe LOC is the only entity.'
    out = _extract_json(raw)
    assert out == {"LOC": ["Tokyo"]}


def test_extract_json_invalid():
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None


def test_extract_json_with_prose_before():
    raw = 'Here is the answer: {"ORG": ["IBM"]}'
    assert _extract_json(raw) == {"ORG": ["IBM"]}


# ---------- _normalize_entity_dict ----------

def test_normalize_aliases():
    obj = {"PERSON": ["Alice"], "Organization": ["Google"], "GPE": ["Berlin"]}
    norm = _normalize_entity_dict(obj)
    assert norm["PER"] == ["Alice"]
    assert norm["ORG"] == ["Google"]
    assert norm["LOC"] == ["Berlin"]


def test_normalize_drops_unknown():
    obj = {"MISC": ["foo"], "PER": ["bar"]}
    norm = _normalize_entity_dict(obj)
    assert norm["PER"] == ["bar"]
    assert "MISC" not in norm


def test_normalize_string_value_coerced_to_list():
    obj = {"ORG": "Apple"}
    norm = _normalize_entity_dict(obj)
    assert norm["ORG"] == ["Apple"]


# ---------- parse_json_to_bio ----------

def test_parse_basic_alignment():
    tokens = ["Apple", "Inc.", "CEO", "Tim", "Cook", "visited", "Beijing", "."]
    raw = '{"PER": ["Tim Cook"], "ORG": ["Apple Inc."], "LOC": ["Beijing"]}'
    bio, ents = parse_json_to_bio(raw, tokens)
    assert bio == ["B-ORG", "I-ORG", "O", "B-PER", "I-PER", "O", "B-LOC", "O"]
    assert {e["type"] for e in ents if e["in_text"]} == {"PER", "ORG", "LOC"}


def test_parse_returns_none_on_garbage():
    bio, ents = parse_json_to_bio("the model emitted prose only", ["hello", "world"])
    assert bio is None
    assert ents == []


def test_parse_hallucinated_entity_marked_in_text_false():
    tokens = ["Apple", "released", "a", "phone", "."]
    raw = '{"PER": ["Elon Musk"], "ORG": ["Apple"]}'
    bio, ents = parse_json_to_bio(raw, tokens)
    # Apple maps to a B-ORG span; Elon Musk should be flagged as in_text=False.
    assert bio[0] == "B-ORG"
    not_in = [e for e in ents if not e["in_text"]]
    assert any(e["type"] == "PER" for e in not_in)


def test_parse_empty_json_all_o():
    tokens = ["nothing", "to", "see"]
    raw = '{"PER": [], "LOC": [], "ORG": []}'
    bio, ents = parse_json_to_bio(raw, tokens)
    assert bio == ["O", "O", "O"]
    assert ents == []


def test_parse_longest_first():
    tokens = ["Bank", "of", "America", "shares", "rose", "."]
    raw = '{"ORG": ["Bank of America", "America"]}'
    bio, ents = parse_json_to_bio(raw, tokens)
    # Longest-first: Bank of America wins, "America" alone overlaps and is
    # not allowed to override.
    assert bio[:3] == ["B-ORG", "I-ORG", "I-ORG"]


def test_parse_case_insensitive_match():
    tokens = ["The", "FED", "raised", "rates"]
    raw = '{"ORG": ["the fed"]}'
    bio, _ = parse_json_to_bio(raw, tokens)
    assert bio[0] == "B-ORG"
    assert bio[1] == "I-ORG"


# ---------- coerce_bio_to_canonical ----------

def test_coerce_drops_unknown_types():
    assert coerce_bio_to_canonical(["B-MISC", "I-MISC", "O", "B-PER"]) == ["O", "O", "O", "B-PER"]


def test_coerce_keeps_canonical():
    assert coerce_bio_to_canonical(["B-ORG", "I-ORG", "B-LOC", "O"]) == ["B-ORG", "I-ORG", "B-LOC", "O"]
