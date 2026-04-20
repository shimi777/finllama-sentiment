"""20 boundary-case tests for src/parser.py."""

import pytest
from src.parser import parse


# --- exact canonical labels ---
def test_exact_positive():
    assert parse("positive") == "positive"

def test_exact_negative():
    assert parse("negative") == "negative"

def test_exact_neutral():
    assert parse("neutral") == "neutral"

# --- case insensitivity ---
def test_uppercase():
    assert parse("POSITIVE") == "positive"

def test_mixed_case():
    assert parse("Negative") == "negative"

# --- label inside a sentence ---
def test_label_in_sentence():
    assert parse("The sentiment is positive.") == "positive"

def test_label_with_hedge():
    assert parse("I'd say positive but there are risks.") == "positive"

def test_label_at_end():
    assert parse("Overall: neutral") == "neutral"

# --- synonyms ---
def test_synonym_bullish():
    assert parse("bullish") == "positive"

def test_synonym_bearish():
    assert parse("bearish") == "negative"

def test_synonym_optimistic():
    assert parse("The text sounds optimistic to me.") == "positive"

def test_synonym_pessimistic():
    assert parse("pessimistic outlook") == "negative"

def test_synonym_mixed():
    assert parse("The tone is mixed.") == "neutral"

# --- first-match wins ---
def test_first_match_wins_pos_neg():
    # "positive" appears before "negative" → should return positive
    assert parse("positive or perhaps negative") == "positive"

def test_first_match_wins_neg_pos():
    assert parse("not negative but actually positive") == "negative"

# --- failure cases → None ---
def test_empty_string():
    assert parse("") is None

def test_no_label():
    assert parse("The stock moved up by 3%.") is None

def test_gibberish():
    assert parse("xyzzy foobar 1234") is None

def test_unclear():
    assert parse("unclear") is None

# --- whitespace / punctuation edge cases ---
def test_trailing_newline():
    assert parse("positive\n") == "positive"

def test_label_with_colon():
    assert parse("Sentiment: negative") == "negative"

def test_quoted_label():
    assert parse('"neutral"') == "neutral"
