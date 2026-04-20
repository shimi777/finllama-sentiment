"""Parse LLM output into a canonical label and track parsing coverage."""

import re

CANONICAL = {"positive", "neutral", "negative"}

SYNONYMS: dict[str, str] = {
    "bullish": "positive",
    "optimistic": "positive",
    "favorable": "positive",
    "upbeat": "positive",
    "good": "positive",
    "bearish": "negative",
    "pessimistic": "negative",
    "unfavorable": "negative",
    "bad": "negative",
    "mixed": "neutral",
    "moderate": "neutral",
    "stable": "neutral",
}

_PATTERN = re.compile(
    r"\b(positive|negative|neutral|bullish|bearish|optimistic|pessimistic"
    r"|favorable|unfavorable|upbeat|mixed|moderate|stable|good|bad)\b",
    re.IGNORECASE,
)


def parse(raw_output: str) -> str | None:
    """Return a canonical label from LLM output, or None if no match.

    Strategy: find first matching token (canonical or synonym).
    Synonyms are resolved to canonical after matching.
    """
    if not raw_output:
        return None
    m = _PATTERN.search(raw_output)
    if m is None:
        return None
    token = m.group(1).lower()
    if token in CANONICAL:
        return token
    return SYNONYMS.get(token)
