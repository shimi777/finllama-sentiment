"""Parse LLM JSON output -> BIO tag sequence aligned to gold tokens.

Returns None on JSON-parse failure so the caller can mark `parse_ok=False`
and exclude it from F1 — same contract as the sentiment parser.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from src.ner.data_loader import CANONICAL_TYPES

_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    """Pull the first balanced JSON object out of `raw`.

    LLMs frequently wrap JSON in ```json ... ``` or trail prose after it.
    Strategy:
      1) try json.loads on the stripped raw output
      2) strip code fences
      3) find the first balanced {...} and try that
    Returns dict or None.
    """
    if not raw:
        return None
    txt = raw.strip()

    # Strip ```json ... ``` fences if present.
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```\s*$", "", txt)

    # 1) direct parse
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) balanced-brace scan for the first object
    depth = 0
    start = -1
    for i, ch in enumerate(txt):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = txt[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError):
                    start = -1
                    continue
    return None


def _normalize_entity_dict(obj: dict) -> dict[str, list[str]]:
    """Lowercase keys, alias verbose types, keep only CANONICAL_TYPES."""
    alias = {
        "PERSON": "PER", "PEOPLE": "PER", "PER": "PER",
        "LOCATION": "LOC", "GPE": "LOC", "LOC": "LOC",
        "ORGANIZATION": "ORG", "ORGANISATION": "ORG", "COMPANY": "ORG",
        "ORG": "ORG",
    }
    out: dict[str, list[str]] = {t: [] for t in CANONICAL_TYPES}
    for k, v in obj.items():
        canon = alias.get(str(k).strip().upper())
        if canon is None:
            continue
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            continue
        for item in v:
            if isinstance(item, dict) and "text" in item:
                item = item["text"]
            if isinstance(item, str) and item.strip():
                out[canon].append(item.strip())
    return out


def _tokens_lower(tokens: list[str]) -> list[str]:
    return [t.lower() for t in tokens]


def _find_span(needle_tokens: list[str], haystack_tokens_lower: list[str]) -> tuple[int, int] | None:
    """Find first contiguous match of `needle_tokens` (lowercased) in haystack.

    Returns (start, end) with end exclusive, or None. Comparison is case-insensitive
    and tolerates trailing punctuation differences via token-equality with a strip.
    """
    if not needle_tokens:
        return None
    n_lo = [t.lower().strip(".,;:'\"") for t in needle_tokens]
    h_lo = [t.strip(".,;:'\"") for t in haystack_tokens_lower]
    n = len(n_lo)
    for i in range(0, len(h_lo) - n + 1):
        if h_lo[i : i + n] == n_lo:
            return (i, i + n)
    return None


def parse_json_to_bio(
    raw_output: str,
    tokens: list[str],
) -> tuple[list[str] | None, list[dict]]:
    """Parse LLM JSON output and align to a BIO tag sequence over `tokens`.

    Args:
        raw_output: the model's raw text emission.
        tokens: gold tokens (whitespace-split words) from the NerSample.

    Returns:
        (bio_tags, pred_entities)
        - bio_tags: list[str] of length len(tokens), one of {"O", "B-PER", ...}.
                    None if JSON could not be parsed at all.
        - pred_entities: list of {type, start_tok, end_tok, text, in_text}
                         in_text=False means the model hallucinated a span
                         that we couldn't anchor in the sentence.
    """
    obj = _extract_json(raw_output)
    if obj is None:
        return None, []

    norm = _normalize_entity_dict(obj)

    bio: list[str] = ["O"] * len(tokens)
    pred_entities: list[dict] = []
    tokens_lower = _tokens_lower(tokens)

    # To avoid double-tagging when the same span is claimed twice, track
    # token indices already covered.
    covered = [False] * len(tokens)

    # Process spans longest-first so "Apple Inc." beats "Apple" alone.
    flat: list[tuple[str, str]] = []
    for etype, ents in norm.items():
        for e in ents:
            flat.append((etype, e))
    flat.sort(key=lambda x: -len(x[1].split()))

    for etype, ent_str in flat:
        # Whitespace-tokenize the entity, mirroring gold tokenization.
        needle = ent_str.split()
        if not needle:
            continue
        span = _find_span(needle, tokens_lower)
        if span is None:
            pred_entities.append({
                "type": etype, "start_tok": -1, "end_tok": -1,
                "text": ent_str, "in_text": False,
            })
            continue
        s, e = span
        if any(covered[s:e]):
            # Overlap with a longer earlier span; record but don't override.
            pred_entities.append({
                "type": etype, "start_tok": s, "end_tok": e,
                "text": " ".join(tokens[s:e]), "in_text": True,
            })
            continue
        bio[s] = f"B-{etype}"
        for j in range(s + 1, e):
            bio[j] = f"I-{etype}"
        for j in range(s, e):
            covered[j] = True
        pred_entities.append({
            "type": etype, "start_tok": s, "end_tok": e,
            "text": " ".join(tokens[s:e]), "in_text": True,
        })

    return bio, pred_entities


def coerce_bio_to_canonical(tags: Iterable[str]) -> list[str]:
    """Force any tag outside {O, B-{PER,LOC,ORG}, I-{PER,LOC,ORG}} to 'O'."""
    out = []
    for t in tags:
        if t == "O":
            out.append("O")
            continue
        if "-" in t:
            p, e = t.split("-", 1)
            if p in ("B", "I") and e in CANONICAL_TYPES:
                out.append(t)
                continue
        out.append("O")
    return out
