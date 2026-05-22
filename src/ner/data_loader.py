"""Load FiNER-ORD and normalize to a unified NerSample schema.

FiNER-ORD (Financial Named Entity Recognition - Open Research Dataset):
  Source: gtfintechlab/finer-ord-bio
  Entities: PER, LOC, ORG (CoNLL-style)
  Splits:   train / validation / test

Why FiNER-ORD over FiNER-139:
  - Universal entity schema every LLM knows zero-shot
  - Compact (~1k test sentences) -> cheap to evaluate
  - Canonical financial-news domain
"""

from __future__ import annotations

import os
from typing import TypedDict

from src.utils import get_logger

logger = get_logger(__name__)


# Canonical entity types after normalization (FiNER-ORD native: PER / LOC / ORG).
CANONICAL_TYPES = ("PER", "LOC", "ORG")


class NerSample(TypedDict):
    id: str
    tokens: list[str]       # whitespace tokenization (gold)
    tags: list[str]         # BIO tags, same length as tokens (e.g. "B-ORG", "I-PER", "O")
    text: str               # " ".join(tokens) — convenience for display
    entities: list[dict]    # [{type, start_tok, end_tok, text}] derived from BIO
    dataset: str            # "FiNER-ORD"
    split: str              # "train" | "validation" | "test"


def _bio_to_spans(tokens: list[str], tags: list[str]) -> list[dict]:
    """Convert parallel (tokens, BIO tags) into a list of entity spans.

    Returns: list of {type, start_tok, end_tok, text}. end_tok is exclusive.
    Robust to malformed BIO (treats I- without matching B- as starting a span).
    """
    out: list[dict] = []
    i = 0
    n = len(tags)
    while i < n:
        t = tags[i]
        if t == "O" or not t:
            i += 1
            continue
        # Strip prefix
        if "-" in t:
            prefix, etype = t.split("-", 1)
        else:
            prefix, etype = "B", t
        start = i
        i += 1
        while i < n and tags[i].startswith("I-") and tags[i].split("-", 1)[1] == etype:
            i += 1
        out.append({
            "type": etype,
            "start_tok": start,
            "end_tok": i,
            "text": " ".join(tokens[start:i]),
        })
    return out


def _normalize_tag(tag: str) -> str:
    """Map FiNER-ORD tag variants onto canonical PER/LOC/ORG BIO tags.

    The HF dataset exposes integer tag IDs that map (per its features.names) to
    something like: ["O","B-PER","I-PER","B-LOC","I-LOC","B-ORG","I-ORG"].
    Also tolerate longer variants like B-PERSON.
    """
    if not tag or tag == "O":
        return "O"
    if "-" not in tag:
        return "O"
    prefix, etype = tag.split("-", 1)
    if prefix not in ("B", "I"):
        return "O"
    etype = etype.upper()
    # Map verbose -> canonical
    alias = {
        "PERSON": "PER",
        "LOCATION": "LOC",
        "ORGANIZATION": "ORG",
        "ORGANISATION": "ORG",
        "GPE": "LOC",
    }
    etype = alias.get(etype, etype)
    if etype not in CANONICAL_TYPES:
        return "O"
    return f"{prefix}-{etype}"


def load_finer_ord(
    split: str = "test",
    max_samples: int | None = None,
    seed: int = 42,
) -> list[NerSample]:
    """Load FiNER-ORD via HuggingFace `datasets`.

    Args:
        split: 'train' | 'validation' | 'test'.
        max_samples: cap (deterministic shuffle by `seed`) to keep cost down.
        seed: shuffling seed.

    Returns: list of NerSample dicts with canonical BIO tags.
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or None
    logger.info("Loading FiNER-ORD split=%s …", split)
    ds = load_dataset("gtfintechlab/finer-ord-bio", split=split, token=token)

    # Discover the tag-id -> tag-name mapping from the dataset features.
    # Different revisions name the column 'ner_tags' or 'tags'; tokens column
    # is usually 'tokens'.
    cols = ds.column_names
    tag_col = "ner_tags" if "ner_tags" in cols else ("tags" if "tags" in cols else None)
    tok_col = "tokens" if "tokens" in cols else None
    if tag_col is None or tok_col is None:
        raise RuntimeError(
            f"Unexpected FiNER-ORD columns {cols}; "
            "expected 'tokens' + 'ner_tags' (or 'tags')."
        )

    # Build tag-id -> tag-name list from the feature schema, or fall back to
    # the canonical FiNER-ORD-BIO id ordering. The HuggingFace export of
    # gtfintechlab/finer-ord-bio stores tags as plain int64 without a
    # ClassLabel wrapper, so the schema-based path returns no names.
    tag_feature = ds.features[tag_col]
    tag_names: list[str]
    try:
        tag_names = list(tag_feature.feature.names)  # type: ignore[attr-defined]
    except AttributeError:
        try:
            tag_names = list(tag_feature.names)  # type: ignore[attr-defined]
        except AttributeError:
            tag_names = []

    # Canonical FiNER-ORD-BIO id mapping (verified empirically: tokens
    # "Robin Lee" -> [1,2] (PER) and "London Overground" -> [3,4] (LOC) in
    # the test split).
    FINER_ORD_BIO_NAMES = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    if not tag_names:
        tag_names = FINER_ORD_BIO_NAMES

    samples: list[NerSample] = []
    for idx, row in enumerate(ds):
        tokens = list(row[tok_col])
        raw_tag_ids = list(row[tag_col])
        if tag_names:
            raw_tags = [tag_names[t] if isinstance(t, int) else str(t) for t in raw_tag_ids]
        else:
            raw_tags = [str(t) for t in raw_tag_ids]
        tags = [_normalize_tag(t) for t in raw_tags]
        if len(tags) != len(tokens):
            # Shouldn't happen, but be defensive — pad/truncate to align.
            min_len = min(len(tokens), len(tags))
            tokens, tags = tokens[:min_len], tags[:min_len]
        entities = _bio_to_spans(tokens, tags)
        samples.append(NerSample(
            id=f"FiNERORD_{idx:05d}",
            tokens=tokens,
            tags=tags,
            text=" ".join(tokens),
            entities=entities,
            dataset="FiNER-ORD",
            split=split,
        ))

    if max_samples is not None and max_samples < len(samples):
        import random
        rng = random.Random(seed)
        rng.shuffle(samples)
        samples = samples[:max_samples]
        # Re-sort by id for stability in downstream artifacts.
        samples.sort(key=lambda s: s["id"])

    logger.info("FiNER-ORD %s: %d samples (after cap)", split, len(samples))
    return samples
