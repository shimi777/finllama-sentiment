"""Load financial NER datasets (Alvarado 2015 FIN, Finer-Ord) and normalize.

Schema (unified NERSample), separate from sentiment Sample so the two tracks do
not collide in `data_loader.py`:

    {
        "id":      str,
        "text":    str,                 # full sentence
        "tokens":  list[str],           # whitespace tokens (informative, not authoritative)
        "tags":    list[str],           # BIO tags aligned to `tokens` if available
        "entities": list[Entity],       # canonical span list: [{text, type}]
        "dataset": "FIN" | "FINER_ORD",
        "split":   str,
    }

`entities` is the metric ground-truth. `tokens` and `tags` are kept for traceability
and for future BIO-based metrics.

Entity types are normalized to the Open-FinLLMs paper set: {PER, ORG, LOC, MISC}.
MISC is collapsed to "MISC" but kept distinct from the PER/ORG/LOC trio the paper's
prompt asks for — `compute_ner_metrics(only={"PER","ORG","LOC"})` enforces the
paper's evaluation choice.
"""

from __future__ import annotations

from typing import TypedDict

from src.utils import get_logger

logger = get_logger(__name__)


class Entity(TypedDict):
    text: str
    type: str


class NERSample(TypedDict):
    id: str
    text: str
    tokens: list[str]
    tags: list[str]
    entities: list[Entity]
    dataset: str
    split: str


# Aliases the Alvarado FIN / Finer-Ord datasets use for entity types.
# All variants map to one canonical set: PER, ORG, LOC, MISC.
_TYPE_ALIAS = {
    "PER": "PER", "PERSON": "PER",
    "ORG": "ORG", "ORGANIZATION": "ORG",
    "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC",
    "MISC": "MISC",
}


def _normalize_type(t: str) -> str | None:
    t = t.upper()
    return _TYPE_ALIAS.get(t)


def _bio_to_entities(tokens: list[str], tags: list[str]) -> list[Entity]:
    """Collapse BIO tags into spans. Tolerates 'I-X' starts (treats as 'B-X')."""
    ents: list[Entity] = []
    cur_type: str | None = None
    cur_toks: list[str] = []

    def flush():
        nonlocal cur_type, cur_toks
        if cur_type and cur_toks:
            ents.append(Entity(text=" ".join(cur_toks), type=cur_type))
        cur_type = None
        cur_toks = []

    for tok, tag in zip(tokens, tags):
        tag = tag.strip() if tag else "O"
        if tag == "O" or tag == "":
            flush()
            continue
        prefix, _, raw_type = tag.partition("-")
        norm = _normalize_type(raw_type) if raw_type else None
        if norm is None:
            flush()
            continue
        if prefix == "B" or cur_type != norm:
            flush()
            cur_type = norm
            cur_toks = [tok]
        else:  # "I" continuing same type
            cur_toks.append(tok)
    flush()
    return ents


def load_fin_alvarado(
    hf_id: str = "TheFinAI/flare-ner",
    split: str = "test",
) -> list[NERSample]:
    """Load the Alvarado 2015 FIN NER dataset as used in PIXIU / Open-FinLLMs.

    Tries the PIXIU FLARE wrapping first (`TheFinAI/flare-ner`). Falls back to a
    no-op error message if unavailable — caller can pass `hf_id` explicitly to
    point at a mirror (e.g. `tner/fin`, `nlpaueb/finer-139` is NOT a substitute).

    Two common HF schemas are supported transparently:
    1. PIXIU FLARE-style: rows have {text/sentence, label, ...} where `label`
       is a string like "Apple, ORG\nTim Cook, PER".
    2. tner/CoNLL-style: rows have {tokens: [...], tags: [...]} with BIO labels.
    """
    from datasets import load_dataset

    logger.info("Loading NER dataset %s split=%s …", hf_id, split)
    ds = load_dataset(hf_id, split=split)

    samples: list[NERSample] = []
    for i, row in enumerate(ds):
        if "tokens" in row and ("tags" in row or "ner_tags" in row):
            tokens = list(row["tokens"])
            raw_tags = row.get("tags") or row.get("ner_tags")
            # Some HF datasets store tags as ints with a ClassLabel feature.
            if raw_tags and isinstance(raw_tags[0], int):
                feats = ds.features.get("tags") or ds.features.get("ner_tags")
                if feats is not None and hasattr(feats, "feature"):
                    names = feats.feature.names
                    tags = [names[t] for t in raw_tags]
                else:
                    tags = [str(t) for t in raw_tags]
            else:
                tags = [str(t) for t in raw_tags]
            text = " ".join(tokens)
            entities = _bio_to_entities(tokens, tags)
        else:
            text = row.get("text") or row.get("sentence") or row.get("query") or ""
            label = row.get("answer") or row.get("label") or row.get("gold") or ""
            tokens = text.split()
            tags = []
            entities = _parse_paper_format(label)

        samples.append(
            NERSample(
                id=f"FIN_{i:05d}",
                text=text,
                tokens=tokens,
                tags=tags,
                entities=entities,
                dataset="FIN",
                split=split,
            )
        )

    logger.info("FIN: loaded %d examples (split=%s)", len(samples), split)
    return samples


def _parse_paper_format(s: str) -> list[Entity]:
    """Parse the paper's gold string format: 'name, TYPE; name, TYPE; …'.

    Also tolerates newline-separated 'name, TYPE' pairs (PIXIU sometimes uses \n).
    Unknown types are dropped (kept loose — final filter happens in metrics).
    """
    if not s:
        return []
    out: list[Entity] = []
    # Split on ';' first, then by newline as a fallback unit.
    chunks: list[str] = []
    for blk in s.split(";"):
        for line in blk.split("\n"):
            line = line.strip()
            if line:
                chunks.append(line)
    for ch in chunks:
        if "," not in ch:
            continue
        name, _, typ = ch.rpartition(",")
        name = name.strip().strip("'\"")
        typ = typ.strip().strip("'\"").upper()
        norm = _normalize_type(typ)
        if not name or norm is None:
            continue
        out.append(Entity(text=name, type=norm))
    return out
