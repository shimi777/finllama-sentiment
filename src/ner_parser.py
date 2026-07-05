"""Parse LLM output into a list of (entity_text, entity_type) spans.

Robust to common deviations:
- "Apple, ORG; Tim Cook, PER"            (paper format, semicolons)
- "Apple, ORG\nTim Cook, PER"            (newlines)
- "Apple (ORG), Tim Cook (PER)"          (parenthetical types)
- "Apple, ORG (a tech company); ..."     (paper format + aside in parens)
- "- Apple, ORG"                         (bulleted)
- "Entities: Apple, ORG; ..."            (leading label kept by some models)
- "NONE" / "No entities" / ""            (empty list)

`parse_ner` returns a list of `Entity` dicts. Empty list means "parsed
successfully, no entities found". Returns `None` only when the output is
clearly malformed beyond the patterns above — in that case the caller's
predictions file gets `parse_ok=False` so evaluation can exclude it.
"""

from __future__ import annotations

import re

VALID_TYPES = {"PER", "ORG", "LOC", "MISC"}

_TYPE_ALIASES = {
    "PERSON": "PER", "PEOPLE": "PER",
    "ORGANIZATION": "ORG", "ORGANISATION": "ORG", "COMPANY": "ORG",
    "LOCATION": "LOC", "PLACE": "LOC", "GPE": "LOC", "COUNTRY": "LOC", "CITY": "LOC",
    "MISCELLANEOUS": "MISC",
}

_NONE_PATTERNS = re.compile(
    r"^\s*(NONE|N/?A|NULL|NO\s+ENTITIES|NO\s+NAMED\s+ENTITIES|\[\]|\(\)|\{\}|EMPTY)\s*\.?\s*$",
    re.IGNORECASE,
)

# Strip a leading "Entities: " / "Answer: " etc that some models emit.
_LEADING_LABEL = re.compile(r"^\s*(entities|answer|output|result)\s*:\s*", re.IGNORECASE)

# Match "name (TYPE)" or "name, TYPE" forms within a chunk.
_PAREN = re.compile(r"^\s*(.+?)\s*\(\s*([A-Za-z]+)\s*\)\s*$")
_COMMA = re.compile(r"^\s*(.+?)\s*,\s*([A-Za-z]+)\s*$")

# A "type-only" parenthetical means the parens contain ONLY a known type word.
_TYPE_PAREN = re.compile(
    r"\(\s*(PER|ORG|LOC|MISC|Person|Organization|Organisation|Location|GPE|"
    r"Place|City|Country|Miscellaneous)\s*\)",
    re.IGNORECASE,
)


def _norm_type(t: str) -> str | None:
    t = t.strip().upper()
    if t in VALID_TYPES:
        return t
    return _TYPE_ALIASES.get(t)


def _split_chunks(s: str) -> list[str]:
    """Split the raw answer into candidate entity chunks.

    Prefers paper-format split (';' / '\n') whenever ';' is present, because
    parenthetical asides like "(a law firm)" inside a paper-format answer would
    otherwise hijack the parser. Pure parenthetical form ("name (TYPE),") is
    only used when there is NO ';' separator AND a TYPE-only parenthetical is
    detected.
    """
    s = _LEADING_LABEL.sub("", s.strip())
    # Stop reading at lines that look like the model started another example.
    s = re.split(r"\n\s*(?:Sentence|Question|Q|Text|Input)\s*:", s, maxsplit=1)[0]

    has_paper_sep = ";" in s
    has_type_parens = _TYPE_PAREN.search(s) is not None

    if has_type_parens and not has_paper_sep:
        SENTINEL = "\x01"
        marked = re.sub(r"\)\s*,?\s*", ")" + SENTINEL, s)
        raw = marked.split(SENTINEL)
    else:
        raw = re.split(r"[;\n]| / ", s)
        # Scrub free-text parenthetical asides inside each chunk:
        # "Apple, ORG (a tech company)" -> "Apple, ORG".
        raw = [re.sub(r"\s*\([^)]*\)\s*", " ", c).strip() for c in raw]

    out: list[str] = []
    for c in raw:
        c = c.strip(" \t-*•·").strip("'\"").strip()
        if not c:
            continue
        if not re.search(r"[A-Za-z]", c):
            continue
        out.append(c)
    return out


def parse_ner(raw_output: str) -> list[dict] | None:
    """Return [{text, type}, ...] or None on hard parse failure.

    Empty list is *valid* (gold may also be empty). None is reserved for
    malformed output we can't reason about — caller should set parse_ok=False.
    """
    if raw_output is None:
        return None
    s = raw_output.strip()
    if not s:
        return []
    if _NONE_PATTERNS.match(s):
        return []

    chunks = _split_chunks(s)
    if not chunks:
        return None

    entities: list[dict] = []
    any_match = False
    for ch in chunks:
        m = _PAREN.match(ch) or _COMMA.match(ch)
        if not m:
            # Allow "Apple ORG" (space-separated) as a last resort.
            sp = re.match(r"^(.+?)\s+([A-Za-z]+)\s*$", ch)
            if not sp:
                continue
            m = sp
        name = m.group(1).strip(" '\"")
        typ = _norm_type(m.group(2))
        if not name or typ is None:
            continue
        any_match = True
        entities.append({"text": name, "type": typ})

    if not any_match:
        return None

    seen = set()
    deduped = []
    for e in entities:
        key = (e["text"].lower(), e["type"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    return deduped
