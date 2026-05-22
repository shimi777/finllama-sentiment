"""Prompts for LLM-based NER on FiNER-ORD.

We ask the model to emit strict JSON of the form:
    {"PER": [...], "LOC": [...], "ORG": [...]}
where each value is a list of *verbatim substrings* from the input text.
Parsing happens in `parser.parse_json_to_bio` — see that module for the
substring -> BIO alignment rules.
"""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a financial NER tagger. Extract named entities of three types from "
    "the input sentence:\n"
    "- PER: people (CEOs, analysts, named individuals)\n"
    "- LOC: locations (cities, countries, regions)\n"
    "- ORG: organizations (companies, agencies, banks, exchanges)\n"
    "\n"
    "Return STRICT JSON with three keys: PER, LOC, ORG. Each value is a list "
    "of strings copied VERBATIM from the input sentence (preserve casing and "
    "punctuation). If a category is empty, return []. Do NOT invent entities "
    "that are not literally in the sentence. Output ONLY the JSON, no prose."
)


# Two templates so we can ablate prompt-sensitivity, matching the sentiment side.
TEMPLATES: dict[str, str] = {
    "A": (
        "Extract PER, LOC, ORG entities from the following sentence. "
        "Return JSON only.\n"
        "\n"
        "{fewshot_block}"
        "Sentence: {text}\n"
        "JSON:"
    ),
    "B": (
        "Sentence: {text}\n"
        "\n"
        "{fewshot_block}"
        "Identify all named entities and output JSON with keys PER, LOC, ORG. "
        "Each value must be a list of substrings copied exactly from the "
        "sentence above. Output JSON only.\n"
        "JSON:"
    ),
}


_FEWSHOT_EXAMPLES = [
    {
        "text": "Apple Inc. CEO Tim Cook will visit Beijing next month.",
        "json": '{"PER": ["Tim Cook"], "LOC": ["Beijing"], "ORG": ["Apple Inc."]}',
    },
    {
        "text": "The Federal Reserve raised interest rates again on Wednesday.",
        "json": '{"PER": [], "LOC": [], "ORG": ["Federal Reserve"]}',
    },
    {
        "text": "Goldman Sachs analyst Maria Chen downgraded Tesla to neutral.",
        "json": '{"PER": ["Maria Chen"], "LOC": [], "ORG": ["Goldman Sachs", "Tesla"]}',
    },
]


def build_prompt(template: str, text: str, n_shots: int = 0) -> str:
    """Build a ready-to-send prompt string for one example.

    Args:
        template: "A" or "B".
        text: the financial sentence to tag.
        n_shots: 0 or 3 (cap to avoid ballooning token bills).
    """
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template '{template}'. Choose from {list(TEMPLATES)}")
    n_shots = max(0, min(n_shots, len(_FEWSHOT_EXAMPLES)))
    fewshot_block = ""
    if n_shots > 0:
        lines = ["Examples:"]
        for ex in _FEWSHOT_EXAMPLES[:n_shots]:
            lines.append(f"Sentence: {ex['text']}")
            lines.append(f"JSON: {ex['json']}")
        fewshot_block = "\n".join(lines) + "\n\n"
    return TEMPLATES[template].format(text=text, fewshot_block=fewshot_block)
