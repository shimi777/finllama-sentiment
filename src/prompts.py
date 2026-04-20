"""Prompt templates (A, B) and few-shot example sampling."""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_loader import Sample

TEMPLATES: dict[str, str] = {
    "A": (
        "Classify the sentiment of the following financial text as positive, negative, or neutral.\n"
        "{fewshot_block}"
        "Text: {text}\n"
        "Sentiment:"
    ),
    "B": (
        "You are a financial analyst. Classify the sentiment of the text below from the perspective of an investor.\n"
        "- Positive: the text suggests favorable conditions, growth, or gains\n"
        "- Negative: the text suggests unfavorable conditions, losses, or risks\n"
        "- Neutral: the text is factual without clear positive or negative implication\n"
        "\n"
        "{fewshot_block}"
        "Text: {text}\n"
        "Answer with one word only (positive / negative / neutral):"
    ),
}

_LABEL_ORDER = ["negative", "neutral", "positive"]


def sample_fewshot(pool: list, n_shots: int, seed: int = 42) -> list:
    """Return a balanced list of n_shots examples sampled from pool.

    Distribution: n=3 → 1/1/1, n=5 → 2/2/1 (extra to negative and neutral).
    """
    if n_shots == 0:
        return []

    rng = random.Random(seed)
    by_label: dict[str, list] = {lbl: [] for lbl in _LABEL_ORDER}
    for s in pool:
        if s["label"] in by_label:
            by_label[s["label"]].append(s)

    if n_shots == 3:
        counts = {"negative": 1, "neutral": 1, "positive": 1}
    elif n_shots == 5:
        counts = {"negative": 2, "neutral": 2, "positive": 1}
    else:
        # generic fallback: round-robin
        per = n_shots // 3
        rem = n_shots % 3
        counts = {lbl: per for lbl in _LABEL_ORDER}
        for lbl in _LABEL_ORDER[:rem]:
            counts[lbl] += 1

    selected: list = []
    for lbl in _LABEL_ORDER:
        k = min(counts[lbl], len(by_label[lbl]))
        selected.extend(rng.sample(by_label[lbl], k))
    rng.shuffle(selected)
    return selected


def build_prompt(template: str, text: str, few_shot: list | None = None) -> str:
    """Build a ready-to-send prompt string.

    Args:
        template: key into TEMPLATES ("A" or "B").
        text: the financial sentence to classify.
        few_shot: list of Sample dicts to prepend as examples (may be empty/None).
    """
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template '{template}'. Choose from: {list(TEMPLATES)}")

    fewshot_block = ""
    if few_shot:
        lines = []
        for ex in few_shot:
            lines.append(f"Text: {ex['text']}")
            lines.append(f"Sentiment: {ex['label']}")
            lines.append("")
        fewshot_block = "\n".join(lines) + "\n"

    return TEMPLATES[template].format(text=text, fewshot_block=fewshot_block)
