"""NER prompts for the Open-FinLLMs reproduction.

Two zero-shot prompts:

- "paper": verbatim from Open-FinLLMs Appendix E.6 (Figure 5 prompt).
- "strict": same instruction but with a tighter "respond with only the list"
  suffix to reduce chatter from instruction-tuned models that like to explain.

A few-shot variant builds the prompt with `k` examples sampled from a train
pool. Examples must already be `NERSample` dicts (or anything with
`text` + `entities`) so this stays decoupled from any specific loader.
"""

from __future__ import annotations

import random


PAPER_PROMPT = (
    "In the sentences extracted from financial agreements in U.S. SEC filings, "
    "identify the named entities that represent a person ('PER'), an organization "
    "('ORG'), or a location ('LOC'). The required answer format is: "
    "'entity name, entity type'. For instance, in 'Elon Musk, CEO of SpaceX, "
    "announced the launch from Cape Canaveral.', the entities would be: "
    "'Elon Musk, PER; SpaceX, ORG; Cape Canaveral, LOC'"
)

STRICT_SUFFIX = (
    "\nReply with ONLY the entity list in the exact format above, "
    "separated by '; '. If there are no entities, reply 'NONE'."
)


def _format_gold(entities: list[dict]) -> str:
    parts = [f"{e['text']}, {e['type']}" for e in entities if e.get("text") and e.get("type")]
    return "; ".join(parts) if parts else "NONE"


def build_ner_prompt(
    template: str,
    text: str,
    few_shot: list[dict] | None = None,
) -> str:
    """Render a NER prompt for one input sentence.

    `template` is "paper" or "strict". `few_shot` is a list of NERSample-like
    dicts with `text` and `entities` keys; pass [] or None for 0-shot.
    """
    base = PAPER_PROMPT
    if template == "strict":
        base = PAPER_PROMPT + STRICT_SUFFIX
    elif template != "paper":
        raise ValueError(f"unknown NER template: {template}")

    parts: list[str] = [base, ""]
    for ex in few_shot or []:
        parts.append(f"Sentence: {ex['text']}")
        parts.append(f"Entities: {_format_gold(ex['entities'])}")
        parts.append("")
    parts.append(f"Sentence: {text}")
    parts.append("Entities:")
    return "\n".join(parts)


def sample_fewshot(
    pool: list[dict],
    k: int,
    seed: int = 42,
    require_entities: bool = True,
) -> list[dict]:
    """Pick `k` examples for few-shot. Default prefers examples with entities."""
    if k <= 0 or not pool:
        return []
    rng = random.Random(seed)
    candidates = [x for x in pool if x.get("entities")] if require_entities else list(pool)
    if not candidates:
        candidates = list(pool)
    if len(candidates) <= k:
        return list(candidates)
    return rng.sample(candidates, k)
