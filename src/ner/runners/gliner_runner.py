"""Local GLiNER zero-shot NER runner — free, runs on CPU.

GLiNER (NAACL 2024) is a compact BERT-encoder NER model that accepts an
arbitrary label list at inference time. We pass ["person", "location",
"organization"] and map back to PER/LOC/ORG.

Lazy-imports gliner so the module is importable even without it installed —
the user only pays the install cost if they actually run this backend.
"""

from __future__ import annotations

import os

# Disable TF code paths before transformers is imported — works around the
# Keras 3 incompatibility (`transformers` <-> Keras 3) on systems without
# `tf-keras` installed. PyTorch is the only backend we need.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import time
from typing import Iterable

from src.utils import get_logger
from src.ner.data_loader import NerSample

logger = get_logger(__name__)


# Maps the GLiNER-friendly labels to canonical types.
_LABELS = ["person", "location", "organization"]
_LABEL_TO_CANON = {
    "person": "PER",
    "location": "LOC",
    "organization": "ORG",
}


def _spans_to_bio(tokens: list[str], spans: list[dict]) -> tuple[list[str], list[dict]]:
    """Align GLiNER character-level spans onto the gold whitespace tokens.

    GLiNER returns {start, end, label, text} with char offsets into the input
    string. We rebuild the token char-ranges by walking " ".join(tokens) and
    then bucket each span into [first_tok, last_tok].
    """
    text = " ".join(tokens)
    # token char ranges
    starts: list[int] = []
    ends: list[int] = []
    pos = 0
    for tok in tokens:
        starts.append(pos)
        pos += len(tok)
        ends.append(pos)
        pos += 1  # space

    bio = ["O"] * len(tokens)
    pred_entities: list[dict] = []
    covered = [False] * len(tokens)

    # Sort spans longest-first (by char length) to avoid sub-span clobber.
    spans = sorted(spans, key=lambda s: -(s.get("end", 0) - s.get("start", 0)))

    for sp in spans:
        label = (sp.get("label") or "").lower()
        canon = _LABEL_TO_CANON.get(label)
        if canon is None:
            continue
        s_char = int(sp.get("start", -1))
        e_char = int(sp.get("end", -1))
        if s_char < 0 or e_char <= s_char:
            continue
        # First token whose end > s_char; last token whose start < e_char.
        tok_start = None
        tok_end = None
        for i, (ts, te) in enumerate(zip(starts, ends)):
            if tok_start is None and te > s_char:
                tok_start = i
            if ts < e_char:
                tok_end = i
        if tok_start is None or tok_end is None or tok_end < tok_start:
            continue
        tok_end += 1  # make exclusive
        if any(covered[tok_start:tok_end]):
            pred_entities.append({
                "type": canon, "start_tok": tok_start, "end_tok": tok_end,
                "text": " ".join(tokens[tok_start:tok_end]), "in_text": True,
            })
            continue
        bio[tok_start] = f"B-{canon}"
        for j in range(tok_start + 1, tok_end):
            bio[j] = f"I-{canon}"
        for j in range(tok_start, tok_end):
            covered[j] = True
        pred_entities.append({
            "type": canon, "start_tok": tok_start, "end_tok": tok_end,
            "text": " ".join(tokens[tok_start:tok_end]), "in_text": True,
        })

    return bio, pred_entities


class GLiNERRunner:
    """Wraps the GLiNER zero-shot model. Cost is always $0."""

    def __init__(self, model_name: str = "urchade/gliner_large-v2.1", threshold: float = 0.5):
        try:
            from gliner import GLiNER
        except ImportError as e:
            raise ImportError(
                "gliner is not installed. `pip install gliner` "
                "(or skip this runner; the dashboard works without it)."
            ) from e
        logger.info("Loading GLiNER %s …", model_name)
        self.model_name = model_name
        self.threshold = threshold
        self.model = GLiNER.from_pretrained(model_name)

    def predict_one(self, sample: NerSample) -> dict:
        text = sample["text"]
        t0 = time.perf_counter()
        try:
            raw_spans = self.model.predict_entities(text, _LABELS, threshold=self.threshold)
        except Exception as e:
            logger.warning("GLiNER predict failed on %s: %s", sample["id"], e)
            return {
                "id": sample["id"], "pred_tags": None, "pred_entities": [],
                "raw_output": f"<<gliner-error: {e}>>", "parse_ok": False,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            }
        bio, pred_entities = _spans_to_bio(sample["tokens"], raw_spans)
        # Stash a compact raw-output for the dashboard explorer.
        raw_str = "; ".join(
            f"{e.get('label','?')}:{e.get('text','?')}" for e in raw_spans
        )
        return {
            "id": sample["id"],
            "pred_tags": bio,
            "pred_entities": pred_entities,
            "raw_output": raw_str,
            "parse_ok": True,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }

    def predict_many(self, samples: Iterable[NerSample]) -> list[dict]:
        out = []
        for s in samples:
            out.append(self.predict_one(s))
        return out
