"""Wrapper for ProsusAI/finbert classification."""

from __future__ import annotations

from src.utils import get_logger

logger = get_logger(__name__)

_FINBERT_LABEL_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral"}


class FinBERTRunner:
    """Run inference with ProsusAI/finbert via HuggingFace pipeline."""

    def __init__(self, hf_id: str = "ProsusAI/finbert", device: int = -1, batch_size: int = 32):
        from transformers import pipeline  # lazy
        self._pipe = pipeline(
            "text-classification",
            model=hf_id,
            device=device,
            truncation=True,
            max_length=512,
            batch_size=batch_size,
        )
        self._batch_size = batch_size
        logger.info("FinBERTRunner ready (device=%d)", device)

    def predict(self, texts: list[str]) -> list[str]:
        """Return canonical labels for a list of texts."""
        results = self._pipe(texts, batch_size=self._batch_size)
        labels: list[str] = []
        for r in results:
            raw = r["label"].lower()
            labels.append(_FINBERT_LABEL_MAP.get(raw, "neutral"))
        return labels
