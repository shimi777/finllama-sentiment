"""Wrapper for VADER sentiment scoring → 3-class label."""

from __future__ import annotations

from src.utils import get_logger

logger = get_logger(__name__)


class VADERRunner:
    """Maps VADER compound score to positive / neutral / negative.

    Thresholds follow the standard VADER recommendation:
      compound ≥  0.05 → positive
      compound ≤ -0.05 → negative
      else             → neutral
    """

    def __init__(self, positive_threshold: float = 0.05, negative_threshold: float = -0.05):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # lazy
        self._sia = SentimentIntensityAnalyzer()
        self._pos_thresh = positive_threshold
        self._neg_thresh = negative_threshold
        logger.info("VADERRunner ready")

    def predict(self, texts: list[str]) -> list[str]:
        """Return a canonical label for each input text."""
        labels: list[str] = []
        for text in texts:
            scores = self._sia.polarity_scores(text)
            c = scores["compound"]
            if c >= self._pos_thresh:
                labels.append("positive")
            elif c <= self._neg_thresh:
                labels.append("negative")
            else:
                labels.append("neutral")
        return labels
