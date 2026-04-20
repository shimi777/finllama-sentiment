"""Smoke tests for model runners — all heavy dependencies are mocked."""

import sys
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Stub out unavailable packages before any import
# ---------------------------------------------------------------------------
for pkg in ["vaderSentiment", "vaderSentiment.vaderSentiment",
            "transformers", "torch", "tqdm", "bitsandbytes"]:
    if pkg not in sys.modules:
        sys.modules[pkg] = MagicMock()


# ---------------------------------------------------------------------------
# VADERRunner
# ---------------------------------------------------------------------------

def test_vader_positive():
    from src.models.vader_runner import VADERRunner
    runner = VADERRunner.__new__(VADERRunner)
    runner._pos_thresh = 0.05
    runner._neg_thresh = -0.05
    mock_sia = MagicMock()
    mock_sia.polarity_scores.return_value = {"compound": 0.8}
    runner._sia = mock_sia
    assert runner.predict(["great earnings"])[0] == "positive"


def test_vader_negative():
    from src.models.vader_runner import VADERRunner
    runner = VADERRunner.__new__(VADERRunner)
    runner._pos_thresh = 0.05
    runner._neg_thresh = -0.05
    mock_sia = MagicMock()
    mock_sia.polarity_scores.return_value = {"compound": -0.6}
    runner._sia = mock_sia
    assert runner.predict(["terrible loss"])[0] == "negative"


def test_vader_neutral():
    from src.models.vader_runner import VADERRunner
    runner = VADERRunner.__new__(VADERRunner)
    runner._pos_thresh = 0.05
    runner._neg_thresh = -0.05
    mock_sia = MagicMock()
    mock_sia.polarity_scores.return_value = {"compound": 0.0}
    runner._sia = mock_sia
    assert runner.predict(["company filed report"])[0] == "neutral"


def test_vader_batch_length():
    from src.models.vader_runner import VADERRunner
    runner = VADERRunner.__new__(VADERRunner)
    runner._pos_thresh = 0.05
    runner._neg_thresh = -0.05
    mock_sia = MagicMock()
    mock_sia.polarity_scores.return_value = {"compound": 0.1}
    runner._sia = mock_sia
    assert len(runner.predict(["a", "b", "c"])) == 3


# ---------------------------------------------------------------------------
# FinBERTRunner
# ---------------------------------------------------------------------------

def test_finbert_label_mapping():
    from src.models.finbert_runner import FinBERTRunner
    runner = FinBERTRunner.__new__(FinBERTRunner)
    mock_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.9}])
    runner._pipe = mock_pipe
    runner._batch_size = 32
    assert runner.predict(["revenue grew 10%"])[0] == "positive"


def test_finbert_unknown_label_falls_back_to_neutral():
    from src.models.finbert_runner import FinBERTRunner
    runner = FinBERTRunner.__new__(FinBERTRunner)
    runner._pipe = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.5}])
    runner._batch_size = 32
    assert runner.predict(["x"])[0] == "neutral"


def test_finbert_batch_length():
    from src.models.finbert_runner import FinBERTRunner
    runner = FinBERTRunner.__new__(FinBERTRunner)
    runner._pipe = MagicMock(return_value=[
        {"label": "positive", "score": 0.9},
        {"label": "negative", "score": 0.8},
    ])
    runner._batch_size = 32
    assert len(runner.predict(["a", "b"])) == 2
