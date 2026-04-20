"""Tests for src/data_loader.py — HuggingFace calls are mocked."""

import io
import sys
import tempfile
import zipfile
from unittest.mock import MagicMock, patch
import pytest

if "datasets" not in sys.modules:
    sys.modules["datasets"] = MagicMock()
if "huggingface_hub" not in sys.modules:
    sys.modules["huggingface_hub"] = MagicMock()

_FPB_LABELS = ["negative", "neutral", "positive"]


def _make_fpb_zip(n: int = 20) -> str:
    """Write a fake Sentences_75Agree.txt inside a zip and return the path."""
    lines = [f"Sentence number {i}@{_FPB_LABELS[i % 3]}" for i in range(n)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
            "\n".join(lines),
        )
    buf.seek(0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.write(buf.read())
    tmp.close()
    return tmp.name


def _fake_fiqa_rows():
    return [
        {"sentence": "strong gains", "score": 0.8},
        {"sentence": "minor losses", "score": -0.5},
        {"sentence": "filed report", "score": 0.0},
        {"sentence": "revenue down slightly", "score": -0.05},
        {"sentence": "massive profit", "score": 0.95},
        {"sentence": "dividend cut", "score": -0.3},
    ]


# ---------------------------------------------------------------------------
# FPB tests
# ---------------------------------------------------------------------------

@patch("huggingface_hub.hf_hub_download", return_value=_make_fpb_zip(20))
def test_fpb_returns_two_splits(_):
    from src.data_loader import load_fpb
    train, test = load_fpb()
    assert len(train) + len(test) == 20


@patch("huggingface_hub.hf_hub_download", return_value=_make_fpb_zip(20))
def test_fpb_test_fraction(_):
    from src.data_loader import load_fpb
    train, test = load_fpb(test_fraction=0.20, seed=42)
    assert len(test) == 4
    assert len(train) == 16


@patch("huggingface_hub.hf_hub_download", return_value=_make_fpb_zip(20))
def test_fpb_splits_tagged(_):
    from src.data_loader import load_fpb
    train, test = load_fpb()
    assert all(s["split"] == "train" for s in train)
    assert all(s["split"] == "test" for s in test)


@patch("huggingface_hub.hf_hub_download", return_value=_make_fpb_zip(3))
def test_fpb_label_mapping(_):
    from src.data_loader import load_fpb
    train, test = load_fpb(test_fraction=0.0, seed=0)
    labels = {s["label"] for s in train}
    assert labels == {"negative", "neutral", "positive"}


@patch("huggingface_hub.hf_hub_download", return_value=_make_fpb_zip(20))
def test_fpb_ids_unique(_):
    from src.data_loader import load_fpb
    train, test = load_fpb()
    all_ids = [s["id"] for s in train + test]
    assert len(all_ids) == len(set(all_ids))


@patch("huggingface_hub.hf_hub_download", return_value=_make_fpb_zip(20))
def test_fpb_dataset_field(_):
    from src.data_loader import load_fpb
    train, test = load_fpb()
    assert all(s["dataset"] == "FPB" for s in train + test)


# ---------------------------------------------------------------------------
# FiQA tests
# ---------------------------------------------------------------------------

@patch("datasets.load_dataset", return_value={"test": _fake_fiqa_rows()})
def test_fiqa_count(_):
    from src.data_loader import load_fiqa
    assert len(load_fiqa()) == 6


@patch("datasets.load_dataset", return_value={"test": _fake_fiqa_rows()})
def test_fiqa_all_test_split(_):
    from src.data_loader import load_fiqa
    assert all(s["split"] == "test" for s in load_fiqa())


@patch("datasets.load_dataset", return_value={"test": [{"sentence": "x", "score": 0.8}]})
def test_fiqa_positive_mapping(_):
    from src.data_loader import load_fiqa
    assert load_fiqa()[0]["label"] == "positive"


@patch("datasets.load_dataset", return_value={"test": [{"sentence": "x", "score": -0.5}]})
def test_fiqa_negative_mapping(_):
    from src.data_loader import load_fiqa
    assert load_fiqa()[0]["label"] == "negative"


@patch("datasets.load_dataset", return_value={"test": [{"sentence": "x", "score": 0.0}]})
def test_fiqa_neutral_mapping(_):
    from src.data_loader import load_fiqa
    assert load_fiqa()[0]["label"] == "neutral"


@patch("datasets.load_dataset", return_value={"test": [{"sentence": "x", "score": -0.05}]})
def test_fiqa_neutral_band_edge(_):
    from src.data_loader import load_fiqa
    assert load_fiqa(neutral_band=0.10)[0]["label"] == "neutral"


@patch("datasets.load_dataset", return_value={"test": _fake_fiqa_rows()})
def test_fiqa_dataset_field(_):
    from src.data_loader import load_fiqa
    assert all(s["dataset"] == "FiQA" for s in load_fiqa())


@patch("datasets.load_dataset", return_value={"test": _fake_fiqa_rows()})
def test_fiqa_ids_unique(_):
    from src.data_loader import load_fiqa
    ids = [s["id"] for s in load_fiqa()]
    assert len(ids) == len(set(ids))
