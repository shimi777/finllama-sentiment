"""Tests for src/data_loader.py — HuggingFace calls are mocked."""

import sys
from unittest.mock import MagicMock, patch, call
import pytest

if "datasets" not in sys.modules:
    sys.modules["datasets"] = MagicMock()
if "huggingface_hub" not in sys.modules:
    sys.modules["huggingface_hub"] = MagicMock()

_FPB_LABELS = ["negative", "neutral", "positive"]


def _make_info_response():
    m = MagicMock()
    m.json.return_value = {
        "dataset_info": {
            "features": {
                "label": {"names": _FPB_LABELS}
            }
        }
    }
    return m


def _make_rows_response(n: int, done: bool = True):
    rows = [
        {"row": {"sentence": f"sentence {i}", "label": i % 3}}
        for i in range(n)
    ]
    m = MagicMock()
    m.json.return_value = {"rows": rows, "num_rows_total": n}
    return m


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
# FPB tests — mock requests.get
# ---------------------------------------------------------------------------

def _fpb_get_side_effect(n):
    """Return info response on first call, rows response on second."""
    responses = [_make_info_response(), _make_rows_response(n)]
    return iter(responses).__next__


@patch("requests.get")
def test_fpb_returns_two_splits(mock_get):
    mock_get.side_effect = [_make_info_response(), _make_rows_response(20)]
    from src.data_loader import load_fpb
    train, test = load_fpb()
    assert len(train) + len(test) == 20


@patch("requests.get")
def test_fpb_test_fraction(mock_get):
    mock_get.side_effect = [_make_info_response(), _make_rows_response(20)]
    from src.data_loader import load_fpb
    train, test = load_fpb(test_fraction=0.20, seed=42)
    assert len(test) == 4
    assert len(train) == 16


@patch("requests.get")
def test_fpb_splits_tagged(mock_get):
    mock_get.side_effect = [_make_info_response(), _make_rows_response(20)]
    from src.data_loader import load_fpb
    train, test = load_fpb()
    assert all(s["split"] == "train" for s in train)
    assert all(s["split"] == "test" for s in test)


@patch("requests.get")
def test_fpb_label_mapping(mock_get):
    mock_get.side_effect = [_make_info_response(), _make_rows_response(3)]
    from src.data_loader import load_fpb
    train, test = load_fpb(test_fraction=0.0, seed=0)
    labels = {s["label"] for s in train}
    assert labels == {"negative", "neutral", "positive"}


@patch("requests.get")
def test_fpb_ids_unique(mock_get):
    mock_get.side_effect = [_make_info_response(), _make_rows_response(20)]
    from src.data_loader import load_fpb
    train, test = load_fpb()
    all_ids = [s["id"] for s in train + test]
    assert len(all_ids) == len(set(all_ids))


@patch("requests.get")
def test_fpb_dataset_field(mock_get):
    mock_get.side_effect = [_make_info_response(), _make_rows_response(20)]
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
