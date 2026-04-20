"""Load FPB + FiQA-SA and normalize to a unified Sample schema."""

from __future__ import annotations

from typing import TypedDict

from src.utils import get_logger, set_seed

logger = get_logger(__name__)

# TypedDict is documentation-only here; dicts are plain at runtime.
class Sample(TypedDict):
    id: str
    text: str
    label: str        # "positive" | "neutral" | "negative"
    dataset: str      # "FPB" | "FiQA"
    split: str        # "train" | "test"


_DS_SERVER = "https://datasets-server.huggingface.co"
_PAGE_SIZE = 100


def _fpb_label_names(config: str, headers: dict) -> list[str]:
    import requests
    resp = requests.get(
        f"{_DS_SERVER}/info",
        params={"dataset": "takala/financial_phrasebank", "config": config},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["dataset_info"]["features"]["label"]["names"]


def _fpb_fetch_rows(config: str, headers: dict) -> list[dict]:
    import requests
    rows, offset = [], 0
    while True:
        resp = requests.get(
            f"{_DS_SERVER}/rows",
            params={
                "dataset": "takala/financial_phrasebank",
                "config": config,
                "split": "train",
                "offset": offset,
                "length": _PAGE_SIZE,
            },
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        page = data.get("rows", [])
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        if offset >= data.get("num_rows_total", offset + 1):
            break
    return rows


def load_fpb(
    config: str = "sentences_75agree",
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Return (train, test) splits from Financial PhraseBank.

    Uses HuggingFace's Datasets Server REST API — no loading script,
    no version dependency on the `datasets` package.
    Reads HF_TOKEN from the environment for auth (optional but recommended).
    """
    import os

    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    logger.info("Loading FPB (%s) via Datasets Server API…", config)
    label_names = _fpb_label_names(config, headers)
    raw_rows = _fpb_fetch_rows(config, headers)
    logger.info("Fetched %d rows", len(raw_rows))

    all_samples: list[Sample] = []
    for i, r in enumerate(raw_rows):
        row = r["row"]
        raw_label = row["label"]
        label = (label_names[int(raw_label)] if isinstance(raw_label, int)
                 else raw_label.lower().strip())
        all_samples.append(
            Sample(
                id=f"FPB_{i:05d}",
                text=row["sentence"],
                label=label,
                dataset="FPB",
                split="",
            )
        )

    set_seed(seed)
    import random
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    n_test = int(len(all_samples) * test_fraction)
    test_samples = all_samples[:n_test]
    train_samples = all_samples[n_test:]

    for s in train_samples:
        s["split"] = "train"
    for s in test_samples:
        s["split"] = "test"

    logger.info("FPB: %d train, %d test", len(train_samples), len(test_samples))
    return train_samples, test_samples


def load_fiqa(neutral_band: float = 0.10) -> list[Sample]:
    """Return all FiQA-SA examples as a test split.

    Continuous score in [-1, 1] is mapped:
      score < -neutral_band  → negative
      -neutral_band ≤ score ≤ neutral_band → neutral
      score > neutral_band   → positive
    """
    from datasets import load_dataset

    logger.info("Loading FiQA-SA…")
    ds = load_dataset("ChanceFocus/fiqa-sentiment-classification")

    samples: list[Sample] = []
    idx = 0
    for split_name in ds.keys():
        for row in ds[split_name]:
            score = float(row["score"])
            if score < -neutral_band:
                label = "negative"
            elif score > neutral_band:
                label = "positive"
            else:
                label = "neutral"
            samples.append(
                Sample(
                    id=f"FiQA_{idx:05d}",
                    text=row["sentence"],
                    label=label,
                    dataset="FiQA",
                    split="test",
                )
            )
            idx += 1

    logger.info("FiQA: %d examples", len(samples))
    return samples
