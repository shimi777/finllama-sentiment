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


# TheFinAI/en-fpb uses string labels directly; no int mapping needed.
_FPB_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_fpb(
    config: str = "sentences_75agree",
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Return (train, test) splits from Financial PhraseBank.

    Loads from takala/financial_phrasebank via its auto-generated Parquet
    export (revision='refs/convert/parquet'), which avoids the loading-script
    restriction in datasets>=3.0.
    The dataset has only a 'train' split, so we carve out test_fraction
    ourselves using a reproducible random seed.
    """
    from datasets import load_dataset

    logger.info("Loading FPB via Parquet export (takala/financial_phrasebank, %s)…", config)
    ds = load_dataset(
        "takala/financial_phrasebank",
        config,
        revision="refs/convert/parquet",
    )
    raw = ds["train"]

    all_samples: list[Sample] = []
    for i, row in enumerate(raw):
        # TheFinAI/en-fpb uses string labels; fall back to int map for safety
        raw_label = row["label"]
        label = (raw_label.lower().strip() if isinstance(raw_label, str)
                 else _FPB_LABEL_MAP[int(raw_label)])
        all_samples.append(
            Sample(
                id=f"FPB_{i:05d}",
                text=row["sentence"],
                label=label,
                dataset="FPB",
                split="",  # filled below
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
