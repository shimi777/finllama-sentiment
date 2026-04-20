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


_CONFIG_TO_FILE = {
    "sentences_50agree": "Sentences_50Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}


def _parse_fpb_zip(zip_path: str, filename: str) -> list[dict]:
    """Extract sentence/label rows from the FPB zip file."""
    import zipfile

    candidates = [
        f"FinancialPhraseBank-v1.0/{filename}",
        filename,
    ]
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        target = next((c for c in candidates if c in names), None)
        if target is None:
            # fuzzy match — case-insensitive
            lower = filename.lower()
            target = next((n for n in names if n.lower().endswith(lower)), None)
        if target is None:
            raise FileNotFoundError(
                f"Could not find {filename} in zip. Contents: {names}"
            )
        with zf.open(target) as f:
            content = f.read().decode("latin-1")

    rows = []
    for line in content.strip().splitlines():
        line = line.strip()
        if "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        rows.append({"sentence": sentence.strip(), "label": label.strip().lower()})
    return rows


def load_fpb(
    config: str = "sentences_75agree",
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Return (train, test) splits from Financial PhraseBank.

    Downloads the raw ZIP from takala/financial_phrasebank via hf_hub_download,
    bypassing the loading script entirely (works with datasets>=3.0).
    """
    from huggingface_hub import hf_hub_download

    filename = _CONFIG_TO_FILE.get(config.lower(), "Sentences_75Agree.txt")
    logger.info("Loading FPB (%s) from raw ZIP…", filename)

    zip_path = hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename="FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    rows = _parse_fpb_zip(zip_path, filename)

    all_samples: list[Sample] = []
    for i, row in enumerate(rows):
        all_samples.append(
            Sample(
                id=f"FPB_{i:05d}",
                text=row["sentence"],
                label=row["label"],
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
