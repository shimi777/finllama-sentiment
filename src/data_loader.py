"""Load FPB + FiQA-SA and normalize to a unified Sample schema."""

from __future__ import annotations

from typing import TypedDict

from src.utils import get_logger

logger = get_logger(__name__)


class Sample(TypedDict):
    id: str
    text: str
    label: str        # "positive" | "neutral" | "negative"
    dataset: str      # "FPB" | "FiQA"
    split: str        # "train" | "test"


def load_fpb(
    config: str = "sentences_75agree",
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Return (train, test) splits from Financial PhraseBank.

    Downloads the raw zip directly from HuggingFace Hub — no loading script,
    no `datasets` library required for FPB.
    Reads HF_TOKEN from the environment for auth (optional but recommended).
    """
    import os
    import random
    import zipfile
    from huggingface_hub import hf_hub_download, list_repo_files

    token = os.environ.get("HF_TOKEN") or None

    logger.info("Loading FPB (%s) via HuggingFace Hub download…", config)

    # Discover the zip filename (may be nested under data/ or at root)
    repo_files = list(
        list_repo_files("takala/financial_phrasebank", repo_type="dataset", token=token)
    )
    zip_in_repo = next((f for f in repo_files if f.endswith(".zip")), None)
    if zip_in_repo is None:
        raise RuntimeError(
            f"No .zip file found in takala/financial_phrasebank. Files: {repo_files}"
        )

    local_zip = hf_hub_download(
        "takala/financial_phrasebank",
        filename=zip_in_repo,
        repo_type="dataset",
        token=token,
    )

    # Map config name → filename fragment inside the zip
    _SUFFIX = {
        "sentences_allagree": "AllAgree",
        "sentences_75agree":  "75Agree",
        "sentences_66agree":  "66Agree",
        "sentences_50agree":  "50Agree",
    }
    suffix = _SUFFIX.get(config, "75Agree")

    with zipfile.ZipFile(local_zip) as zf:
        names = zf.namelist()
        txt_match = next(
            (n for n in names if suffix in n and n.endswith(".txt")), None
        )
        if txt_match is None:
            raise RuntimeError(
                f"No *{suffix}*.txt in zip. Contents: {names}"
            )
        with zf.open(txt_match) as fh:
            lines = fh.read().decode("latin-1").splitlines()

    all_samples: list[Sample] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or "@" not in line:
            continue
        text, label_str = line.rsplit("@", 1)
        all_samples.append(
            Sample(
                id=f"FPB_{i:05d}",
                text=text.strip(),
                label=label_str.strip().lower(),
                dataset="FPB",
                split="",
            )
        )

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    n_test = int(len(all_samples) * test_fraction)
    test_samples  = all_samples[:n_test]
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
