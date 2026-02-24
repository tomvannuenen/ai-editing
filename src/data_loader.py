"""Load and sample creative writing from the WritingPrompts dataset.

Expected layout after Kaggle download & unzip into data/raw/:
    data/raw/writingPrompts/
        train.wp_source   (prompts, one per line)
        train.wp_target   (stories, one per line)
        valid.wp_source / valid.wp_target
        test.wp_source  / test.wp_target

Each line in wp_target is a story. Newlines within stories are encoded as
<newline> tokens.
"""

import random
import re
from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    N_SAMPLES,
    MIN_WORD_COUNT,
    MAX_WORD_COUNT,
    RANDOM_SEED,
)


def _clean_story(text: str) -> str:
    """Decode <newline> tokens and clean up whitespace."""
    text = text.replace("<newline>", "\n")
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def load_writing_prompts(split: str = "train") -> pd.DataFrame:
    """Load prompts + stories for a given split and return a DataFrame."""
    base = RAW_DIR / "writingPrompts"
    src_path = base / f"{split}.wp_source"
    tgt_path = base / f"{split}.wp_target"

    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {base}. Download from "
            "https://www.kaggle.com/datasets/ratthachat/writing-prompts "
            f"and unzip into {RAW_DIR}/writingPrompts/"
        )

    prompts = src_path.read_text(encoding="utf-8").strip().split("\n")
    stories = tgt_path.read_text(encoding="utf-8").strip().split("\n")

    df = pd.DataFrame({"prompt": prompts, "story_raw": stories})
    df["story"] = df["story_raw"].apply(_clean_story)
    df["word_count"] = df["story"].apply(_word_count)
    return df


def sample_stories(
    df: pd.DataFrame | None = None,
    n: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Filter by length and draw a random sample.

    Returns a DataFrame with columns: id, prompt, story, word_count.
    """
    if df is None:
        df = load_writing_prompts("train")

    filtered = df[
        (df["word_count"] >= MIN_WORD_COUNT)
        & (df["word_count"] <= MAX_WORD_COUNT)
    ].copy()

    if len(filtered) < n:
        print(
            f"Warning: only {len(filtered)} stories meet length criteria "
            f"({MIN_WORD_COUNT}–{MAX_WORD_COUNT} words). Using all of them."
        )
        sample = filtered
    else:
        sample = filtered.sample(n=n, random_state=seed)

    sample = sample[["prompt", "story", "word_count"]].reset_index(drop=True)
    sample.index.name = "id"
    sample = sample.reset_index()
    return sample


def save_sample(df: pd.DataFrame, tag: str = "sample") -> Path:
    """Persist the sampled stories to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{tag}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} stories → {out}")
    return out


def load_sample(tag: str = "sample") -> pd.DataFrame:
    """Load a previously saved sample."""
    path = PROCESSED_DIR / f"{tag}.parquet"
    return pd.read_parquet(path)
