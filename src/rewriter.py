"""Send stories to LLM APIs for rewriting and collect the results."""

import json
import os
import time
from pathlib import Path

import pandas as pd

from src.config import (
    PROCESSED_DIR,
    REWRITE_MODELS,
    REWRITE_PROMPT,
    REWRITE_MAX_TOKENS,
    REWRITE_TEMPERATURE,
)

# ── Provider helpers ───────────────────────────────────────────────────

def _call_openai(text: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(text=text)}],
        max_tokens=REWRITE_MAX_TOKENS,
        temperature=REWRITE_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(text: str, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()  # uses ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model=model,
        max_tokens=REWRITE_MAX_TOKENS,
        temperature=REWRITE_TEMPERATURE,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(text=text)}],
    )
    return response.content[0].text.strip()


PROVIDERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


# ── Main rewriting loop ───────────────────────────────────────────────

def rewrite_stories(
    df: pd.DataFrame,
    models: list[dict] | None = None,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """Rewrite each story with each model. Adds columns like `rewrite_gpt4o`.

    Supports resumption: if a checkpoint file exists, already-completed rows
    are skipped.

    Parameters
    ----------
    df : DataFrame with columns ``id`` and ``story``.
    models : list of model dicts (provider, model, label). Defaults to
             ``config.REWRITE_MODELS``.
    checkpoint_every : save progress every N stories.
    """
    if models is None:
        models = REWRITE_MODELS

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = PROCESSED_DIR / "rewrite_checkpoint.parquet"

    # Resume from checkpoint if available
    if ckpt_path.exists():
        result = pd.read_parquet(ckpt_path)
        print(f"Resuming from checkpoint ({len(result)} rows on disk).")
    else:
        result = df.copy()
        for m in models:
            col = f"rewrite_{m['label']}"
            if col not in result.columns:
                result[col] = None

    total = len(result)

    for m in models:
        label = m["label"]
        col = f"rewrite_{label}"
        provider_fn = PROVIDERS[m["provider"]]

        # Find rows that still need rewriting for this model
        todo_mask = result[col].isna()
        todo_ids = result.loc[todo_mask].index.tolist()

        if not todo_ids:
            print(f"[{label}] All {total} stories already rewritten. Skipping.")
            continue

        print(f"[{label}] Rewriting {len(todo_ids)}/{total} stories with {m['model']}…")

        for count, idx in enumerate(todo_ids, 1):
            story = result.at[idx, "story"]
            sid = result.at[idx, "id"]

            try:
                rewrite = provider_fn(story, m["model"])
                result.at[idx, col] = rewrite
            except Exception as e:
                print(f"  [!] id={sid} failed: {e}")
                # leave as None so it can be retried later

            # Rate-limit courtesy pause
            time.sleep(0.5)

            if count % checkpoint_every == 0:
                result.to_parquet(ckpt_path, index=False)
                print(f"  [{label}] checkpoint at {count}/{len(todo_ids)}")

        # Final save after each model
        result.to_parquet(ckpt_path, index=False)
        print(f"[{label}] Done.")

    return result


def save_rewrites(df: pd.DataFrame, tag: str = "rewrites") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{tag}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved rewrites → {out}")
    return out


def load_rewrites(tag: str = "rewrites") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{tag}.parquet")
