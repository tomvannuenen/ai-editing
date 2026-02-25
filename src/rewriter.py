"""Send stories to LLM APIs for rewriting and collect the results.

Supports two modes:
- **batch** (default): Uses the Batch API for both OpenAI and Anthropic.
  50% cheaper, results within 24 hours.
- **sync**: Calls the standard API one-at-a-time with checkpointing.
"""

import json
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

# ═══════════════════════════════════════════════════════════════════════
# SYNCHRONOUS (one-at-a-time) helpers
# ═══════════════════════════════════════════════════════════════════════

def _call_openai(text: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(text=text)}],
        max_tokens=REWRITE_MAX_TOKENS,
        temperature=REWRITE_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(text: str, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
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


# ═══════════════════════════════════════════════════════════════════════
# BATCH API — OpenAI
# ═══════════════════════════════════════════════════════════════════════

def _submit_openai_batch(
    stories: list[tuple[int, str]],
    model: str,
    label: str,
) -> str:
    """Create a .jsonl file of requests and submit an OpenAI batch job.

    Returns the batch ID.
    """
    from openai import OpenAI

    client = OpenAI()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = PROCESSED_DIR / f"batch_input_{label}.jsonl"

    with open(jsonl_path, "w") as f:
        for story_id, text in stories:
            request = {
                "custom_id": str(story_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": REWRITE_PROMPT.format(text=text)}
                    ],
                    "max_tokens": REWRITE_MAX_TOKENS,
                    "temperature": REWRITE_TEMPERATURE,
                },
            }
            f.write(json.dumps(request) + "\n")

    # Upload the file
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    # Create the batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"  [OpenAI] Batch submitted: {batch.id}")
    # Persist batch ID for later retrieval
    meta_path = PROCESSED_DIR / f"batch_meta_{label}.json"
    meta_path.write_text(json.dumps({"batch_id": batch.id, "label": label}))
    return batch.id


def _poll_openai_batch(label: str, poll_interval: int = 60) -> dict[int, str]:
    """Poll until the OpenAI batch completes. Returns {story_id: rewrite}."""
    from openai import OpenAI

    client = OpenAI()
    meta_path = PROCESSED_DIR / f"batch_meta_{label}.json"
    meta = json.loads(meta_path.read_text())
    batch_id = meta["batch_id"]

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"  [OpenAI] Batch {batch_id}: {status}")

        if status == "completed":
            break
        elif status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"OpenAI batch {batch_id} ended with status: {status}")

        time.sleep(poll_interval)

    # Download results
    content = client.files.content(batch.output_file_id)
    results = {}
    for line in content.text.strip().split("\n"):
        obj = json.loads(line)
        story_id = int(obj["custom_id"])
        if obj.get("error"):
            print(f"  [!] story {story_id} failed: {obj['error']}")
            continue
        text = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
        results[story_id] = text

    return results


# ═══════════════════════════════════════════════════════════════════════
# BATCH API — Anthropic
# ═══════════════════════════════════════════════════════════════════════

def _submit_anthropic_batch(
    stories: list[tuple[int, str]],
    model: str,
    label: str,
) -> str:
    """Submit an Anthropic Message Batch. Returns the batch ID."""
    from anthropic import Anthropic

    client = Anthropic()

    requests = []
    for story_id, text in stories:
        requests.append({
            "custom_id": str(story_id),
            "params": {
                "model": model,
                "max_tokens": REWRITE_MAX_TOKENS,
                "temperature": REWRITE_TEMPERATURE,
                "messages": [
                    {"role": "user", "content": REWRITE_PROMPT.format(text=text)}
                ],
            },
        })

    batch = client.messages.batches.create(requests=requests)
    print(f"  [Anthropic] Batch submitted: {batch.id}")

    meta_path = PROCESSED_DIR / f"batch_meta_{label}.json"
    meta_path.write_text(json.dumps({"batch_id": batch.id, "label": label}))
    return batch.id


def _poll_anthropic_batch(label: str, poll_interval: int = 60) -> dict[int, str]:
    """Poll until the Anthropic batch completes. Returns {story_id: rewrite}."""
    from anthropic import Anthropic

    client = Anthropic()
    meta_path = PROCESSED_DIR / f"batch_meta_{label}.json"
    meta = json.loads(meta_path.read_text())
    batch_id = meta["batch_id"]

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        print(f"  [Anthropic] Batch {batch_id}: {status}")

        if status == "ended":
            break

        time.sleep(poll_interval)

    # Stream results
    results = {}
    for result in client.messages.batches.results(batch_id):
        story_id = int(result.custom_id)
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text.strip()
            results[story_id] = text
        else:
            print(f"  [!] story {story_id} failed: {result.result.type}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# BATCH DISPATCH
# ═══════════════════════════════════════════════════════════════════════

BATCH_SUBMIT = {
    "openai": _submit_openai_batch,
    "anthropic": _submit_anthropic_batch,
}

BATCH_POLL = {
    "openai": _poll_openai_batch,
    "anthropic": _poll_anthropic_batch,
}


def rewrite_stories_batch(
    df: pd.DataFrame,
    models: list[dict] | None = None,
    poll_interval: int = 60,
) -> pd.DataFrame:
    """Submit batch jobs for all models, poll for completion, merge results.

    Uses the Batch API (50% discount) for both OpenAI and Anthropic.
    """
    if models is None:
        models = REWRITE_MODELS

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = PROCESSED_DIR / "rewrite_checkpoint.parquet"

    if ckpt_path.exists():
        result = pd.read_parquet(ckpt_path)
        print(f"Resuming from checkpoint ({len(result)} rows).")
    else:
        result = df.copy()

    for m in models:
        label = m["label"]
        col = f"rewrite_{label}"
        if col not in result.columns:
            result[col] = None

    # ── Submit phase ──────────────────────────────────────────────────
    pending_batches = []
    for m in models:
        label = m["label"]
        col = f"rewrite_{label}"

        # Check if results already exist for this model
        todo_mask = result[col].isna()
        if not todo_mask.any():
            print(f"[{label}] All stories already rewritten. Skipping.")
            continue

        # Check if a batch was already submitted (resume support)
        meta_path = PROCESSED_DIR / f"batch_meta_{label}.json"
        if meta_path.exists():
            print(f"[{label}] Batch already submitted. Will poll for results.")
            pending_batches.append(m)
            continue

        stories = [
            (int(row["id"]), row["story"])
            for _, row in result[todo_mask].iterrows()
        ]
        print(f"[{label}] Submitting batch of {len(stories)} stories…")
        BATCH_SUBMIT[m["provider"]](stories, m["model"], label)
        pending_batches.append(m)

    # ── Poll phase ────────────────────────────────────────────────────
    for m in pending_batches:
        label = m["label"]
        col = f"rewrite_{label}"
        print(f"\n[{label}] Polling for batch results…")

        batch_results = BATCH_POLL[m["provider"]](label, poll_interval)

        # Merge results into the DataFrame
        id_to_idx = {int(row["id"]): idx for idx, row in result.iterrows()}
        filled = 0
        for story_id, rewrite_text in batch_results.items():
            if story_id in id_to_idx:
                result.at[id_to_idx[story_id], col] = rewrite_text
                filled += 1

        print(f"[{label}] Got {filled}/{len(batch_results)} results.")
        result.to_parquet(ckpt_path, index=False)

        # Clean up batch metadata
        meta_path = PROCESSED_DIR / f"batch_meta_{label}.json"
        if meta_path.exists():
            meta_path.unlink()

    return result


# ═══════════════════════════════════════════════════════════════════════
# SYNCHRONOUS FALLBACK
# ═══════════════════════════════════════════════════════════════════════

def rewrite_stories_sync(
    df: pd.DataFrame,
    models: list[dict] | None = None,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """Rewrite stories one-at-a-time with checkpointing. No batch discount."""
    if models is None:
        models = REWRITE_MODELS

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = PROCESSED_DIR / "rewrite_checkpoint.parquet"

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

            time.sleep(0.5)

            if count % checkpoint_every == 0:
                result.to_parquet(ckpt_path, index=False)
                print(f"  [{label}] checkpoint at {count}/{len(todo_ids)}")

        result.to_parquet(ckpt_path, index=False)
        print(f"[{label}] Done.")

    return result


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def rewrite_stories(
    df: pd.DataFrame,
    models: list[dict] | None = None,
    mode: str = "batch",
    **kwargs,
) -> pd.DataFrame:
    """Rewrite stories. Set mode='batch' (default, 50% off) or 'sync'."""
    if mode == "batch":
        return rewrite_stories_batch(df, models, **kwargs)
    else:
        return rewrite_stories_sync(df, models, **kwargs)


def save_rewrites(df: pd.DataFrame, tag: str = "rewrites") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{tag}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved rewrites → {out}")
    return out


def load_rewrites(tag: str = "rewrites") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{tag}.parquet")
