#!/usr/bin/env python3
"""Main pipeline for the AI text sanitization study.

Usage
-----
    # Run the full pipeline end-to-end:
    python run_pipeline.py

    # Run individual stages:
    python run_pipeline.py --stage sample
    python run_pipeline.py --stage rewrite
    python run_pipeline.py --stage analyze
    python run_pipeline.py --stage compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import PROCESSED_DIR, RESULTS_DIR, REWRITE_MODELS


# ── Stage 1: Sample stories ───────────────────────────────────────────

def stage_sample():
    from src.data_loader import load_writing_prompts, sample_stories, save_sample

    print("=" * 60)
    print("STAGE 1: Sampling stories from WritingPrompts dataset")
    print("=" * 60)

    df = load_writing_prompts("train")
    print(f"Loaded {len(df)} stories from training set.")

    sample = sample_stories(df)
    save_sample(sample)
    print(f"Sampled {len(sample)} stories.")
    return sample


# ── Stage 2: Rewrite with LLMs ────────────────────────────────────────

def stage_rewrite():
    from src.data_loader import load_sample
    from src.rewriter import rewrite_stories, save_rewrites

    print("=" * 60)
    print("STAGE 2: Rewriting stories with LLMs")
    print("=" * 60)

    sample = load_sample("sample")
    result = rewrite_stories(sample)
    save_rewrites(result)
    return result


# ── Stage 3: Compute linguistic markers ────────────────────────────────

def stage_analyze():
    from src.rewriter import load_rewrites
    from src.markers import compute_all_markers

    print("=" * 60)
    print("STAGE 3: Computing linguistic markers")
    print("=" * 60)

    df = load_rewrites()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Identify text columns to analyze
    text_cols = ["story"]  # original
    for m in REWRITE_MODELS:
        col = f"rewrite_{m['label']}"
        if col in df.columns:
            text_cols.append(col)

    for col in text_cols:
        out_path = PROCESSED_DIR / f"markers_{col}.parquet"

        # Resume support: skip if already computed
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            if len(existing) == len(df):
                print(f"[{col}] markers already computed. Skipping.")
                continue

        print(f"\n[{col}] Computing markers for {len(df)} texts…")
        records = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=col):
            text = row[col]
            if pd.isna(text) or not text.strip():
                records.append({"id": row["id"]})
                continue
            markers = compute_all_markers(text)
            markers["id"] = row["id"]
            records.append(markers)

        markers_df = pd.DataFrame(records)
        markers_df.to_parquet(out_path, index=False)
        print(f"[{col}] Saved → {out_path}")


# ── Stage 4: Statistical comparison & visualisation ────────────────────

def stage_compare():
    from src.stats import compare_markers, summary_by_dimension
    from src.visualize import (
        plot_effect_sizes,
        plot_dimension_summary,
        plot_paired_distributions,
    )

    print("=" * 60)
    print("STAGE 4: Statistical comparison & visualisation")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    orig_path = PROCESSED_DIR / "markers_story.parquet"
    if not orig_path.exists():
        print("Original markers not found. Run --stage analyze first.")
        return

    orig = pd.read_parquet(orig_path)

    for m in REWRITE_MODELS:
        label = m["label"]
        rewrite_path = PROCESSED_DIR / f"markers_rewrite_{label}.parquet"
        if not rewrite_path.exists():
            print(f"[{label}] Rewrite markers not found. Skipping.")
            continue

        rewrite = pd.read_parquet(rewrite_path)
        print(f"\n{'─' * 40}")
        print(f"Comparing: original vs. {label}")
        print(f"{'─' * 40}")

        comparison = compare_markers(orig, rewrite)
        comparison.to_csv(RESULTS_DIR / f"comparison_{label}.csv", index=False)

        summary = summary_by_dimension(comparison)
        summary.to_csv(RESULTS_DIR / f"summary_{label}.csv", index=False)

        # Print highlights
        sig = comparison[comparison.get("significant_fdr", comparison["significant"])]
        print(f"\n{len(sig)} significant markers (of {len(comparison)}):\n")
        for _, row in sig.iterrows():
            arrow = "↓" if row["direction"] == "decrease" else "↑"
            print(
                f"  {arrow} {row['marker']:35s}  "
                f"d={row['cohens_d']:+.3f}  "
                f"Δ={row['pct_change']:+.1f}%  "
                f"p={row['p_value']:.2e}"
            )

        print(f"\nDimension summary:")
        print(summary.to_string(index=False))

        # Plots
        plot_effect_sizes(comparison, label)
        plot_dimension_summary(summary, label)

        # Distribution plots for the top-6 largest effects
        top_markers = comparison.head(6)["marker"].tolist()
        available = [c for c in top_markers if c in orig.columns and c in rewrite.columns]
        if available:
            plot_paired_distributions(orig, rewrite, available, label)

    # ── Cross-model comparison (if >1 model) ──────────────────────────
    model_labels = [m["label"] for m in REWRITE_MODELS]
    comparisons = {}
    for label in model_labels:
        path = RESULTS_DIR / f"comparison_{label}.csv"
        if path.exists():
            comparisons[label] = pd.read_csv(path)

    if len(comparisons) > 1:
        print(f"\n{'=' * 40}")
        print("Cross-model effect size comparison")
        print(f"{'=' * 40}")

        merged = None
        for label, comp in comparisons.items():
            subset = comp[["marker", "cohens_d"]].rename(columns={"cohens_d": f"d_{label}"})
            if merged is None:
                merged = subset
            else:
                merged = merged.merge(subset, on="marker", how="outer")

        if merged is not None:
            merged.to_csv(RESULTS_DIR / "cross_model_effects.csv", index=False)
            print(merged.to_string(index=False))


# ── CLI ────────────────────────────────────────────────────────────────

STAGES = {
    "sample": stage_sample,
    "rewrite": stage_rewrite,
    "analyze": stage_analyze,
    "compare": stage_compare,
}


def main():
    parser = argparse.ArgumentParser(description="AI Text Sanitization Study Pipeline")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        default=None,
        help="Run a single stage. Omit to run the full pipeline.",
    )
    args = parser.parse_args()

    if args.stage:
        STAGES[args.stage]()
    else:
        for name, fn in STAGES.items():
            fn()

    print("\nDone.")


if __name__ == "__main__":
    main()
