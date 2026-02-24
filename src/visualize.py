"""Visualisations for the AI text sanitization study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import RESULTS_DIR


def _savefig(fig, name: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_effect_sizes(comparison: pd.DataFrame, model_label: str = ""):
    """Horizontal bar chart of Cohen's d for each marker."""
    df = comparison.sort_values("cohens_d")
    sig = df["significant_fdr"] if "significant_fdr" in df.columns else df["significant"]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
    colors = ["#e74c3c" if s else "#95a5a6" for s in sig]
    ax.barh(df["marker"], df["cohens_d"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (positive = original higher)")
    ax.set_title(f"Effect Sizes: Original vs. Rewrite{' — ' + model_label if model_label else ''}")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", label="Significant (FDR)"),
            plt.Rectangle((0, 0), 1, 1, fc="#95a5a6", label="Not significant"),
        ],
        loc="lower right",
    )
    _savefig(fig, f"effect_sizes_{model_label or 'all'}")


def plot_dimension_summary(summary: pd.DataFrame, model_label: str = ""):
    """Grouped bar chart showing n_significant and mean effect size per dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: number of significant markers
    axes[0].barh(summary["dimension"], summary["n_significant"], color="#3498db")
    axes[0].set_xlabel("# Significant markers (FDR-corrected)")
    axes[0].set_title("Significant Changes by Dimension")

    # Right: mean |Cohen's d|
    axes[1].barh(summary["dimension"], summary["mean_abs_cohens_d"], color="#2ecc71")
    axes[1].set_xlabel("Mean |Cohen's d|")
    axes[1].set_title("Effect Size by Dimension")

    fig.suptitle(
        f"Dimension-Level Summary{' — ' + model_label if model_label else ''}",
        fontsize=14,
    )
    fig.tight_layout()
    _savefig(fig, f"dimension_summary_{model_label or 'all'}")


def plot_paired_distributions(
    orig_markers: pd.DataFrame,
    rewrite_markers: pd.DataFrame,
    cols: list[str],
    model_label: str = "",
):
    """Overlay KDE plots of selected markers for original vs. rewrite."""
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        sns.kdeplot(orig_markers[col].dropna(), ax=ax, label="Original", color="#2c3e50")
        sns.kdeplot(rewrite_markers[col].dropna(), ax=ax, label="Rewrite", color="#e74c3c")
        ax.set_title(col)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Distribution Comparison{' — ' + model_label if model_label else ''}",
        fontsize=14,
    )
    fig.tight_layout()
    _savefig(fig, f"distributions_{model_label or 'all'}")
