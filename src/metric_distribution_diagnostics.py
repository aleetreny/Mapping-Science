from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.embedding_space_metrics import CORE_EMBEDDING_METRIC_COLUMNS, DIAGNOSTIC_COLUMNS
from src.morphology_metrics import CORE_METRIC_COLUMNS_V2, DIAGNOSTIC_METRIC_COLUMNS

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


UMAP_METRIC_COLUMNS = CORE_METRIC_COLUMNS_V2 + DIAGNOSTIC_METRIC_COLUMNS
EMBEDDING_METRIC_COLUMNS = CORE_EMBEDDING_METRIC_COLUMNS + DIAGNOSTIC_COLUMNS

SUMMARY_COLUMNS = [
    "metric_name",
    "metric_family",
    "n_rows",
    "n_non_missing",
    "missing_share",
    "n_unique",
    "unique_share",
    "zero_share",
    "mean",
    "std",
    "min",
    "p01",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "max",
    "iqr",
    "is_constant",
    "is_low_unique",
    "is_high_missing",
    "is_zero_dominated",
    "recommended_review_flag",
]


def summarize_metric_series(
    values: pd.Series,
    *,
    metric_name: str,
    metric_family: str,
    high_missing_threshold: float = 0.30,
    low_unique_threshold: int = 5,
    low_unique_share_threshold: float = 0.05,
    zero_dominated_threshold: float = 0.80,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    n_rows = int(len(numeric))
    non_missing = numeric.dropna()
    n_non_missing = int(len(non_missing))
    missing_share = float(1.0 - (n_non_missing / n_rows)) if n_rows else np.nan
    n_unique = int(non_missing.nunique(dropna=True))
    unique_share = float(n_unique / n_non_missing) if n_non_missing else np.nan
    zero_share = float((non_missing == 0).mean()) if n_non_missing else np.nan

    if n_non_missing:
        quantiles = non_missing.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        mean = float(non_missing.mean())
        std = float(non_missing.std(ddof=0))
        minimum = float(non_missing.min())
        maximum = float(non_missing.max())
        p01 = float(quantiles.loc[0.01])
        p05 = float(quantiles.loc[0.05])
        p25 = float(quantiles.loc[0.25])
        median = float(quantiles.loc[0.50])
        p75 = float(quantiles.loc[0.75])
        p95 = float(quantiles.loc[0.95])
        p99 = float(quantiles.loc[0.99])
        iqr = float(p75 - p25)
    else:
        mean = std = minimum = maximum = p01 = p05 = p25 = median = p75 = p95 = p99 = iqr = np.nan

    is_constant = n_unique <= 1
    is_low_unique = (n_unique <= low_unique_threshold) or (
        pd.notna(unique_share) and unique_share <= low_unique_share_threshold
    )
    is_high_missing = pd.notna(missing_share) and missing_share >= high_missing_threshold
    is_zero_dominated = pd.notna(zero_share) and zero_share >= zero_dominated_threshold
    recommended = bool(
        is_constant or is_low_unique or is_high_missing or is_zero_dominated
    )

    return {
        "metric_name": metric_name,
        "metric_family": metric_family,
        "n_rows": n_rows,
        "n_non_missing": n_non_missing,
        "missing_share": missing_share,
        "n_unique": n_unique,
        "unique_share": unique_share,
        "zero_share": zero_share,
        "mean": mean,
        "std": std,
        "min": minimum,
        "p01": p01,
        "p05": p05,
        "p25": p25,
        "median": median,
        "p75": p75,
        "p95": p95,
        "p99": p99,
        "max": maximum,
        "iqr": iqr,
        "is_constant": bool(is_constant),
        "is_low_unique": bool(is_low_unique),
        "is_high_missing": bool(is_high_missing),
        "is_zero_dominated": bool(is_zero_dominated),
        "recommended_review_flag": recommended,
    }


def summarize_metric_frame(
    frame: pd.DataFrame,
    *,
    metric_family: str,
    metric_columns: list[str],
    high_missing_threshold: float = 0.30,
    low_unique_threshold: int = 5,
    low_unique_share_threshold: float = 0.05,
    zero_dominated_threshold: float = 0.80,
) -> pd.DataFrame:
    rows = []
    for column in metric_columns:
        if column not in frame.columns:
            continue
        rows.append(
            summarize_metric_series(
                frame[column],
                metric_name=column,
                metric_family=metric_family,
                high_missing_threshold=high_missing_threshold,
                low_unique_threshold=low_unique_threshold,
                low_unique_share_threshold=low_unique_share_threshold,
                zero_dominated_threshold=zero_dominated_threshold,
            )
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_distribution_summary(
    umap_metrics: pd.DataFrame,
    embedding_metrics: pd.DataFrame,
    *,
    high_missing_threshold: float = 0.30,
    low_unique_threshold: int = 5,
    low_unique_share_threshold: float = 0.05,
    zero_dominated_threshold: float = 0.80,
) -> pd.DataFrame:
    umap_summary = summarize_metric_frame(
        umap_metrics,
        metric_family="umap_projected",
        metric_columns=UMAP_METRIC_COLUMNS,
        high_missing_threshold=high_missing_threshold,
        low_unique_threshold=low_unique_threshold,
        low_unique_share_threshold=low_unique_share_threshold,
        zero_dominated_threshold=zero_dominated_threshold,
    )
    embedding_summary = summarize_metric_frame(
        embedding_metrics,
        metric_family="embedding_space",
        metric_columns=EMBEDDING_METRIC_COLUMNS,
        high_missing_threshold=high_missing_threshold,
        low_unique_threshold=low_unique_threshold,
        low_unique_share_threshold=low_unique_share_threshold,
        zero_dominated_threshold=zero_dominated_threshold,
    )
    return pd.concat([umap_summary, embedding_summary], ignore_index=True)


def low_information_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    return (
        summary.loc[summary["recommended_review_flag"].fillna(False)]
        .sort_values(
            ["metric_family", "is_constant", "is_high_missing", "is_zero_dominated", "metric_name"],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def plot_metric_histograms(
    frame: pd.DataFrame,
    metric_columns: list[str],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    available = [column for column in metric_columns if column in frame.columns]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not available:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    n_cols = 4
    n_rows = ceil(len(available) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 2.7),
        dpi=150,
        squeeze=False,
    )
    for ax, column in zip(axes.ravel(), available):
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            ax.text(0.5, 0.5, "all missing", ha="center", va="center")
        else:
            ax.hist(numeric.to_numpy(), bins=min(30, max(5, numeric.nunique())), color="#2f6f8f", alpha=0.85)
        ax.set_title(column, fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path)
    plt.close(fig)


def zscore_metric_values(frame: pd.DataFrame, metric_columns: list[str]) -> tuple[list[np.ndarray], list[str]]:
    values: list[np.ndarray] = []
    labels: list[str] = []
    for column in metric_columns:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        std = float(numeric.std(ddof=0))
        if std <= 0 or not np.isfinite(std):
            z = np.zeros(len(numeric), dtype=float)
        else:
            z = ((numeric - float(numeric.mean())) / std).to_numpy(dtype=float)
        values.append(z)
        labels.append(column)
    return values, labels


def plot_zscore_boxplots(
    umap_metrics: pd.DataFrame,
    embedding_metrics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    umap_values, umap_labels = zscore_metric_values(umap_metrics, UMAP_METRIC_COLUMNS)
    embedding_values, embedding_labels = zscore_metric_values(
        embedding_metrics,
        EMBEDDING_METRIC_COLUMNS,
    )
    values = umap_values + embedding_values
    labels = [f"u:{label}" for label in umap_labels] + [
        f"e:{label}" for label in embedding_labels
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    fig_width = max(12, len(values) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_width, 7), dpi=150)
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Within-metric z-score")
    ax.set_title("Z-Scored Metric Distributions")
    ax.axhline(0, color="#444444", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def markdown_distribution_summary(summary: pd.DataFrame, low_info: pd.DataFrame) -> str:
    n_umap = int((summary["metric_family"] == "umap_projected").sum()) if not summary.empty else 0
    n_embedding = int((summary["metric_family"] == "embedding_space").sum()) if not summary.empty else 0
    lines = [
        "# Metric Distribution Diagnostics",
        "",
        "This diagnostic summarizes missingness, uniqueness, zero dominance, and "
        "basic quantiles for metric columns. A review flag means inspect the metric; "
        "it does not mean delete it automatically.",
        "",
        f"- UMAP/projected metrics summarized: {n_umap}",
        f"- Embedding-space metrics summarized: {n_embedding}",
        f"- Metrics flagged for review: {len(low_info)}",
        "",
        "## Flagged Metrics",
        "",
    ]
    if low_info.empty:
        lines.append("No metrics crossed the review thresholds.")
    else:
        for row in low_info.itertuples(index=False):
            reasons = []
            if row.is_constant:
                reasons.append("constant")
            if row.is_low_unique:
                reasons.append("low unique")
            if row.is_high_missing:
                reasons.append("high missing")
            if row.is_zero_dominated:
                reasons.append("zero dominated")
            lines.append(
                f"- `{row.metric_name}` ({row.metric_family}): "
                f"{', '.join(reasons)}; missing={row.missing_share:.2f}, "
                f"unique={row.n_unique}, zero={row.zero_share:.2f}"
            )
    lines.extend(
        [
            "",
            "Default thresholds: high missing >= 0.30, low unique <= 5 or "
            "unique share <= 0.05, zero dominated >= 0.80.",
            "",
        ]
    )
    return "\n".join(lines)
