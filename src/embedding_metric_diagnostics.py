from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.embedding_space_metrics import (
    CANDIDATE_EMBEDDING_METRIC_COLUMNS,
    CORE_EMBEDDING_METRIC_COLUMNS,
    EMBEDDING_METRIC_ROLE_BY_COLUMN,
    EXCLUDED_FROM_CORE_COLUMNS,
)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DIAGNOSTIC_OUTPUT_FILENAMES = {
    "histograms_core": "embedding_metric_histograms_core.png",
    "spearman_heatmap": "embedding_metric_spearman_correlation_heatmap.png",
    "pearson_heatmap": "embedding_metric_pearson_correlation_heatmap.png",
    "spearman_matrix": "embedding_metric_spearman_correlation_matrix.csv",
    "pearson_matrix": "embedding_metric_pearson_correlation_matrix.csv",
    "top_abs_spearman_pairs": "embedding_metric_top_abs_spearman_pairs.csv",
    "distribution_diagnostics": "embedding_metric_distribution_diagnostics.csv",
    "summary_md": "embedding_metric_diagnostics_summary.md",
    "summary_json": "embedding_metric_diagnostics_summary.json",
}

DISTRIBUTION_COLUMNS = [
    "metric",
    "role",
    "n_non_missing",
    "missing_share",
    "n_unique",
    "mean",
    "std",
    "median",
    "iqr",
    "min",
    "max",
    "skewness",
    "p01",
    "p99",
    "outlier_share_iqr_rule",
    "low_information_flag",
    "notes",
]


def diagnostic_output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        key: output_dir / filename
        for key, filename in DIAGNOSTIC_OUTPUT_FILENAMES.items()
    }


def display_metric_name(metric: str) -> str:
    cleaned = metric
    if cleaned.startswith("embedding_"):
        cleaned = cleaned[len("embedding_") :]
    return "\n".join(wrap(cleaned.replace("_", " "), width=24))


def available_core_metric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in CORE_EMBEDDING_METRIC_COLUMNS if column in frame.columns]


def numeric_metric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.DataFrame(index=frame.index)
    return frame[available].apply(pd.to_numeric, errors="coerce")


def plot_core_metric_histograms(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    dpi: int = 220,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = numeric_metric_frame(frame, available_core_metric_columns(frame))
    if numeric.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No core metrics available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    n_cols = 4
    n_rows = ceil(len(numeric.columns) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.1, n_rows * 2.8),
        dpi=dpi,
        squeeze=False,
    )
    for ax, column in zip(axes.ravel(), numeric.columns):
        values = numeric[column].dropna()
        if values.empty:
            ax.text(0.5, 0.5, "all missing", ha="center", va="center", fontsize=8)
        else:
            bins = min(30, max(5, int(values.nunique())))
            ax.hist(values.to_numpy(), bins=bins, color="#2d6a74", alpha=0.88)
        ax.set_title(display_metric_name(column), fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes.ravel()[len(numeric.columns) :]:
        ax.axis("off")
    fig.suptitle("Core Embedding-Space Metric Distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path)
    plt.close(fig)


def plot_correlation_heatmap(
    matrix: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    dpi: int = 220,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No correlations available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    values = matrix.to_numpy(dtype=float)
    labels = [display_metric_name(column) for column in matrix.columns]
    fig_size = max(10, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.86), dpi=dpi)
    image = ax.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title, fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="correlation")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def correlation_matrices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = numeric_metric_frame(frame, available_core_metric_columns(frame))
    if numeric.empty:
        return numeric, numeric
    return numeric.corr(method="spearman"), numeric.corr(method="pearson")


def top_abs_spearman_pairs(
    frame: pd.DataFrame,
    spearman: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(spearman.columns)
    numeric = numeric_metric_frame(frame, columns)
    rows: list[dict[str, Any]] = []
    for left_idx, metric_a in enumerate(columns):
        for metric_b in columns[left_idx + 1 :]:
            correlation = spearman.loc[metric_a, metric_b]
            pair = numeric[[metric_a, metric_b]].dropna()
            rows.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "spearman_correlation": float(correlation)
                    if pd.notna(correlation)
                    else np.nan,
                    "abs_spearman_correlation": float(abs(correlation))
                    if pd.notna(correlation)
                    else np.nan,
                    "n_pairwise_complete": int(len(pair)),
                }
            )
    result = pd.DataFrame(
        rows,
        columns=[
            "metric_a",
            "metric_b",
            "spearman_correlation",
            "abs_spearman_correlation",
            "n_pairwise_complete",
        ],
    )
    if result.empty:
        return result
    return result.sort_values(
        ["abs_spearman_correlation", "metric_a", "metric_b"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def distribution_diagnostic_row(
    values: pd.Series,
    *,
    metric: str,
    role: str,
    high_missing_threshold: float,
    low_unique_threshold: int,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    n_rows = int(len(numeric))
    non_missing = numeric.dropna()
    n_non_missing = int(len(non_missing))
    missing_share = float(1.0 - (n_non_missing / n_rows)) if n_rows else np.nan
    n_unique = int(non_missing.nunique(dropna=True))

    if n_non_missing:
        quantiles = non_missing.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        p01 = float(quantiles.loc[0.01])
        p25 = float(quantiles.loc[0.25])
        median = float(quantiles.loc[0.50])
        p75 = float(quantiles.loc[0.75])
        p99 = float(quantiles.loc[0.99])
        iqr = float(p75 - p25)
        lower = p25 - (1.5 * iqr)
        upper = p75 + (1.5 * iqr)
        outlier_share = float(((non_missing < lower) | (non_missing > upper)).mean())
        mean = float(non_missing.mean())
        std = float(non_missing.std(ddof=0))
        minimum = float(non_missing.min())
        maximum = float(non_missing.max())
        skewness = float(non_missing.skew()) if n_non_missing >= 3 else np.nan
    else:
        mean = std = median = iqr = minimum = maximum = skewness = p01 = p99 = np.nan
        outlier_share = np.nan

    notes: list[str] = []
    if n_unique <= 1:
        notes.append("constant_or_all_missing")
    elif n_unique <= low_unique_threshold:
        notes.append("low_unique")
    if pd.notna(missing_share) and missing_share >= high_missing_threshold:
        notes.append("high_missing")
    if role != "core":
        notes.append("not_core_feature")

    return {
        "metric": metric,
        "role": role,
        "n_non_missing": n_non_missing,
        "missing_share": missing_share,
        "n_unique": n_unique,
        "mean": mean,
        "std": std,
        "median": median,
        "iqr": iqr,
        "min": minimum,
        "max": maximum,
        "skewness": skewness,
        "p01": p01,
        "p99": p99,
        "outlier_share_iqr_rule": outlier_share,
        "low_information_flag": bool(any(note != "not_core_feature" for note in notes)),
        "notes": ";".join(notes),
    }


def distribution_diagnostics(
    frame: pd.DataFrame,
    *,
    high_missing_threshold: float = 0.30,
    low_unique_threshold: int = 3,
) -> pd.DataFrame:
    rows = []
    for metric in CANDIDATE_EMBEDDING_METRIC_COLUMNS:
        if metric not in frame.columns:
            continue
        rows.append(
            distribution_diagnostic_row(
                frame[metric],
                metric=metric,
                role=EMBEDDING_METRIC_ROLE_BY_COLUMN.get(metric, "diagnostic"),
                high_missing_threshold=high_missing_threshold,
                low_unique_threshold=low_unique_threshold,
            )
        )
    return pd.DataFrame(rows, columns=DISTRIBUTION_COLUMNS)


def diagnostics_summary_markdown(
    distribution: pd.DataFrame,
    top_pairs: pd.DataFrame,
) -> str:
    role_counts = (
        distribution["role"].value_counts().to_dict() if not distribution.empty else {}
    )
    low_info = (
        distribution.loc[distribution["low_information_flag"].fillna(False)]
        if not distribution.empty
        else pd.DataFrame()
    )
    lines = [
        "# Embedding-Space Metric Diagnostics",
        "",
        "This folder summarizes the substantive core embedding-space metrics and "
        "keeps controls/diagnostics separate from the core feature set.",
        "",
        f"- Core metrics: {len(CORE_EMBEDDING_METRIC_COLUMNS)}",
        f"- Candidate columns summarized: {len(distribution)}",
        f"- Low-information flags: {len(low_info)}",
        f"- Excluded-from-core columns: {len(EXCLUDED_FROM_CORE_COLUMNS)}",
        "",
        "## Role Counts",
        "",
    ]
    if role_counts:
        for role, count in sorted(role_counts.items()):
            lines.append(f"- `{role}`: {count}")
    else:
        lines.append("No candidate metric columns were available.")

    lines.extend(["", "## Highest Absolute Spearman Pairs", ""])
    available_pairs = top_pairs.dropna(subset=["abs_spearman_correlation"]).head(10)
    if available_pairs.empty:
        lines.append("No pairwise Spearman correlations were available.")
    else:
        for row in available_pairs.itertuples(index=False):
            lines.append(
                f"- `{row.metric_a}` vs `{row.metric_b}`: "
                f"rho={row.spearman_correlation:.3f}, "
                f"n={row.n_pairwise_complete}"
            )

    lines.extend(["", "## Low-Information Flags", ""])
    if low_info.empty:
        lines.append("No candidate columns crossed the low-information thresholds.")
    else:
        for row in low_info.itertuples(index=False):
            lines.append(
                f"- `{row.metric}` ({row.role}): {row.notes or 'flagged'}; "
                f"missing={row.missing_share:.2f}, unique={row.n_unique}"
            )

    lines.extend(
        [
            "",
            "Spearman correlations are the primary diagnostic for interpretation. "
            "All correlations use pairwise-complete observations.",
            "",
        ]
    )
    return "\n".join(lines)


def diagnostics_summary_json(
    distribution: pd.DataFrame,
    top_pairs: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, Any]:
    role_counts = (
        {str(key): int(value) for key, value in distribution["role"].value_counts().items()}
        if not distribution.empty
        else {}
    )
    low_info = (
        distribution.loc[distribution["low_information_flag"].fillna(False), "metric"]
        .astype(str)
        .tolist()
        if not distribution.empty
        else []
    )
    return {
        "core_metric_columns": CORE_EMBEDDING_METRIC_COLUMNS,
        "n_core_metric_columns": len(CORE_EMBEDDING_METRIC_COLUMNS),
        "excluded_from_core_columns": EXCLUDED_FROM_CORE_COLUMNS,
        "role_counts": role_counts,
        "low_information_metrics": low_info,
        "top_abs_spearman_pairs_preview": top_pairs.head(20).to_dict("records"),
        "output_paths": {key: str(path) for key, path in paths.items()},
        "correlation_missing_value_strategy": "pairwise_complete_observations",
    }


def write_embedding_metric_diagnostics(
    metrics: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    paths = diagnostic_output_paths(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_core_metric_histograms(metrics, paths["histograms_core"])

    spearman, pearson = correlation_matrices(metrics)
    spearman.to_csv(paths["spearman_matrix"])
    pearson.to_csv(paths["pearson_matrix"])
    plot_correlation_heatmap(
        spearman,
        paths["spearman_heatmap"],
        title="Core Embedding-Space Spearman Correlations",
    )
    plot_correlation_heatmap(
        pearson,
        paths["pearson_heatmap"],
        title="Core Embedding-Space Pearson Correlations",
    )

    top_pairs = top_abs_spearman_pairs(metrics, spearman)
    top_pairs.to_csv(paths["top_abs_spearman_pairs"], index=False)

    distribution = distribution_diagnostics(metrics)
    distribution.to_csv(paths["distribution_diagnostics"], index=False)

    paths["summary_md"].write_text(
        diagnostics_summary_markdown(distribution, top_pairs),
        encoding="utf-8",
    )
    summary = diagnostics_summary_json(distribution, top_pairs, paths)
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
