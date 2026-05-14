from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.storage import save_parquet

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


VALID_METRIC_STATUSES = {"completed", "completed_with_warnings"}

REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS = [
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_iqr",
    "embedding_distance_to_centroid_p90",
    "embedding_knn_median_distance",
    "embedding_knn_distance_cv",
    "embedding_knn_indegree_gini",
    "embedding_pca_dim_80",
    "embedding_pca_spectral_entropy",
    "embedding_centroid_drift_early_late",
    "embedding_radial_expansion_slope",
    "embedding_recent_novelty_score",
]

REDUCED_INTERPRETABLE_EMBEDDING_CORE_GROUPS = {
    "global_semantic_dispersion": [
        "embedding_distance_to_centroid_median",
        "embedding_distance_to_centroid_iqr",
        "embedding_distance_to_centroid_p90",
    ],
    "local_semantic_density_and_hubness": [
        "embedding_knn_median_distance",
        "embedding_knn_distance_cv",
        "embedding_knn_indegree_gini",
    ],
    "intrinsic_dimensionality": [
        "embedding_pca_dim_80",
        "embedding_pca_spectral_entropy",
    ],
    "temporal_semantic_movement": [
        "embedding_centroid_drift_early_late",
        "embedding_radial_expansion_slope",
        "embedding_recent_novelty_score",
    ],
}

EXCLUDED_FROM_CLEAN_CORE_V1 = [
    "embedding_centroid_norm",
    "embedding_knn_p90_distance",
    "embedding_pca_top5_variance_share",
]

EXCLUSION_RATIONALES = {
    "embedding_centroid_norm": (
        "Nearly inverse of centroid-distance metrics, especially median distance "
        "to centroid."
    ),
    "embedding_knn_p90_distance": (
        "Highly redundant with kNN median distance in the clean-core diagnostics."
    ),
    "embedding_pca_top5_variance_share": (
        "Highly redundant or anticorrelated with spectral entropy and dim80."
    ),
}

OPTIONAL_METADATA_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "subfield_label_unique",
    "subfield_label_short",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "n_available",
    "n_used",
    "year_min",
    "year_max",
    "metric_status",
    "metric_warning_message",
]

OUTPUT_FILENAMES = {
    "metrics_csv": "reduced_interpretable_core_metrics.csv",
    "metrics_parquet": "reduced_interpretable_core_metrics.parquet",
    "spearman_matrix": "reduced_core_spearman_correlation_matrix.csv",
    "pearson_matrix": "reduced_core_pearson_correlation_matrix.csv",
    "spearman_heatmap": "reduced_core_spearman_correlation_heatmap.png",
    "pearson_heatmap": "reduced_core_pearson_correlation_heatmap.png",
    "histograms": "reduced_core_histograms.png",
    "top_spearman": "reduced_core_top_absolute_spearman_pairs.csv",
    "top_pearson": "reduced_core_top_absolute_pearson_pairs.csv",
    "distribution": "reduced_core_distribution_diagnostics.csv",
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}

METHODOLOGICAL_NOTE = (
    "This reduced core is intended for the main downstream analysis, while the "
    "full embedding-space metric table is retained for sensitivity checks."
)


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}


def existing_outputs(output_dir: str | Path) -> list[Path]:
    return [path for path in output_paths(output_dir).values() if path.exists()]


def validate_reduced_core_definition() -> None:
    if len(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS) != 11:
        raise ValueError("reduced interpretable embedding core must have exactly 11 metrics")
    if len(set(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)) != len(
        REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS
    ):
        raise ValueError("reduced interpretable embedding core contains duplicates")
    grouped = [
        metric
        for metrics in REDUCED_INTERPRETABLE_EMBEDDING_CORE_GROUPS.values()
        for metric in metrics
    ]
    if grouped != REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS:
        raise ValueError(
            "reduced core groups must cover all reduced metrics exactly once "
            "and preserve metric order"
        )
    overlap = set(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS) & set(
        EXCLUDED_FROM_CLEAN_CORE_V1
    )
    if overlap:
        raise ValueError(
            "metrics excluded from clean-core v1 cannot appear in the reduced core: "
            f"{', '.join(sorted(overlap))}"
        )


def available_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def missing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


def filter_eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "metric_status" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["metric_status"].isin(VALID_METRIC_STATUSES)].copy()


def build_reduced_metric_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available_metrics = available_columns(
        frame,
        REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
    )
    metadata_columns = available_columns(frame, OPTIONAL_METADATA_COLUMNS)
    reduced = frame[metadata_columns + available_metrics].copy()
    for metric in available_metrics:
        reduced[metric] = pd.to_numeric(reduced[metric], errors="coerce")
    return reduced, missing_columns(frame, REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)


def reduced_numeric_frame(reduced: pd.DataFrame) -> pd.DataFrame:
    available_metrics = available_columns(
        reduced,
        REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
    )
    if not available_metrics:
        return pd.DataFrame(index=reduced.index)
    return reduced[available_metrics].apply(pd.to_numeric, errors="coerce")


def correlation_matrices(reduced: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = reduced_numeric_frame(reduced)
    if numeric.empty:
        return numeric, numeric
    return numeric.corr(method="spearman"), numeric.corr(method="pearson")


def top_absolute_correlation_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = list(matrix.columns)
    for left_idx, metric_a in enumerate(columns):
        for metric_b in columns[left_idx + 1 :]:
            correlation = matrix.loc[metric_a, metric_b]
            rows.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "correlation": float(correlation)
                    if pd.notna(correlation)
                    else np.nan,
                    "abs_correlation": float(abs(correlation))
                    if pd.notna(correlation)
                    else np.nan,
                }
            )
    result = pd.DataFrame(
        rows,
        columns=["metric_a", "metric_b", "correlation", "abs_correlation"],
    )
    if result.empty:
        return result
    return result.sort_values(
        ["abs_correlation", "metric_a", "metric_b"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def readable_metric_label(metric: str, *, width: int = 18) -> str:
    label = metric.removeprefix("embedding_").replace("_", " ")
    return "\n".join(wrap(label, width=width))


def group_boundaries(columns: list[str]) -> list[int]:
    boundaries: list[int] = []
    seen = 0
    for metrics in REDUCED_INTERPRETABLE_EMBEDDING_CORE_GROUPS.values():
        group_count = sum(metric in columns for metric in metrics)
        seen += group_count
        if group_count and seen < len(columns):
            boundaries.append(seen)
    return boundaries


def plot_correlation_heatmap(
    matrix: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=240)
        ax.text(0.5, 0.5, "No reduced-core correlations available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    values = matrix.to_numpy(dtype=float)
    labels = [readable_metric_label(column) for column in matrix.columns]
    fig_size = max(7.5, len(labels) * 0.64)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=240)
    image = ax.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=12)
    for boundary in group_boundaries(list(matrix.columns)):
        ax.axhline(boundary - 0.5, color="#222222", linewidth=0.9)
        ax.axvline(boundary - 0.5, color="#222222", linewidth=0.9)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="correlation")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_histograms(reduced: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = reduced_numeric_frame(reduced)
    if numeric.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=240)
        ax.text(0.5, 0.5, "No reduced-core metrics available", ha="center", va="center")
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
        figsize=(n_cols * 4.0, n_rows * 2.8),
        dpi=220,
        squeeze=False,
    )
    for ax, metric in zip(axes.ravel(), numeric.columns):
        values = numeric[metric].dropna()
        if values.empty:
            ax.text(0.5, 0.5, "all missing", ha="center", va="center", fontsize=8)
        else:
            bins = min(30, max(5, int(values.nunique())))
            ax.hist(values.to_numpy(), bins=bins, color="#2d6a74", alpha=0.88)
        ax.set_title(readable_metric_label(metric, width=24), fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes.ravel()[len(numeric.columns) :]:
        ax.axis("off")
    fig.suptitle("Reduced Interpretable Embedding Core Distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path)
    plt.close(fig)


def distribution_diagnostics(
    reduced: pd.DataFrame,
    *,
    high_missing_threshold: float = 0.30,
    near_constant_unique_threshold: int = 3,
    high_skew_threshold: float = 2.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric = reduced_numeric_frame(reduced)
    n_rows = len(reduced)
    for metric in numeric.columns:
        values = pd.to_numeric(numeric[metric], errors="coerce")
        non_missing = values.dropna()
        n_non_missing = int(len(non_missing))
        missing_count = int(n_rows - n_non_missing)
        missing_share = float(missing_count / n_rows) if n_rows else np.nan
        unique_count = int(non_missing.nunique(dropna=True))
        if n_non_missing:
            quantiles = non_missing.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            p01 = float(quantiles.loc[0.01])
            p05 = float(quantiles.loc[0.05])
            p25 = float(quantiles.loc[0.25])
            median = float(quantiles.loc[0.50])
            p75 = float(quantiles.loc[0.75])
            p95 = float(quantiles.loc[0.95])
            p99 = float(quantiles.loc[0.99])
            mean = float(non_missing.mean())
            std = float(non_missing.std(ddof=0))
            minimum = float(non_missing.min())
            maximum = float(non_missing.max())
            iqr = float(p75 - p25)
            skewness = float(non_missing.skew()) if n_non_missing >= 3 else np.nan
        else:
            mean = std = median = iqr = minimum = p01 = p05 = p25 = p75 = p95 = p99 = maximum = np.nan
            skewness = np.nan
        near_constant = unique_count <= near_constant_unique_threshold
        high_missing = pd.notna(missing_share) and missing_share >= high_missing_threshold
        high_outlier_risk = pd.notna(skewness) and abs(skewness) >= high_skew_threshold
        rows.append(
            {
                "metric": metric,
                "n": n_rows,
                "missing_count": missing_count,
                "missing_share": missing_share,
                "unique_count": unique_count,
                "mean": mean,
                "std": std,
                "median": median,
                "iqr": iqr,
                "min": minimum,
                "p01": p01,
                "p05": p05,
                "p25": p25,
                "p75": p75,
                "p95": p95,
                "p99": p99,
                "max": maximum,
                "skewness": skewness,
                "near_constant": bool(near_constant),
                "high_missing": bool(high_missing),
                "high_outlier_risk": bool(high_outlier_risk),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "n",
            "missing_count",
            "missing_share",
            "unique_count",
            "mean",
            "std",
            "median",
            "iqr",
            "min",
            "p01",
            "p05",
            "p25",
            "p75",
            "p95",
            "p99",
            "max",
            "skewness",
            "near_constant",
            "high_missing",
            "high_outlier_risk",
        ],
    )


def flagged_distribution_metrics(distribution: pd.DataFrame) -> list[str]:
    if distribution.empty:
        return []
    mask = (
        distribution["near_constant"].fillna(False)
        | distribution["high_missing"].fillna(False)
        | distribution["high_outlier_risk"].fillna(False)
    )
    return distribution.loc[mask, "metric"].astype(str).tolist()


def markdown_group_lines() -> list[str]:
    lines: list[str] = []
    for group, metrics in REDUCED_INTERPRETABLE_EMBEDDING_CORE_GROUPS.items():
        lines.append(f"### {group.replace('_', ' ').title()}")
        lines.append("")
        for metric in metrics:
            lines.append(f"- `{metric}`")
        lines.append("")
    return lines


def markdown_summary(
    *,
    input_path: str,
    n_rows_input: int,
    n_rows_eligible: int,
    missing_metrics: list[str],
    top_spearman: pd.DataFrame,
    top_pearson: pd.DataFrame,
    flagged_metrics: list[str],
) -> str:
    lines = [
        "# Reduced Interpretable Embedding Core",
        "",
        "This downstream step selects a compact embedding-space feature set for the "
        "main thesis analysis. It does not recompute embeddings or remove columns "
        "from the full script-12 metric table.",
        "",
        f"- Input table: `{input_path}`",
        f"- Input rows: {n_rows_input}",
        f"- Eligible subfields: {n_rows_eligible}",
        f"- Reduced core metrics: {len(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)}",
        "",
        "## Reduced Core By Block",
        "",
    ]
    lines.extend(markdown_group_lines())
    lines.extend(
        [
            "## Removed From The 14-Metric Clean Core",
            "",
        ]
    )
    for metric in EXCLUDED_FROM_CLEAN_CORE_V1:
        lines.append(f"- `{metric}`: {EXCLUSION_RATIONALES[metric]}")

    lines.extend(["", "## Top Absolute Spearman Correlations", ""])
    if top_spearman.empty:
        lines.append("No Spearman correlations were available.")
    else:
        for row in top_spearman.head(10).itertuples(index=False):
            lines.append(
                f"- `{row.metric_a}` vs `{row.metric_b}`: {row.correlation:.3f}"
            )

    lines.extend(["", "## Top Absolute Pearson Correlations", ""])
    if top_pearson.empty:
        lines.append("No Pearson correlations were available.")
    else:
        for row in top_pearson.head(10).itertuples(index=False):
            lines.append(
                f"- `{row.metric_a}` vs `{row.metric_b}`: {row.correlation:.3f}"
            )

    lines.extend(["", "## Distribution Warnings", ""])
    if flagged_metrics:
        lines.append(
            "Metrics with near-constant, high-missing, or high-skew/outlier-risk flags: "
            + ", ".join(f"`{metric}`" for metric in flagged_metrics)
        )
    else:
        lines.append("No reduced-core metrics crossed the simple distribution flag thresholds.")
    if missing_metrics:
        lines.append(
            "Missing reduced-core metrics: "
            + ", ".join(f"`{metric}`" for metric in missing_metrics)
        )
    else:
        lines.append("All required reduced-core metrics were present in the input table.")

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            METHODOLOGICAL_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


def summary_payload(
    *,
    input_path: str,
    output_dir: str,
    n_rows_input: int,
    n_rows_eligible: int,
    missing_metrics: list[str],
    top_spearman: pd.DataFrame,
    top_pearson: pd.DataFrame,
    flagged_metrics: list[str],
    paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "input_path": input_path,
        "output_dir": output_dir,
        "n_rows_input": int(n_rows_input),
        "n_rows_eligible": int(n_rows_eligible),
        "core_metric_count": len(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS),
        "core_metrics": REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
        "core_metric_groups": REDUCED_INTERPRETABLE_EMBEDDING_CORE_GROUPS,
        "excluded_from_clean_core_v1": EXCLUDED_FROM_CLEAN_CORE_V1,
        "missing_core_metrics": missing_metrics,
        "distribution_flagged_metrics": flagged_metrics,
        "top_abs_spearman_pairs": top_spearman.head(20).to_dict("records"),
        "top_abs_pearson_pairs": top_pearson.head(20).to_dict("records"),
        "created_outputs": [str(path) for path in paths.values()],
    }


def build_reduced_interpretable_embedding_core_outputs(
    frame: pd.DataFrame,
    *,
    input_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    validate_reduced_core_definition()
    paths = output_paths(output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible = filter_eligible_rows(frame)
    reduced, missing_metrics = build_reduced_metric_table(eligible)
    reduced.to_csv(paths["metrics_csv"], index=False)
    save_parquet(reduced, paths["metrics_parquet"])

    spearman, pearson = correlation_matrices(reduced)
    spearman.to_csv(paths["spearman_matrix"])
    pearson.to_csv(paths["pearson_matrix"])
    plot_correlation_heatmap(
        spearman,
        paths["spearman_heatmap"],
        title="Reduced Embedding Core: Spearman Correlations",
    )
    plot_correlation_heatmap(
        pearson,
        paths["pearson_heatmap"],
        title="Reduced Embedding Core: Pearson Correlations",
    )
    plot_histograms(reduced, paths["histograms"])

    top_spearman = top_absolute_correlation_pairs(spearman)
    top_pearson = top_absolute_correlation_pairs(pearson)
    top_spearman.to_csv(paths["top_spearman"], index=False)
    top_pearson.to_csv(paths["top_pearson"], index=False)

    distribution = distribution_diagnostics(reduced)
    distribution.to_csv(paths["distribution"], index=False)
    flagged_metrics = flagged_distribution_metrics(distribution)

    summary_md = markdown_summary(
        input_path=str(input_path),
        n_rows_input=len(frame),
        n_rows_eligible=len(eligible),
        missing_metrics=missing_metrics,
        top_spearman=top_spearman,
        top_pearson=top_pearson,
        flagged_metrics=flagged_metrics,
    )
    paths["summary_md"].write_text(summary_md, encoding="utf-8")

    payload = summary_payload(
        input_path=str(input_path),
        output_dir=str(output_dir),
        n_rows_input=len(frame),
        n_rows_eligible=len(eligible),
        missing_metrics=missing_metrics,
        top_spearman=top_spearman,
        top_pearson=top_pearson,
        flagged_metrics=flagged_metrics,
        paths=paths,
    )
    paths["summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
