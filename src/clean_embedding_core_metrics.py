from __future__ import annotations

import json
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.storage import save_parquet

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


CLEAN_CORE_EMBEDDING_METRIC_COLUMNS = [
    "embedding_centroid_norm",
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_iqr",
    "embedding_distance_to_centroid_p90",
    "embedding_knn_median_distance",
    "embedding_knn_p90_distance",
    "embedding_knn_distance_cv",
    "embedding_knn_indegree_gini",
    "embedding_pca_top5_variance_share",
    "embedding_pca_dim_80",
    "embedding_pca_spectral_entropy",
    "embedding_centroid_drift_early_late",
    "embedding_radial_expansion_slope",
    "embedding_recent_novelty_score",
]

CLEAN_CORE_EMBEDDING_METRIC_GROUPS = {
    "semantic_coherence": [
        "embedding_centroid_norm",
    ],
    "semantic_dispersion": [
        "embedding_distance_to_centroid_median",
        "embedding_distance_to_centroid_iqr",
        "embedding_distance_to_centroid_p90",
    ],
    "local_density_and_hubness": [
        "embedding_knn_median_distance",
        "embedding_knn_p90_distance",
        "embedding_knn_distance_cv",
        "embedding_knn_indegree_gini",
    ],
    "intrinsic_dimensionality": [
        "embedding_pca_top5_variance_share",
        "embedding_pca_dim_80",
        "embedding_pca_spectral_entropy",
    ],
    "temporal_semantic_change": [
        "embedding_centroid_drift_early_late",
        "embedding_radial_expansion_slope",
        "embedding_recent_novelty_score",
    ],
}

EXCLUDED_FROM_CLEAN_CORE = [
    "embedding_distance_to_centroid_mean",
    "embedding_distance_to_centroid_std",
    "embedding_tail_index_p90_median",
    "embedding_knn_mean_distance",
    "embedding_pca_first_component_share",
    "embedding_pca_top3_variance_share",
    "embedding_pca_dim_50",
    "embedding_pca_participation_ratio",
    "embedding_graph_edge_distance_median",
    "embedding_graph_edge_distance_p90",
    "embedding_annual_centroid_path_length",
    "embedding_directionality_ratio",
]

DIAGNOSTIC_OR_CONTROL_COLUMNS = [
    "embedding_radial_expansion_r2",
    "n_valid_embedding_rows",
    "n_years_available",
    "n_early_points",
    "n_late_points",
    "n_annual_centroids",
    "n_pca_components_used",
]

OPTIONAL_IDENTIFIER_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "subfield_label_unique",
    "subfield_label_short",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
]

OPTIONAL_STATUS_CONTROL_COLUMNS = [
    "metric_status",
    "metric_warning_message",
    "n_available",
    "n_used",
    "year_min",
    "year_max",
]

OUTPUT_FILENAMES = {
    "clean_csv": "clean_embedding_core_metrics.csv",
    "clean_parquet": "clean_embedding_core_metrics.parquet",
    "pearson_matrix": "clean_core_pearson_correlation_matrix.csv",
    "spearman_matrix": "clean_core_spearman_correlation_matrix.csv",
    "pearson_heatmap": "clean_core_pearson_correlation_heatmap.png",
    "spearman_heatmap": "clean_core_spearman_correlation_heatmap.png",
    "top_spearman": "clean_core_top_abs_spearman_pairs.csv",
    "top_pearson": "clean_core_top_abs_pearson_pairs.csv",
    "distribution": "clean_core_distribution_diagnostics.csv",
    "groups": "clean_core_metric_groups.csv",
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}

METRIC_INTERPRETATIONS = {
    "embedding_centroid_norm": "Semantic coherence / strength of common direction.",
    "embedding_distance_to_centroid_median": "Typical semantic dispersion.",
    "embedding_distance_to_centroid_iqr": "Heterogeneity of dispersion.",
    "embedding_distance_to_centroid_p90": "Peripheral semantic spread.",
    "embedding_knn_median_distance": "Typical local sparsity or density.",
    "embedding_knn_p90_distance": "Sparse-neighborhood tail.",
    "embedding_knn_distance_cv": "Inequality of local density.",
    "embedding_knn_indegree_gini": "Hubness / concentration of neighbor centrality.",
    "embedding_pca_top5_variance_share": "Concentration of variance in few dimensions.",
    "embedding_pca_dim_80": "Dimensional complexity.",
    "embedding_pca_spectral_entropy": "Entropy of the variance spectrum.",
    "embedding_centroid_drift_early_late": "Long-run semantic movement.",
    "embedding_radial_expansion_slope": "Semantic expansion or contraction over time.",
    "embedding_recent_novelty_score": "Novelty of recent papers relative to historical core.",
}

EXCLUDED_METRIC_RATIONALES = {
    "embedding_distance_to_centroid_mean": "Less robust and highly redundant with the median distance.",
    "embedding_distance_to_centroid_std": "Less robust and highly redundant with the interquartile range.",
    "embedding_tail_index_p90_median": "Conceptually useful but more outlier-prone than direct percentile summaries.",
    "embedding_knn_mean_distance": "Highly redundant with kNN median distance.",
    "embedding_pca_first_component_share": "Part of a highly correlated PCA block; retained only for sensitivity checks.",
    "embedding_pca_top3_variance_share": "Highly correlated with top-five variance share.",
    "embedding_pca_dim_50": "More compressed dimensionality threshold than the retained dim-80 metric.",
    "embedding_pca_participation_ratio": "Highly correlated or anticorrelated with retained PCA summaries.",
    "embedding_graph_edge_distance_median": "Highly redundant with kNN median distance.",
    "embedding_graph_edge_distance_p90": "Highly redundant with kNN P90 distance.",
    "embedding_annual_centroid_path_length": "Temporal metric retained for sensitivity because it can be outlier-sensitive.",
    "embedding_directionality_ratio": "Temporal ratio retained for sensitivity because it can be unstable when path length is small.",
}

METHODOLOGICAL_NOTE = (
    "The broad embedding-space metric table is retained for sensitivity analysis. "
    "The clean core set is used for the main analysis because it reduces redundancy, "
    "avoids near-constant or unstable variables, and preserves interpretable coverage "
    "of semantic coherence, dispersion, local density, hubness, intrinsic "
    "dimensionality, and temporal semantic change."
)


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}


def existing_outputs(output_dir: str | Path) -> list[Path]:
    paths = output_paths(output_dir)
    return [path for path in paths.values() if path.exists()]


def validate_clean_core_definition() -> None:
    if len(CLEAN_CORE_EMBEDDING_METRIC_COLUMNS) != 14:
        raise ValueError("clean core embedding metric list must contain exactly 14 metrics")
    if len(set(CLEAN_CORE_EMBEDDING_METRIC_COLUMNS)) != len(
        CLEAN_CORE_EMBEDDING_METRIC_COLUMNS
    ):
        raise ValueError("clean core embedding metric list contains duplicate metrics")
    grouped = [
        metric
        for metrics in CLEAN_CORE_EMBEDDING_METRIC_GROUPS.values()
        for metric in metrics
    ]
    if grouped != CLEAN_CORE_EMBEDDING_METRIC_COLUMNS:
        raise ValueError(
            "clean core metric groups must cover the clean core metrics exactly once "
            "and in the same order"
        )
    overlap = set(CLEAN_CORE_EMBEDDING_METRIC_COLUMNS) & set(EXCLUDED_FROM_CLEAN_CORE)
    if overlap:
        raise ValueError(
            "excluded metrics cannot appear in the clean core list: "
            f"{', '.join(sorted(overlap))}"
        )


def available_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def missing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


def build_clean_metric_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    metric_columns = available_columns(frame, CLEAN_CORE_EMBEDDING_METRIC_COLUMNS)
    metadata_columns = available_columns(
        frame,
        OPTIONAL_IDENTIFIER_COLUMNS + OPTIONAL_STATUS_CONTROL_COLUMNS,
    )
    clean = frame[metadata_columns + metric_columns].copy()
    for column in metric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean, missing_columns(frame, CLEAN_CORE_EMBEDDING_METRIC_COLUMNS)


def clean_numeric_metrics(clean_table: pd.DataFrame) -> pd.DataFrame:
    columns = available_columns(clean_table, CLEAN_CORE_EMBEDDING_METRIC_COLUMNS)
    if not columns:
        return pd.DataFrame(index=clean_table.index)
    return clean_table[columns].apply(pd.to_numeric, errors="coerce")


def correlation_matrices(clean_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = clean_numeric_metrics(clean_table)
    if numeric.empty:
        return numeric, numeric
    return numeric.corr(method="pearson"), numeric.corr(method="spearman")


def top_abs_correlation_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
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


def short_metric_label(metric: str) -> str:
    label = metric
    if label.startswith("embedding_"):
        label = label[len("embedding_") :]
    return "\n".join(wrap(label.replace("_", " "), width=18))


def group_boundaries(columns: list[str]) -> list[int]:
    boundaries: list[int] = []
    seen = 0
    for metrics in CLEAN_CORE_EMBEDDING_METRIC_GROUPS.values():
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
        fig, ax = plt.subplots(figsize=(8, 4), dpi=220)
        ax.text(0.5, 0.5, "No clean core correlations available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    values = matrix.to_numpy(dtype=float)
    labels = [short_metric_label(column) for column in matrix.columns]
    fig_size = max(7.5, len(labels) * 0.58)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=220)
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


def distribution_diagnostics(clean_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in available_columns(clean_table, CLEAN_CORE_EMBEDDING_METRIC_COLUMNS):
        values = pd.to_numeric(clean_table[metric], errors="coerce")
        non_missing = values.dropna()
        n_rows = len(values)
        n_non_missing = int(len(non_missing))
        missing_share = float(1.0 - (n_non_missing / n_rows)) if n_rows else np.nan
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
            unique_count = int(non_missing.nunique(dropna=True))
        else:
            mean = std = median = iqr = minimum = p01 = p05 = p95 = p99 = maximum = np.nan
            unique_count = 0
        rows.append(
            {
                "metric": metric,
                "n_non_missing": n_non_missing,
                "missing_share": missing_share,
                "mean": mean,
                "std": std,
                "median": median,
                "iqr": iqr,
                "min": minimum,
                "p01": p01,
                "p05": p05,
                "p95": p95,
                "p99": p99,
                "max": maximum,
                "unique_count": unique_count,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "n_non_missing",
            "missing_share",
            "mean",
            "std",
            "median",
            "iqr",
            "min",
            "p01",
            "p05",
            "p95",
            "p99",
            "max",
            "unique_count",
        ],
    )


def metric_group_dictionary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, metrics in CLEAN_CORE_EMBEDDING_METRIC_GROUPS.items():
        for metric in metrics:
            rows.append(
                {
                    "metric": metric,
                    "group": group,
                    "interpretation_short": METRIC_INTERPRETATIONS[metric],
                    "included_in_clean_core": True,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["metric", "group", "interpretation_short", "included_in_clean_core"],
    )


def high_missing_metrics(
    diagnostics: pd.DataFrame,
    *,
    min_non_missing_share: float,
) -> list[str]:
    if diagnostics.empty:
        return []
    max_missing_share = 1.0 - min_non_missing_share
    return (
        diagnostics.loc[diagnostics["missing_share"] > max_missing_share, "metric"]
        .astype(str)
        .tolist()
    )


def markdown_summary(
    *,
    input_path: str,
    rows_read: int,
    clean_rows: int,
    missing_final_metrics: list[str],
    high_missing_final_metrics: list[str],
    top_spearman: pd.DataFrame,
    top_pearson: pd.DataFrame,
    min_non_missing_share: float,
) -> str:
    lines = [
        "# Clean Embedding Core Metrics",
        "",
        f"- Input file: `{input_path}`",
        f"- Rows read: {rows_read}",
        f"- Rows in clean table: {clean_rows}",
        f"- Final clean core metrics: {len(CLEAN_CORE_EMBEDDING_METRIC_COLUMNS)}",
        f"- Minimum non-missing share threshold: {min_non_missing_share:.2f}",
        "",
        "## Final Clean Core Metrics",
        "",
    ]
    for metric in CLEAN_CORE_EMBEDDING_METRIC_COLUMNS:
        lines.append(f"- `{metric}`")

    lines.extend(["", "## Excluded From Clean Core", ""])
    for metric in EXCLUDED_FROM_CLEAN_CORE:
        lines.append(f"- `{metric}`: {EXCLUDED_METRIC_RATIONALES[metric]}")

    lines.extend(["", "## Diagnostic Or Control Only", ""])
    for metric in DIAGNOSTIC_OR_CONTROL_COLUMNS:
        lines.append(f"- `{metric}`")

    lines.extend(["", "## Warnings", ""])
    if missing_final_metrics:
        lines.append(
            "Missing final metrics: "
            + ", ".join(f"`{metric}`" for metric in missing_final_metrics)
        )
    else:
        lines.append("No final clean core metrics were missing from the input table.")
    if high_missing_final_metrics:
        lines.append(
            "High-missing final metrics: "
            + ", ".join(f"`{metric}`" for metric in high_missing_final_metrics)
        )
    else:
        lines.append("No final clean core metrics exceeded the missingness threshold.")

    lines.extend(["", "## Top Absolute Spearman Correlations", ""])
    if top_spearman.empty:
        lines.append("No Spearman correlations were available.")
    else:
        for row in top_spearman.head(10).itertuples(index=False):
            lines.append(
                f"- `{row.metric_a}` vs `{row.metric_b}`: "
                f"{row.correlation:.3f}"
            )

    lines.extend(["", "## Top Absolute Pearson Correlations", ""])
    if top_pearson.empty:
        lines.append("No Pearson correlations were available.")
    else:
        for row in top_pearson.head(10).itertuples(index=False):
            lines.append(
                f"- `{row.metric_a}` vs `{row.metric_b}`: "
                f"{row.correlation:.3f}"
            )

    lines.extend(["", "## Methodological Note", "", METHODOLOGICAL_NOTE, ""])
    return "\n".join(lines)


def summary_payload(
    *,
    input_path: str,
    rows_read: int,
    clean_rows: int,
    missing_final_metrics: list[str],
    high_missing_final_metrics: list[str],
    top_spearman: pd.DataFrame,
    top_pearson: pd.DataFrame,
    min_non_missing_share: float,
    paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "input_path": input_path,
        "rows_read": int(rows_read),
        "clean_rows": int(clean_rows),
        "clean_core_metric_columns": CLEAN_CORE_EMBEDDING_METRIC_COLUMNS,
        "n_clean_core_metric_columns": len(CLEAN_CORE_EMBEDDING_METRIC_COLUMNS),
        "clean_core_metric_groups": CLEAN_CORE_EMBEDDING_METRIC_GROUPS,
        "excluded_from_clean_core": EXCLUDED_FROM_CLEAN_CORE,
        "diagnostic_or_control_columns": DIAGNOSTIC_OR_CONTROL_COLUMNS,
        "missing_final_metrics": missing_final_metrics,
        "high_missing_final_metrics": high_missing_final_metrics,
        "min_non_missing_share": float(min_non_missing_share),
        "top_abs_spearman_pairs_preview": top_spearman.head(10).to_dict("records"),
        "top_abs_pearson_pairs_preview": top_pearson.head(10).to_dict("records"),
        "methodological_note": METHODOLOGICAL_NOTE,
        "output_paths": {key: str(path) for key, path in paths.items()},
    }


def build_clean_embedding_core_outputs(
    frame: pd.DataFrame,
    *,
    input_path: str | Path,
    output_dir: str | Path,
    min_non_missing_share: float = 0.70,
) -> dict[str, Any]:
    validate_clean_core_definition()
    paths = output_paths(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    clean_table, missing_final_metrics = build_clean_metric_table(frame)
    clean_table.to_csv(paths["clean_csv"], index=False)
    save_parquet(clean_table, paths["clean_parquet"])

    pearson, spearman = correlation_matrices(clean_table)
    pearson.to_csv(paths["pearson_matrix"])
    spearman.to_csv(paths["spearman_matrix"])
    plot_correlation_heatmap(
        pearson,
        paths["pearson_heatmap"],
        title="Clean Core Embedding Metrics: Pearson Correlations",
    )
    plot_correlation_heatmap(
        spearman,
        paths["spearman_heatmap"],
        title="Clean Core Embedding Metrics: Spearman Correlations",
    )

    top_pearson = top_abs_correlation_pairs(pearson)
    top_spearman = top_abs_correlation_pairs(spearman)
    top_pearson.to_csv(paths["top_pearson"], index=False)
    top_spearman.to_csv(paths["top_spearman"], index=False)

    distributions = distribution_diagnostics(clean_table)
    distributions.to_csv(paths["distribution"], index=False)
    high_missing_final_metrics = high_missing_metrics(
        distributions,
        min_non_missing_share=min_non_missing_share,
    )

    groups = metric_group_dictionary()
    groups.to_csv(paths["groups"], index=False)

    summary_md = markdown_summary(
        input_path=str(input_path),
        rows_read=len(frame),
        clean_rows=len(clean_table),
        missing_final_metrics=missing_final_metrics,
        high_missing_final_metrics=high_missing_final_metrics,
        top_spearman=top_spearman,
        top_pearson=top_pearson,
        min_non_missing_share=min_non_missing_share,
    )
    paths["summary_md"].write_text(summary_md, encoding="utf-8")
    payload = summary_payload(
        input_path=str(input_path),
        rows_read=len(frame),
        clean_rows=len(clean_table),
        missing_final_metrics=missing_final_metrics,
        high_missing_final_metrics=high_missing_final_metrics,
        top_spearman=top_spearman,
        top_pearson=top_pearson,
        min_non_missing_share=min_non_missing_share,
        paths=paths,
    )
    paths["summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
