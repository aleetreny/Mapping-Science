from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphology_metrics import CORE_METRIC_COLUMNS_V2, DIAGNOSTIC_METRIC_COLUMNS
from src.storage import load_parquet, save_parquet
from src.subfield_labels import add_subfield_label_columns, duplicate_subfield_names_report


METADATA_COLUMNS = [
    "subfield_id",
    "subfield_label_unique",
    "subfield_label_short",
    "subfield_display_name",
    "subfield_display_name_is_duplicated",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "n_points",
]

METRIC_FAMILIES = {
    "radial_dispersion": [
        "radial_tail_index",
        "radial_iqr_index",
        "knn_median_distance",
        "knn_distance_cv",
    ],
    "density_concentration": [
        "density_entropy",
        "peak_dominance",
        "effective_area_90",
        "core_periphery_ratio",
    ],
    "fragmentation": [
        "density_peak_count",
        "peak_mass_entropy",
        "dense_component_count",
        "largest_component_mass_share",
        "component_separation_index",
        "mst_gap_index",
    ],
    "shape": [
        "anisotropy_ratio",
        "support_solidity",
        "boundary_complexity",
        "max_normalized_radius",
    ],
    "temporal": [
        "centroid_drift_early_late",
        "annual_centroid_path_length",
        "directionality_ratio",
        "radial_expansion_slope",
        "annual_centroid_step_cv",
        "radial_expansion_r2",
        "density_entropy_slope_by_year",
    ],
}

FAMILY_SCORE_DEFINITIONS = {
    "diffuseness_score": [
        ("density_entropy", 1.0),
        ("effective_area_90", 1.0),
        ("radial_tail_index", 1.0),
    ],
    "concentration_score": [
        ("peak_dominance", 1.0),
        ("core_periphery_ratio", -1.0),
        ("density_entropy", -1.0),
        ("effective_area_90", -1.0),
    ],
    "fragmentation_score": [
        ("dense_component_count", 1.0),
        ("density_peak_count", 1.0),
        ("component_separation_index", 1.0),
        ("mst_gap_index", 1.0),
        ("largest_component_mass_share", -1.0),
    ],
    "elongation_score": [
        ("anisotropy_ratio", 1.0),
        ("boundary_complexity", 1.0),
        ("support_solidity", -1.0),
    ],
    "temporal_dynamism_score": [
        ("centroid_drift_early_late", 1.0),
        ("annual_centroid_path_length", 1.0),
        ("annual_centroid_step_cv", 1.0),
    ],
    "directional_change_score": [
        ("directionality_ratio", 1.0),
        ("centroid_drift_early_late", 1.0),
        ("radial_expansion_r2", 1.0),
    ],
    "expansion_score": [
        ("radial_expansion_slope", 1.0),
    ],
    "diversification_score": [
        ("density_entropy_slope_by_year", 1.0),
    ],
}

FAMILY_SCORE_COLUMNS = list(FAMILY_SCORE_DEFINITIONS)

KEY_FAMILY_SCATTERS = [
    ("diffuseness_score", "concentration_score"),
    ("fragmentation_score", "elongation_score"),
    ("temporal_dynamism_score", "directional_change_score"),
    ("expansion_score", "diversification_score"),
    ("concentration_score", "elongation_score"),
]

SELECTED_RANKING_METRICS = [
    "density_entropy",
    "peak_dominance",
    "effective_area_90",
    "dense_component_count",
    "component_separation_index",
    "anisotropy_ratio",
    "centroid_drift_early_late",
    "directionality_ratio",
    "radial_expansion_slope",
    "density_entropy_slope_by_year",
]

STALE_LEGACY_FIGURE_NAMES = [
    "core_metric_correlation_heatmap.png",
    "core_metric_histograms.png",
    "domain_family_score_boxplots.png",
    "metric_pca_loadings.png",
    "metric_pca_scatter.png",
]

STALE_LEGACY_ROOT_OUTPUT_NAMES = [
    "curated_model_features.csv",
    "curated_model_features.parquet",
    "domain_metric_summary.csv",
    "duplicate_subfield_names_report.csv",
    "family_scores.csv",
    "field_metric_summary.csv",
    "high_correlation_pairs.csv",
    "low_variance_metrics.csv",
    "metric_correlation_matrix.csv",
    "metric_descriptive_stats.csv",
    "metric_missingness.csv",
    "metric_pca_loadings.csv",
    "metric_pca_scores.csv",
    "metric_spearman_correlation_matrix.csv",
    "top_bottom_subfields_by_metric.csv",
]

QUALITY_FIGURES = "00_quality"
DIST_FIGURES = "01_distributions"
CORR_FIGURES = "02_correlations"
FAMILY_FIGURES = "03_family_scores"
PROFILE_FIGURES = "04_domain_field_profiles"
PCA_FIGURES = "05_metric_space_pca"
RANKING_FIGURES = "06_rankings_extremes"
CASE_FIGURES = "07_case_study_atlas"


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Explore the final subfield morphology feature layer."
    )
    parser.add_argument(
        "--input-path",
        default="data/processed/subfield_morphology_metrics.parquet",
    )
    parser.add_argument(
        "--dictionary-path",
        default="outputs/metrics/subfield_morphology_metrics_dictionary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/metrics/morphology_analysis",
    )
    parser.add_argument("--high-correlation-threshold", type=float, default=0.85)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--label-top-n", type=int, default=8)
    parser.add_argument("--limit-subfields", type=int, default=None)
    parser.add_argument("--subfield-id", default=None)
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--skip-heavy-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def output_relative(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir)).replace("\\", "/")
    except ValueError:
        return display_path(path)


def metric_family(metric_name: str) -> str:
    for family, metrics in METRIC_FAMILIES.items():
        if metric_name in metrics:
            return family
    if metric_name in DIAGNOSTIC_METRIC_COLUMNS:
        return "diagnostic"
    return "unknown"


def short_text(value: object, max_len: int = 44) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def safe_filename(text: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in text.lower())
    return "_".join(part for part in safe.split("_") if part)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame, written_tables: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    written_tables.append(path)


def write_matrix_csv(path: Path, frame: pd.DataFrame, written_tables: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    written_tables.append(path)


def expected_paths(output_dir: Path) -> dict[str, Path]:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    return {
        "data_quality_summary": tables / "data_quality_summary.csv",
        "metric_missingness": tables / "metric_missingness.csv",
        "low_variance_metrics": tables / "low_variance_metrics.csv",
        "duplicate_subfield_names_report": tables / "duplicate_subfield_names_report.csv",
        "domain_counts": tables / "domain_counts.csv",
        "field_counts": tables / "field_counts.csv",
        "metric_descriptive_stats": tables / "metric_descriptive_stats.csv",
        "metric_quantiles_long": tables / "metric_quantiles_long.csv",
        "metric_correlation_matrix": tables / "metric_correlation_matrix.csv",
        "metric_spearman_correlation_matrix": tables / "metric_spearman_correlation_matrix.csv",
        "high_correlation_pairs": tables / "high_correlation_pairs.csv",
        "redundancy_clusters": tables / "redundancy_clusters.csv",
        "family_scores": tables / "family_scores.csv",
        "family_score_descriptive_stats": tables / "family_score_descriptive_stats.csv",
        "family_score_correlation_matrix": tables / "family_score_correlation_matrix.csv",
        "top_bottom_subfields_by_family_score": tables
        / "top_bottom_subfields_by_family_score.csv",
        "domain_metric_summary": tables / "domain_metric_summary.csv",
        "field_metric_summary": tables / "field_metric_summary.csv",
        "domain_family_score_summary": tables / "domain_family_score_summary.csv",
        "field_family_score_summary": tables / "field_family_score_summary.csv",
        "metric_pca_scores": tables / "metric_pca_scores.csv",
        "metric_pca_loadings": tables / "metric_pca_loadings.csv",
        "metric_pca_explained_variance": tables / "metric_pca_explained_variance.csv",
        "top_pca_contributors": tables / "top_pca_contributors.csv",
        "top_bottom_subfields_by_metric": tables / "top_bottom_subfields_by_metric.csv",
        "case_study_candidates": tables / "case_study_candidates.csv",
        "morphology_archetypes": tables / "morphology_archetypes.csv",
        "curated_model_features_csv": tables / "curated_model_features.csv",
        "curated_model_features_parquet": tables / "curated_model_features.parquet",
        "quality_dashboard": figures / QUALITY_FIGURES / "data_quality_dashboard.png",
        "core_histograms_by_family": figures
        / DIST_FIGURES
        / "core_metric_histograms_by_family.png",
        "core_boxplots_by_family": figures
        / DIST_FIGURES
        / "core_metric_boxplots_by_family.png",
        "diagnostic_histograms": figures
        / DIST_FIGURES
        / "diagnostic_metric_histograms.png",
        "pearson_heatmap": figures
        / CORR_FIGURES
        / "core_metric_correlation_heatmap_clustered.png",
        "spearman_heatmap": figures
        / CORR_FIGURES
        / "core_metric_spearman_heatmap_clustered.png",
        "high_correlation_network": figures
        / CORR_FIGURES
        / "high_correlation_network.png",
        "redundancy_cluster_summary": figures
        / CORR_FIGURES
        / "redundancy_cluster_summary.png",
        "family_distributions": figures
        / FAMILY_FIGURES
        / "family_score_distributions.png",
        "family_correlation_heatmap": figures
        / FAMILY_FIGURES
        / "family_score_correlation_heatmap.png",
        "family_pairplots": figures
        / FAMILY_FIGURES
        / "family_score_pairplots_key_axes.png",
        "top_bottom_family_scores": figures
        / FAMILY_FIGURES
        / "top_bottom_family_scores.png",
        "domain_family_heatmap": figures
        / PROFILE_FIGURES
        / "domain_family_score_heatmap.png",
        "domain_family_boxplots": figures
        / PROFILE_FIGURES
        / "domain_family_score_boxplots.png",
        "domain_metric_profile_heatmap": figures
        / PROFILE_FIGURES
        / "domain_metric_profile_heatmap.png",
        "top_fields_by_family_scores": figures
        / PROFILE_FIGURES
        / "top_fields_by_family_scores.png",
        "pca_explained_variance": figures
        / PCA_FIGURES
        / "pca_explained_variance.png",
        "pca_scatter_by_domain": figures
        / PCA_FIGURES
        / "pca_scatter_by_domain.png",
        "pca_scatter_labeled_extremes": figures
        / PCA_FIGURES
        / "pca_scatter_labeled_extremes.png",
        "pca_loadings_heatmap": figures
        / PCA_FIGURES
        / "pca_loadings_heatmap.png",
        "pca_biplot_pc1_pc2": figures
        / PCA_FIGURES
        / "pca_biplot_pc1_pc2.png",
        "pca_biplot_pc2_pc3": figures
        / PCA_FIGURES
        / "pca_biplot_pc2_pc3.png",
        "top_bottom_selected_metrics": figures
        / RANKING_FIGURES
        / "top_bottom_by_selected_metrics.png",
        "ranking_top_bottom_family_scores": figures
        / RANKING_FIGURES
        / "top_bottom_by_family_scores.png",
        "case_study_matrix": figures
        / CASE_FIGURES
        / "case_study_candidate_matrix.png",
        "archetype_counts": figures / CASE_FIGURES / "archetype_counts.png",
        "archetype_domain_distribution": figures
        / CASE_FIGURES
        / "archetype_domain_distribution.png",
        "summary": output_dir / "morphology_analysis_summary.json",
        "report": output_dir / "morphology_analysis_report.md",
        "figure_index": output_dir / "morphology_analysis_figure_index.csv",
    }


def ensure_output_dirs(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path) for path in existing[:10])
        raise FileExistsError(
            "Refusing to overwrite morphology analysis outputs without --overwrite: "
            f"{formatted}"
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def cleanup_stale_legacy_outputs(output_dir: Path, overwrite: bool) -> None:
    if not overwrite:
        return
    figures_dir = output_dir / "figures"
    for file_name in STALE_LEGACY_FIGURE_NAMES:
        stale_path = figures_dir / file_name
        if stale_path.exists() and stale_path.is_file():
            stale_path.unlink()
    for file_name in STALE_LEGACY_ROOT_OUTPUT_NAMES:
        stale_path = output_dir / file_name
        if stale_path.exists() and stale_path.is_file():
            stale_path.unlink()


def select_rows(
    metrics: pd.DataFrame,
    *,
    subfield_id: str | None,
    limit_subfields: int | None,
) -> pd.DataFrame:
    selected = metrics.copy()
    if "metric_status" in selected.columns:
        selected = selected[selected["metric_status"] != "failed"].copy()
    selected["_subfield_sort_key"] = selected["subfield_id"].astype(str)
    selected = selected.sort_values("_subfield_sort_key", kind="mergesort").drop(
        columns="_subfield_sort_key"
    )
    if subfield_id is not None:
        selected = selected[selected["subfield_id"].astype(str) == str(subfield_id)]
        if selected.empty:
            raise ValueError(f"No morphology row matched --subfield-id {subfield_id}")
    if limit_subfields is not None:
        if limit_subfields <= 0:
            raise ValueError("limit_subfields must be positive when provided")
        selected = selected.head(limit_subfields)
    if selected.empty:
        raise ValueError("No morphology rows are available to analyze")
    selected = add_subfield_label_columns(selected.reset_index(drop=True))
    for column in METADATA_COLUMNS:
        if column not in selected.columns:
            selected[column] = np.nan
    return selected


def validate_core_columns(metrics: pd.DataFrame) -> None:
    missing = [column for column in CORE_METRIC_COLUMNS_V2 if column not in metrics.columns]
    if missing:
        raise ValueError(f"morphology table missing core metrics: {', '.join(missing)}")


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce")


def standardize_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = numeric_frame(frame, columns)
    values = values.fillna(values.median(numeric_only=True))
    mean = values.mean()
    std = values.std(ddof=0).replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def robust_z_scores(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = numeric_frame(frame, columns)
    medians = values.median()
    iqrs = (values.quantile(0.75) - values.quantile(0.25)).replace(0, np.nan)
    return ((values - medians) / iqrs).replace([np.inf, -np.inf], np.nan).clip(-5, 5)


def descriptive_stats_for_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    metric_kind: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_rows = len(frame)
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        q = values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        mean = float(values.mean()) if len(valid) else np.nan
        std = float(values.std()) if len(valid) else np.nan
        iqr = float(q.loc[0.75] - q.loc[0.25]) if len(valid) else np.nan
        lower = q.loc[0.25] - 1.5 * iqr if np.isfinite(iqr) else np.nan
        upper = q.loc[0.75] + 1.5 * iqr if np.isfinite(iqr) else np.nan
        outlier_share = (
            float(((values < lower) | (values > upper)).mean())
            if np.isfinite(lower) and np.isfinite(upper)
            else np.nan
        )
        skewness = float(valid.skew()) if len(valid) >= 3 else np.nan
        rows.append(
            {
                "metric_name": column,
                "metric_kind": metric_kind,
                "metric_family": metric_family(column),
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "missing_share": float(values.isna().mean()) if n_rows else np.nan,
                "mean": mean,
                "std": std,
                "min": float(values.min()) if len(valid) else np.nan,
                "p01": float(q.loc[0.01]) if len(valid) else np.nan,
                "p05": float(q.loc[0.05]) if len(valid) else np.nan,
                "p25": float(q.loc[0.25]) if len(valid) else np.nan,
                "median": float(q.loc[0.50]) if len(valid) else np.nan,
                "p75": float(q.loc[0.75]) if len(valid) else np.nan,
                "p95": float(q.loc[0.95]) if len(valid) else np.nan,
                "p99": float(q.loc[0.99]) if len(valid) else np.nan,
                "max": float(values.max()) if len(valid) else np.nan,
                "iqr": iqr,
                "coefficient_of_variation": float(std / abs(mean))
                if np.isfinite(mean) and np.isfinite(std) and abs(mean) > 1e-12
                else np.nan,
                "n_unique": int(values.nunique(dropna=True)),
                "skewness": skewness,
                "outlier_share_iqr_rule": outlier_share,
                "skewness_flag": bool(np.isfinite(skewness) and abs(skewness) >= 1.0),
                "extreme_outlier_flag": bool(
                    np.isfinite(outlier_share) and outlier_share >= 0.05
                ),
            }
        )
    return pd.DataFrame(rows)


def metric_descriptive_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    diagnostic_present = [
        column for column in DIAGNOSTIC_METRIC_COLUMNS if column in metrics.columns
    ]
    parts = [
        descriptive_stats_for_columns(
            metrics,
            CORE_METRIC_COLUMNS_V2,
            metric_kind="core",
        )
    ]
    if diagnostic_present:
        parts.append(
            descriptive_stats_for_columns(
                metrics,
                diagnostic_present,
                metric_kind="diagnostic",
            )
        )
    return pd.concat(parts, ignore_index=True)


def metric_quantiles_long(stats: pd.DataFrame) -> pd.DataFrame:
    quantile_columns = ["p01", "p05", "p25", "median", "p75", "p95", "p99"]
    rows = []
    for row in stats.itertuples(index=False):
        for column in quantile_columns:
            rows.append(
                {
                    "metric_name": row.metric_name,
                    "metric_kind": row.metric_kind,
                    "metric_family": row.metric_family,
                    "quantile": column,
                    "value": getattr(row, column),
                }
            )
    return pd.DataFrame(rows)


def metric_missingness(stats: pd.DataFrame) -> pd.DataFrame:
    return stats[
        [
            "metric_name",
            "metric_kind",
            "metric_family",
            "count",
            "missing",
            "missing_share",
            "n_unique",
        ]
    ].copy()


def low_variance_metrics(stats: pd.DataFrame) -> pd.DataFrame:
    flagged = stats[
        (stats["n_unique"] <= 3)
        | (stats["std"].abs() <= 1e-12)
        | (stats["iqr"].abs() <= 1e-12)
    ].copy()
    if flagged.empty:
        return pd.DataFrame(columns=list(stats.columns) + ["low_variance_reason"])
    reasons = []
    for row in flagged.itertuples(index=False):
        row_reasons = []
        if row.n_unique <= 3:
            row_reasons.append("n_unique <= 3")
        if np.isfinite(row.std) and abs(row.std) <= 1e-12:
            row_reasons.append("std close to 0")
        if np.isfinite(row.iqr) and abs(row.iqr) <= 1e-12:
            row_reasons.append("iqr close to 0")
        reasons.append("; ".join(row_reasons))
    flagged["low_variance_reason"] = reasons
    return flagged


def domain_counts(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("domain_display_name", dropna=False)
        .size()
        .reset_index(name="n_subfields")
        .sort_values(["n_subfields", "domain_display_name"], ascending=[False, True])
    )


def field_counts(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["domain_display_name", "field_display_name"], dropna=False)
        .size()
        .reset_index(name="n_subfields")
        .sort_values(
            ["n_subfields", "domain_display_name", "field_display_name"],
            ascending=[False, True, True],
        )
    )


def data_quality_summary(
    metrics: pd.DataFrame,
    stats: pd.DataFrame,
    duplicate_report: pd.DataFrame,
) -> pd.DataFrame:
    duplicated_names = (
        int(duplicate_report["subfield_display_name"].nunique())
        if len(duplicate_report)
        else 0
    )
    rows = [
        ("n_rows_analyzed", len(metrics)),
        ("n_domains", metrics["domain_display_name"].nunique(dropna=True)),
        ("n_fields", metrics["field_display_name"].nunique(dropna=True)),
        ("n_core_metrics", len(CORE_METRIC_COLUMNS_V2)),
        (
            "n_diagnostic_metrics_present",
            len([c for c in DIAGNOSTIC_METRIC_COLUMNS if c in metrics.columns]),
        ),
        ("n_family_scores", len(FAMILY_SCORE_COLUMNS)),
        ("n_duplicate_display_name_rows", len(duplicate_report)),
        ("n_duplicated_display_names", duplicated_names),
        ("n_metrics_with_missing_values", int((stats["missing"] > 0).sum())),
        ("median_n_points", float(pd.to_numeric(metrics["n_points"], errors="coerce").median())),
        ("min_n_points", float(pd.to_numeric(metrics["n_points"], errors="coerce").min())),
        ("max_n_points", float(pd.to_numeric(metrics["n_points"], errors="coerce").max())),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def clustered_correlation(correlation: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    corr = correlation.copy().fillna(0.0)
    if corr.shape[0] <= 2:
        return corr, list(corr.columns)
    values = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, 1.0)
    distance = 1.0 - np.abs(values)
    distance = np.clip(distance, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if not np.isfinite(condensed).all() or np.allclose(condensed, 0.0):
        return corr, list(corr.columns)
    order = leaves_list(linkage(condensed, method="average"))
    ordered_columns = [corr.columns[i] for i in order]
    return corr.loc[ordered_columns, ordered_columns], ordered_columns


def high_correlation_pairs(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    columns = list(pearson.columns)
    for i, metric_a in enumerate(columns):
        for metric_b in columns[i + 1 :]:
            pearson_value = float(pearson.loc[metric_a, metric_b])
            spearman_value = float(spearman.loc[metric_a, metric_b])
            max_abs = np.nanmax([abs(pearson_value), abs(spearman_value)])
            if not np.isfinite(max_abs) or max_abs < threshold:
                continue
            family_a = metric_family(metric_a)
            family_b = metric_family(metric_b)
            if family_a == "diagnostic" or family_b == "diagnostic":
                action = "diagnostic_only"
            elif family_a == family_b and abs(pearson_value) >= threshold:
                action = "keep_one_for_modeling"
            else:
                action = "keep_both_for_interpretation"
            rows.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "pearson": pearson_value,
                    "spearman": spearman_value,
                    "abs_pearson": abs(pearson_value),
                    "abs_spearman": abs(spearman_value),
                    "family_a": family_a,
                    "family_b": family_b,
                    "suggested_action": action,
                }
            )
    columns_out = [
        "metric_a",
        "metric_b",
        "pearson",
        "spearman",
        "abs_pearson",
        "abs_spearman",
        "family_a",
        "family_b",
        "suggested_action",
    ]
    if not rows:
        return pd.DataFrame(columns=columns_out)
    return pd.DataFrame(rows).sort_values(
        ["abs_pearson", "abs_spearman", "metric_a", "metric_b"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )[columns_out]


def redundancy_clusters(
    high_pairs: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    adjacency: dict[str, set[str]] = {metric: set() for metric in CORE_METRIC_COLUMNS_V2}
    for row in high_pairs.itertuples(index=False):
        adjacency[row.metric_a].add(row.metric_b)
        adjacency[row.metric_b].add(row.metric_a)

    stat_index = stats.set_index("metric_name")
    visited: set[str] = set()
    rows: list[dict[str, Any]] = []
    cluster_id = 0
    for metric in CORE_METRIC_COLUMNS_V2:
        if metric in visited:
            continue
        cluster_id += 1
        queue = deque([metric])
        component = []
        visited.add(metric)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in sorted(adjacency[current]):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        component = [m for m in CORE_METRIC_COLUMNS_V2 if m in component]
        primary = sorted(
            component,
            key=lambda name: (
                float(stat_index.loc[name, "missing_share"])
                if name in stat_index.index
                else 1.0,
                -int(stat_index.loc[name, "n_unique"])
                if name in stat_index.index
                else 0,
                name,
            ),
        )[0]
        for name in component:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "metric_name": name,
                    "metric_family": metric_family(name),
                    "n_cluster_metrics": len(component),
                    "cluster_metrics": ", ".join(component),
                    "suggested_primary_metric": primary,
                    "has_high_correlation_edge": bool(adjacency[name]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["n_cluster_metrics", "cluster_id", "metric_name"],
        ascending=[False, True, True],
        kind="mergesort",
    )


def compute_family_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    z = robust_z_scores(metrics, CORE_METRIC_COLUMNS_V2)
    output = metrics[METADATA_COLUMNS].copy()
    for score_name, components in FAMILY_SCORE_DEFINITIONS.items():
        component_values = []
        used_names = []
        for metric_name, sign in components:
            if metric_name not in z.columns:
                continue
            component_values.append(z[metric_name] * sign)
            used_names.append(metric_name)
        if component_values:
            component_frame = pd.concat(component_values, axis=1)
            output[score_name] = component_frame.mean(axis=1, skipna=True)
            output[f"{score_name}_n_components"] = component_frame.notna().sum(axis=1)
            output[f"{score_name}_components"] = ", ".join(used_names)
        else:
            output[score_name] = np.nan
            output[f"{score_name}_n_components"] = 0
            output[f"{score_name}_components"] = ""
    return output


def curated_model_features(metrics: pd.DataFrame, family_scores: pd.DataFrame) -> pd.DataFrame:
    return metrics[METADATA_COLUMNS + CORE_METRIC_COLUMNS_V2].merge(
        family_scores[["subfield_id"] + FAMILY_SCORE_COLUMNS],
        on="subfield_id",
        how="left",
        validate="one_to_one",
    )


def top_bottom_subfields(
    metrics: pd.DataFrame,
    columns: list[str],
    *,
    n: int,
    value_name: str,
) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in metrics.columns:
            continue
        ranked = metrics.dropna(subset=[column]).copy()
        for direction, ascending in [("bottom", True), ("top", False)]:
            subset = ranked.sort_values(
                [column, "subfield_id"],
                ascending=[ascending, True],
                kind="mergesort",
            ).head(n)
            for rank, row in enumerate(subset.itertuples(index=False), start=1):
                rows.append(
                    {
                        "metric_name": column,
                        "rank_direction": direction,
                        "rank": rank,
                        "subfield_id": row.subfield_id,
                        "subfield_label_unique": row.subfield_label_unique,
                        "subfield_label_short": row.subfield_label_short,
                        "subfield_display_name": row.subfield_display_name,
                        "field_display_name": row.field_display_name,
                        "domain_display_name": row.domain_display_name,
                        value_name: float(getattr(row, column)),
                    }
                )
    return pd.DataFrame(rows)


def group_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
    *,
    metric_kind: str,
) -> pd.DataFrame:
    z = standardize_frame(frame, value_columns)
    z_frame = pd.concat([frame[group_columns].reset_index(drop=True), z], axis=1)
    rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        z_group = z_frame.loc[group.index]
        base = dict(zip(group_columns, group_key))
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row = {
                **base,
                "metric_name": column,
                "metric_kind": metric_kind,
                "metric_family": metric_family(column)
                if metric_kind == "core"
                else "family_score",
                "n_subfields": int(len(group)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean_z": float(z_group[column].mean()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def run_pca(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = standardize_frame(metrics, CORE_METRIC_COLUMNS_V2)
    n_components = min(5, len(x), len(CORE_METRIC_COLUMNS_V2))
    if n_components < 2:
        raise ValueError("PCA requires at least two analyzed rows")
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(x)
    score_columns = [f"PC{i + 1}" for i in range(n_components)]
    score_frame = metrics[METADATA_COLUMNS].copy()
    for idx, column in enumerate(score_columns):
        score_frame[column] = scores[:, idx]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=score_columns,
        index=CORE_METRIC_COLUMNS_V2,
    )
    loadings.insert(0, "metric_family", [metric_family(m) for m in loadings.index])
    loadings = loadings.reset_index().rename(columns={"index": "metric_name"})

    explained = pd.DataFrame(
        {
            "component": score_columns,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )

    contributor_rows = []
    for component in score_columns[: min(3, len(score_columns))]:
        component_loadings = loadings[["metric_name", "metric_family", component]].copy()
        component_loadings["abs_loading"] = component_loadings[component].abs()
        for direction, ascending in [("positive", False), ("negative", True)]:
            subset = component_loadings.sort_values(
                [component, "metric_name"],
                ascending=[ascending, True],
                kind="mergesort",
            ).head(8)
            for rank, row in enumerate(subset.itertuples(index=False), start=1):
                contributor_rows.append(
                    {
                        "component": component,
                        "direction": direction,
                        "rank": rank,
                        "metric_name": row.metric_name,
                        "metric_family": row.metric_family,
                        "loading": float(getattr(row, component)),
                        "abs_loading": float(row.abs_loading),
                    }
                )
    contributors = pd.DataFrame(contributor_rows)
    return score_frame, loadings, explained, contributors


def domain_color_map(domains: Iterable[object]) -> dict[str, Any]:
    labels = sorted(str(label) for label in pd.Series(list(domains)).dropna().unique())
    if not labels:
        return {}
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    return {label: cmap(idx) for idx, label in enumerate(labels)}


def scatter_by_domain(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    colors: dict[str, Any],
    alpha: float = 0.80,
    size: float = 34,
) -> None:
    for domain in sorted(frame["domain_display_name"].dropna().astype(str).unique()):
        subset = frame[frame["domain_display_name"].astype(str) == domain]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=size,
            alpha=alpha,
            color=colors.get(domain, "#4c78a8"),
            label=domain,
            edgecolor="white",
            linewidth=0.4,
        )


def label_extreme_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    label_top_n: int,
    label_column: str = "subfield_label_short",
) -> None:
    if label_top_n <= 0 or frame.empty:
        return
    values = frame[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
    if values.notna().sum().min() == 0:
        return
    zx = (values[x_col] - values[x_col].mean()) / values[x_col].std(ddof=0)
    zy = (values[y_col] - values[y_col].mean()) / values[y_col].std(ddof=0)
    distance = np.sqrt(zx.fillna(0.0) ** 2 + zy.fillna(0.0) ** 2)
    label_rows = frame.assign(_distance=distance).sort_values(
        ["_distance", "subfield_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(label_top_n)
    for row in label_rows.itertuples(index=False):
        x = getattr(row, x_col)
        y = getattr(row, y_col)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        ax.annotate(
            short_text(getattr(row, label_column), 34),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            color="#222222",
        )


def save_figure(
    fig: plt.Figure,
    path: Path,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    *,
    group: str,
    title: str,
    description: str,
    question: str,
    related_tables: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figure_rows.append(
        {
            "figure_path": output_relative(path, output_dir),
            "figure_group": group,
            "title": title,
            "description": description,
            "primary_question_answered": question,
            "related_tables": "; ".join(output_relative(p, output_dir) for p in related_tables),
        }
    )


def plot_data_quality_dashboard(
    metrics: pd.DataFrame,
    stats: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=dpi)
    axes = axes.ravel()

    counts = domain_counts(metrics)
    axes[0].bar(counts["domain_display_name"].astype(str), counts["n_subfields"], color="#4c78a8")
    axes[0].set_title("Subfields by Domain")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].set_ylabel("Subfields")

    points = pd.to_numeric(metrics["n_points"], errors="coerce").dropna()
    axes[1].hist(points, bins=min(20, max(5, int(math.sqrt(max(len(points), 1))))), color="#59a14f")
    axes[1].set_title("Distribution of n_points")
    axes[1].set_xlabel("Papers in morphology map")
    axes[1].set_ylabel("Subfields")

    duplicate_values = [
        int(metrics["subfield_display_name_is_duplicated"].sum()),
        int((~metrics["subfield_display_name_is_duplicated"]).sum()),
    ]
    axes[2].bar(["Duplicated name", "Unique name"], duplicate_values, color=["#e15759", "#76b7b2"])
    axes[2].set_title("Display-Name Ambiguity")
    axes[2].set_ylabel("Rows")

    missing = stats.sort_values(["missing_share", "metric_name"], ascending=[False, True]).head(12)
    axes[3].barh(missing["metric_name"], missing["missing_share"], color="#f28e2b")
    axes[3].invert_yaxis()
    axes[3].set_xlim(0, max(1.0, float(missing["missing_share"].max() if len(missing) else 0)))
    axes[3].set_title("Missingness Overview")
    axes[3].set_xlabel("Missing share")
    axes[3].tick_params(axis="y", labelsize=7)

    family_counts = pd.Series({family: len(cols) for family, cols in METRIC_FAMILIES.items()})
    family_counts.loc["diagnostic"] = len([c for c in DIAGNOSTIC_METRIC_COLUMNS if c in metrics.columns])
    axes[4].bar(family_counts.index, family_counts.values, color="#b07aa1")
    axes[4].set_title("Metrics by Family/Role")
    axes[4].tick_params(axis="x", rotation=35, labelsize=8)
    axes[4].set_ylabel("Metrics")

    axes[5].axis("off")
    text = (
        f"Rows analyzed: {len(metrics)}\n"
        f"Domains: {metrics['domain_display_name'].nunique(dropna=True)}\n"
        f"Fields: {metrics['field_display_name'].nunique(dropna=True)}\n"
        f"Core metrics: {len(CORE_METRIC_COLUMNS_V2)}\n"
        f"Diagnostics present: {family_counts.loc['diagnostic']}\n"
        f"Duplicate-name rows: {len(duplicate_report)}"
    )
    axes[5].text(0.05, 0.75, text, va="top", fontsize=12)

    fig.suptitle("Morphology Metric Data Quality Dashboard", fontsize=16, y=1.02)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=QUALITY_FIGURES,
        title="Data Quality Dashboard",
        description="Domain coverage, point counts, duplicate display names, missingness, and metric counts.",
        question="Is the morphology feature layer complete enough to explore?",
        related_tables=related_tables,
    )


def plot_core_histograms_by_family(
    metrics: pd.DataFrame,
    stats: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    max_cols = max(len(cols) for cols in METRIC_FAMILIES.values())
    fig, axes = plt.subplots(len(METRIC_FAMILIES), max_cols, figsize=(19, 13), dpi=dpi)
    stat_index = stats.set_index("metric_name")
    for row_idx, (family, columns) in enumerate(METRIC_FAMILIES.items()):
        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(columns):
                ax.axis("off")
                continue
            metric_name = columns[col_idx]
            values = pd.to_numeric(metrics[metric_name], errors="coerce").dropna()
            ax.hist(values, bins=min(18, max(6, int(math.sqrt(max(len(values), 1))))), color="#4c78a8", alpha=0.85)
            median = float(values.median()) if len(values) else np.nan
            if np.isfinite(median):
                ax.axvline(median, color="#e15759", linewidth=1.0)
            flags = []
            if metric_name in stat_index.index:
                if bool(stat_index.loc[metric_name, "skewness_flag"]):
                    flags.append("skewed")
                if bool(stat_index.loc[metric_name, "extreme_outlier_flag"]):
                    flags.append("outliers")
            suffix = f" ({', '.join(flags)})" if flags else ""
            ax.set_title(f"{metric_name}{suffix}", fontsize=8)
            ax.tick_params(labelsize=7)
            if col_idx == 0:
                ax.set_ylabel(family.replace("_", " "), fontsize=9)
    fig.suptitle("Core Metric Distributions by Conceptual Family", fontsize=16, y=1.02)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=DIST_FIGURES,
        title="Core Metric Histograms by Family",
        description="Small multiples of core metric distributions with median reference lines.",
        question="Which morphology metrics vary meaningfully or look skewed?",
        related_tables=related_tables,
    )


def plot_core_boxplots_by_family(
    metrics: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    z = standardize_frame(metrics, CORE_METRIC_COLUMNS_V2)
    fig, axes = plt.subplots(len(METRIC_FAMILIES), 1, figsize=(15, 12), dpi=dpi)
    for ax, (family, columns) in zip(axes, METRIC_FAMILIES.items()):
        data = [z[column].dropna() for column in columns]
        ax.boxplot(data, vert=False, tick_labels=columns, showfliers=True, patch_artist=True)
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.set_title(family.replace("_", " ").title(), fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Standardized value")
    fig.suptitle("Standardized Core Metric Boxplots by Family", fontsize=16, y=1.01)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=DIST_FIGURES,
        title="Core Metric Boxplots by Family",
        description="Standardized boxplots make metric spread and outliers comparable across scales.",
        question="Which core metrics have broad spread or many outliers?",
        related_tables=related_tables,
    )


def plot_diagnostic_histograms(
    metrics: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    diagnostic_present = [c for c in DIAGNOSTIC_METRIC_COLUMNS if c in metrics.columns]
    n_cols = 4
    n_rows = max(1, int(math.ceil(len(diagnostic_present) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax, metric_name in zip(axes, diagnostic_present):
        values = pd.to_numeric(metrics[metric_name], errors="coerce").dropna()
        ax.hist(values, bins=min(18, max(5, int(math.sqrt(max(len(values), 1))))), color="#f28e2b", alpha=0.85)
        ax.set_title(metric_name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(diagnostic_present) :]:
        ax.axis("off")
    fig.suptitle("Diagnostic Metric Distributions", fontsize=16, y=1.02)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=DIST_FIGURES,
        title="Diagnostic Metric Histograms",
        description="Distribution checks for diagnostic metrics that are not part of the 25 core features.",
        question="Do diagnostic-only metrics reveal artifacts or sparse controls?",
        related_tables=related_tables,
    )


def plot_correlation_heatmap(
    correlation: pd.DataFrame,
    path: Path,
    *,
    title: str,
    description: str,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    clustered, _ = clustered_correlation(correlation)
    fig, ax = plt.subplots(figsize=(13, 11), dpi=dpi)
    image = ax.imshow(clustered.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(clustered.columns)))
    ax.set_yticks(range(len(clustered.index)))
    ax.set_xticklabels(clustered.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(clustered.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=CORR_FIGURES,
        title=title,
        description=description,
        question="Which morphology metrics are redundant or strongly related?",
        related_tables=related_tables,
    )


def plot_high_correlation_network(
    high_pairs: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(13, 10), dpi=dpi)
    metrics_in_edges = sorted(set(high_pairs.get("metric_a", [])) | set(high_pairs.get("metric_b", [])))
    if not metrics_in_edges:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No core metric pairs exceeded the high-correlation threshold.",
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        angles = np.linspace(0, 2 * np.pi, len(metrics_in_edges), endpoint=False)
        positions = {
            metric: (math.cos(angle), math.sin(angle))
            for metric, angle in zip(metrics_in_edges, angles)
        }
        family_colors = {
            family: color
            for family, color in zip(
                METRIC_FAMILIES,
                ["#4c78a8", "#f28e2b", "#59a14f", "#e15759", "#b07aa1"],
            )
        }
        for row in high_pairs.itertuples(index=False):
            x1, y1 = positions[row.metric_a]
            x2, y2 = positions[row.metric_b]
            weight = max(row.abs_pearson, row.abs_spearman)
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="#777777",
                alpha=0.25 + 0.55 * min(weight, 1.0),
                linewidth=1.0 + 2.0 * min(weight, 1.0),
            )
        for metric in metrics_in_edges:
            x, y = positions[metric]
            family = metric_family(metric)
            ax.scatter(
                x,
                y,
                s=260,
                color=family_colors.get(family, "#999999"),
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            ax.text(
                x * 1.10,
                y * 1.10,
                metric,
                ha="center",
                va="center",
                fontsize=8,
            )
        ax.set_title("High-Correlation Network of Core Metrics")
        ax.axis("off")
        ax.set_aspect("equal")
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=CORR_FIGURES,
        title="High-Correlation Network",
        description="Network-like view where nodes are core metrics and edges are high absolute correlations.",
        question="Which metric redundancies form connected groups?",
        related_tables=related_tables,
    )


def plot_redundancy_cluster_summary(
    clusters: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    summary = (
        clusters.drop_duplicates("cluster_id")
        .sort_values(["n_cluster_metrics", "cluster_id"], ascending=[False, True])
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(11, 6), dpi=dpi)
    labels = [f"C{row.cluster_id}: {short_text(row.cluster_metrics, 52)}" for row in summary.itertuples(index=False)]
    ax.barh(labels, summary["n_cluster_metrics"], color="#76b7b2")
    ax.invert_yaxis()
    ax.set_xlabel("Metrics in redundancy cluster")
    ax.set_title("Largest Redundancy Clusters")
    ax.tick_params(axis="y", labelsize=8)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=CORR_FIGURES,
        title="Redundancy Cluster Summary",
        description="Connected components from high-correlation pairs, including singleton clusters.",
        question="Which clusters might need feature-selection attention later?",
        related_tables=related_tables,
    )


def plot_family_score_distributions(
    family_scores: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=dpi)
    for ax, score in zip(axes.ravel(), FAMILY_SCORE_COLUMNS):
        values = pd.to_numeric(family_scores[score], errors="coerce").dropna()
        ax.hist(values, bins=min(18, max(6, int(math.sqrt(max(len(values), 1))))), color="#4c78a8", alpha=0.85)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_title(score, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Family Score Distributions", fontsize=16, y=1.02)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=FAMILY_FIGURES,
        title="Family Score Distributions",
        description="Distributions for the eight robust-z family scores.",
        question="Which interpretable morphology dimensions vary most?",
        related_tables=related_tables,
    )


def plot_family_correlation_heatmap(
    family_scores: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    corr = family_scores[FAMILY_SCORE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=dpi)
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Family Score Correlations")
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=FAMILY_FIGURES,
        title="Family Score Correlation Heatmap",
        description="Correlation among the eight interpretable family scores.",
        question="Which morphology families overlap conceptually?",
        related_tables=related_tables,
    )


def plot_family_scatter(
    family_scores: pd.DataFrame,
    x_col: str,
    y_col: str,
    path: Path,
    *,
    dpi: int,
    label_top_n: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
    register: bool = True,
) -> None:
    colors = domain_color_map(family_scores["domain_display_name"])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    scatter_by_domain(ax, family_scores, x_col, y_col, colors=colors)
    ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="#666666", linewidth=0.8, alpha=0.6)
    label_extreme_points(ax, family_scores, x_col, y_col, label_top_n=label_top_n)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    ax.legend(fontsize=7, frameon=False, loc="best")
    if register:
        save_figure(
            fig,
            path,
            figure_rows,
            output_dir,
            group=FAMILY_FIGURES,
            title=f"{x_col} vs {y_col}",
            description="Domain-colored family-score scatter with only extreme points labeled.",
            question="Which subfields sit at extremes of this morphology-family contrast?",
            related_tables=related_tables,
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def plot_family_pair_panel(
    family_scores: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    label_top_n: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    colors = domain_color_map(family_scores["domain_display_name"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=dpi)
    axes = axes.ravel()
    for ax, (x_col, y_col) in zip(axes, KEY_FAMILY_SCATTERS):
        scatter_by_domain(ax, family_scores, x_col, y_col, colors=colors, size=30)
        ax.axhline(0, color="#666666", linewidth=0.7, alpha=0.6)
        ax.axvline(0, color="#666666", linewidth=0.7, alpha=0.6)
        label_extreme_points(
            ax,
            family_scores,
            x_col,
            y_col,
            label_top_n=max(2, min(label_top_n, 5)),
        )
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)
        ax.set_title(f"{x_col} vs {y_col}", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, fontsize=8, frameon=False, loc="center")
    fig.suptitle("Key Family-Score Axes", fontsize=16, y=1.02)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=FAMILY_FIGURES,
        title="Family Score Pairplots on Key Axes",
        description="Combined panel for interpretable family-score contrasts.",
        question="Which dimensions structure the subfields in interpretable pairs?",
        related_tables=related_tables,
    )


def plot_top_bottom_bars(
    rankings: pd.DataFrame,
    path: Path,
    *,
    metric_columns: list[str],
    value_name: str,
    title: str,
    group: str,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    n_metrics = len(metric_columns)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, max(4, 2.4 * n_metrics)), dpi=dpi)
    axes = np.atleast_1d(axes)
    for ax, metric_name in zip(axes, metric_columns):
        subset = rankings[rankings["metric_name"] == metric_name].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset = subset[subset["rank"] <= 5].copy()
        subset["plot_label"] = (
            subset["rank_direction"].str[0].str.upper()
            + subset["rank"].astype(str)
            + " "
            + subset["subfield_label_short"].map(lambda x: short_text(x, 42))
        )
        subset["signed_value"] = np.where(
            subset["rank_direction"] == "bottom",
            -subset[value_name].abs(),
            subset[value_name].abs(),
        )
        colors = np.where(subset["rank_direction"] == "bottom", "#76b7b2", "#e15759")
        ax.barh(subset["plot_label"], subset["signed_value"], color=colors)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_title(metric_name, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(title, fontsize=16, y=1.005)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=group,
        title=title,
        description="Top and bottom ranked subfields for selected metrics or family scores.",
        question="Which subfields are the clearest extreme examples?",
        related_tables=related_tables,
    )


def heatmap_from_summary(
    summary: pd.DataFrame,
    row_column: str,
    value_columns: list[str],
    path: Path,
    *,
    title: str,
    group: str,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
    max_rows: int | None = None,
) -> None:
    pivot = summary.pivot_table(
        index=row_column,
        columns="metric_name",
        values="mean_z",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=[c for c in value_columns if c in pivot.columns])
    counts = summary.groupby(row_column)["n_subfields"].max()
    pivot = pivot.loc[counts.sort_values(ascending=False).index]
    if max_rows is not None:
        pivot = pivot.head(max_rows)
    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(pivot.columns)), max(5, 0.42 * len(pivot))), dpi=dpi)
    image = ax.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    labels = [f"{idx} (n={int(counts.loc[idx])})" for idx in pivot.index]
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Mean standardized score")
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=group,
        title=title,
        description="Group-level mean standardized morphology profile.",
        question="Which domains or fields have distinctive morphology profiles?",
        related_tables=related_tables,
    )


def plot_domain_family_boxplots(
    family_scores: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    domains = sorted(family_scores["domain_display_name"].dropna().astype(str).unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=dpi)
    for ax, score_name in zip(axes.ravel(), FAMILY_SCORE_COLUMNS):
        data = [
            family_scores.loc[
                family_scores["domain_display_name"].astype(str) == domain,
                score_name,
            ].dropna()
            for domain in domains
        ]
        ax.boxplot(data, tick_labels=domains, showfliers=False)
        ax.axhline(0, color="#666666", linewidth=0.7, alpha=0.6)
        ax.set_title(score_name, fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Domain Family-Score Distributions", fontsize=16, y=1.02)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=PROFILE_FIGURES,
        title="Domain Family Score Boxplots",
        description="Family-score distributions by OpenAlex domain.",
        question="Which domain differences are broad versus driven by a few subfields?",
        related_tables=related_tables,
    )


def plot_top_fields_by_family_scores(
    field_summary: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), dpi=dpi)
    for ax, score in zip(axes.ravel(), FAMILY_SCORE_COLUMNS):
        subset = field_summary[field_summary["metric_name"] == score].copy()
        subset = subset.sort_values(["mean_z", "field_display_name"], ascending=[False, True]).head(8)
        labels = [
            f"{short_text(row.field_display_name, 30)} (n={int(row.n_subfields)})"
            for row in subset.itertuples(index=False)
        ]
        ax.barh(labels, subset["mean_z"], color="#4c78a8")
        ax.invert_yaxis()
        ax.axvline(0, color="#666666", linewidth=0.7)
        ax.set_title(score, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Top Fields by Family Scores", fontsize=16, y=1.02)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=PROFILE_FIGURES,
        title="Top Fields by Family Scores",
        description="Highest-mean fields for each family score, with field subfield counts.",
        question="Which fields stand out on each morphology family?",
        related_tables=related_tables,
    )


def plot_pca_explained_variance(
    explained: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    ax.bar(explained["component"], explained["explained_variance_ratio"], color="#4c78a8")
    ax.plot(
        explained["component"],
        explained["cumulative_explained_variance_ratio"],
        color="#e15759",
        marker="o",
        label="Cumulative",
    )
    ax.set_ylim(0, min(1.0, max(0.1, float(explained["cumulative_explained_variance_ratio"].max()) * 1.1)))
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Metric-Space PCA Explained Variance")
    ax.legend(frameon=False)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=PCA_FIGURES,
        title="PCA Explained Variance",
        description="Explained and cumulative variance for exploratory PCA over standardized core metrics.",
        question="How many broad dimensions summarize the metric table?",
        related_tables=related_tables,
    )


def plot_pca_scatter(
    scores: pd.DataFrame,
    path: Path,
    *,
    title: str,
    label_extremes: bool,
    dpi: int,
    label_top_n: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=dpi)
    colors = domain_color_map(scores["domain_display_name"])
    scatter_by_domain(ax, scores, "PC1", "PC2", colors=colors)
    ax.axhline(0, color="#666666", linewidth=0.7, alpha=0.6)
    ax.axvline(0, color="#666666", linewidth=0.7, alpha=0.6)
    if label_extremes:
        label_extreme_points(ax, scores, "PC1", "PC2", label_top_n=label_top_n)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False, loc="best")
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=PCA_FIGURES,
        title=title,
        description="Exploratory PCA scatter of subfields in standardized morphology-metric space.",
        question="Which subfields occupy different regions of morphology-metric space?",
        related_tables=related_tables,
    )


def plot_pca_loadings_heatmap(
    loadings: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    pc_columns = [column for column in loadings.columns if column.startswith("PC")]
    matrix = loadings.set_index("metric_name")[pc_columns]
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=dpi)
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pc_columns)))
    ax.set_xticklabels(pc_columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title("PCA Loadings Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Loading")
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=PCA_FIGURES,
        title="PCA Loadings Heatmap",
        description="Metric loadings for exploratory PCA components.",
        question="Which metrics define each broad morphology dimension?",
        related_tables=related_tables,
    )


def plot_pca_biplot(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    component_x: str,
    component_y: str,
    path: Path,
    *,
    dpi: int,
    label_top_n: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)
    if component_x not in scores.columns or component_y not in scores.columns:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"{component_x}/{component_y} not available for this run.",
            ha="center",
            va="center",
        )
    else:
        colors = domain_color_map(scores["domain_display_name"])
        scatter_by_domain(ax, scores, component_x, component_y, colors=colors, alpha=0.55, size=26)
        label_extreme_points(
            ax,
            scores,
            component_x,
            component_y,
            label_top_n=max(2, min(label_top_n, 6)),
        )
        loading_subset = loadings[["metric_name", component_x, component_y]].copy()
        loading_subset["length"] = np.sqrt(
            loading_subset[component_x] ** 2 + loading_subset[component_y] ** 2
        )
        loading_subset = loading_subset.sort_values(
            ["length", "metric_name"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)
        score_range = max(
            scores[component_x].max() - scores[component_x].min(),
            scores[component_y].max() - scores[component_y].min(),
            1.0,
        )
        scale = 0.32 * score_range
        for row in loading_subset.itertuples(index=False):
            lx = getattr(row, component_x) * scale
            ly = getattr(row, component_y) * scale
            ax.arrow(0, 0, lx, ly, color="#333333", alpha=0.65, head_width=0.05 * scale, length_includes_head=True)
            ax.text(lx * 1.08, ly * 1.08, row.metric_name, fontsize=7, ha="center", va="center")
        ax.axhline(0, color="#666666", linewidth=0.7, alpha=0.6)
        ax.axvline(0, color="#666666", linewidth=0.7, alpha=0.6)
        ax.set_xlabel(component_x)
        ax.set_ylabel(component_y)
        ax.legend(fontsize=7, frameon=False, loc="best")
    title = f"PCA Biplot {component_x} / {component_y}"
    ax.set_title(title)
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=PCA_FIGURES,
        title=title,
        description="Biplot with subfield scores and strongest metric loading arrows.",
        question="How do subfield positions relate to the metrics defining this PCA plane?",
        related_tables=related_tables,
    )


def build_case_study_candidates(
    metrics: pd.DataFrame,
    family_scores: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    combined = metrics[METADATA_COLUMNS + CORE_METRIC_COLUMNS_V2].merge(
        family_scores[["subfield_id"] + FAMILY_SCORE_COLUMNS],
        on="subfield_id",
        how="left",
        validate="one_to_one",
    )
    metric_z = standardize_frame(combined, CORE_METRIC_COLUMNS_V2)
    family_z = standardize_frame(combined, FAMILY_SCORE_COLUMNS)
    metric_threshold = 1.75
    family_threshold = 1.15
    rows = []
    for idx, row in combined.iterrows():
        extreme_metrics = [
            metric
            for metric in CORE_METRIC_COLUMNS_V2
            if abs(float(metric_z.loc[idx, metric])) >= metric_threshold
        ]
        extreme_families = [
            score
            for score in FAMILY_SCORE_COLUMNS
            if abs(float(family_z.loc[idx, score])) >= family_threshold
        ]
        reason_parts: list[str] = []
        if (
            "temporal_dynamism_score" in extreme_families
            and "directional_change_score" in extreme_families
            and row["temporal_dynamism_score"] > 0
            and row["directional_change_score"] > 0
        ):
            reason_parts.append("high temporal dynamism and high directional change")
        if (
            "concentration_score" in extreme_families
            and "elongation_score" in extreme_families
            and row["concentration_score"] > 0
            and row["elongation_score"] > 0
        ):
            reason_parts.append("high concentration and high elongation")
        if "fragmentation_score" in extreme_families and row["fragmentation_score"] > 0:
            reason_parts.append("high fragmentation")
        if (
            "expansion_score" in extreme_families
            and row["expansion_score"] > 0
            and (
                "diversification_score" not in extreme_families
                or row["diversification_score"] < 0
            )
        ):
            reason_parts.append("strong expansion but low diversification")
        if "diversification_score" in extreme_families and row["diversification_score"] > 0:
            reason_parts.append("strong diversification")
        if (
            "diffuseness_score" in extreme_families
            and row["diffuseness_score"] > 0
            and row.get("concentration_score", 0) < 0
        ):
            reason_parts.append("diffuse and low concentration")
        if not reason_parts and extreme_families:
            readable = [
                score.replace("_score", "").replace("_", " ")
                for score in extreme_families[:3]
            ]
            reason_parts.append("extreme family scores: " + ", ".join(readable))
        if not reason_parts and extreme_metrics:
            reason_parts.append("extreme core metrics: " + ", ".join(extreme_metrics[:3]))
        if not reason_parts:
            continue
        rows.append(
            {
                "subfield_id": row["subfield_id"],
                "subfield_label_unique": row["subfield_label_unique"],
                "subfield_label_short": row["subfield_label_short"],
                "domain_display_name": row["domain_display_name"],
                "field_display_name": row["field_display_name"],
                "n_extreme_core_metrics": len(extreme_metrics),
                "n_extreme_family_scores": len(extreme_families),
                "extreme_metric_names": ", ".join(extreme_metrics),
                "extreme_family_names": ", ".join(extreme_families),
                "case_study_reason": "; ".join(reason_parts),
                "candidate_score": len(extreme_metrics) + 2 * len(extreme_families),
            }
        )
    columns = [
        "subfield_id",
        "subfield_label_unique",
        "subfield_label_short",
        "domain_display_name",
        "field_display_name",
        "n_extreme_core_metrics",
        "n_extreme_family_scores",
        "extreme_metric_names",
        "extreme_family_names",
        "case_study_reason",
        "candidate_score",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["candidate_score", "n_extreme_family_scores", "n_extreme_core_metrics", "subfield_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        .head(max(top_n * 3, top_n))
        .reset_index(drop=True)[columns]
    )


def plot_case_study_matrix(
    candidates: pd.DataFrame,
    family_scores: pd.DataFrame,
    path: Path,
    *,
    top_n: int,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * top_n)), dpi=dpi)
    if candidates.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No case-study candidates met the extreme-score rule.", ha="center", va="center")
    else:
        subset = candidates.head(top_n).merge(
            family_scores[["subfield_id"] + FAMILY_SCORE_COLUMNS],
            on="subfield_id",
            how="left",
        )
        matrix = subset.set_index("subfield_label_short")[FAMILY_SCORE_COLUMNS]
        image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([short_text(label, 58) for label in matrix.index], fontsize=8)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
        ax.set_title("Case-Study Candidate Family-Score Matrix")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Family score")
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=CASE_FIGURES,
        title="Case Study Candidate Matrix",
        description="Family-score heatmap for the strongest rule-based case-study candidates.",
        question="Which subfields are useful thesis case studies and why?",
        related_tables=related_tables,
    )


def build_archetypes(family_scores: pd.DataFrame) -> pd.DataFrame:
    output = family_scores[
        [
            "subfield_id",
            "subfield_label_unique",
            "subfield_label_short",
            "domain_display_name",
            "field_display_name",
        ]
        + FAMILY_SCORE_COLUMNS
    ].copy()
    z = standardize_frame(output, FAMILY_SCORE_COLUMNS)
    thresholds = {
        score: {
            "upper": max(1.0, float(z[score].quantile(0.85))),
            "lower": min(-1.0, float(z[score].quantile(0.15))),
        }
        for score in FAMILY_SCORE_COLUMNS
    }
    tag_rules = [
        ("Diffuse", "diffuseness_score", "upper", "high diffuseness"),
        ("Concentrated", "concentration_score", "upper", "high concentration"),
        ("Fragmented", "fragmentation_score", "upper", "high fragmentation"),
        ("Elongated", "elongation_score", "upper", "high elongation"),
        ("Temporally dynamic", "temporal_dynamism_score", "upper", "high temporal dynamism"),
        ("Directionally changing", "directional_change_score", "upper", "high directional change"),
        ("Expanding", "expansion_score", "upper", "high radial expansion"),
        ("Diversifying", "diversification_score", "upper", "increasing density entropy"),
        ("Contracting", "expansion_score", "lower", "low radial expansion or contraction"),
        ("Consolidating", "diversification_score", "lower", "decreasing density entropy"),
    ]
    rows = []
    for idx, row in output.iterrows():
        tags = []
        reasons = []
        for tag, score, side, reason in tag_rules:
            value = float(z.loc[idx, score])
            if side == "upper" and value >= thresholds[score]["upper"]:
                tags.append(tag)
                reasons.append(reason)
            elif side == "lower" and value <= thresholds[score]["lower"]:
                tags.append(tag)
                reasons.append(reason)
        if not tags:
            tags = ["Mixed / no dominant archetype"]
            reasons = ["no family score crossed the transparent tail threshold"]
        rows.append(
            {
                "subfield_id": row["subfield_id"],
                "subfield_label_unique": row["subfield_label_unique"],
                "subfield_label_short": row["subfield_label_short"],
                "domain_display_name": row["domain_display_name"],
                "field_display_name": row["field_display_name"],
                "archetype_tags": "; ".join(tags),
                "primary_archetype": tags[0],
                "archetype_reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def plot_archetype_counts(
    archetypes: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    counts: defaultdict[str, int] = defaultdict(int)
    for tags in archetypes["archetype_tags"].fillna(""):
        for tag in [tag.strip() for tag in str(tags).split(";") if tag.strip()]:
            counts[tag] += 1
    count_frame = pd.DataFrame(
        sorted(counts.items(), key=lambda item: (-item[1], item[0])),
        columns=["archetype", "n_subfields"],
    )
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.barh(count_frame["archetype"], count_frame["n_subfields"], color="#59a14f")
    ax.invert_yaxis()
    ax.set_xlabel("Subfields with tag")
    ax.set_title("Rule-Based Morphology Archetype Counts")
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=CASE_FIGURES,
        title="Archetype Counts",
        description="Counts of transparent rule-based morphology archetype tags.",
        question="Which descriptive morphology archetypes are common?",
        related_tables=related_tables,
    )


def plot_archetype_domain_distribution(
    archetypes: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    figure_rows: list[dict[str, str]],
    output_dir: Path,
    related_tables: list[Path],
) -> None:
    crosstab = pd.crosstab(
        archetypes["domain_display_name"],
        archetypes["primary_archetype"],
    )
    crosstab = crosstab.loc[crosstab.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    bottom = np.zeros(len(crosstab))
    cmap = plt.get_cmap("tab20", max(len(crosstab.columns), 1))
    for idx, column in enumerate(crosstab.columns):
        values = crosstab[column].to_numpy()
        ax.bar(crosstab.index, values, bottom=bottom, label=column, color=cmap(idx))
        bottom += values
    ax.set_ylabel("Subfields")
    ax.set_title("Primary Archetype Distribution by Domain")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(
        fig,
        path,
        figure_rows,
        output_dir,
        group=CASE_FIGURES,
        title="Archetype Domain Distribution",
        description="Stacked counts of primary archetype by OpenAlex domain.",
        question="Are descriptive archetypes concentrated in particular domains?",
        related_tables=related_tables,
    )


def write_figure_index(
    path: Path,
    figure_rows: list[dict[str, str]],
    written_tables: list[Path],
) -> pd.DataFrame:
    index = pd.DataFrame(
        figure_rows,
        columns=[
            "figure_path",
            "figure_group",
            "title",
            "description",
            "primary_question_answered",
            "related_tables",
        ],
    )
    write_csv(path, index, written_tables)
    return index


def pca_interpretation_rows(contributors: pd.DataFrame) -> list[str]:
    rows = []
    if contributors.empty:
        return ["PCA was skipped or unavailable for this run."]
    for component in ["PC1", "PC2", "PC3"]:
        subset = contributors[contributors["component"] == component]
        if subset.empty:
            continue
        positive = subset[subset["direction"] == "positive"].head(4)["metric_name"].tolist()
        negative = subset[subset["direction"] == "negative"].head(4)["metric_name"].tolist()
        rows.append(
            f"- {component} appears to contrast positive-loading metrics "
            f"({', '.join(positive)}) against negative-loading metrics "
            f"({', '.join(negative)})."
        )
    return rows


def build_report(
    *,
    output_dir: Path,
    input_path: Path,
    dictionary_path: Path,
    n_input_rows: int,
    metrics: pd.DataFrame,
    stats: pd.DataFrame,
    high_pairs: pd.DataFrame,
    low_variance: pd.DataFrame,
    family_score_stats: pd.DataFrame,
    domain_family_summary: pd.DataFrame,
    pca_explained: pd.DataFrame,
    pca_contributors: pd.DataFrame,
    candidates: pd.DataFrame,
    figure_index: pd.DataFrame,
    paths: dict[str, Path],
    skip_pca: bool,
) -> str:
    top_variable = (
        stats[stats["metric_kind"] == "core"]
        .sort_values(["coefficient_of_variation", "metric_name"], ascending=[False, True])
        .head(5)["metric_name"]
        .tolist()
    )
    top_high_pairs = (
        high_pairs.head(5)[["metric_a", "metric_b", "pearson"]].to_dict("records")
        if not high_pairs.empty
        else []
    )
    domain_patterns = (
        domain_family_summary.assign(abs_mean_z=domain_family_summary["mean_z"].abs())
        .sort_values(["abs_mean_z", "domain_display_name"], ascending=[False, True])
        .head(6)
    )
    candidate_lines = [
        f"- {row.subfield_label_short}: {row.case_study_reason}"
        for row in candidates.head(10).itertuples(index=False)
    ]
    if not candidate_lines:
        candidate_lines = ["- No case-study candidates crossed the current extreme-score rule."]

    high_pair_lines = [
        f"- {row['metric_a']} / {row['metric_b']}: Pearson {row['pearson']:.2f}"
        for row in top_high_pairs
    ]
    if not high_pair_lines:
        high_pair_lines = ["- No pairs exceeded the configured high-correlation threshold."]

    domain_lines = [
        f"- {row.domain_display_name}: {row.metric_name} mean_z={row.mean_z:.2f} (n={int(row.n_subfields)})"
        for row in domain_patterns.itertuples(index=False)
    ]
    if not domain_lines:
        domain_lines = ["- Domain summaries were not available."]

    pca_lines = pca_interpretation_rows(pca_contributors)
    if skip_pca:
        pca_summary = "PCA was skipped with `--skip-pca`."
    elif not pca_explained.empty:
        explained_text = ", ".join(
            f"{row.component}={row.explained_variance_ratio:.2%}"
            for row in pca_explained.head(3).itertuples(index=False)
        )
        pca_summary = f"First components explained: {explained_text}."
    else:
        pca_summary = "PCA outputs were unavailable for this run."

    figure_count = len(figure_index)
    report = f"""# Morphology Analysis Report

## 1. Executive Summary

Stage 12 explored `{display_path(input_path)}` with {len(metrics)} analyzed subfields out of {n_input_rows} input rows. It produced {figure_count} figures plus organized tables under `{output_relative(output_dir / "tables", output_dir)}`.

No growth targets are joined in this stage.  
No prediction model is estimated in this stage.  
These analyses are descriptive and exploratory.

The most variable core metrics by coefficient of variation are: {", ".join(top_variable) if top_variable else "not available"}.

## 2. Run Information

- Input table: `{display_path(input_path)}`
- Metric dictionary: `{display_path(dictionary_path) if dictionary_path.exists() else "not found"}`
- Output directory: `{display_path(output_dir)}`
- Figure index: `{output_relative(paths["figure_index"], output_dir)}`
- Summary JSON: `{output_relative(paths["summary"], output_dir)}`

## 3. Dataset Coverage

See `{output_relative(paths["data_quality_summary"], output_dir)}`, `{output_relative(paths["domain_counts"], output_dir)}`, and `{output_relative(paths["field_counts"], output_dir)}`.

The dashboard at `{output_relative(paths["quality_dashboard"], output_dir)}` summarizes domain counts, point counts, duplicate display names, missingness, and metric-family coverage.

## 4. Core Metrics and Diagnostic Metrics

The analysis treats the 25 `CORE_METRIC_COLUMNS_V2` metrics as the main morphology feature layer. Diagnostic metrics remain available for artifact checks but are not promoted into the core feature set. Descriptive statistics are in `{output_relative(paths["metric_descriptive_stats"], output_dir)}` and quantiles are in `{output_relative(paths["metric_quantiles_long"], output_dir)}`.

## 5. Main Distributional Findings

Open `{output_relative(paths["core_histograms_by_family"], output_dir)}` and `{output_relative(paths["core_boxplots_by_family"], output_dir)}` for metric distributions by conceptual family. Metrics flagged for skewness or outlier-heavy distributions are marked in `{output_relative(paths["metric_descriptive_stats"], output_dir)}`.

Low-variance metrics are listed in `{output_relative(paths["low_variance_metrics"], output_dir)}`; this table has {len(low_variance)} rows in the current run.

## 6. Correlation and Redundancy Findings

Pearson and Spearman matrices are saved in `{output_relative(paths["metric_correlation_matrix"], output_dir)}` and `{output_relative(paths["metric_spearman_correlation_matrix"], output_dir)}`. The high-correlation table has {len(high_pairs)} pairs.

Top high-correlation examples:

{chr(10).join(high_pair_lines)}

Use `{output_relative(paths["redundancy_clusters"], output_dir)}` as a later feature-selection aid. These are interpretation diagnostics, not model decisions.

## 7. Family Score Interpretation

Family scores are robust-z composites over the morphology metrics. `expansion_score` and `diversification_score` are single-metric standardized families based on `radial_expansion_slope` and `density_entropy_slope_by_year`, respectively.

Family-score tables are `{output_relative(paths["family_scores"], output_dir)}` and `{output_relative(paths["family_score_descriptive_stats"], output_dir)}`. Key visual contrasts are in `{output_relative(paths["family_pairplots"], output_dir)}`.

## 8. Domain-Level Patterns

Domain and field summaries are descriptive, not causal. They compare standardized family scores and metrics while reporting group counts.

Largest absolute domain-family means:

{chr(10).join(domain_lines)}

See `{output_relative(paths["domain_family_heatmap"], output_dir)}`, `{output_relative(paths["domain_family_boxplots"], output_dir)}`, and `{output_relative(paths["top_fields_by_family_scores"], output_dir)}`.

## 9. PCA Interpretation

PCA here is allowed because it is applied after the morphology metrics are computed. It is exploratory and is not part of UMAP map construction.

{pca_summary}

{chr(10).join(pca_lines)}

PCA tables and biplots are under `{output_relative(output_dir / "figures" / PCA_FIGURES, output_dir)}`.

## 10. Extreme Subfields and Case-Study Candidates

Case-study candidates are rule-based examples that are extreme on one or more core metrics or family scores. See `{output_relative(paths["case_study_candidates"], output_dir)}` and `{output_relative(paths["case_study_matrix"], output_dir)}`.

Top candidates:

{chr(10).join(candidate_lines)}

The archetype table at `{output_relative(paths["morphology_archetypes"], output_dir)}` assigns transparent multi-tag labels; no subfield is forced into exactly one substantive type.

## 11. Caveats

- The inputs are separately fitted per-subfield UMAP maps, so these are normalized morphology descriptors, not direct physical distances in original embedding space.
- PCA and family scores are explanatory summaries of the metric table; they should not be treated as proof of causal structure.
- Domain and field profiles are descriptive and can be influenced by the number and composition of subfields in each group.
- Later prediction work should re-check redundancy and leakage once growth targets are constructed.

## 12. Recommended Next Step

Use the case-study candidates and archetypes to choose a small set of qualitative visual examples from the Stage 10 maps, then build growth targets separately without changing these morphology metrics based on future outcomes.
"""
    return report


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.label_top_n < 0:
        raise ValueError("--label-top-n must be non-negative")
    if not 0 < args.high_correlation_threshold <= 1:
        raise ValueError("--high-correlation-threshold must be in (0, 1]")

    input_path = resolve_path(args.input_path)
    dictionary_path = resolve_path(args.dictionary_path)
    output_dir = resolve_path(args.output_dir)
    paths = expected_paths(output_dir)
    ensure_output_dirs(paths, args.overwrite)
    cleanup_stale_legacy_outputs(output_dir, args.overwrite)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing morphology metrics table: {display_path(input_path)}. "
            "Run scripts/11_compute_subfield_morphology_metrics.py first."
        )

    metrics = load_parquet(input_path)
    validate_core_columns(metrics)
    selected = select_rows(
        metrics,
        subfield_id=args.subfield_id,
        limit_subfields=args.limit_subfields,
    )

    written_tables: list[Path] = []
    figure_rows: list[dict[str, str]] = []

    stats = metric_descriptive_stats(selected)
    quantiles = metric_quantiles_long(stats)
    missingness = metric_missingness(stats)
    low_variance = low_variance_metrics(stats)
    duplicate_report = duplicate_subfield_names_report(selected)
    domain_count_frame = domain_counts(selected)
    field_count_frame = field_counts(selected)
    quality = data_quality_summary(selected, stats, duplicate_report)

    core_values = numeric_frame(selected, CORE_METRIC_COLUMNS_V2)
    pearson = core_values.corr(method="pearson")
    spearman = core_values.corr(method="spearman")
    high_pairs = high_correlation_pairs(
        pearson,
        spearman,
        threshold=args.high_correlation_threshold,
    )
    clusters = redundancy_clusters(
        high_pairs,
        stats[stats["metric_kind"] == "core"],
    )

    family_scores = compute_family_scores(selected)
    family_score_stats = descriptive_stats_for_columns(
        family_scores,
        FAMILY_SCORE_COLUMNS,
        metric_kind="family_score",
    )
    family_score_corr = family_scores[FAMILY_SCORE_COLUMNS].corr()
    family_rankings = top_bottom_subfields(
        family_scores,
        FAMILY_SCORE_COLUMNS,
        n=args.top_n,
        value_name="family_score_value",
    )
    metric_rankings = top_bottom_subfields(
        selected,
        CORE_METRIC_COLUMNS_V2,
        n=args.top_n,
        value_name="metric_value",
    )
    curated = curated_model_features(selected, family_scores)

    domain_metric_summary = group_summary(
        selected,
        ["domain_display_name"],
        CORE_METRIC_COLUMNS_V2,
        metric_kind="core",
    )
    field_metric_summary = group_summary(
        selected,
        ["domain_display_name", "field_display_name"],
        CORE_METRIC_COLUMNS_V2,
        metric_kind="core",
    )
    domain_family_summary = group_summary(
        family_scores,
        ["domain_display_name"],
        FAMILY_SCORE_COLUMNS,
        metric_kind="family_score",
    )
    field_family_summary = group_summary(
        family_scores,
        ["domain_display_name", "field_display_name"],
        FAMILY_SCORE_COLUMNS,
        metric_kind="family_score",
    )

    candidates = build_case_study_candidates(
        selected,
        family_scores,
        top_n=args.top_n,
    )
    archetypes = build_archetypes(family_scores)

    write_csv(paths["data_quality_summary"], quality, written_tables)
    write_csv(paths["metric_missingness"], missingness, written_tables)
    write_csv(paths["low_variance_metrics"], low_variance, written_tables)
    write_csv(paths["duplicate_subfield_names_report"], duplicate_report, written_tables)
    write_csv(paths["domain_counts"], domain_count_frame, written_tables)
    write_csv(paths["field_counts"], field_count_frame, written_tables)
    write_csv(paths["metric_descriptive_stats"], stats, written_tables)
    write_csv(paths["metric_quantiles_long"], quantiles, written_tables)
    write_matrix_csv(paths["metric_correlation_matrix"], pearson, written_tables)
    write_matrix_csv(paths["metric_spearman_correlation_matrix"], spearman, written_tables)
    write_csv(paths["high_correlation_pairs"], high_pairs, written_tables)
    write_csv(paths["redundancy_clusters"], clusters, written_tables)
    write_csv(paths["family_scores"], family_scores, written_tables)
    write_csv(paths["family_score_descriptive_stats"], family_score_stats, written_tables)
    write_matrix_csv(
        paths["family_score_correlation_matrix"],
        family_score_corr,
        written_tables,
    )
    write_csv(paths["top_bottom_subfields_by_family_score"], family_rankings, written_tables)
    write_csv(paths["domain_metric_summary"], domain_metric_summary, written_tables)
    write_csv(paths["field_metric_summary"], field_metric_summary, written_tables)
    write_csv(paths["domain_family_score_summary"], domain_family_summary, written_tables)
    write_csv(paths["field_family_score_summary"], field_family_summary, written_tables)
    write_csv(paths["top_bottom_subfields_by_metric"], metric_rankings, written_tables)
    write_csv(paths["case_study_candidates"], candidates, written_tables)
    write_csv(paths["morphology_archetypes"], archetypes, written_tables)
    write_csv(paths["curated_model_features_csv"], curated, written_tables)
    save_parquet(curated, paths["curated_model_features_parquet"])
    written_tables.append(paths["curated_model_features_parquet"])

    pca_scores = pd.DataFrame()
    pca_loadings = pd.DataFrame()
    pca_explained = pd.DataFrame()
    pca_contributors = pd.DataFrame()
    if not args.skip_pca:
        pca_scores, pca_loadings, pca_explained, pca_contributors = run_pca(selected)
        write_csv(paths["metric_pca_scores"], pca_scores, written_tables)
        write_csv(paths["metric_pca_loadings"], pca_loadings, written_tables)
        write_csv(paths["metric_pca_explained_variance"], pca_explained, written_tables)
        write_csv(paths["top_pca_contributors"], pca_contributors, written_tables)

    plot_data_quality_dashboard(
        selected,
        stats,
        duplicate_report,
        paths["quality_dashboard"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[
            paths["data_quality_summary"],
            paths["metric_missingness"],
            paths["domain_counts"],
            paths["field_counts"],
        ],
    )
    plot_core_histograms_by_family(
        selected,
        stats,
        paths["core_histograms_by_family"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["metric_descriptive_stats"], paths["metric_quantiles_long"]],
    )
    plot_core_boxplots_by_family(
        selected,
        paths["core_boxplots_by_family"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["metric_descriptive_stats"]],
    )
    plot_diagnostic_histograms(
        selected,
        paths["diagnostic_histograms"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["metric_descriptive_stats"]],
    )
    plot_correlation_heatmap(
        pearson,
        paths["pearson_heatmap"],
        title="Clustered Pearson Correlation of Core Metrics",
        description="Hierarchically ordered Pearson correlation matrix for core morphology metrics.",
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["metric_correlation_matrix"], paths["high_correlation_pairs"]],
    )
    plot_correlation_heatmap(
        spearman,
        paths["spearman_heatmap"],
        title="Clustered Spearman Correlation of Core Metrics",
        description="Hierarchically ordered Spearman rank-correlation matrix for core morphology metrics.",
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[
            paths["metric_spearman_correlation_matrix"],
            paths["high_correlation_pairs"],
        ],
    )
    plot_high_correlation_network(
        high_pairs,
        paths["high_correlation_network"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["high_correlation_pairs"]],
    )
    plot_redundancy_cluster_summary(
        clusters,
        paths["redundancy_cluster_summary"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["redundancy_clusters"]],
    )
    plot_family_score_distributions(
        family_scores,
        paths["family_distributions"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["family_scores"], paths["family_score_descriptive_stats"]],
    )
    plot_family_correlation_heatmap(
        family_scores,
        paths["family_correlation_heatmap"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["family_score_correlation_matrix"]],
    )

    if not args.skip_heavy_plots:
        for x_col, y_col in KEY_FAMILY_SCATTERS:
            plot_family_scatter(
                family_scores,
                x_col,
                y_col,
                output_dir
                / "figures"
                / FAMILY_FIGURES
                / f"{safe_filename(x_col)}_vs_{safe_filename(y_col)}.png",
                dpi=args.dpi,
                label_top_n=args.label_top_n,
                figure_rows=figure_rows,
                output_dir=output_dir,
                related_tables=[paths["family_scores"]],
            )
    plot_family_pair_panel(
        family_scores,
        paths["family_pairplots"],
        dpi=args.dpi,
        label_top_n=args.label_top_n,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["family_scores"]],
    )
    plot_top_bottom_bars(
        family_rankings,
        paths["top_bottom_family_scores"],
        metric_columns=FAMILY_SCORE_COLUMNS,
        value_name="family_score_value",
        title="Top and Bottom Subfields by Family Score",
        group=FAMILY_FIGURES,
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["top_bottom_subfields_by_family_score"]],
    )
    heatmap_from_summary(
        domain_family_summary,
        "domain_display_name",
        FAMILY_SCORE_COLUMNS,
        paths["domain_family_heatmap"],
        title="Domain Mean Family-Score Profile",
        group=PROFILE_FIGURES,
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["domain_family_score_summary"]],
    )
    plot_domain_family_boxplots(
        family_scores,
        paths["domain_family_boxplots"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["family_scores"], paths["domain_family_score_summary"]],
    )
    heatmap_from_summary(
        domain_metric_summary,
        "domain_display_name",
        CORE_METRIC_COLUMNS_V2,
        paths["domain_metric_profile_heatmap"],
        title="Domain Mean Core-Metric Profile",
        group=PROFILE_FIGURES,
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["domain_metric_summary"]],
    )
    plot_top_fields_by_family_scores(
        field_family_summary,
        paths["top_fields_by_family_scores"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["field_family_score_summary"]],
    )
    if not args.skip_pca:
        plot_pca_explained_variance(
            pca_explained,
            paths["pca_explained_variance"],
            dpi=args.dpi,
            figure_rows=figure_rows,
            output_dir=output_dir,
            related_tables=[paths["metric_pca_explained_variance"]],
        )
        plot_pca_scatter(
            pca_scores,
            paths["pca_scatter_by_domain"],
            title="PCA Scatter by Domain",
            label_extremes=False,
            dpi=args.dpi,
            label_top_n=args.label_top_n,
            figure_rows=figure_rows,
            output_dir=output_dir,
            related_tables=[paths["metric_pca_scores"]],
        )
        plot_pca_scatter(
            pca_scores,
            paths["pca_scatter_labeled_extremes"],
            title="PCA Scatter with Labeled Extremes",
            label_extremes=True,
            dpi=args.dpi,
            label_top_n=args.label_top_n,
            figure_rows=figure_rows,
            output_dir=output_dir,
            related_tables=[paths["metric_pca_scores"], paths["top_pca_contributors"]],
        )
        plot_pca_loadings_heatmap(
            pca_loadings,
            paths["pca_loadings_heatmap"],
            dpi=args.dpi,
            figure_rows=figure_rows,
            output_dir=output_dir,
            related_tables=[paths["metric_pca_loadings"]],
        )
        plot_pca_biplot(
            pca_scores,
            pca_loadings,
            "PC1",
            "PC2",
            paths["pca_biplot_pc1_pc2"],
            dpi=args.dpi,
            label_top_n=args.label_top_n,
            figure_rows=figure_rows,
            output_dir=output_dir,
            related_tables=[paths["metric_pca_scores"], paths["metric_pca_loadings"]],
        )
        plot_pca_biplot(
            pca_scores,
            pca_loadings,
            "PC2",
            "PC3",
            paths["pca_biplot_pc2_pc3"],
            dpi=args.dpi,
            label_top_n=args.label_top_n,
            figure_rows=figure_rows,
            output_dir=output_dir,
            related_tables=[paths["metric_pca_scores"], paths["metric_pca_loadings"]],
        )
    plot_top_bottom_bars(
        metric_rankings,
        paths["top_bottom_selected_metrics"],
        metric_columns=[m for m in SELECTED_RANKING_METRICS if m in CORE_METRIC_COLUMNS_V2],
        value_name="metric_value",
        title="Top and Bottom Subfields by Selected Core Metrics",
        group=RANKING_FIGURES,
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["top_bottom_subfields_by_metric"]],
    )
    plot_top_bottom_bars(
        family_rankings,
        paths["ranking_top_bottom_family_scores"],
        metric_columns=FAMILY_SCORE_COLUMNS,
        value_name="family_score_value",
        title="Top and Bottom Subfields by Family Scores",
        group=RANKING_FIGURES,
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["top_bottom_subfields_by_family_score"]],
    )
    plot_case_study_matrix(
        candidates,
        family_scores,
        paths["case_study_matrix"],
        top_n=args.top_n,
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["case_study_candidates"], paths["family_scores"]],
    )
    plot_archetype_counts(
        archetypes,
        paths["archetype_counts"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["morphology_archetypes"]],
    )
    plot_archetype_domain_distribution(
        archetypes,
        paths["archetype_domain_distribution"],
        dpi=args.dpi,
        figure_rows=figure_rows,
        output_dir=output_dir,
        related_tables=[paths["morphology_archetypes"]],
    )

    figure_index = write_figure_index(
        paths["figure_index"],
        figure_rows,
        written_tables,
    )
    report = build_report(
        output_dir=output_dir,
        input_path=input_path,
        dictionary_path=dictionary_path,
        n_input_rows=len(metrics),
        metrics=selected,
        stats=stats,
        high_pairs=high_pairs,
        low_variance=low_variance,
        family_score_stats=family_score_stats,
        domain_family_summary=domain_family_summary,
        pca_explained=pca_explained,
        pca_contributors=pca_contributors,
        candidates=candidates,
        figure_index=figure_index,
        paths=paths,
        skip_pca=args.skip_pca,
    )
    write_markdown(paths["report"], report)
    written_tables.append(paths["report"])

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": display_path(input_path),
        "dictionary_path": display_path(dictionary_path) if dictionary_path.exists() else "",
        "output_dir": display_path(output_dir),
        "n_input_rows": int(len(metrics)),
        "n_rows_analyzed": int(len(selected)),
        "n_core_metrics": int(len(CORE_METRIC_COLUMNS_V2)),
        "n_diagnostic_metrics": int(
            len([column for column in DIAGNOSTIC_METRIC_COLUMNS if column in selected.columns])
        ),
        "n_family_scores": int(len(FAMILY_SCORE_COLUMNS)),
        "n_high_correlation_pairs": int(len(high_pairs)),
        "n_low_variance_metrics": int(len(low_variance)),
        "n_duplicate_display_name_rows": int(len(duplicate_report)),
        "n_duplicated_display_names": int(
            duplicate_report["subfield_display_name"].nunique()
        )
        if len(duplicate_report)
        else 0,
        "pca_explained_variance": (
            {
                row.component: float(row.explained_variance_ratio)
                for row in pca_explained.itertuples(index=False)
            }
            if not pca_explained.empty
            else {}
        ),
        "top_case_study_candidates": candidates.head(10)[
            [
                "subfield_id",
                "subfield_label_short",
                "domain_display_name",
                "case_study_reason",
            ]
        ].to_dict("records"),
        "outputs": {
            "tables": [display_path(path) for path in written_tables if path.exists()],
            "figures": [
                display_path(output_dir / row["figure_path"])
                for row in figure_rows
                if (output_dir / row["figure_path"]).exists()
            ],
            "summary": display_path(paths["summary"]),
            "report": display_path(paths["report"]),
            "figure_index": display_path(paths["figure_index"]),
        },
        "notes": [
            "No growth targets are joined in this stage.",
            "No prediction model is estimated in this stage.",
            "These analyses are descriptive and exploratory.",
            "PCA is applied only after morphology metrics are computed.",
            "Expansion and diversification family scores are single-metric standardized families.",
        ],
    }
    write_json(paths["summary"], summary)

    print(f"Analyzed {len(selected)} subfields")
    print(f"Wrote outputs under {display_path(output_dir)}")
    print(f"Figures: {len(figure_rows)}")
    print(f"Tables/report files: {len([p for p in written_tables if p.exists()])}")
    print(f"High-correlation pairs: {len(high_pairs)}")
    print(f"Case-study candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
