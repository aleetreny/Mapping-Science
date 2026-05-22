from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.temporal_common import (
    REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
    WINDOW_STRUCTURAL_METRICS,
    display_path,
    ensure_outputs_do_not_exist,
    utc_now_iso,
)


VALID_METRIC_STATUSES = {"completed", "completed_with_warnings"}

REDUCED_METRICS: list[str] = list(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)
STATIC_STRUCTURAL_METRICS: list[str] = list(WINDOW_STRUCTURAL_METRICS)
TEMPORAL_SUMMARY_METRICS: list[str] = [
    metric for metric in REDUCED_METRICS if metric not in STATIC_STRUCTURAL_METRICS
]

METRIC_FAMILIES: dict[str, list[str]] = {
    "Dispersion": [
        "embedding_distance_to_centroid_median",
        "embedding_distance_to_centroid_iqr",
        "embedding_distance_to_centroid_p90",
    ],
    "Local density and hubness": [
        "embedding_knn_median_distance",
        "embedding_knn_distance_cv",
        "embedding_knn_indegree_gini",
    ],
    "Spectral structure": [
        "embedding_pca_dim_80",
        "embedding_pca_spectral_entropy",
    ],
    "Temporal movement": [
        "embedding_centroid_drift_early_late",
        "embedding_radial_expansion_slope",
        "embedding_recent_novelty_score",
    ],
}

METRIC_SHORT_LABELS: dict[str, str] = {
    "embedding_distance_to_centroid_median": "Centroid\nmedian",
    "embedding_distance_to_centroid_iqr": "Centroid\nIQR",
    "embedding_distance_to_centroid_p90": "Centroid\nP90",
    "embedding_knn_median_distance": "kNN\nmedian",
    "embedding_knn_distance_cv": "kNN\nCV",
    "embedding_knn_indegree_gini": "Hub\nGini",
    "embedding_pca_dim_80": "PCA\nD80",
    "embedding_pca_spectral_entropy": "PCA\nentropy",
    "embedding_centroid_drift_early_late": "Centroid\ndrift",
    "embedding_radial_expansion_slope": "Radial\nslope",
    "embedding_recent_novelty_score": "Recent\nnovelty",
}

METRIC_TEXT_LABELS: dict[str, str] = {
    "embedding_distance_to_centroid_median": "centroid median",
    "embedding_distance_to_centroid_iqr": "centroid IQR",
    "embedding_distance_to_centroid_p90": "centroid P90",
    "embedding_knn_median_distance": "kNN median",
    "embedding_knn_distance_cv": "kNN CV",
    "embedding_knn_indegree_gini": "hub Gini",
    "embedding_pca_dim_80": "PCA D80",
    "embedding_pca_spectral_entropy": "PCA entropy",
    "embedding_centroid_drift_early_late": "centroid drift",
    "embedding_radial_expansion_slope": "radial slope",
    "embedding_recent_novelty_score": "recent novelty",
}

TYPOLOGY_ORDER: list[tuple[str, str]] = [
    ("T1", "Broad sparse profiles"),
    ("T2", "Compact low-dispersion profiles"),
    ("T3", "Low-dimensional locally uneven profiles"),
    ("T4", "Temporal novelty profiles"),
    ("T5", "Uneven dispersion outliers"),
]

TYPOLOGY_COLORS: dict[str, str] = {
    "T1": "#3776ab",
    "T2": "#5f8f3a",
    "T3": "#8e5ea2",
    "T4": "#c47a2c",
    "T5": "#b5483a",
}

DOMAIN_COLORS: dict[str, str] = {
    "Life Sciences": "#2f7f5f",
    "Social Sciences": "#8b6bb5",
    "Physical Sciences": "#ba6a36",
    "Health Sciences": "#4c78a8",
}

OUTPUT_FILENAMES: dict[str, str] = {
    "model_comparison": "cluster_model_comparison.csv",
    "selected_solution": "cluster_solution_selected.csv",
    "assignments": "subfield_typology_assignments.csv",
    "profile_summary": "typology_profile_summary.csv",
    "domain_composition": "typology_domain_composition.csv",
    "stability_summary": "typology_stability_summary.csv",
    "bootstrap_stability": "typology_bootstrap_stability.csv",
    "pca_scores": "typology_pca_scores.csv",
    "summary_md": "cluster_exploration_summary.md",
    "summary_json": "summary.json",
}

MAIN_FIGURE_STEMS: list[str] = [
    "fig_09_cluster_quality_by_k",
    "fig_09_typology_profile_heatmap",
    "fig_09_typology_pca_map",
    "fig_09_typology_domain_composition",
]

DIAGNOSTIC_FIGURE_STEMS: list[str] = [
    "fig_09_typology_dendrogram",
]

FIGURE_STEMS: list[str] = [*MAIN_FIGURE_STEMS, *DIAGNOSTIC_FIGURE_STEMS]


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}


def figure_paths(output_dir: str | Path, figure_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    paths: dict[str, Path] = {}
    for stem in FIGURE_STEMS:
        paths[f"{stem}_output_png"] = output_dir / f"{stem}.png"
        paths[f"{stem}_output_pdf"] = output_dir / f"{stem}.pdf"
    for stem in MAIN_FIGURE_STEMS:
        paths[f"{stem}_thesis_png"] = figure_dir / f"{stem}.png"
    return paths


def latex_table_paths(table_dir: str | Path) -> dict[str, Path]:
    table_dir = Path(table_dir)
    return {
        "summary": table_dir / "tab_09_typology_summary.tex",
        "examples": table_dir / "tab_09_typology_examples.tex",
    }


def existing_outputs(
    output_dir: str | Path,
    figure_dir: str | Path,
    table_dir: str | Path,
) -> list[Path]:
    paths = list(output_paths(output_dir).values())
    paths.extend(figure_paths(output_dir, figure_dir).values())
    paths.extend(latex_table_paths(table_dir).values())
    return [path for path in paths if path.exists()]


def ensure_morphological_typology_outputs(
    output_dir: str | Path,
    figure_dir: str | Path,
    table_dir: str | Path,
    *,
    overwrite: bool,
    root: Path,
) -> None:
    ensure_outputs_do_not_exist(
        existing_outputs(output_dir, figure_dir, table_dir),
        overwrite=overwrite,
        root=root,
        description="morphological typology outputs",
    )


def read_metric_core(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing reduced metric core: {input_path}")
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)
    raise ValueError("Reduced metric core must be a .csv or .parquet file.")


def load_reduced_metric_core(path: str | Path) -> pd.DataFrame:
    frame = read_metric_core(path)
    required = {
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        *REDUCED_METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    if "metric_status" in frame.columns:
        frame = frame.loc[frame["metric_status"].isin(VALID_METRIC_STATUSES)].copy()

    for metric in REDUCED_METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    missing_counts = frame[REDUCED_METRICS].isna().sum()
    if int(missing_counts.sum()) > 0:
        detail = ", ".join(
            f"{metric}={int(count)}"
            for metric, count in missing_counts.items()
            if int(count) > 0
        )
        raise ValueError("Reduced metric core contains missing metric values: " + detail)

    frame["subfield_id"] = frame["subfield_id"].astype(str)
    frame["field_id"] = frame["field_id"].astype(str)
    frame["domain_id"] = frame["domain_id"].astype(str)
    return frame.sort_values("subfield_id", kind="mergesort").reset_index(drop=True)


def feature_columns_for(feature_set: str) -> list[str]:
    if feature_set == "all11":
        return list(REDUCED_METRICS)
    if feature_set == "static8":
        return list(STATIC_STRUCTURAL_METRICS)
    raise ValueError("feature_set must be all11 or static8")


def _family_for_metric(metric: str) -> str:
    for family, metrics in METRIC_FAMILIES.items():
        if metric in metrics:
            return family
    raise ValueError(f"Metric is not assigned to a family: {metric}")


def scale_feature_matrix(
    frame: pd.DataFrame,
    *,
    metrics: list[str],
    scaling: str,
    family_balanced: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    values = frame[metrics].to_numpy(dtype=float)
    if scaling == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))
    elif scaling == "zscore":
        scaler = StandardScaler()
    else:
        raise ValueError("scaling must be robust or zscore")

    matrix = scaler.fit_transform(values)
    parameters: list[dict[str, Any]] = []
    for idx, metric in enumerate(metrics):
        if scaling == "robust":
            center = float(scaler.center_[idx])
            scale = float(scaler.scale_[idx])
        else:
            center = float(scaler.mean_[idx])
            scale = float(scaler.scale_[idx])
        parameters.append(
            {
                "metric": metric,
                "scaling": scaling,
                "family_balanced": family_balanced,
                "center": center,
                "scale": scale,
                "family": _family_for_metric(metric),
            }
        )

    if family_balanced:
        family_sizes = {
            family: len([metric for metric in metrics if metric in family_metrics])
            for family, family_metrics in METRIC_FAMILIES.items()
        }
        weights = np.array(
            [1.0 / math.sqrt(family_sizes[_family_for_metric(metric)]) for metric in metrics],
            dtype=float,
        )
        matrix = matrix * weights
        for row, weight in zip(parameters, weights):
            row["family_weight"] = float(weight)
    else:
        for row in parameters:
            row["family_weight"] = 1.0

    return np.asarray(matrix, dtype=float), pd.DataFrame(parameters)


def _hierarchical_labels(
    matrix: np.ndarray,
    *,
    k: int,
    method: str,
    distance_metric: str,
) -> np.ndarray:
    if method == "ward":
        clustered = linkage(matrix, method="ward")
    elif distance_metric == "correlation":
        distances = pdist(matrix, metric="correlation")
        distances = np.nan_to_num(distances, nan=0.0, posinf=2.0, neginf=0.0)
        clustered = linkage(distances, method=method)
    else:
        clustered = linkage(matrix, method=method, metric=distance_metric)
    return fcluster(clustered, t=k, criterion="maxclust").astype(int)


def _hierarchical_linkage(
    matrix: np.ndarray,
    *,
    method: str,
    distance_metric: str,
) -> np.ndarray:
    if method == "ward":
        return linkage(matrix, method="ward")
    if distance_metric == "correlation":
        distances = pdist(matrix, metric="correlation")
        distances = np.nan_to_num(distances, nan=0.0, posinf=2.0, neginf=0.0)
        return linkage(distances, method=method)
    return linkage(matrix, method=method, metric=distance_metric)


def _silhouette(matrix: np.ndarray, labels: np.ndarray, *, distance_metric: str) -> float:
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return float("nan")
    try:
        if distance_metric == "correlation":
            distances = squareform(pdist(matrix, metric="correlation"))
            distances = np.nan_to_num(distances, nan=0.0, posinf=2.0, neginf=0.0)
            return float(silhouette_score(distances, labels, metric="precomputed"))
        return float(silhouette_score(matrix, labels, metric=distance_metric))
    except ValueError:
        return float("nan")


def _euclidean_internal_scores(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return float("nan"), float("nan")
    try:
        ch = float(calinski_harabasz_score(matrix, labels))
    except ValueError:
        ch = float("nan")
    try:
        db = float(davies_bouldin_score(matrix, labels))
    except ValueError:
        db = float("nan")
    return ch, db


def _cluster_size_summary(labels: np.ndarray) -> tuple[int, int, str]:
    sizes = pd.Series(labels).value_counts().sort_values().astype(int).tolist()
    return int(min(sizes)), int(max(sizes)), ";".join(str(size) for size in sizes)


def _candidate_row(
    *,
    solution_id: str,
    algorithm: str,
    scaling: str,
    feature_set: str,
    family_balanced: bool,
    k: int,
    labels: np.ndarray,
    matrix: np.ndarray,
    distance_metric: str,
    linkage_method: str | None = None,
    bic: float | None = None,
    aic: float | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    min_size, max_size, size_text = _cluster_size_summary(labels)
    ch, db = _euclidean_internal_scores(matrix, labels)
    return {
        "solution_id": solution_id,
        "algorithm": algorithm,
        "linkage": linkage_method or "",
        "scaling": scaling,
        "feature_set": feature_set,
        "family_balanced": bool(family_balanced),
        "distance_metric": distance_metric,
        "k": int(k),
        "silhouette": _silhouette(matrix, labels, distance_metric=distance_metric),
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "cluster_sizes_sorted": size_text,
        "bic": float(bic) if bic is not None else np.nan,
        "aic": float(aic) if aic is not None else np.nan,
        "status": status,
    }


def explore_candidate_models(
    frame: pd.DataFrame,
    *,
    random_seed: int,
    k_range: range = range(3, 11),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    matrices: dict[tuple[str, str, bool], np.ndarray] = {}

    for feature_set in ["all11", "static8"]:
        metrics = feature_columns_for(feature_set)
        for scaling in ["robust", "zscore"]:
            for family_balanced in ([False, True] if feature_set == "all11" and scaling == "robust" else [False]):
                matrix, _params = scale_feature_matrix(
                    frame,
                    metrics=metrics,
                    scaling=scaling,
                    family_balanced=family_balanced,
                )
                matrices[(feature_set, scaling, family_balanced)] = matrix

    for k in k_range:
        for feature_set in ["all11", "static8"]:
            for scaling in ["robust", "zscore"]:
                matrix = matrices[(feature_set, scaling, False)]
                labels = _hierarchical_labels(matrix, k=k, method="ward", distance_metric="euclidean")
                rows.append(
                    _candidate_row(
                        solution_id=f"ward_{scaling}_{feature_set}_k{k}",
                        algorithm="hierarchical",
                        linkage_method="ward",
                        scaling=scaling,
                        feature_set=feature_set,
                        family_balanced=False,
                        k=k,
                        labels=labels,
                        matrix=matrix,
                        distance_metric="euclidean",
                    )
                )

        matrix = matrices[("all11", "robust", True)]
        labels = _hierarchical_labels(matrix, k=k, method="ward", distance_metric="euclidean")
        rows.append(
            _candidate_row(
                solution_id=f"ward_robust_all11_family_balanced_k{k}",
                algorithm="hierarchical",
                linkage_method="ward",
                scaling="robust",
                feature_set="all11",
                family_balanced=True,
                k=k,
                labels=labels,
                matrix=matrix,
                distance_metric="euclidean",
            )
        )

        robust_all11 = matrices[("all11", "robust", False)]
        for distance_metric in ["euclidean", "correlation"]:
            labels = _hierarchical_labels(
                robust_all11,
                k=k,
                method="average",
                distance_metric=distance_metric,
            )
            rows.append(
                _candidate_row(
                    solution_id=f"average_{distance_metric}_robust_all11_k{k}",
                    algorithm="hierarchical",
                    linkage_method="average",
                    scaling="robust",
                    feature_set="all11",
                    family_balanced=False,
                    k=k,
                    labels=labels,
                    matrix=robust_all11,
                    distance_metric=distance_metric,
                )
            )

        for feature_set in ["all11", "static8"]:
            matrix = matrices[(feature_set, "robust", False)]
            kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=50)
            labels = kmeans.fit_predict(matrix) + 1
            rows.append(
                _candidate_row(
                    solution_id=f"kmeans_robust_{feature_set}_k{k}",
                    algorithm="kmeans",
                    scaling="robust",
                    feature_set=feature_set,
                    family_balanced=False,
                    k=k,
                    labels=labels,
                    matrix=matrix,
                    distance_metric="euclidean",
                )
            )

            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=random_seed,
                    n_init=10,
                    reg_covar=1e-6,
                )
                labels = gmm.fit_predict(matrix) + 1
                status = "ok"
                if pd.Series(labels).value_counts().min() <= 1:
                    status = "fragmented"
                rows.append(
                    _candidate_row(
                        solution_id=f"gmm_robust_{feature_set}_k{k}",
                        algorithm="gmm",
                        scaling="robust",
                        feature_set=feature_set,
                        family_balanced=False,
                        k=k,
                        labels=labels,
                        matrix=matrix,
                        distance_metric="euclidean",
                        bic=gmm.bic(matrix),
                        aic=gmm.aic(matrix),
                        status=status,
                    )
                )
            except Exception as exc:  # pragma: no cover - numerical fallback.
                rows.append(
                    {
                        "solution_id": f"gmm_robust_{feature_set}_k{k}",
                        "algorithm": "gmm",
                        "linkage": "",
                        "scaling": "robust",
                        "feature_set": feature_set,
                        "family_balanced": False,
                        "distance_metric": "euclidean",
                        "k": int(k),
                        "silhouette": np.nan,
                        "calinski_harabasz": np.nan,
                        "davies_bouldin": np.nan,
                        "min_cluster_size": np.nan,
                        "max_cluster_size": np.nan,
                        "cluster_sizes_sorted": "",
                        "bic": np.nan,
                        "aic": np.nan,
                        "status": f"failed: {type(exc).__name__}",
                    }
                )

    rows.append(
        {
            "solution_id": "hdbscan_robust_all11",
            "algorithm": "hdbscan",
            "linkage": "",
            "scaling": "robust",
            "feature_set": "all11",
            "family_balanced": False,
            "distance_metric": "euclidean",
            "k": np.nan,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "min_cluster_size": np.nan,
            "max_cluster_size": np.nan,
            "cluster_sizes_sorted": "",
            "bic": np.nan,
            "aic": np.nan,
            "status": "not run: optional hdbscan package unavailable",
        }
    )
    return pd.DataFrame(rows)


def selected_raw_solution(frame: pd.DataFrame, *, k: int = 5) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    metrics = feature_columns_for("all11")
    matrix, parameters = scale_feature_matrix(
        frame,
        metrics=metrics,
        scaling="robust",
        family_balanced=False,
    )
    labels = _hierarchical_labels(matrix, k=k, method="ward", distance_metric="euclidean")
    return labels, matrix, parameters


def _pick_and_remove(
    profile: pd.DataFrame,
    remaining: set[int],
    metric: str,
    *,
    direction: str,
) -> int:
    candidates = profile.loc[profile["raw_cluster"].isin(remaining)]
    if direction == "max":
        raw_cluster = int(candidates.sort_values(metric, ascending=False).iloc[0]["raw_cluster"])
    elif direction == "min":
        raw_cluster = int(candidates.sort_values(metric, ascending=True).iloc[0]["raw_cluster"])
    else:
        raise ValueError("direction must be max or min")
    remaining.remove(raw_cluster)
    return raw_cluster


def map_typology_labels(raw_labels: np.ndarray, matrix: np.ndarray, metrics: list[str]) -> dict[int, tuple[str, str]]:
    scaled = pd.DataFrame(matrix, columns=metrics)
    scaled["raw_cluster"] = raw_labels
    profile = scaled.groupby("raw_cluster", sort=True)[metrics].mean().reset_index()
    remaining = set(int(value) for value in profile["raw_cluster"])

    temporal = _pick_and_remove(
        profile,
        remaining,
        "embedding_centroid_drift_early_late",
        direction="max",
    )
    outlier = _pick_and_remove(
        profile,
        remaining,
        "embedding_distance_to_centroid_iqr",
        direction="max",
    )
    broad = _pick_and_remove(
        profile,
        remaining,
        "embedding_knn_median_distance",
        direction="max",
    )
    compact = _pick_and_remove(
        profile,
        remaining,
        "embedding_distance_to_centroid_median",
        direction="min",
    )
    low_dimensional = int(next(iter(remaining)))

    return {
        broad: ("T1", "Broad sparse profiles"),
        compact: ("T2", "Compact low-dispersion profiles"),
        low_dimensional: ("T3", "Low-dimensional locally uneven profiles"),
        temporal: ("T4", "Temporal novelty profiles"),
        outlier: ("T5", "Uneven dispersion outliers"),
    }


def selected_assignments(
    frame: pd.DataFrame,
    raw_labels: np.ndarray,
    matrix: np.ndarray,
    *,
    metrics: list[str],
) -> pd.DataFrame:
    label_map = map_typology_labels(raw_labels, matrix, metrics)
    result = frame.copy()
    result["raw_cluster"] = raw_labels
    result["typology_id"] = [label_map[int(label)][0] for label in raw_labels]
    result["typology_label"] = [label_map[int(label)][1] for label in raw_labels]
    result["typology_short_label"] = result["typology_id"] + " " + result["typology_label"]

    scaled = pd.DataFrame(matrix, columns=metrics)
    scaled["typology_id"] = result["typology_id"].to_numpy()
    centroids = scaled.groupby("typology_id", sort=True)[metrics].mean()
    ordered_ids = [typology_id for typology_id, _label in TYPOLOGY_ORDER]
    centroid_matrix = centroids.loc[ordered_ids, metrics].to_numpy(dtype=float)
    distance_matrix = np.linalg.norm(matrix[:, None, :] - centroid_matrix[None, :, :], axis=2)
    own_index = np.array([ordered_ids.index(value) for value in result["typology_id"]])
    result["distance_to_typology_centroid"] = distance_matrix[np.arange(len(result)), own_index]
    other_distances = distance_matrix.copy()
    other_distances[np.arange(len(result)), own_index] = np.inf
    result["nearest_other_typology_distance"] = np.min(other_distances, axis=1)
    result["border_margin"] = (
        result["nearest_other_typology_distance"] - result["distance_to_typology_centroid"]
    )
    result["is_typology_prototype"] = False
    for typology_id in ordered_ids:
        idx = (
            result.loc[result["typology_id"] == typology_id]
            .sort_values("distance_to_typology_centroid", kind="mergesort")
            .head(1)
            .index
        )
        result.loc[idx, "is_typology_prototype"] = True

    for metric in metrics:
        result[f"scaled_{metric}"] = scaled[metric].to_numpy()
    return result


def summarize_typologies(assignments: pd.DataFrame, *, metrics: list[str]) -> pd.DataFrame:
    scaled_metrics = [f"scaled_{metric}" for metric in metrics]
    rows: list[dict[str, Any]] = []
    for typology_id, label in TYPOLOGY_ORDER:
        subset = assignments.loc[assignments["typology_id"] == typology_id].copy()
        profile = subset[scaled_metrics].mean()
        profile.index = metrics
        positive = profile.sort_values(ascending=False).head(3)
        negative = profile.sort_values(ascending=True).head(3)
        prototypes = (
            subset.sort_values("distance_to_typology_centroid", kind="mergesort")
            .head(3)["subfield_display_name"]
            .tolist()
        )
        domain_counts = subset["domain_display_name"].value_counts()
        domain_mix = "; ".join(
            f"{domain} {int(count)}"
            for domain, count in domain_counts.head(3).items()
        )
        row: dict[str, Any] = {
            "typology_id": typology_id,
            "typology_label": label,
            "n_subfields": int(len(subset)),
            "dominant_positive_metrics": "; ".join(
                f"{METRIC_TEXT_LABELS[metric]} {value:+.2f}"
                for metric, value in positive.items()
            ),
            "dominant_negative_metrics": "; ".join(
                f"{METRIC_TEXT_LABELS[metric]} {value:+.2f}"
                for metric, value in negative.items()
            ),
            "representative_subfields": "; ".join(prototypes),
            "domain_mix": domain_mix,
            "mean_distance_to_centroid": float(subset["distance_to_typology_centroid"].mean()),
            "median_border_margin": float(subset["border_margin"].median()),
        }
        for metric in metrics:
            row[metric] = float(profile[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def domain_composition(assignments: pd.DataFrame) -> pd.DataFrame:
    counts = (
        assignments.groupby(
            ["typology_id", "typology_label", "domain_id", "domain_display_name"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_subfields")
    )
    totals = (
        counts.groupby(["typology_id", "typology_label"], dropna=False)["n_subfields"]
        .sum()
        .reset_index(name="typology_total")
    )
    result = counts.merge(totals, on=["typology_id", "typology_label"], how="left")
    result["share"] = result["n_subfields"] / result["typology_total"]
    return result.sort_values(["typology_id", "domain_id"], kind="mergesort").reset_index(drop=True)


def selected_solution_summary(
    labels: np.ndarray,
    matrix: np.ndarray,
    *,
    solution_id: str = "ward_robust_all11_k5_selected",
) -> pd.DataFrame:
    row = _candidate_row(
        solution_id=solution_id,
        algorithm="hierarchical",
        linkage_method="ward",
        scaling="robust",
        feature_set="all11",
        family_balanced=False,
        k=len(np.unique(labels)),
        labels=labels,
        matrix=matrix,
        distance_metric="euclidean",
    )
    return pd.DataFrame([row])


def stability_against_selected(
    frame: pd.DataFrame,
    selected_labels: np.ndarray,
    *,
    selected_k: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = [
        ("z-score Ward, all 11 metrics", "ward", "zscore", "all11", False, "euclidean"),
        ("robust Ward, static 8 metrics", "ward", "robust", "static8", False, "euclidean"),
        ("family-balanced robust Ward", "ward", "robust", "all11", True, "euclidean"),
        ("robust k-means, all 11 metrics", "kmeans", "robust", "all11", False, "euclidean"),
        ("average-link correlation", "average", "robust", "all11", False, "correlation"),
        ("average-link Euclidean", "average", "robust", "all11", False, "euclidean"),
    ]
    for label, algorithm, scaling, feature_set, family_balanced, distance_metric in variants:
        metrics = feature_columns_for(feature_set)
        matrix, _params = scale_feature_matrix(
            frame,
            metrics=metrics,
            scaling=scaling,
            family_balanced=family_balanced,
        )
        if algorithm == "kmeans":
            variant_labels = (
                KMeans(n_clusters=selected_k, random_state=random_seed, n_init=50)
                .fit_predict(matrix)
                + 1
            )
        else:
            variant_labels = _hierarchical_labels(
                matrix,
                k=selected_k,
                method=algorithm,
                distance_metric=distance_metric,
            )
        min_size, max_size, sizes = _cluster_size_summary(variant_labels)
        rows.append(
            {
                "comparison": label,
                "algorithm": algorithm,
                "scaling": scaling,
                "feature_set": feature_set,
                "family_balanced": family_balanced,
                "distance_metric": distance_metric,
                "adjusted_rand_index": adjusted_rand_score(selected_labels, variant_labels),
                "adjusted_mutual_information": adjusted_mutual_info_score(
                    selected_labels,
                    variant_labels,
                ),
                "min_cluster_size": min_size,
                "max_cluster_size": max_size,
                "cluster_sizes_sorted": sizes,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_stability(
    matrix: np.ndarray,
    selected_labels: np.ndarray,
    *,
    k: int,
    random_seed: int,
    n_iterations: int = 100,
    sample_fraction: float = 0.8,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    n_rows = matrix.shape[0]
    n_sample = int(round(n_rows * sample_fraction))
    rows: list[dict[str, Any]] = []
    for iteration in range(n_iterations):
        indices = np.sort(rng.choice(n_rows, size=n_sample, replace=False))
        labels = _hierarchical_labels(
            matrix[indices, :],
            k=k,
            method="ward",
            distance_metric="euclidean",
        )
        rows.append(
            {
                "iteration": iteration + 1,
                "sample_fraction": sample_fraction,
                "n_sample": n_sample,
                "adjusted_rand_index": adjusted_rand_score(selected_labels[indices], labels),
                "adjusted_mutual_information": adjusted_mutual_info_score(
                    selected_labels[indices],
                    labels,
                ),
                "min_cluster_size": int(pd.Series(labels).value_counts().min()),
                "max_cluster_size": int(pd.Series(labels).value_counts().max()),
            }
        )
    return pd.DataFrame(rows)


def pca_scores(assignments: pd.DataFrame, matrix: np.ndarray) -> tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(matrix)
    result = assignments[
        [
            "subfield_id",
            "subfield_display_name",
            "field_display_name",
            "domain_display_name",
            "typology_id",
            "typology_label",
            "distance_to_typology_centroid",
            "border_margin",
        ]
    ].copy()
    result["pca_1"] = scores[:, 0]
    result["pca_2"] = scores[:, 1]
    result["pca_1_explained_variance"] = float(pca.explained_variance_ratio_[0])
    result["pca_2_explained_variance"] = float(pca.explained_variance_ratio_[1])
    return result, pca


def _save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    output_dir: Path,
    figure_dir: Path,
    write_thesis_copy: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    if write_thesis_copy:
        fig.savefig(figure_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_quality_by_k(
    model_comparison: pd.DataFrame,
    stability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    output_dir: Path,
    figure_dir: Path,
    selected_k: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    line_specs = [
        ("ward_robust_all11", "Ward robust, all 11", "#1f77b4", "o"),
        ("ward_zscore_all11", "Ward z-score, all 11", "#7f7f7f", "s"),
        ("ward_robust_static8", "Ward robust, static 8", "#2ca02c", "^"),
        ("kmeans_robust_all11", "k-means robust, all 11", "#ff7f0e", "D"),
        ("average_correlation_robust_all11", "Average correlation", "#9467bd", "v"),
    ]
    for prefix, label, color, marker in line_specs:
        subset = model_comparison.loc[
            model_comparison["solution_id"].str.startswith(prefix + "_k")
        ].sort_values("k")
        if subset.empty:
            continue
        ax.plot(
            subset["k"],
            subset["silhouette"],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4,
        )
    ax.axvline(selected_k, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Number of typologies (k)", fontsize=10)
    ax.set_ylabel("Silhouette score", fontsize=10)
    ax.set_title("A. Cluster separation is modest", fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7.8, frameon=False, loc="best")

    ax = axes[1]
    stability_plot = stability.copy()
    stability_plot["comparison_short"] = stability_plot["comparison"].replace(
        {
            "z-score Ward, all 11 metrics": "z-score Ward",
            "robust Ward, static 8 metrics": "static 8 Ward",
            "family-balanced robust Ward": "family-balanced Ward",
            "robust k-means, all 11 metrics": "k-means",
            "average-link correlation": "avg. correlation",
            "average-link Euclidean": "avg. Euclidean",
        }
    )
    y = np.arange(len(stability_plot))
    ax.barh(
        y - 0.18,
        stability_plot["adjusted_rand_index"],
        height=0.32,
        color="#4c78a8",
        label="ARI",
    )
    ax.barh(
        y + 0.18,
        stability_plot["adjusted_mutual_information"],
        height=0.32,
        color="#f58518",
        label="AMI",
    )
    boot_ari = bootstrap["adjusted_rand_index"].mean()
    ax.axvline(
        boot_ari,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="subsample ARI mean",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(stability_plot["comparison_short"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Agreement with selected solution", fontsize=10)
    ax.set_title("B. Sensitivity checks qualify the partition", fontsize=11, loc="left")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, "fig_09_cluster_quality_by_k", output_dir=output_dir, figure_dir=figure_dir)


def plot_profile_heatmap(
    profile_summary: pd.DataFrame,
    *,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    ordered_ids = [typology_id for typology_id, _label in TYPOLOGY_ORDER]
    matrix = (
        profile_summary.set_index("typology_id")
        .loc[ordered_ids, REDUCED_METRICS]
        .to_numpy(dtype=float)
    )
    ordered_summary = profile_summary.set_index("typology_id").loc[ordered_ids].reset_index()
    y_labels = [
        f"{row.typology_id} ({int(row.n_subfields)})\n{row.typology_label}"
        for row in ordered_summary.itertuples()
    ]
    x_labels = [METRIC_SHORT_LABELS[metric] for metric in REDUCED_METRICS]

    fig, ax = plt.subplots(figsize=(12, 5.4))
    vmax = max(1.8, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8.4)
    ax.set_title("Selected typology mean profiles", fontsize=12)
    for y_idx in range(matrix.shape[0]):
        for x_idx in range(matrix.shape[1]):
            value = matrix[y_idx, x_idx]
            color = "white" if abs(value) > vmax * 0.62 else "#222222"
            ax.text(x_idx, y_idx, f"{value:+.1f}", ha="center", va="center", fontsize=7, color=color)
    separators = [2.5, 5.5, 7.5]
    for separator in separators:
        ax.axvline(separator, color="black", linewidth=0.8, alpha=0.45)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.015)
    cbar.set_label("Mean robust-scaled metric value", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save_figure(
        fig,
        "fig_09_typology_profile_heatmap",
        output_dir=output_dir,
        figure_dir=figure_dir,
    )


def _short_name(value: str, max_chars: int = 28) -> str:
    value = str(value)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "."


def plot_pca_map(
    pca_frame: pd.DataFrame,
    *,
    pca: PCA,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    for typology_id, label in TYPOLOGY_ORDER:
        subset = pca_frame.loc[pca_frame["typology_id"] == typology_id]
        ax.scatter(
            subset["pca_1"],
            subset["pca_2"],
            s=34,
            c=TYPOLOGY_COLORS[typology_id],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
            label=f"{typology_id} {label}",
        )
    prototypes = (
        pca_frame.sort_values("distance_to_typology_centroid", kind="mergesort")
        .groupby("typology_id", sort=True)
        .head(1)
    )
    for row in prototypes.itertuples():
        ax.scatter(
            [row.pca_1],
            [row.pca_2],
            s=92,
            facecolors="none",
            edgecolors="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            _short_name(row.subfield_display_name),
            xy=(row.pca_1, row.pca_2),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
        )
    ax.axhline(0, color="#dddddd", linewidth=0.8)
    ax.axvline(0, color="#dddddd", linewidth=0.8)
    ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)", fontsize=10)
    ax.set_title("Subfields in metric-profile PCA space", fontsize=12)
    ax.grid(True, alpha=0.18)
    ax.legend(fontsize=7.1, frameon=False, loc="best", ncol=1)
    fig.tight_layout()
    _save_figure(fig, "fig_09_typology_pca_map", output_dir=output_dir, figure_dir=figure_dir)


def plot_domain_composition(
    composition: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    domain_order = (
        assignments[["domain_id", "domain_display_name"]]
        .drop_duplicates()
        .sort_values("domain_id")["domain_display_name"]
        .tolist()
    )
    typology_ids = [typology_id for typology_id, _label in TYPOLOGY_ORDER]
    pivot = (
        composition.pivot_table(
            index="typology_id",
            columns="domain_display_name",
            values="share",
            fill_value=0.0,
        )
        .reindex(index=typology_ids, columns=domain_order)
        .fillna(0.0)
    )
    counts = assignments.groupby("typology_id").size().reindex(typology_ids)
    y = np.arange(len(typology_ids))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    left = np.zeros(len(typology_ids))
    for domain in domain_order:
        values = pivot[domain].to_numpy(dtype=float)
        color = DOMAIN_COLORS.get(domain, "#999999")
        bars = ax.barh(y, values, left=left, color=color, edgecolor="white", linewidth=0.6, label=domain)
        for bar, share in zip(bars, values):
            if share >= 0.12:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{share * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                )
        left += values
    y_labels = [
        f"{typology_id} ({int(counts.loc[typology_id])}) {label}"
        for typology_id, label in TYPOLOGY_ORDER
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of subfields", fontsize=10)
    ax.set_title("OpenAlex domain composition by typology", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4)
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    _save_figure(
        fig,
        "fig_09_typology_domain_composition",
        output_dir=output_dir,
        figure_dir=figure_dir,
    )


def plot_dendrogram(
    matrix: np.ndarray,
    assignments: pd.DataFrame,
    *,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    clustered = _hierarchical_linkage(matrix, method="ward", distance_metric="euclidean")
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    dendrogram(
        clustered,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#999999",
        ax=ax,
    )
    ax.set_title("Ward dendrogram of subfield metric profiles", fontsize=12)
    ax.set_xlabel("Subfields (labels suppressed)", fontsize=10)
    ax.set_ylabel("Ward distance", fontsize=10)
    ax.text(
        0.01,
        0.94,
        f"n={len(assignments)} subfields; selected cut: k=5",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    fig.tight_layout()
    _save_figure(
        fig,
        "fig_09_typology_dendrogram",
        output_dir=output_dir,
        figure_dir=figure_dir,
        write_thesis_copy=False,
    )


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for original, escaped in replacements.items():
        text = text.replace(original, escaped)
    return text


def _compact_list(values: list[str], *, max_items: int = 3, max_chars: int = 30) -> str:
    return "; ".join(_short_name(value, max_chars=max_chars) for value in values[:max_items])


def _dominant_traits(profile_row: pd.Series) -> str:
    values = profile_row[REDUCED_METRICS].astype(float)
    positive = values.sort_values(ascending=False).head(2)
    negative = values.sort_values(ascending=True).head(2)
    parts = [
        "high " + ", ".join(METRIC_TEXT_LABELS[metric] for metric in positive.index),
        "low " + ", ".join(METRIC_TEXT_LABELS[metric] for metric in negative.index),
    ]
    return "; ".join(parts)


def write_typology_summary_table(
    path: str | Path,
    profile_summary: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for row in profile_summary.itertuples():
        subset = assignments.loc[assignments["typology_id"] == row.typology_id]
        representatives = _compact_list(
            subset.sort_values("distance_to_typology_centroid", kind="mergesort")[
                "subfield_display_name"
            ].tolist(),
            max_items=3,
            max_chars=31,
        )
        top_domains = (
            subset["domain_display_name"].value_counts().head(2).items()
        )
        domain_text = "; ".join(f"{domain} {count}" for domain, count in top_domains)
        traits = _dominant_traits(pd.Series(row._asdict()))
        rows.append(
            " & ".join(
                [
                    _latex_escape(f"{row.typology_id} {row.typology_label}"),
                    str(int(row.n_subfields)),
                    _latex_escape(traits),
                    _latex_escape(representatives),
                    _latex_escape(domain_text),
                ]
            )
            + r" \\"
        )
    table = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption[Morphological typology summary]{Selected exploratory morphological typology}",
            r"\label{tab:typology_summary}",
            r"\scriptsize",
            r"\begingroup",
            r"\setlength{\tabcolsep}{2.4pt}",
            r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.20\textwidth} >{\raggedleft\arraybackslash}p{0.06\textwidth} >{\raggedright\arraybackslash}p{0.29\textwidth} >{\raggedright\arraybackslash}p{0.26\textwidth} >{\raggedright\arraybackslash}p{0.14\textwidth}@{}}",
            r"\hline",
            r"\textbf{Typology} & \textbf{$n$} & \textbf{Dominant profile traits} & \textbf{Representative subfields} & \textbf{Largest domains} \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            r"\endgroup",
            r"\par\vspace{0.15cm}",
            r"\footnotesize\textit{Note.} Typologies are obtained from Ward clustering of robust-scaled eleven-metric subfield profiles. Representative subfields are closest to each typology centroid in the standardized metric space.",
            r"\end{table}",
        ]
    )
    path.write_text(table + "\n", encoding="utf-8")


def write_typology_examples_table(path: str | Path, assignments: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for typology_id, label in TYPOLOGY_ORDER:
        subset = assignments.loc[assignments["typology_id"] == typology_id]
        prototypes = _compact_list(
            subset.sort_values("distance_to_typology_centroid", kind="mergesort")[
                "subfield_display_name"
            ].tolist(),
            max_items=3,
            max_chars=34,
        )
        borderline = _compact_list(
            subset.sort_values("border_margin", kind="mergesort")[
                "subfield_display_name"
            ].tolist(),
            max_items=3,
            max_chars=34,
        )
        distant = _compact_list(
            subset.sort_values("distance_to_typology_centroid", ascending=False, kind="mergesort")[
                "subfield_display_name"
            ].tolist(),
            max_items=2,
            max_chars=34,
        )
        rows.append(
            " & ".join(
                [
                    _latex_escape(f"{typology_id} {label}"),
                    _latex_escape(prototypes),
                    _latex_escape(borderline),
                    _latex_escape(distant),
                ]
            )
            + r" \\"
        )
    table = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption[Prototypical and borderline typology cases]{Prototypical, borderline, and internally distant subfields}",
            r"\label{tab:typology_examples}",
            r"\scriptsize",
            r"\begingroup",
            r"\setlength{\tabcolsep}{2.5pt}",
            r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.22\textwidth} >{\raggedright\arraybackslash}p{0.25\textwidth} >{\raggedright\arraybackslash}p{0.25\textwidth} >{\raggedright\arraybackslash}p{0.20\textwidth}@{}}",
            r"\hline",
            r"\textbf{Typology} & \textbf{Prototypical cases} & \textbf{Border cases} & \textbf{Distant members} \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            r"\endgroup",
            r"\par\vspace{0.15cm}",
            r"\footnotesize\textit{Note.} Border cases have the smallest margin between their assigned typology centroid and the nearest alternative centroid. Distant members are far from their own typology centroid and should be read as internal variation rather than errors.",
            r"\end{table}",
        ]
    )
    path.write_text(table + "\n", encoding="utf-8")


def write_summary_report(
    path: str | Path,
    *,
    input_path: str,
    frame: pd.DataFrame,
    selected_solution: pd.DataFrame,
    profile_summary: pd.DataFrame,
    stability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pca: PCA,
) -> None:
    selected = selected_solution.iloc[0]
    bootstrap_ari = bootstrap["adjusted_rand_index"].mean()
    bootstrap_ami = bootstrap["adjusted_mutual_information"].mean()
    lines = [
        "# Exploratory Morphological Typologies",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "## Input",
        "",
        f"- Input table: `{input_path}`",
        f"- Unit clustered: {len(frame)} subfields.",
        f"- Feature set: {len(REDUCED_METRICS)} reduced embedding-space morphology metrics.",
        "- Missing values in selected metrics: 0.",
        "- Scaling used for the selected solution: robust median/IQR scaling across subfields.",
        "- Projection coordinates are not used for clustering.",
        "",
        "## Selected solution",
        "",
        "- Algorithm: Ward hierarchical clustering.",
        "- Distance geometry: Euclidean distance in robust-scaled eleven-metric profile space.",
        f"- Selected k: {int(selected['k'])}.",
        f"- Silhouette: {selected['silhouette']:.3f}.",
        f"- Cluster sizes: {selected['cluster_sizes_sorted']}.",
        f"- PCA visualization variance: PC1 {pca.explained_variance_ratio_[0] * 100:.1f}%, "
        f"PC2 {pca.explained_variance_ratio_[1] * 100:.1f}%.",
        "",
        "## Typologies",
        "",
    ]
    for row in profile_summary.itertuples():
        lines.append(
            f"- {row.typology_id} {row.typology_label}: {int(row.n_subfields)} subfields; "
            f"{_dominant_traits(pd.Series(row._asdict()))}."
        )
    lines.extend(
        [
            "",
            "## Robustness",
            "",
            f"- Mean subsample ARI over {len(bootstrap)} 80% subsamples: {bootstrap_ari:.3f}.",
            f"- Mean subsample AMI over {len(bootstrap)} 80% subsamples: {bootstrap_ami:.3f}.",
            "- Agreement with plausible alternatives:",
            "",
        ]
    )
    for row in stability.itertuples():
        lines.append(
            f"  - {row.comparison}: ARI {row.adjusted_rand_index:.3f}, "
            f"AMI {row.adjusted_mutual_information:.3f}; sizes {row.cluster_sizes_sorted}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The evidence supports a weak, descriptive typology rather than a strong natural partition. "
            "The selected solution avoids singleton clusters and gives interpretable profiles, but "
            "silhouette and sensitivity diagnostics show that boundaries are fuzzy.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_docs_report(path: str | Path, summary_report: str | Path) -> None:
    text = Path(summary_report).read_text(encoding="utf-8")
    extra = [
        "",
        "## Reproducibility note",
        "",
        "Run `python scripts/18_explore_morphological_typologies.py --overwrite` "
        "from the repository root to regenerate the typology outputs, figures, and thesis tables.",
        "",
        "The active thesis evidence is the reduced eleven-metric core. Domain and field labels "
        "are metadata for interpretation only; they are not used as clustering targets.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text + "\n".join(extra) + "\n", encoding="utf-8")


def build_morphological_typology_outputs(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    table_dir: str | Path,
    docs_path: str | Path,
    root: Path,
    selected_k: int = 5,
    random_seed: int = 42,
    n_bootstrap: int = 100,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    table_dir = Path(table_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    frame = load_reduced_metric_core(input_path)
    model_comparison = explore_candidate_models(frame, random_seed=random_seed)
    raw_labels, matrix, scaling_parameters = selected_raw_solution(frame, k=selected_k)
    assignments = selected_assignments(
        frame,
        raw_labels,
        matrix,
        metrics=REDUCED_METRICS,
    )
    selected = selected_solution_summary(raw_labels, matrix)
    profiles = summarize_typologies(assignments, metrics=REDUCED_METRICS)
    composition = domain_composition(assignments)
    stability = stability_against_selected(
        frame,
        assignments["typology_id"].to_numpy(),
        selected_k=selected_k,
        random_seed=random_seed,
    )
    bootstrap = bootstrap_stability(
        matrix,
        assignments["typology_id"].to_numpy(),
        k=selected_k,
        random_seed=random_seed,
        n_iterations=n_bootstrap,
    )
    pca_frame, pca = pca_scores(assignments, matrix)

    paths = output_paths(output_dir)
    model_comparison.to_csv(paths["model_comparison"], index=False)
    selected.to_csv(paths["selected_solution"], index=False)
    assignments.to_csv(paths["assignments"], index=False)
    profiles.to_csv(paths["profile_summary"], index=False)
    composition.to_csv(paths["domain_composition"], index=False)
    stability.to_csv(paths["stability_summary"], index=False)
    bootstrap.to_csv(paths["bootstrap_stability"], index=False)
    pca_frame.to_csv(paths["pca_scores"], index=False)
    scaling_parameters.to_csv(output_dir / "typology_scaling_parameters.csv", index=False)

    plot_quality_by_k(
        model_comparison,
        stability,
        bootstrap,
        output_dir=output_dir,
        figure_dir=figure_dir,
        selected_k=selected_k,
    )
    plot_profile_heatmap(profiles, output_dir=output_dir, figure_dir=figure_dir)
    plot_pca_map(pca_frame, pca=pca, output_dir=output_dir, figure_dir=figure_dir)
    plot_domain_composition(composition, assignments, output_dir=output_dir, figure_dir=figure_dir)
    plot_dendrogram(matrix, assignments, output_dir=output_dir, figure_dir=figure_dir)

    table_paths = latex_table_paths(table_dir)
    write_typology_summary_table(table_paths["summary"], profiles, assignments)
    write_typology_examples_table(table_paths["examples"], assignments)

    input_display = display_path(input_path, root=root)
    write_summary_report(
        paths["summary_md"],
        input_path=input_display,
        frame=frame,
        selected_solution=selected,
        profile_summary=profiles,
        stability=stability,
        bootstrap=bootstrap,
        pca=pca,
    )
    write_docs_report(docs_path, paths["summary_md"])

    summary_payload = {
        "generated_at": utc_now_iso(),
        "input_path": input_display,
        "n_subfields": int(len(frame)),
        "n_metrics": len(REDUCED_METRICS),
        "selected_solution": selected.iloc[0].to_dict(),
        "typology_counts": {
            row.typology_id: int(row.n_subfields) for row in profiles.itertuples()
        },
        "stability_mean_subsample_ari": float(bootstrap["adjusted_rand_index"].mean()),
        "stability_mean_subsample_ami": float(bootstrap["adjusted_mutual_information"].mean()),
        "pca_explained_variance": [float(value) for value in pca.explained_variance_ratio_],
        "output_paths": {
            key: display_path(path, root=root) for key, path in paths.items()
        },
        "figure_paths": {
            key: display_path(path, root=root)
            for key, path in figure_paths(output_dir, figure_dir).items()
        },
        "table_paths": {
            key: display_path(path, root=root) for key, path in table_paths.items()
        },
    }
    paths["summary_json"].write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload
