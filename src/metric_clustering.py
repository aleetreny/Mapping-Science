from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.embedding_space_metrics import CORE_EMBEDDING_METRIC_COLUMNS
from src.morphology_metrics import CORE_METRIC_COLUMNS_V2

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


VALID_STATUSES = {"completed", "completed_with_warnings"}
METADATA_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "subfield_label_unique",
    "subfield_label_short",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
]
SUPPORTED_SPACES = {"embedding_only", "umap_only", "combined"}


@dataclass(frozen=True)
class ClusteringConfig:
    max_missing_share: float = 0.30
    min_unique_values: int = 4
    pca_variance_threshold: float = 0.90
    k_min: int = 3
    k_max: int = 8
    default_k: int = 5
    scaler: str = "standard"
    random_state: int = 42
    combined_strategy: str = "block_pca"


@dataclass
class MetricSpaceData:
    name: str
    metadata: pd.DataFrame
    raw_metrics: pd.DataFrame
    feature_matrix: pd.DataFrame
    preprocessing_report: pd.DataFrame
    pca_scores: pd.DataFrame
    pca_loadings: pd.DataFrame
    pca_explained_variance: pd.DataFrame
    warnings: list[str]
    component_blocks: list[str]


@dataclass
class ClusteringResult:
    assignments: pd.DataFrame
    quality_by_k: pd.DataFrame
    cluster_profiles: pd.DataFrame
    representatives: pd.DataFrame
    main_cluster_column: str
    effective_default_k: int


def validate_config(config: ClusteringConfig) -> None:
    if not 0 <= config.max_missing_share <= 1:
        raise ValueError("max_missing_share must be between 0 and 1")
    if config.min_unique_values < 1:
        raise ValueError("min_unique_values must be positive")
    if not 0 < config.pca_variance_threshold <= 1:
        raise ValueError("pca_variance_threshold must be in (0, 1]")
    if config.k_min < 2 or config.k_max < config.k_min:
        raise ValueError("k range must satisfy 2 <= k_min <= k_max")
    if config.default_k < 2:
        raise ValueError("default_k must be at least 2")
    if config.scaler not in {"standard", "robust"}:
        raise ValueError("scaler must be standard or robust")
    if config.combined_strategy not in {"block_pca", "raw"}:
        raise ValueError("combined_strategy must be block_pca or raw")


def filter_valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "metric_status" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["metric_status"].isin(VALID_STATUSES)].copy()


def available_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "subfield_id" not in frame.columns:
        raise ValueError("metric table must contain subfield_id")
    metadata = frame.copy()
    for column in METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = ""
    return metadata[METADATA_COLUMNS].drop_duplicates("subfield_id").reset_index(drop=True)


def metric_block(frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    columns = available_columns(frame, metric_columns)
    if not columns:
        raise ValueError("none of the requested metric columns are available")
    return frame[columns].apply(pd.to_numeric, errors="coerce")


def scaler_for_name(name: str) -> StandardScaler | RobustScaler:
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError("scaler must be standard or robust")


def preprocess_metric_block(
    block: pd.DataFrame,
    *,
    metric_family: str,
    config: ClusteringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_config(config)
    reports: list[dict[str, Any]] = []
    retained_columns: list[str] = []
    for column in block.columns:
        values = pd.to_numeric(block[column], errors="coerce")
        n_rows = int(len(values))
        n_non_missing = int(values.notna().sum())
        missing_share = float(1.0 - (n_non_missing / n_rows)) if n_rows else 1.0
        n_unique = int(values.dropna().nunique())
        reasons: list[str] = []
        if missing_share > config.max_missing_share:
            reasons.append("high_missing")
        if n_unique <= 1:
            reasons.append("constant")
        elif n_unique < config.min_unique_values:
            reasons.append("low_unique")
        retained = not reasons
        if retained:
            retained_columns.append(column)
        reports.append(
            {
                "metric_name": column,
                "metric_family": metric_family,
                "n_rows": n_rows,
                "n_non_missing": n_non_missing,
                "missing_share": missing_share,
                "n_unique": n_unique,
                "retained": bool(retained),
                "drop_reason": ";".join(reasons),
                "imputation_value": float(values.median()) if n_non_missing else np.nan,
            }
        )

    report = pd.DataFrame(reports)
    if not retained_columns:
        raise ValueError(f"no metrics retained for {metric_family}")
    retained = block[retained_columns].copy()
    for column in retained.columns:
        median = retained[column].median()
        retained[column] = retained[column].fillna(median)
    scaler = scaler_for_name(config.scaler)
    scaled = pd.DataFrame(
        scaler.fit_transform(retained),
        columns=retained.columns,
        index=retained.index,
    )
    return scaled, report


def fit_pca_scores(
    features: pd.DataFrame,
    *,
    variance_threshold: float,
    block_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(features) < 2:
        raise ValueError("PCA requires at least two rows")
    if features.shape[1] < 1:
        raise ValueError("PCA requires at least one feature")
    n_components = min(features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full")
    all_scores = pca.fit_transform(features)
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cumulative = np.cumsum(ratios)
    retained_count = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    retained_count = max(1, min(retained_count, n_components))
    component_names = [f"{block_name}_PC{idx + 1}" for idx in range(retained_count)]
    scores = pd.DataFrame(
        all_scores[:, :retained_count],
        columns=component_names,
        index=features.index,
    )

    loadings_rows: list[dict[str, Any]] = []
    for component_idx, component_name in enumerate(component_names):
        for metric_name, loading in zip(features.columns, pca.components_[component_idx]):
            loadings_rows.append(
                {
                    "block": block_name,
                    "component": component_name,
                    "metric_name": metric_name,
                    "loading": float(loading),
                }
            )
    loadings = pd.DataFrame(loadings_rows)
    explained = pd.DataFrame(
        {
            "block": block_name,
            "component": [f"{block_name}_PC{idx + 1}" for idx in range(n_components)],
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": cumulative,
            "retained": [idx < retained_count for idx in range(n_components)],
        }
    )
    return scores, loadings, explained


def valid_k_values(n_rows: int, config: ClusteringConfig) -> list[int]:
    upper = min(config.k_max, n_rows)
    return [k for k in range(config.k_min, upper + 1)]


def effective_default_k(n_rows: int, config: ClusteringConfig) -> int:
    return min(config.default_k, n_rows)


def ward_labels(features: pd.DataFrame, k: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(features.to_numpy()) + 1


def kmeans_labels(features: pd.DataFrame, k: int, *, random_state: int) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    return model.fit_predict(features.to_numpy()) + 1


def gmm_labels_and_probability(
    features: pd.DataFrame,
    k: int,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, GaussianMixture]:
    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=random_state,
    )
    probabilities = model.fit_predict(features.to_numpy())
    membership = model.predict_proba(features.to_numpy())
    return probabilities + 1, membership.max(axis=1), model


def cluster_size_summary(labels: np.ndarray) -> tuple[int, int]:
    _, counts = np.unique(labels, return_counts=True)
    return int(counts.min()), int(counts.max())


def clustering_quality(
    features: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, float]:
    n_rows = len(features)
    n_clusters = int(len(np.unique(labels)))
    min_size, max_size = cluster_size_summary(labels)
    valid_for_scores = 2 <= n_clusters < n_rows
    return {
        "silhouette_score": float(silhouette_score(features, labels))
        if valid_for_scores
        else np.nan,
        "calinski_harabasz_score": float(calinski_harabasz_score(features, labels))
        if valid_for_scores
        else np.nan,
        "davies_bouldin_score": float(davies_bouldin_score(features, labels))
        if valid_for_scores
        else np.nan,
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "n_clusters": n_clusters,
    }


def run_cluster_assignments(
    space_data: MetricSpaceData,
    *,
    config: ClusteringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, str, int]:
    features = space_data.feature_matrix
    if len(features) < 2:
        raise ValueError("clustering requires at least two rows")
    k_values = valid_k_values(len(features), config)
    if not k_values:
        raise ValueError("no valid k values are available for clustering")
    main_k = effective_default_k(len(features), config)
    if main_k < config.k_min:
        main_k = k_values[0]

    assignments = space_data.metadata.reset_index(drop=True).copy()
    quality_rows: list[dict[str, Any]] = []
    for k in k_values:
        labels = ward_labels(features, k)
        column = f"cluster_ward_k{k}"
        assignments[column] = labels
        quality_rows.append(
            {
                "method": "ward",
                "k": k,
                **clustering_quality(features, labels),
                "bic": np.nan,
                "aic": np.nan,
            }
        )

        kmeans = kmeans_labels(features, k, random_state=config.random_state)
        quality_rows.append(
            {
                "method": "kmeans",
                "k": k,
                **clustering_quality(features, kmeans),
                "bic": np.nan,
                "aic": np.nan,
            }
        )

        try:
            gmm_labels, _, gmm_model = gmm_labels_and_probability(
                features,
                k,
                random_state=config.random_state,
            )
            gmm_quality = clustering_quality(features, gmm_labels)
            quality_rows.append(
                {
                    "method": "gmm",
                    "k": k,
                    **gmm_quality,
                    "bic": float(gmm_model.bic(features.to_numpy())),
                    "aic": float(gmm_model.aic(features.to_numpy())),
                }
            )
        except Exception:
            quality_rows.append(
                {
                    "method": "gmm",
                    "k": k,
                    "silhouette_score": np.nan,
                    "calinski_harabasz_score": np.nan,
                    "davies_bouldin_score": np.nan,
                    "min_cluster_size": np.nan,
                    "max_cluster_size": np.nan,
                    "n_clusters": np.nan,
                    "bic": np.nan,
                    "aic": np.nan,
                }
            )

    if main_k not in k_values:
        main_k = k_values[-1]
    kmeans_main = kmeans_labels(features, main_k, random_state=config.random_state)
    assignments[f"cluster_kmeans_k{main_k}"] = kmeans_main
    try:
        gmm_main, gmm_probability, _ = gmm_labels_and_probability(
            features,
            main_k,
            random_state=config.random_state,
        )
        assignments[f"cluster_gmm_k{main_k}"] = gmm_main
        assignments[f"gmm_max_probability_k{main_k}"] = gmm_probability
    except Exception:
        assignments[f"cluster_gmm_k{main_k}"] = np.nan
        assignments[f"gmm_max_probability_k{main_k}"] = np.nan

    main_column = f"cluster_ward_k{main_k}"
    return assignments, pd.DataFrame(quality_rows), main_column, main_k


def cluster_profiles(
    assignments: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    *,
    cluster_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = assignments[cluster_column].to_numpy()
    for cluster_id in sorted(pd.Series(labels).dropna().unique()):
        mask = labels == cluster_id
        n_subfields = int(mask.sum())
        for metric_name in raw_metrics.columns:
            values = pd.to_numeric(raw_metrics.loc[mask, metric_name], errors="coerce")
            global_values = pd.to_numeric(raw_metrics[metric_name], errors="coerce")
            median = values.median()
            mean = values.mean()
            global_mean = global_values.mean()
            global_std = global_values.std(ddof=0)
            z_median = (
                float((median - global_mean) / global_std)
                if pd.notna(median) and pd.notna(global_std) and global_std > 0
                else np.nan
            )
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "n_subfields": n_subfields,
                    "metric_name": metric_name,
                    "median": float(median) if pd.notna(median) else np.nan,
                    "mean": float(mean) if pd.notna(mean) else np.nan,
                    "standardized_median": z_median,
                }
            )
    return pd.DataFrame(rows)


def cluster_representatives(
    assignments: pd.DataFrame,
    features: pd.DataFrame,
    *,
    cluster_column: str,
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    values = features.to_numpy(dtype=float)
    labels = assignments[cluster_column].to_numpy()
    for cluster_id in sorted(pd.Series(labels).dropna().unique()):
        positions = np.flatnonzero(labels == cluster_id)
        centroid = values[positions].mean(axis=0)
        distances = np.linalg.norm(values[positions] - centroid, axis=1)
        order = np.argsort(distances)[:top_n]
        for rank, local_position in enumerate(order, start=1):
            absolute_position = int(positions[local_position])
            row = assignments.iloc[absolute_position]
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "rank": rank,
                    "subfield_id": row.get("subfield_id", ""),
                    "subfield_display_name": row.get("subfield_display_name", ""),
                    "subfield_label_unique": row.get("subfield_label_unique", ""),
                    "field_display_name": row.get("field_display_name", ""),
                    "domain_display_name": row.get("domain_display_name", ""),
                    "distance_to_cluster_center": float(distances[local_position]),
                }
            )
    return pd.DataFrame(rows)


def run_metric_space_clustering(
    space_data: MetricSpaceData,
    *,
    config: ClusteringConfig,
) -> ClusteringResult:
    assignments, quality, main_column, main_k = run_cluster_assignments(
        space_data,
        config=config,
    )
    profiles = cluster_profiles(
        assignments,
        space_data.raw_metrics,
        cluster_column=main_column,
    )
    representatives = cluster_representatives(
        assignments,
        space_data.feature_matrix,
        cluster_column=main_column,
    )
    return ClusteringResult(
        assignments=assignments,
        quality_by_k=quality,
        cluster_profiles=profiles,
        representatives=representatives,
        main_cluster_column=main_column,
        effective_default_k=main_k,
    )


def build_single_family_space(
    frame: pd.DataFrame,
    *,
    name: str,
    metric_columns: list[str],
    metric_family: str,
    config: ClusteringConfig,
) -> MetricSpaceData:
    valid = filter_valid_rows(frame).reset_index(drop=True)
    metadata = metadata_frame(valid)
    block = metric_block(valid, metric_columns)
    scaled, report = preprocess_metric_block(
        block,
        metric_family=metric_family,
        config=config,
    )
    pca_scores, loadings, explained = fit_pca_scores(
        scaled,
        variance_threshold=config.pca_variance_threshold,
        block_name=name,
    )
    return MetricSpaceData(
        name=name,
        metadata=metadata,
        raw_metrics=block[scaled.columns].reset_index(drop=True),
        feature_matrix=pca_scores.reset_index(drop=True),
        preprocessing_report=report,
        pca_scores=pca_scores.reset_index(drop=True),
        pca_loadings=loadings,
        pca_explained_variance=explained,
        warnings=[],
        component_blocks=[name],
    )


def joined_metric_tables(
    umap_metrics: pd.DataFrame,
    embedding_metrics: pd.DataFrame,
) -> pd.DataFrame:
    umap = filter_valid_rows(umap_metrics).reset_index(drop=True)
    embedding = filter_valid_rows(embedding_metrics).reset_index(drop=True)
    if "subfield_id" not in umap.columns or "subfield_id" not in embedding.columns:
        raise ValueError("both metric tables must contain subfield_id")
    return umap.merge(
        embedding,
        on="subfield_id",
        how="inner",
        suffixes=("_umap", "_embedding"),
    )


def combined_metadata(joined: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.DataFrame({"subfield_id": joined["subfield_id"].astype(str)})
    for column in METADATA_COLUMNS:
        if column == "subfield_id":
            continue
        left = f"{column}_umap"
        right = f"{column}_embedding"
        if left in joined.columns:
            metadata[column] = joined[left]
        elif right in joined.columns:
            metadata[column] = joined[right]
        elif column in joined.columns:
            metadata[column] = joined[column]
        else:
            metadata[column] = ""
    return metadata[METADATA_COLUMNS]


def build_combined_space(
    umap_metrics: pd.DataFrame,
    embedding_metrics: pd.DataFrame,
    *,
    config: ClusteringConfig,
) -> MetricSpaceData:
    joined = joined_metric_tables(umap_metrics, embedding_metrics).reset_index(drop=True)
    metadata = combined_metadata(joined)
    umap_block = metric_block(joined, CORE_METRIC_COLUMNS_V2)
    embedding_block = metric_block(joined, CORE_EMBEDDING_METRIC_COLUMNS)
    raw_metrics = pd.concat([umap_block, embedding_block], axis=1)
    warnings: list[str] = []

    if config.combined_strategy == "raw":
        scaled, report = preprocess_metric_block(
            raw_metrics,
            metric_family="combined_raw",
            config=config,
        )
        pca_scores, loadings, explained = fit_pca_scores(
            scaled,
            variance_threshold=config.pca_variance_threshold,
            block_name="combined",
        )
        feature_matrix = pca_scores
        retained_raw = raw_metrics[scaled.columns]
        component_blocks = ["combined"]
    else:
        scaled_umap, report_umap = preprocess_metric_block(
            umap_block,
            metric_family="umap_projected",
            config=config,
        )
        scaled_embedding, report_embedding = preprocess_metric_block(
            embedding_block,
            metric_family="embedding_space",
            config=config,
        )
        umap_scores, umap_loadings, umap_explained = fit_pca_scores(
            scaled_umap,
            variance_threshold=config.pca_variance_threshold,
            block_name="umap",
        )
        embedding_scores, embedding_loadings, embedding_explained = fit_pca_scores(
            scaled_embedding,
            variance_threshold=config.pca_variance_threshold,
            block_name="embedding",
        )
        feature_matrix = pd.concat([umap_scores, embedding_scores], axis=1)
        report = pd.concat([report_umap, report_embedding], ignore_index=True)
        loadings = pd.concat([umap_loadings, embedding_loadings], ignore_index=True)
        explained = pd.concat([umap_explained, embedding_explained], ignore_index=True)
        retained_raw = pd.concat(
            [umap_block[scaled_umap.columns], embedding_block[scaled_embedding.columns]],
            axis=1,
        )
        component_blocks = ["umap", "embedding"]

    if config.combined_strategy == "raw":
        warnings.append("combined raw strategy lets all retained metrics enter one PCA block")

    return MetricSpaceData(
        name="combined",
        metadata=metadata.reset_index(drop=True),
        raw_metrics=retained_raw.reset_index(drop=True),
        feature_matrix=feature_matrix.reset_index(drop=True),
        preprocessing_report=report,
        pca_scores=feature_matrix.reset_index(drop=True),
        pca_loadings=loadings,
        pca_explained_variance=explained,
        warnings=warnings,
        component_blocks=component_blocks,
    )


def build_metric_space(
    space: str,
    umap_metrics: pd.DataFrame,
    embedding_metrics: pd.DataFrame,
    *,
    config: ClusteringConfig,
) -> MetricSpaceData:
    if space not in SUPPORTED_SPACES:
        raise ValueError(
            f"space must be one of {', '.join(sorted(SUPPORTED_SPACES))}: {space}"
        )
    if space == "embedding_only":
        return build_single_family_space(
            embedding_metrics,
            name="embedding_only",
            metric_columns=CORE_EMBEDDING_METRIC_COLUMNS,
            metric_family="embedding_space",
            config=config,
        )
    if space == "umap_only":
        return build_single_family_space(
            umap_metrics,
            name="umap_only",
            metric_columns=CORE_METRIC_COLUMNS_V2,
            metric_family="umap_projected",
            config=config,
        )
    return build_combined_space(umap_metrics, embedding_metrics, config=config)


def partition_agreement(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    cluster_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"subfield_id", cluster_column}
    if not required.issubset(left.columns) or not required.issubset(right.columns):
        raise ValueError(f"both assignment frames must contain subfield_id and {cluster_column}")
    joined = left[["subfield_id", cluster_column]].merge(
        right[["subfield_id", cluster_column]],
        on="subfield_id",
        how="inner",
        suffixes=("_left", "_right"),
    )
    if joined.empty:
        ari = nmi = np.nan
    else:
        ari = adjusted_rand_score(
            joined[f"{cluster_column}_left"],
            joined[f"{cluster_column}_right"],
        )
        nmi = normalized_mutual_info_score(
            joined[f"{cluster_column}_left"],
            joined[f"{cluster_column}_right"],
        )
    contingency = pd.crosstab(
        joined[f"{cluster_column}_left"],
        joined[f"{cluster_column}_right"],
    )
    contingency.index.name = f"{left_name}_{cluster_column}"
    contingency.columns.name = f"{right_name}_{cluster_column}"
    return (
        {
            "left_space": left_name,
            "right_space": right_name,
            "adjusted_rand_index": float(ari) if pd.notna(ari) else np.nan,
            "normalized_mutual_info": float(nmi) if pd.notna(nmi) else np.nan,
            "n_common_subfields": int(len(joined)),
        },
        contingency,
    )


def compare_cluster_partitions(
    assignments_by_space: dict[str, pd.DataFrame],
    *,
    cluster_column: str,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    pairs = [
        ("embedding_only", "umap_only"),
        ("embedding_only", "combined"),
        ("umap_only", "combined"),
    ]
    rows = []
    contingencies: dict[tuple[str, str], pd.DataFrame] = {}
    for left_name, right_name in pairs:
        if left_name not in assignments_by_space or right_name not in assignments_by_space:
            continue
        row, contingency = partition_agreement(
            assignments_by_space[left_name],
            assignments_by_space[right_name],
            left_name=left_name,
            right_name=right_name,
            cluster_column=cluster_column,
        )
        rows.append(row)
        contingencies[(left_name, right_name)] = contingency
    return pd.DataFrame(rows), contingencies


def plot_pca_scatter(
    pca_scores: pd.DataFrame,
    assignments: pd.DataFrame,
    output_path: str | Path,
    *,
    cluster_column: str,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = pca_scores.iloc[:, 0].to_numpy()
    y = pca_scores.iloc[:, 1].to_numpy() if pca_scores.shape[1] > 1 else np.zeros(len(x))
    labels = assignments[cluster_column].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    scatter = ax.scatter(x, y, c=labels, cmap="tab10", s=28, alpha=0.85, linewidths=0)
    ax.set_xlabel(pca_scores.columns[0])
    ax.set_ylabel(pca_scores.columns[1] if pca_scores.shape[1] > 1 else "0")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label=cluster_column)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def metric_space_umap_coordinates(
    features: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[np.ndarray, str]:
    try:
        import umap

        n_neighbors = min(15, max(2, len(features) - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=random_state,
        )
        coords = reducer.fit_transform(features.to_numpy())
        warning = ""
    except Exception as exc:
        coords = np.full((len(features), 2), np.nan, dtype=float)
        warning = f"metric-space UMAP skipped: {exc}"
    return coords, warning


def metric_space_umap_output_frame(
    assignments: pd.DataFrame,
    coordinates: np.ndarray,
    *,
    cluster_column: str,
) -> pd.DataFrame:
    required = {"subfield_id", "subfield_display_name", cluster_column}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(
            "assignment table missing columns for metric-space UMAP output: "
            f"{', '.join(sorted(missing))}"
        )
    coords = np.asarray(coordinates, dtype=float)
    if coords.shape != (len(assignments), 2):
        raise ValueError(
            "metric-space UMAP coordinates must have shape "
            f"({len(assignments)}, 2), got {coords.shape}"
        )
    frame = assignments[["subfield_id", "subfield_display_name", cluster_column]].copy()
    frame["metric_umap_x"] = coords[:, 0]
    frame["metric_umap_y"] = coords[:, 1]
    frame["cluster_label"] = frame[cluster_column]
    columns = [
        "subfield_id",
        "subfield_display_name",
        "metric_umap_x",
        "metric_umap_y",
        cluster_column,
        "cluster_label",
    ]
    return frame[columns]


def plot_metric_space_umap(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    output_path: str | Path,
    *,
    cluster_column: str,
    random_state: int,
    coordinates: np.ndarray | None = None,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if coordinates is None:
        coords, warning = metric_space_umap_coordinates(
            features,
            random_state=random_state,
        )
    else:
        coords = np.asarray(coordinates, dtype=float)
        warning = ""
        if coords.shape != (len(features), 2) or not np.isfinite(coords).all():
            warning = "metric-space UMAP coordinates unavailable"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    if warning:
        ax.text(0.5, 0.5, warning, ha="center", va="center", wrap=True)
        ax.axis("off")
    else:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=assignments[cluster_column].to_numpy(),
            cmap="tab10",
            s=28,
            alpha=0.85,
            linewidths=0,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("UMAP Visualization Of Subfields In Metric Space")
        fig.colorbar(scatter, ax=ax, label=cluster_column)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return warning


def plot_dendrogram(features: pd.DataFrame, output_path: str | Path, *, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    if len(features) < 2:
        ax.text(0.5, 0.5, "Need at least two rows for dendrogram", ha="center", va="center")
        ax.axis("off")
    else:
        tree = linkage(features.to_numpy(), method="ward")
        dendrogram(tree, no_labels=True, ax=ax, color_threshold=None)
        ax.set_title(title)
        ax.set_xlabel("Subfields")
        ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cluster_profile_heatmap(
    profiles: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if profiles.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        ax.text(0.5, 0.5, "No cluster profiles available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    matrix = profiles.pivot(
        index="cluster_id",
        columns="metric_name",
        values="standardized_median",
    ).fillna(0.0)
    fig_width = max(10, len(matrix.columns) * 0.28)
    fig_height = max(4, len(matrix.index) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([f"cluster {idx}" for idx in matrix.index], fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="z-scored median")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_agreement_heatmap(agreement: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if agreement.empty:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
        ax.text(0.5, 0.5, "No agreement comparisons available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    labels = [f"{row.left_space}\nvs\n{row.right_space}" for row in agreement.itertuples()]
    values = agreement["adjusted_rand_index"].to_numpy(dtype=float)[None, :]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.2), 3), dpi=160)
    image = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels(["ARI"])
    ax.set_title("Cluster Partition Agreement")
    for idx, value in enumerate(values.ravel()):
        text = "NaN" if pd.isna(value) else f"{value:.2f}"
        ax.text(idx, 0, text, ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def space_summary_markdown(
    space_data: MetricSpaceData,
    result: ClusteringResult,
    *,
    config: ClusteringConfig,
) -> str:
    retained = space_data.preprocessing_report.loc[
        space_data.preprocessing_report["retained"],
        "metric_name",
    ].tolist()
    dropped = space_data.preprocessing_report.loc[
        ~space_data.preprocessing_report["retained"],
        ["metric_name", "drop_reason"],
    ]
    lines = [
        f"# Metric Clustering: {space_data.name}",
        "",
        "This is an exploratory typology of OpenAlex subfields based on selected "
        "morphology/structure metrics. It is descriptive, not a claim about true "
        "natural kinds.",
        "",
        f"- Subfields clustered: {len(space_data.metadata)}",
        f"- Retained metrics: {len(retained)}",
        f"- Main method: Ward hierarchical clustering on PCA scores",
        f"- Main assignment column: `{result.main_cluster_column}`",
        f"- Effective default k: {result.effective_default_k}",
        f"- PCA variance threshold: {config.pca_variance_threshold:.2f}",
        f"- Scaler: {config.scaler}",
        "",
        "## Dropped Metrics",
        "",
    ]
    if dropped.empty:
        lines.append("No metrics were dropped by preprocessing.")
    else:
        for row in dropped.itertuples(index=False):
            lines.append(f"- `{row.metric_name}`: {row.drop_reason}")
    if space_data.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in space_data.warnings)
    lines.extend(
        [
            "",
            "KMeans and GaussianMixture assignments are included as robustness checks. "
            "Quality metrics are diagnostics and do not choose a true k automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def comparison_summary_markdown(agreement: pd.DataFrame) -> str:
    lines = [
        "# Metric Clustering Comparison",
        "",
        "This compares Ward cluster assignments across metric spaces. High agreement "
        "means the spaces produce similar typologies; low agreement does not "
        "automatically mean failure because projected morphology and embedding-space "
        "structure measure different aspects.",
        "",
    ]
    if agreement.empty:
        lines.append("No partition comparisons were available.")
    else:
        for row in agreement.itertuples(index=False):
            lines.append(
                f"- `{row.left_space}` vs `{row.right_space}`: "
                f"ARI={row.adjusted_rand_index:.3f}, "
                f"NMI={row.normalized_mutual_info:.3f}, "
                f"n={row.n_common_subfields}"
            )
    lines.extend(
        [
            "",
            "Combined clusters should be read as a compromise/summary typology over "
            "both metric families.",
            "",
        ]
    )
    return "\n".join(lines)
