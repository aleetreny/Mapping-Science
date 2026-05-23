from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


EPS = 1e-12


def canonicalize_cluster_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    counts = pd.Series(values).value_counts()
    ordered = sorted(counts.index.tolist(), key=lambda value: (-int(counts[value]), str(value)))
    mapping = {old: idx + 1 for idx, old in enumerate(ordered) if int(old) != -1}
    if -1 in counts.index:
        mapping[-1] = 0
    return np.array([mapping[value] for value in values], dtype=int)


def cluster_size_summary(labels: np.ndarray) -> tuple[int, int, str]:
    counts = pd.Series(labels).value_counts().sort_values()
    return int(counts.min()), int(counts.max()), ";".join(str(int(value)) for value in counts)


def domain_purity(cluster_labels: np.ndarray, domain_labels: np.ndarray) -> float:
    frame = pd.DataFrame({"cluster": cluster_labels, "domain": domain_labels})
    majority_counts = frame.groupby("cluster")["domain"].agg(lambda values: values.value_counts().iloc[0])
    return float(majority_counts.sum() / len(frame))


def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return StandardScaler().fit_transform(values)


def condensed_from_square(distance_matrix: np.ndarray) -> np.ndarray:
    return squareform(np.asarray(distance_matrix, dtype=float), checks=False)


def normalize_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(distance_matrix, dtype=float).copy()
    np.fill_diagonal(values, 0.0)
    upper = values[np.triu_indices_from(values, k=1)]
    finite = upper[np.isfinite(upper)]
    scale = float(np.mean(finite)) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
    return values / scale


def labels_hierarchical_features(
    matrix: np.ndarray,
    *,
    k: int,
    method: str,
    metric: str,
) -> np.ndarray:
    if method == "ward":
        tree = linkage(matrix, method="ward", metric="euclidean")
    else:
        tree = linkage(pdist(matrix, metric=metric), method=method)
    return canonicalize_cluster_labels(fcluster(tree, t=k, criterion="maxclust"))


def labels_hierarchical_precomputed(
    distance_matrix: np.ndarray,
    *,
    k: int,
    method: str = "average",
) -> np.ndarray:
    tree = linkage(condensed_from_square(distance_matrix), method=method)
    return canonicalize_cluster_labels(fcluster(tree, t=k, criterion="maxclust"))


def labels_kmeans(matrix: np.ndarray, *, k: int, random_seed: int) -> np.ndarray:
    return canonicalize_cluster_labels(
        KMeans(n_clusters=k, n_init=50, random_state=random_seed).fit_predict(matrix)
    )


def labels_spectral_features(matrix: np.ndarray, *, k: int, random_seed: int) -> np.ndarray:
    return canonicalize_cluster_labels(
        SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=min(12, max(2, len(matrix) // 10)),
            assign_labels="kmeans",
            random_state=random_seed,
            n_init=20,
        ).fit_predict(matrix)
    )


def labels_spectral_precomputed(distance_matrix: np.ndarray, *, k: int, random_seed: int) -> np.ndarray:
    upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    sigma = float(np.median(upper[np.isfinite(upper)]))
    if not np.isfinite(sigma) or sigma <= EPS:
        sigma = 1.0
    affinity = np.exp(-((distance_matrix / sigma) ** 2))
    np.fill_diagonal(affinity, 1.0)
    return canonicalize_cluster_labels(
        SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_seed,
            n_init=20,
        ).fit_predict(affinity)
    )


def safe_silhouette(
    matrix_or_distance: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    precomputed: bool = False,
) -> float:
    if len(np.unique(labels)) <= 1 or len(np.unique(labels)) >= len(labels):
        return float("nan")
    try:
        return float(
            silhouette_score(
                matrix_or_distance,
                labels,
                metric="precomputed" if precomputed else metric,
            )
        )
    except ValueError:
        return float("nan")


def evaluate_labels(
    *,
    solution_id: str,
    representation: str,
    algorithm: str,
    distance_metric: str,
    k: int,
    labels: np.ndarray,
    feature_matrix: np.ndarray,
    morph_labels: np.ndarray,
    domain_labels: np.ndarray,
    centroid_labels: np.ndarray | None = None,
    silhouette_matrix: np.ndarray | None = None,
    silhouette_metric: str = "euclidean",
    precomputed_silhouette: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels = canonicalize_cluster_labels(labels)
    min_size, max_size, sizes = cluster_size_summary(labels)
    row: dict[str, Any] = {
        "solution_id": solution_id,
        "representation": representation,
        "algorithm": algorithm,
        "distance_metric": distance_metric,
        "k": int(k),
        "silhouette": safe_silhouette(
            silhouette_matrix if silhouette_matrix is not None else feature_matrix,
            labels,
            metric=silhouette_metric,
            precomputed=precomputed_silhouette,
        ),
        "calinski_harabasz": _safe_score(calinski_harabasz_score, feature_matrix, labels),
        "davies_bouldin": _safe_score(davies_bouldin_score, feature_matrix, labels),
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "cluster_sizes_sorted": sizes,
        "ari_vs_morphology": float(adjusted_rand_score(morph_labels, labels)),
        "ami_vs_morphology": float(adjusted_mutual_info_score(morph_labels, labels)),
        "ari_vs_domain": float(adjusted_rand_score(domain_labels, labels)),
        "ami_vs_domain": float(adjusted_mutual_info_score(domain_labels, labels)),
        "domain_purity": domain_purity(labels, domain_labels),
        "status": "ok",
    }
    if centroid_labels is not None:
        row["ari_vs_centroid"] = float(adjusted_rand_score(centroid_labels, labels))
        row["ami_vs_centroid"] = float(adjusted_mutual_info_score(centroid_labels, labels))
    if extra:
        row.update(extra)
    return row


def _safe_score(function: Any, matrix: np.ndarray, labels: np.ndarray) -> float:
    try:
        if len(np.unique(labels)) <= 1 or len(np.unique(labels)) >= len(labels):
            return float("nan")
        return float(function(matrix, labels))
    except ValueError:
        return float("nan")


def error_row(
    *,
    solution_id: str,
    representation: str,
    algorithm: str,
    distance_metric: str,
    k: int,
    exc: Exception,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "solution_id": solution_id,
        "representation": representation,
        "algorithm": algorithm,
        "distance_metric": distance_metric,
        "k": int(k),
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
        "min_cluster_size": np.nan,
        "max_cluster_size": np.nan,
        "cluster_sizes_sorted": "",
        "ari_vs_morphology": np.nan,
        "ami_vs_morphology": np.nan,
        "ari_vs_domain": np.nan,
        "ami_vs_domain": np.nan,
        "domain_purity": np.nan,
        "status": f"error: {type(exc).__name__}: {exc}",
    }
    if extra:
        row.update(extra)
    return row


def bootstrap_subsample_stability(
    matrix: np.ndarray,
    selected_labels: np.ndarray,
    *,
    label_function: Any,
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
        labels = label_function(indices)
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


def dataframe_to_markdown(frame: pd.DataFrame, *, float_digits: int = 3) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value)

    rows = [[format_value(value) for value in row] for row in frame[columns].to_numpy()]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows))
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def save_figure(fig: plt.Figure, stem: str, *, output_dir: Path, figure_dir: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
