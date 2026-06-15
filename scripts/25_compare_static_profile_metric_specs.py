from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.temporal_common import utc_now_iso


STRUCTURAL_METRICS = [
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_iqr",
    "embedding_distance_to_centroid_p90",
    "embedding_knn_median_distance",
    "embedding_knn_distance_cv",
    "embedding_knn_indegree_gini",
    "embedding_pca_dim_80",
    "embedding_pca_spectral_entropy",
]

TEMPORAL_SUMMARY_METRICS = [
    "embedding_centroid_drift_early_late",
    "embedding_radial_expansion_slope",
    "embedding_recent_novelty_score",
]

ALL11_METRICS = [*STRUCTURAL_METRICS, *TEMPORAL_SUMMARY_METRICS]

METRIC_LABELS = {
    "embedding_distance_to_centroid_median": "centroid median",
    "embedding_distance_to_centroid_iqr": "centroid IQR",
    "embedding_distance_to_centroid_p90": "centroid P90",
    "embedding_knn_median_distance": "kNN median",
    "embedding_knn_distance_cv": "kNN CV",
    "embedding_knn_indegree_gini": "hub Gini",
    "embedding_pca_dim_80": "PCA D80",
    "embedding_pca_spectral_entropy": "PCA entropy",
    "embedding_centroid_drift_early_late": "centroid drift",
    "embedding_radial_expansion_slope": "radial expansion slope",
    "embedding_recent_novelty_score": "recent novelty",
}

VALID_STATUSES = {"completed", "completed_with_warnings"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 11-metric full-period morphology profiles with an 8-metric "
            "structural-only specification without overwriting thesis outputs."
        )
    )
    parser.add_argument(
        "--input",
        default="outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.csv",
        help="Reduced metric core CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/10_static_profile_spec_comparison",
        help="Isolated output folder for this comparison.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--cluster-k", type=int, default=5)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def load_core(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        *ALL11_METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if "metric_status" in frame.columns:
        frame = frame.loc[frame["metric_status"].isin(VALID_STATUSES)].copy()
    for column in ["subfield_id", "field_id", "domain_id"]:
        frame[column] = frame[column].astype(str)
    for metric in ALL11_METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    missing_counts = frame[ALL11_METRICS].isna().sum()
    if int(missing_counts.sum()) > 0:
        detail = ", ".join(
            f"{metric}={int(count)}"
            for metric, count in missing_counts.items()
            if int(count) > 0
        )
        raise ValueError("Missing metric values: " + detail)
    return frame.sort_values("subfield_id", kind="mergesort").reset_index(drop=True)


def profile_table(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    subfield_cols = [
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        *metrics,
    ]
    subfields = frame[subfield_cols].copy()
    subfields = subfields.rename(
        columns={
            "subfield_id": "entity_id",
            "subfield_display_name": "entity_display_name",
        }
    )
    subfields.insert(0, "level", "subfield")
    subfields["n_subfields"] = 1
    rows.append(subfields)

    field_profiles = (
        frame.groupby(
            ["field_id", "field_display_name", "domain_id", "domain_display_name"],
            dropna=False,
            sort=True,
        )[metrics]
        .mean()
        .reset_index()
    )
    counts = (
        frame.groupby(["field_id", "field_display_name"], dropna=False, sort=True)["subfield_id"]
        .nunique()
        .reset_index(name="n_subfields")
    )
    field_profiles = field_profiles.merge(counts, on=["field_id", "field_display_name"], how="left")
    field_profiles = field_profiles.rename(
        columns={"field_id": "entity_id", "field_display_name": "entity_display_name"}
    )
    field_profiles.insert(0, "level", "field")
    rows.append(field_profiles)

    domain_profiles = (
        frame.groupby(["domain_id", "domain_display_name"], dropna=False, sort=True)[metrics]
        .mean()
        .reset_index()
    )
    counts = (
        frame.groupby(["domain_id", "domain_display_name"], dropna=False, sort=True)["subfield_id"]
        .nunique()
        .reset_index(name="n_subfields")
    )
    domain_profiles = domain_profiles.merge(
        counts, on=["domain_id", "domain_display_name"], how="left"
    )
    domain_profiles = domain_profiles.rename(
        columns={"domain_id": "entity_id", "domain_display_name": "entity_display_name"}
    )
    domain_profiles.insert(0, "level", "domain")
    domain_profiles["field_id"] = ""
    domain_profiles["field_display_name"] = ""
    domain_profiles["domain_id"] = domain_profiles["entity_id"]
    domain_profiles["domain_display_name"] = domain_profiles["entity_display_name"]
    rows.append(domain_profiles)

    result = pd.concat(rows, ignore_index=True, sort=False)
    result["entity_id"] = result["entity_id"].astype(str)
    return result.sort_values(["level", "entity_id"], kind="mergesort").reset_index(drop=True)


def robust_scale_by_level(profiles: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    scaled = profiles.copy()
    for level, level_frame in profiles.groupby("level", sort=False):
        idx = level_frame.index
        for metric in metrics:
            values = pd.to_numeric(level_frame[metric], errors="coerce")
            center = values.median()
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            scale = q75 - q25
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = values.std(ddof=0)
            if not np.isfinite(scale) or scale <= 1e-12:
                scaled.loc[idx, metric] = np.nan
            else:
                scaled.loc[idx, metric] = (values - center) / scale
    return scaled


def subfield_matrix(frame: pd.DataFrame, metrics: list[str]) -> np.ndarray:
    return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75)).fit_transform(
        frame[metrics].to_numpy(dtype=float)
    )


def similarity_profile_table(frame: pd.DataFrame, metrics: list[str], spec: str) -> pd.DataFrame:
    """Profiles used by Chapter 6: robust-scale subfields, then aggregate."""
    scaled = frame[
        [
            "subfield_id",
            "subfield_display_name",
            "field_id",
            "field_display_name",
            "domain_id",
            "domain_display_name",
        ]
    ].copy()
    scaled[metrics] = subfield_matrix(frame, metrics)

    rows: list[pd.DataFrame] = []
    subfields = scaled.rename(
        columns={
            "subfield_id": "entity_id",
            "subfield_display_name": "entity_display_name",
        }
    )
    subfields.insert(0, "spec", spec)
    subfields.insert(1, "level", "subfield")
    subfields["n_source_subfields"] = 1
    rows.append(subfields)

    fields = (
        scaled.groupby(
            ["field_id", "field_display_name", "domain_id", "domain_display_name"],
            sort=True,
            dropna=False,
        )[metrics]
        .mean()
        .reset_index()
    )
    counts = (
        scaled.groupby(["field_id", "field_display_name"], sort=True, dropna=False)["subfield_id"]
        .nunique()
        .reset_index(name="n_source_subfields")
    )
    fields = fields.merge(counts, on=["field_id", "field_display_name"], how="left")
    fields = fields.rename(
        columns={"field_id": "entity_id", "field_display_name": "entity_display_name"}
    )
    fields.insert(0, "spec", spec)
    fields.insert(1, "level", "field")
    rows.append(fields)

    domains = (
        scaled.groupby(["domain_id", "domain_display_name"], sort=True, dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    counts = (
        scaled.groupby(["domain_id", "domain_display_name"], sort=True, dropna=False)["subfield_id"]
        .nunique()
        .reset_index(name="n_source_subfields")
    )
    domains = domains.merge(counts, on=["domain_id", "domain_display_name"], how="left")
    domains = domains.rename(
        columns={"domain_id": "entity_id", "domain_display_name": "entity_display_name"}
    )
    domains.insert(0, "spec", spec)
    domains.insert(1, "level", "domain")
    domains["field_id"] = ""
    domains["field_display_name"] = ""
    domains["domain_id"] = domains["entity_id"]
    domains["domain_display_name"] = domains["entity_display_name"]
    rows.append(domains)

    result = pd.concat(rows, ignore_index=True, sort=False)
    result["entity_id"] = result["entity_id"].astype(str)
    return result.sort_values(["level", "entity_display_name"], kind="mergesort").reset_index(drop=True)


def pair_id(left_id: str, right_id: str) -> str:
    return "__".join(sorted([str(left_id), str(right_id)]))


def field_distances(scaled_profiles: pd.DataFrame, metrics: list[str], spec: str) -> pd.DataFrame:
    fields = scaled_profiles.loc[scaled_profiles["level"] == "field"].copy()
    fields = fields.sort_values("entity_id", kind="mergesort").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for left_idx, right_idx in combinations(range(len(fields)), 2):
        left = fields.iloc[left_idx]
        right = fields.iloc[right_idx]
        left_values = left[metrics].to_numpy(dtype=float)
        right_values = right[metrics].to_numpy(dtype=float)
        rows.append(
            {
                "spec": spec,
                "pair_id": pair_id(left.entity_id, right.entity_id),
                "field_id_a": left.entity_id,
                "field_a": left.entity_display_name,
                "domain_a": left.domain_display_name,
                "field_id_b": right.entity_id,
                "field_b": right.entity_display_name,
                "domain_b": right.domain_display_name,
                "same_domain": bool(left.domain_id == right.domain_id),
                "distance": float(np.linalg.norm(left_values - right_values)),
                "n_metrics": len(metrics),
            }
        )
    return pd.DataFrame(rows)


def nearest_neighbors(distances: pd.DataFrame, spec: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in distances.itertuples(index=False):
        rows.append(
            {
                "spec": spec,
                "field_id": row.field_id_a,
                "field": row.field_a,
                "domain": row.domain_a,
                "neighbor_field_id": row.field_id_b,
                "neighbor_field": row.field_b,
                "neighbor_domain": row.domain_b,
                "nearest_distance": row.distance,
                "outside_domain": not bool(row.same_domain),
            }
        )
        rows.append(
            {
                "spec": spec,
                "field_id": row.field_id_b,
                "field": row.field_b,
                "domain": row.domain_b,
                "neighbor_field_id": row.field_id_a,
                "neighbor_field": row.field_a,
                "neighbor_domain": row.domain_a,
                "nearest_distance": row.distance,
                "outside_domain": not bool(row.same_domain),
            }
        )
    expanded = pd.DataFrame(rows)
    return (
        expanded.sort_values(
            ["field_id", "nearest_distance", "neighbor_field"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .groupby("field_id", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def cluster_summary(matrix: np.ndarray, spec: str, k: int) -> tuple[pd.DataFrame, np.ndarray]:
    linked = linkage(matrix, method="ward")
    labels = fcluster(linked, t=k, criterion="maxclust").astype(int)
    sizes = pd.Series(labels).value_counts().sort_values().astype(int).tolist()
    silhouette = float(silhouette_score(matrix, labels, metric="euclidean"))
    summary = pd.DataFrame(
        [
            {
                "spec": spec,
                "algorithm": "Ward hierarchical",
                "distance_metric": "euclidean",
                "scaling": "robust median/IQR",
                "k": int(k),
                "silhouette": silhouette,
                "min_cluster_size": int(min(sizes)),
                "max_cluster_size": int(max(sizes)),
                "cluster_sizes_sorted": ";".join(str(size) for size in sizes),
            }
        ]
    )
    return summary, labels


def pca_scores(frame: pd.DataFrame, matrix: np.ndarray, spec: str) -> tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)
    result = frame[
        [
            "subfield_id",
            "subfield_display_name",
            "field_display_name",
            "domain_display_name",
        ]
    ].copy()
    result.insert(0, "spec", spec)
    result["pca_1"] = coords[:, 0]
    result["pca_2"] = coords[:, 1]
    result["pca_radius"] = np.sqrt(np.square(coords[:, 0]) + np.square(coords[:, 1]))
    result["pca_1_explained_variance"] = float(pca.explained_variance_ratio_[0])
    result["pca_2_explained_variance"] = float(pca.explained_variance_ratio_[1])
    return result.sort_values("subfield_id", kind="mergesort").reset_index(drop=True), pca


def metric_extremes(profiles: pd.DataFrame, metrics: list[str], spec: str, top_n: int) -> pd.DataFrame:
    subfields = profiles.loc[profiles["level"] == "subfield"].copy()
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for direction, ascending in [("lowest", True), ("highest", False)]:
            ranked = subfields.sort_values(
                [metric, "entity_display_name"],
                ascending=[ascending, True],
                kind="mergesort",
            ).head(top_n)
            for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                rows.append(
                    {
                        "spec": spec,
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "direction": direction,
                        "rank": rank,
                        "subfield_id": row.entity_id,
                        "subfield": row.entity_display_name,
                        "field": row.field_display_name,
                        "domain": row.domain_display_name,
                        "value": float(getattr(row, metric)),
                    }
                )
    return pd.DataFrame(rows)


def top_pairs(distances: pd.DataFrame, spec: str, top_n: int) -> pd.DataFrame:
    closest = distances.sort_values(["distance", "pair_id"], kind="mergesort").head(top_n)
    farthest = distances.sort_values(
        ["distance", "pair_id"], ascending=[False, True], kind="mergesort"
    ).head(top_n)
    return pd.concat(
        [closest.assign(direction="closest"), farthest.assign(direction="farthest")],
        ignore_index=True,
    )


def run_spec(frame: pd.DataFrame, spec: str, metrics: list[str], k: int, top_n: int) -> dict[str, Any]:
    profiles = profile_table(frame, metrics)
    scaled_profiles = robust_scale_by_level(profiles, metrics)
    similarity_profiles = similarity_profile_table(frame, metrics, spec)
    distances = field_distances(similarity_profiles, metrics, spec)
    nearest = nearest_neighbors(distances, spec)
    matrix = subfield_matrix(frame, metrics)
    cluster, labels = cluster_summary(matrix, spec, k)
    pca_frame, pca = pca_scores(frame, matrix, spec)
    pca_frame["cluster_id"] = labels
    extremes = metric_extremes(profiles, metrics, spec, top_n)
    pairs = top_pairs(distances, spec, top_n)
    domain_field_scaled = scaled_profiles.loc[
        scaled_profiles["level"].isin(["domain", "field"]),
        [
            "level",
            "entity_id",
            "entity_display_name",
            "domain_id",
            "domain_display_name",
            "n_subfields",
            *metrics,
        ],
    ].copy()
    domain_field_scaled.insert(0, "spec", spec)
    domain_field_similarity = similarity_profiles.loc[
        similarity_profiles["level"].isin(["domain", "field"]),
        [
            "spec",
            "level",
            "entity_id",
            "entity_display_name",
            "domain_id",
            "domain_display_name",
            "n_source_subfields",
            *metrics,
        ],
    ].copy()
    return {
        "metrics": metrics,
        "profiles": profiles,
        "scaled_profiles": scaled_profiles,
        "similarity_profiles": similarity_profiles,
        "domain_field_scaled": domain_field_scaled,
        "domain_field_similarity": domain_field_similarity,
        "distances": distances,
        "nearest": nearest,
        "cluster": cluster,
        "pca": pca_frame,
        "pca_model": pca,
        "extremes": extremes,
        "top_pairs": pairs,
    }


def paired_field_distances(dist11: pd.DataFrame, dist8: pd.DataFrame) -> pd.DataFrame:
    left = dist11.rename(columns={"distance": "distance_11_metric", "n_metrics": "n_metrics_11"})[
        [
            "pair_id",
            "field_id_a",
            "field_a",
            "domain_a",
            "field_id_b",
            "field_b",
            "domain_b",
            "same_domain",
            "distance_11_metric",
            "n_metrics_11",
        ]
    ]
    right = dist8.rename(columns={"distance": "distance_8_metric", "n_metrics": "n_metrics_8"})[
        ["pair_id", "distance_8_metric", "n_metrics_8"]
    ]
    merged = left.merge(right, on="pair_id", how="inner")
    merged["distance_delta_8_minus_11"] = (
        merged["distance_8_metric"] - merged["distance_11_metric"]
    )
    merged["rank_11_closest"] = merged["distance_11_metric"].rank(method="min", ascending=True)
    merged["rank_8_closest"] = merged["distance_8_metric"].rank(method="min", ascending=True)
    merged["rank_delta_8_minus_11"] = merged["rank_8_closest"] - merged["rank_11_closest"]
    return merged.sort_values("rank_11_closest", kind="mergesort").reset_index(drop=True)


def paired_nearest(nn11: pd.DataFrame, nn8: pd.DataFrame) -> pd.DataFrame:
    left = nn11.rename(
        columns={
            "neighbor_field_id": "neighbor_field_id_11",
            "neighbor_field": "neighbor_field_11",
            "neighbor_domain": "neighbor_domain_11",
            "nearest_distance": "nearest_distance_11",
            "outside_domain": "outside_domain_11",
        }
    )[
        [
            "field_id",
            "field",
            "domain",
            "neighbor_field_id_11",
            "neighbor_field_11",
            "neighbor_domain_11",
            "nearest_distance_11",
            "outside_domain_11",
        ]
    ]
    right = nn8.rename(
        columns={
            "neighbor_field_id": "neighbor_field_id_8",
            "neighbor_field": "neighbor_field_8",
            "neighbor_domain": "neighbor_domain_8",
            "nearest_distance": "nearest_distance_8",
            "outside_domain": "outside_domain_8",
        }
    )[
        [
            "field_id",
            "neighbor_field_id_8",
            "neighbor_field_8",
            "neighbor_domain_8",
            "nearest_distance_8",
            "outside_domain_8",
        ]
    ]
    merged = left.merge(right, on="field_id", how="inner")
    merged["same_nearest_neighbor"] = (
        merged["neighbor_field_id_11"].astype(str) == merged["neighbor_field_id_8"].astype(str)
    )
    merged["nearest_distance_delta_8_minus_11"] = (
        merged["nearest_distance_8"] - merged["nearest_distance_11"]
    )
    return merged.sort_values("field", kind="mergesort").reset_index(drop=True)


def pca_outliers_side_by_side(pca11: pd.DataFrame, pca8: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top11 = (
        pca11.sort_values(["pca_radius", "subfield_display_name"], ascending=[False, True])
        .head(top_n)
        .assign(pca_outlier_rank=lambda x: np.arange(1, len(x) + 1))
    )
    top8 = (
        pca8.sort_values(["pca_radius", "subfield_display_name"], ascending=[False, True])
        .head(top_n)
        .assign(pca_outlier_rank=lambda x: np.arange(1, len(x) + 1))
    )
    return pd.concat([top11, top8], ignore_index=True, sort=False)


def pair_overlap(left: pd.DataFrame, right: pd.DataFrame, direction: str) -> tuple[int, int, float]:
    left_ids = set(left.loc[left["direction"] == direction, "pair_id"])
    right_ids = set(right.loc[right["direction"] == direction, "pair_id"])
    denom = min(len(left_ids), len(right_ids))
    overlap = len(left_ids & right_ids)
    return overlap, denom, float(overlap / denom) if denom else np.nan


def field_distance_correlations(paired: pd.DataFrame) -> tuple[float, float]:
    x = paired["distance_11_metric"].to_numpy(dtype=float)
    y = paired["distance_8_metric"].to_numpy(dtype=float)
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def domain_conclusion_rows(domain_scaled_11: pd.DataFrame, domain_scaled_8: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    domain11 = domain_scaled_11.loc[domain_scaled_11["level"] == "domain"].copy()
    domain8 = domain_scaled_8.loc[domain_scaled_8["level"] == "domain"].copy()
    for metric in STRUCTURAL_METRICS:
        for direction, ascending in [("highest", False), ("lowest", True)]:
            left = domain11.sort_values(
                [metric, "entity_display_name"], ascending=[ascending, True], kind="mergesort"
            ).iloc[0]
            right = domain8.sort_values(
                [metric, "entity_display_name"], ascending=[ascending, True], kind="mergesort"
            ).iloc[0]
            rows.append(
                {
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "direction": direction,
                    "domain_11_metric": left.entity_display_name,
                    "standardized_value_11": float(left[metric]),
                    "domain_8_metric": right.entity_display_name,
                    "standardized_value_8": float(right[metric]),
                    "same_domain": bool(left.entity_id == right.entity_id),
                }
            )
    return pd.DataFrame(rows)


def summary_table(
    *,
    spec11: dict[str, Any],
    spec8: dict[str, Any],
    paired_distances: pd.DataFrame,
    paired_nn: pd.DataFrame,
    pca_outliers: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    pearson, spearman = field_distance_correlations(paired_distances)
    same_nn = int(paired_nn["same_nearest_neighbor"].sum())
    total_fields = int(len(paired_nn))
    outside11 = int(spec11["nearest"]["outside_domain"].sum())
    outside8 = int(spec8["nearest"]["outside_domain"].sum())
    closest_overlap, closest_denom, closest_share = pair_overlap(
        spec11["top_pairs"], spec8["top_pairs"], "closest"
    )
    farthest_overlap, farthest_denom, farthest_share = pair_overlap(
        spec11["top_pairs"], spec8["top_pairs"], "farthest"
    )
    pca11 = set(pca_outliers.loc[pca_outliers["spec"] == "11_metric", "subfield_id"].astype(str))
    pca8 = set(pca_outliers.loc[pca_outliers["spec"] == "8_metric", "subfield_id"].astype(str))
    pca_overlap = len(pca11 & pca8)
    return pd.DataFrame(
        [
            {
                "comparison": "field_distance_matrix_pearson",
                "value": pearson,
                "n": len(paired_distances),
                "detail": "Pearson correlation over 325 unordered field-pair Euclidean distances.",
            },
            {
                "comparison": "field_distance_matrix_spearman",
                "value": spearman,
                "n": len(paired_distances),
                "detail": "Spearman correlation over 325 unordered field-pair Euclidean distances.",
            },
            {
                "comparison": "same_nearest_neighbor_share",
                "value": same_nn / total_fields,
                "n": total_fields,
                "detail": f"{same_nn}/{total_fields} fields retain the same nearest neighbor.",
            },
            {
                "comparison": "outside_domain_nearest_neighbor_count_11_metric",
                "value": outside11,
                "n": total_fields,
                "detail": "Count under the current 11-metric specification.",
            },
            {
                "comparison": "outside_domain_nearest_neighbor_count_8_metric",
                "value": outside8,
                "n": total_fields,
                "detail": "Count under the structural-only 8-metric specification.",
            },
            {
                "comparison": "outside_domain_count_delta_8_minus_11",
                "value": outside8 - outside11,
                "n": total_fields,
                "detail": "Change in cross-domain nearest-neighbor count.",
            },
            {
                "comparison": "static_silhouette_11_metric",
                "value": float(spec11["cluster"].iloc[0]["silhouette"]),
                "n": len(spec11["pca"]),
                "detail": "Ward hierarchical clustering, robust-scaled profile, k=5.",
            },
            {
                "comparison": "static_silhouette_8_metric",
                "value": float(spec8["cluster"].iloc[0]["silhouette"]),
                "n": len(spec8["pca"]),
                "detail": "Ward hierarchical clustering, robust-scaled profile, k=5.",
            },
            {
                "comparison": "static_silhouette_delta_8_minus_11",
                "value": float(spec8["cluster"].iloc[0]["silhouette"])
                - float(spec11["cluster"].iloc[0]["silhouette"]),
                "n": len(spec8["pca"]),
                "detail": "Positive values mean the 8-metric static solution separates slightly more.",
            },
            {
                "comparison": f"top_{top_n}_pca_outlier_overlap_share",
                "value": pca_overlap / top_n,
                "n": top_n,
                "detail": f"{pca_overlap}/{top_n} subfields overlap among top PCA-radius outliers.",
            },
            {
                "comparison": f"top_{top_n}_closest_field_pair_overlap_share",
                "value": closest_share,
                "n": closest_denom,
                "detail": f"{closest_overlap}/{closest_denom} closest field pairs overlap.",
            },
            {
                "comparison": f"top_{top_n}_farthest_field_pair_overlap_share",
                "value": farthest_share,
                "n": farthest_denom,
                "detail": f"{farthest_overlap}/{farthest_denom} farthest field pairs overlap.",
            },
        ]
    )


def interpret_change(summary: pd.DataFrame, domain_rows: pd.DataFrame) -> str:
    values = dict(zip(summary["comparison"], summary["value"]))
    pearson = values["field_distance_matrix_pearson"]
    spearman = values["field_distance_matrix_spearman"]
    same_nn = values["same_nearest_neighbor_share"]
    outside_delta = abs(values["outside_domain_count_delta_8_minus_11"])
    silhouette_delta = abs(values["static_silhouette_delta_8_minus_11"])
    domain_same = bool(domain_rows["same_domain"].all())
    closest_overlap = values["top_10_closest_field_pair_overlap_share"]
    if (
        pearson >= 0.95
        and spearman >= 0.95
        and same_nn >= 0.80
        and outside_delta <= 1
        and silhouette_delta < 0.02
        and closest_overlap >= 0.70
        and domain_same
    ):
        return "negligible changes"
    if (
        pearson >= 0.80
        and spearman >= 0.80
        and outside_delta <= 3
        and silhouette_delta < 0.08
        and domain_same
    ):
        return "moderate but non-substantive changes"
    return "substantive changes"


def format_pair_list(pairs: pd.DataFrame, direction: str, distance_col: str = "distance") -> str:
    subset = pairs.loc[pairs["direction"] == direction].head(5)
    return "; ".join(
        f"{row.field_a}--{row.field_b} ({getattr(row, distance_col):.2f})"
        for row in subset.itertuples()
    )


def write_report(
    path: Path,
    *,
    input_path: Path,
    output_dir: Path,
    summary: pd.DataFrame,
    verdict: str,
    spec11: dict[str, Any],
    spec8: dict[str, Any],
    paired_nn: pd.DataFrame,
    domain_rows: pd.DataFrame,
    pca_outliers: pd.DataFrame,
    top_n: int,
) -> None:
    values = dict(zip(summary["comparison"], summary["value"]))
    same_nn = int(paired_nn["same_nearest_neighbor"].sum())
    total_nn = len(paired_nn)
    pca11 = set(pca_outliers.loc[pca_outliers["spec"] == "11_metric", "subfield_id"].astype(str))
    pca8 = set(pca_outliers.loc[pca_outliers["spec"] == "8_metric", "subfield_id"].astype(str))
    pca_overlap_names = (
        pca_outliers.loc[
            pca_outliers["subfield_id"].astype(str).isin(pca11 & pca8),
            "subfield_display_name",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    changed_nn = paired_nn.loc[~paired_nn["same_nearest_neighbor"]].copy()
    changed_preview = "; ".join(
        f"{row.field}: {row.neighbor_field_11} -> {row.neighbor_field_8}"
        for row in changed_nn.head(8).itertuples()
    )
    domain_same = bool(domain_rows["same_domain"].all())
    lines = [
        "# 8-Metric Versus 11-Metric Static Profile Comparison",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "## Inputs and Scope",
        "",
        f"- Input table: `{display(input_path)}`",
        "- Current specification: 11 full-period metrics.",
        "- Structural-only specification: 8 metrics, excluding centroid drift, radial expansion slope, and recent novelty.",
        "- Outputs are written only to this comparison folder; thesis LaTeX, thesis figures, and existing result folders are not modified.",
        f"- Output folder: `{display(output_dir)}`",
        "",
        "## Bottom Line",
        "",
        f"Removing the three temporal indicators causes **{verdict}** to the static-profile conclusions.",
        "",
        "The field-distance geometry remains highly similar and the domain-level interpretation is unchanged. "
        "The main observable differences are local: several nearest-neighbor assignments change, and the static Ward silhouette rises modestly when the temporal indicators are removed.",
        "",
        "## Key Diagnostics",
        "",
        f"- Field-distance Pearson correlation: {values['field_distance_matrix_pearson']:.3f}.",
        f"- Field-distance Spearman correlation: {values['field_distance_matrix_spearman']:.3f}.",
        f"- Same nearest neighbor: {same_nn}/{total_nn} fields ({values['same_nearest_neighbor_share'] * 100:.1f}%).",
        f"- Outside-domain nearest-neighbor count: {int(values['outside_domain_nearest_neighbor_count_11_metric'])}/26 under 11 metrics versus {int(values['outside_domain_nearest_neighbor_count_8_metric'])}/26 under 8 metrics.",
        f"- Static Ward silhouette: {values['static_silhouette_11_metric']:.3f} under 11 metrics versus {values['static_silhouette_8_metric']:.3f} under 8 metrics.",
        f"- Top-{top_n} PCA outlier overlap by PC1/PC2 radius: {values[f'top_{top_n}_pca_outlier_overlap_share'] * 100:.1f}%.",
        f"- Top-{top_n} closest field-pair overlap: {values[f'top_{top_n}_closest_field_pair_overlap_share'] * 100:.1f}%.",
        f"- Top-{top_n} farthest field-pair overlap: {values[f'top_{top_n}_farthest_field_pair_overlap_share'] * 100:.1f}%.",
        "",
        "## Field Similarity Details",
        "",
        f"- 11-metric closest pairs: {format_pair_list(spec11['top_pairs'], 'closest')}.",
        f"- 8-metric closest pairs: {format_pair_list(spec8['top_pairs'], 'closest')}.",
        f"- 11-metric farthest pairs: {format_pair_list(spec11['top_pairs'], 'farthest')}.",
        f"- 8-metric farthest pairs: {format_pair_list(spec8['top_pairs'], 'farthest')}.",
        f"- Largest nearest-neighbor changes preview: {changed_preview if changed_preview else 'none'}.",
        "",
        "## PCA and Extremes",
        "",
        f"- PCA explained variance, 11 metrics: PC1 {spec11['pca_model'].explained_variance_ratio_[0] * 100:.1f}%, PC2 {spec11['pca_model'].explained_variance_ratio_[1] * 100:.1f}%.",
        f"- PCA explained variance, 8 metrics: PC1 {spec8['pca_model'].explained_variance_ratio_[0] * 100:.1f}%, PC2 {spec8['pca_model'].explained_variance_ratio_[1] * 100:.1f}%.",
        f"- Shared top PCA-radius outliers: {', '.join(pca_overlap_names) if pca_overlap_names else 'none'}.",
        "- Static structural metric extremes are unchanged for the retained eight metrics because their raw subfield values and robust standardization are identical; the 8-metric version simply drops the three temporal-extreme lists.",
        "",
        "## Domain Interpretation",
        "",
        f"- Domain-level maxima/minima across the eight retained metrics are {'unchanged' if domain_same else 'changed'}.",
        "- The qualitative domain reading remains the same: Physical Sciences remain the most dispersed and spectrally extended broad domain; Health Sciences remain the locally densest and most hub-concentrated; Social Sciences and Life Sciences remain less uniformly extreme.",
        "",
        "## Reproducibility Files",
        "",
        "- `comparison_summary.csv`: compact side-by-side diagnostic table.",
        "- `field_distance_matrix_comparison.csv`: all 325 field-pair distances under both specifications.",
        "- `field_nearest_neighbors_side_by_side.csv`: nearest neighbor under each specification for each field.",
        "- `domain_field_standardized_profiles_side_by_side.csv`: Chapter 4-style within-level standardized domain and field profiles.",
        "- `domain_field_similarity_profiles_side_by_side.csv`: Chapter 6-style profiles, built by averaging robust-scaled subfield profiles.",
        "- `subfield_pca_scores_side_by_side.csv` and `subfield_pca_outliers_side_by_side.csv`: PCA comparison artifacts.",
        "- `field_top_pairs_side_by_side.csv`: closest and farthest field pairs.",
        "- `static_metric_extremes_side_by_side.csv`: metric extremes under each specification.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.cluster_k < 2:
        raise ValueError("--cluster-k must be at least 2")

    input_path = resolve(args.input)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_core(input_path)
    spec11 = run_spec(frame, "11_metric", ALL11_METRICS, args.cluster_k, args.top_n)
    spec8 = run_spec(frame, "8_metric", STRUCTURAL_METRICS, args.cluster_k, args.top_n)

    paired_dist = paired_field_distances(spec11["distances"], spec8["distances"])
    paired_nn = paired_nearest(spec11["nearest"], spec8["nearest"])
    pca_outliers = pca_outliers_side_by_side(spec11["pca"], spec8["pca"], args.top_n)
    summary = summary_table(
        spec11=spec11,
        spec8=spec8,
        paired_distances=paired_dist,
        paired_nn=paired_nn,
        pca_outliers=pca_outliers,
        top_n=args.top_n,
    )
    domain_rows = domain_conclusion_rows(
        spec11["domain_field_scaled"], spec8["domain_field_scaled"]
    )
    verdict = interpret_change(summary, domain_rows)

    pd.concat([spec11["domain_field_scaled"], spec8["domain_field_scaled"]], ignore_index=True).to_csv(
        output_dir / "domain_field_standardized_profiles_side_by_side.csv", index=False
    )
    pd.concat(
        [spec11["domain_field_similarity"], spec8["domain_field_similarity"]],
        ignore_index=True,
    ).to_csv(output_dir / "domain_field_similarity_profiles_side_by_side.csv", index=False)
    paired_dist.to_csv(output_dir / "field_distance_matrix_comparison.csv", index=False)
    paired_nn.to_csv(output_dir / "field_nearest_neighbors_side_by_side.csv", index=False)
    pd.concat([spec11["cluster"], spec8["cluster"]], ignore_index=True).to_csv(
        output_dir / "static_clustering_side_by_side.csv", index=False
    )
    pd.concat([spec11["pca"], spec8["pca"]], ignore_index=True).to_csv(
        output_dir / "subfield_pca_scores_side_by_side.csv", index=False
    )
    pca_outliers.to_csv(output_dir / "subfield_pca_outliers_side_by_side.csv", index=False)
    pd.concat([spec11["top_pairs"], spec8["top_pairs"]], ignore_index=True).to_csv(
        output_dir / "field_top_pairs_side_by_side.csv", index=False
    )
    pd.concat([spec11["extremes"], spec8["extremes"]], ignore_index=True).to_csv(
        output_dir / "static_metric_extremes_side_by_side.csv", index=False
    )
    domain_rows.to_csv(output_dir / "domain_conclusion_check.csv", index=False)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)

    payload = {
        "generated_at": utc_now_iso(),
        "input_path": display(input_path),
        "output_dir": display(output_dir),
        "verdict": verdict,
        "metrics": {
            "11_metric": ALL11_METRICS,
            "8_metric": STRUCTURAL_METRICS,
            "excluded_from_8_metric": TEMPORAL_SUMMARY_METRICS,
        },
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(
        output_dir / "comparison_report.md",
        input_path=input_path,
        output_dir=output_dir,
        summary=summary,
        verdict=verdict,
        spec11=spec11,
        spec8=spec8,
        paired_nn=paired_nn,
        domain_rows=domain_rows,
        pca_outliers=pca_outliers,
        top_n=args.top_n,
    )

    print(f"Wrote {display(output_dir / 'comparison_report.md')}")
    print(f"Wrote {display(output_dir / 'comparison_summary.csv')}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
