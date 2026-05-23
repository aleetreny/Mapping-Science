from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)

from src.morphological_typologies import TYPOLOGY_COLORS, TYPOLOGY_ORDER
from src.temporal_common import display_path, ensure_outputs_do_not_exist, utc_now_iso


EPS = 1e-12
MORPHOLOGICAL_SILHOUETTE = 0.13259888874044007
FIGURE_STEM = "fig_09_centroid_embedding_clustering_sensitivity"

OUTPUT_FILENAMES: dict[str, str] = {
    "centroid_embeddings": "subfield_centroid_embeddings_unit.npy",
    "centroid_metadata": "subfield_centroid_metadata.csv",
    "model_comparison": "centroid_cluster_model_comparison.csv",
    "selected_solution": "centroid_cluster_solution_selected.csv",
    "assignments": "centroid_cluster_assignments.csv",
    "domain_alignment": "centroid_domain_alignment.csv",
    "morphology_comparison": "centroid_vs_morphology_comparison.csv",
    "summary_md": "centroid_embedding_clustering_summary.md",
    "summary_json": "summary.json",
}


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}


def figure_paths(output_dir: str | Path, figure_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    return {
        "output_png": output_dir / f"{FIGURE_STEM}.png",
        "output_pdf": output_dir / f"{FIGURE_STEM}.pdf",
        "thesis_png": figure_dir / f"{FIGURE_STEM}.png",
    }


def existing_outputs(output_dir: str | Path, figure_dir: str | Path) -> list[Path]:
    paths = list(output_paths(output_dir).values())
    paths.extend(figure_paths(output_dir, figure_dir).values())
    return [path for path in paths if path.exists()]


def ensure_centroid_embedding_clustering_outputs(
    output_dir: str | Path,
    figure_dir: str | Path,
    *,
    overwrite: bool,
    root: Path,
) -> None:
    ensure_outputs_do_not_exist(
        existing_outputs(output_dir, figure_dir),
        overwrite=overwrite,
        root=root,
        description="centroid embedding clustering outputs",
    )


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe_norms = np.where(norms > EPS, norms, 1.0)
    return values / safe_norms


def _subfield_key(series: pd.Series) -> pd.Series:
    return series.astype(str)


def compute_subfield_centroids(
    analysis_index: pd.DataFrame,
    embedding_matrix: np.ndarray,
    typology_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    required_index = {
        "analysis_row_id",
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        "publication_year",
    }
    missing_index = sorted(required_index - set(analysis_index.columns))
    if missing_index:
        raise ValueError(f"analysis index is missing columns: {missing_index}")
    required_assignments = {
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        "typology_id",
        "typology_label",
    }
    missing_assignments = sorted(required_assignments - set(typology_assignments.columns))
    if missing_assignments:
        raise ValueError(f"typology assignments are missing columns: {missing_assignments}")

    assignments = typology_assignments.copy()
    assignments["subfield_id_key"] = _subfield_key(assignments["subfield_id"])
    target_keys = set(assignments["subfield_id_key"])

    index = analysis_index.copy()
    index["subfield_id_key"] = _subfield_key(index["subfield_id"])
    index = index.loc[index["subfield_id_key"].isin(target_keys)].copy()
    if "main_analysis_eligible" in index.columns:
        index = index.loc[index["main_analysis_eligible"].fillna(False).astype(bool)].copy()
    if index.empty:
        raise ValueError("No embedding-index rows match the morphology typology subfields.")

    grouped = {key: group for key, group in index.groupby("subfield_id_key", sort=False)}
    centroids: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for row in assignments.itertuples(index=False):
        key = str(row.subfield_id_key)
        group = grouped.get(key)
        if group is None or group.empty:
            raise ValueError(f"No embedding rows found for subfield_id={row.subfield_id}")
        row_ids = np.sort(group["analysis_row_id"].to_numpy(dtype=np.int64, copy=True))
        vectors = np.asarray(embedding_matrix[row_ids], dtype=np.float32)
        normalized_vectors = l2_normalize_rows(vectors)
        centroid = normalized_vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= EPS:
            unit_centroid = centroid
        else:
            unit_centroid = centroid / centroid_norm
        centroids.append(unit_centroid.astype(np.float32, copy=False))
        records.append(
            {
                "subfield_id": row.subfield_id,
                "subfield_display_name": row.subfield_display_name,
                "field_display_name": row.field_display_name,
                "domain_display_name": row.domain_display_name,
                "typology_id": row.typology_id,
                "typology_label": row.typology_label,
                "n_centroid_papers": int(len(row_ids)),
                "year_min": int(group["publication_year"].min()),
                "year_max": int(group["publication_year"].max()),
                "mean_l2_centroid_norm_before_unit_scaling": centroid_norm,
            }
        )

    metadata = pd.DataFrame(records)
    centroid_matrix = np.vstack(centroids).astype(np.float32)
    return metadata, centroid_matrix


def _canonicalize_cluster_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    counts = pd.Series(labels).value_counts()
    ordered = sorted(counts.index.tolist(), key=lambda value: (-int(counts[value]), str(value)))
    mapping = {old: idx + 1 for idx, old in enumerate(ordered) if int(old) != -1}
    if -1 in counts.index:
        mapping[-1] = 0
    return np.array([mapping[value] for value in labels], dtype=int)


def _cluster_size_summary(labels: np.ndarray) -> tuple[int, int, str]:
    counts = pd.Series(labels).value_counts().sort_values()
    return int(counts.min()), int(counts.max()), ";".join(str(int(value)) for value in counts)


def domain_purity(cluster_labels: np.ndarray, domain_labels: np.ndarray) -> float:
    frame = pd.DataFrame({"cluster": cluster_labels, "domain": domain_labels})
    majority_counts = frame.groupby("cluster")["domain"].agg(lambda values: values.value_counts().iloc[0])
    return float(majority_counts.sum() / len(frame))


def _safe_silhouette(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    distance_metric: str,
) -> float:
    unique = np.unique(labels)
    if len(unique) <= 1 or len(unique) >= len(labels):
        return float("nan")
    metric = "cosine" if distance_metric == "cosine" else "euclidean"
    try:
        return float(silhouette_score(matrix, labels, metric=metric))
    except ValueError:
        return float("nan")


def _evaluate_solution(
    *,
    solution_id: str,
    algorithm: str,
    linkage_method: str | None,
    distance_metric: str,
    k: int | None,
    labels: np.ndarray,
    matrix: np.ndarray,
    morph_labels: np.ndarray,
    domain_labels: np.ndarray,
    status: str = "ok",
) -> dict[str, Any]:
    labels = _canonicalize_cluster_labels(labels)
    min_size, max_size, sizes = _cluster_size_summary(labels)
    return {
        "solution_id": solution_id,
        "algorithm": algorithm,
        "linkage": linkage_method or "",
        "distance_metric": distance_metric,
        "k": int(k) if k is not None else np.nan,
        "silhouette": _safe_silhouette(matrix, labels, distance_metric=distance_metric),
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "cluster_sizes_sorted": sizes,
        "ari_vs_morphology": float(adjusted_rand_score(morph_labels, labels)),
        "ami_vs_morphology": float(adjusted_mutual_info_score(morph_labels, labels)),
        "ari_vs_domain": float(adjusted_rand_score(domain_labels, labels)),
        "ami_vs_domain": float(adjusted_mutual_info_score(domain_labels, labels)),
        "domain_purity": domain_purity(labels, domain_labels),
        "status": status,
    }


def _hierarchical_labels(
    matrix: np.ndarray,
    *,
    k: int,
    method: str,
    distance_metric: str,
) -> np.ndarray:
    if method == "ward":
        if distance_metric != "euclidean":
            raise ValueError("Ward linkage requires Euclidean distance.")
        tree = linkage(matrix, method="ward", metric="euclidean")
    else:
        condensed = pdist(matrix, metric=distance_metric)
        tree = linkage(condensed, method=method)
    return fcluster(tree, t=k, criterion="maxclust")


def explore_centroid_cluster_models(
    centroid_matrix: np.ndarray,
    *,
    morph_labels: np.ndarray,
    domain_labels: np.ndarray,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    labels_by_solution: dict[str, np.ndarray] = {}

    def add_candidate(
        *,
        solution_id: str,
        algorithm: str,
        linkage_method: str | None,
        distance_metric: str,
        k: int,
        labels: np.ndarray,
    ) -> None:
        canonical = _canonicalize_cluster_labels(labels)
        labels_by_solution[solution_id] = canonical
        rows.append(
            _evaluate_solution(
                solution_id=solution_id,
                algorithm=algorithm,
                linkage_method=linkage_method,
                distance_metric=distance_metric,
                k=k,
                labels=canonical,
                matrix=centroid_matrix,
                morph_labels=morph_labels,
                domain_labels=domain_labels,
            )
        )

    for k in range(3, 11):
        for method, metric in [
            ("ward", "euclidean"),
            ("average", "euclidean"),
            ("average", "cosine"),
        ]:
            solution_id = f"centroid_{method}_{metric}_k{k}"
            try:
                labels = _hierarchical_labels(
                    centroid_matrix,
                    k=k,
                    method=method,
                    distance_metric=metric,
                )
                add_candidate(
                    solution_id=solution_id,
                    algorithm="hierarchical",
                    linkage_method=method,
                    distance_metric=metric,
                    k=k,
                    labels=labels,
                )
            except Exception as exc:
                rows.append(_error_row(solution_id, "hierarchical", method, metric, k, exc))

        solution_id = f"centroid_kmeans_euclidean_k{k}"
        try:
            labels = KMeans(n_clusters=k, n_init=50, random_state=random_seed).fit_predict(
                centroid_matrix
            )
            add_candidate(
                solution_id=solution_id,
                algorithm="kmeans",
                linkage_method=None,
                distance_metric="euclidean",
                k=k,
                labels=labels,
            )
        except Exception as exc:
            rows.append(_error_row(solution_id, "kmeans", None, "euclidean", k, exc))

        solution_id = f"centroid_spectral_neighbors_k{k}"
        try:
            labels = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=12,
                assign_labels="kmeans",
                random_state=random_seed,
                n_init=20,
            ).fit_predict(centroid_matrix)
            add_candidate(
                solution_id=solution_id,
                algorithm="spectral",
                linkage_method=None,
                distance_metric="nearest_neighbors",
                k=k,
                labels=labels,
            )
        except Exception as exc:
            rows.append(_error_row(solution_id, "spectral", None, "nearest_neighbors", k, exc))

    if importlib.util.find_spec("hdbscan") is None:
        rows.append(
            {
                "solution_id": "centroid_hdbscan_euclidean_skipped",
                "algorithm": "hdbscan",
                "linkage": "",
                "distance_metric": "euclidean",
                "k": np.nan,
                "silhouette": np.nan,
                "min_cluster_size": np.nan,
                "max_cluster_size": np.nan,
                "cluster_sizes_sorted": "",
                "ari_vs_morphology": np.nan,
                "ami_vs_morphology": np.nan,
                "ari_vs_domain": np.nan,
                "ami_vs_domain": np.nan,
                "domain_purity": np.nan,
                "status": "skipped: hdbscan is not installed",
            }
        )
    else:
        import hdbscan  # type: ignore[import-not-found]

        for min_cluster_size in [5, 10, 15]:
            solution_id = f"centroid_hdbscan_euclidean_min{min_cluster_size}"
            try:
                labels = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    metric="euclidean",
                ).fit_predict(centroid_matrix)
                canonical = _canonicalize_cluster_labels(labels)
                labels_by_solution[solution_id] = canonical
                n_clusters = len(set(canonical) - {0})
                rows.append(
                    _evaluate_solution(
                        solution_id=solution_id,
                        algorithm="hdbscan",
                        linkage_method=None,
                        distance_metric="euclidean",
                        k=n_clusters,
                        labels=canonical,
                        matrix=centroid_matrix,
                        morph_labels=morph_labels,
                        domain_labels=domain_labels,
                    )
                )
            except Exception as exc:
                rows.append(_error_row(solution_id, "hdbscan", None, "euclidean", None, exc))

    return pd.DataFrame(rows), labels_by_solution


def _error_row(
    solution_id: str,
    algorithm: str,
    linkage_method: str | None,
    distance_metric: str,
    k: int | None,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "solution_id": solution_id,
        "algorithm": algorithm,
        "linkage": linkage_method or "",
        "distance_metric": distance_metric,
        "k": int(k) if k is not None else np.nan,
        "silhouette": np.nan,
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


def select_centroid_solution(model_comparison: pd.DataFrame) -> pd.Series:
    ok = model_comparison.loc[model_comparison["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No centroid clustering candidate completed successfully.")
    ok["k"] = pd.to_numeric(ok["k"], errors="coerce")
    ok["silhouette"] = pd.to_numeric(ok["silhouette"], errors="coerce")
    if "cluster_sizes_sorted" in ok.columns:
        parsed_totals = ok["cluster_sizes_sorted"].fillna("").map(
            lambda value: sum(int(part) for part in str(value).split(";") if part)
        )
        n_subfields = int(parsed_totals.max()) if parsed_totals.max() > 0 else 241
    else:
        n_subfields = 241
    balanced = ok.loc[
        (ok["min_cluster_size"] >= 5)
        & (ok["max_cluster_size"] <= max(30, int(round(0.75 * n_subfields))))
        & ok["silhouette"].notna()
    ].copy()
    pool = balanced if not balanced.empty else ok.loc[ok["silhouette"].notna()].copy()
    if pool.empty:
        pool = ok
    pool["k_distance_from_5"] = (pool["k"] - 5).abs()
    return pool.sort_values(
        ["silhouette", "domain_purity", "ami_vs_domain", "k_distance_from_5"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def centroid_cluster_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in assignments.groupby("centroid_cluster_id", sort=True):
        domain_counts = group["domain_display_name"].value_counts()
        typology_counts = group["typology_id"].value_counts()
        rows.append(
            {
                "centroid_cluster_id": int(cluster_id),
                "n_subfields": int(len(group)),
                "dominant_domain": str(domain_counts.index[0]),
                "dominant_domain_share": float(domain_counts.iloc[0] / len(group)),
                "dominant_morphology_typology": str(typology_counts.index[0]),
                "dominant_morphology_typology_share": float(typology_counts.iloc[0] / len(group)),
            }
        )
    return pd.DataFrame(rows)


def selected_pairwise_tables(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain_counts = (
        assignments.groupby(["centroid_cluster_id", "domain_display_name"], as_index=False)
        .size()
        .rename(columns={"size": "n_subfields"})
    )
    totals = domain_counts.groupby("centroid_cluster_id")["n_subfields"].transform("sum")
    domain_counts["share_within_centroid_cluster"] = domain_counts["n_subfields"] / totals

    morphology_counts = (
        assignments.groupby(["centroid_cluster_id", "typology_id", "typology_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_subfields"})
    )
    totals = morphology_counts.groupby("centroid_cluster_id")["n_subfields"].transform("sum")
    morphology_counts["share_within_centroid_cluster"] = morphology_counts["n_subfields"] / totals
    return domain_counts, morphology_counts


def centroid_pca_scores(
    centroid_metadata: pd.DataFrame,
    centroid_matrix: np.ndarray,
) -> tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(centroid_matrix)
    frame = centroid_metadata.copy()
    frame["centroid_pca_1"] = scores[:, 0]
    frame["centroid_pca_2"] = scores[:, 1]
    frame["centroid_pca_1_explained_variance"] = float(pca.explained_variance_ratio_[0])
    frame["centroid_pca_2_explained_variance"] = float(pca.explained_variance_ratio_[1])
    return frame, pca


def plot_sensitivity_figure(
    *,
    morphology_pca: pd.DataFrame,
    centroid_assignments: pd.DataFrame,
    selected_solution: pd.Series,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), constrained_layout=True)

    ax = axes[0]
    for typology_id, label in TYPOLOGY_ORDER:
        subset = morphology_pca.loc[morphology_pca["typology_id"] == typology_id]
        ax.scatter(
            subset["pca_1"],
            subset["pca_2"],
            s=30,
            c=TYPOLOGY_COLORS[typology_id],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            label=f"{typology_id} ({len(subset)})",
        )
    pca1 = float(morphology_pca["pca_1_explained_variance"].iloc[0]) * 100
    pca2 = float(morphology_pca["pca_2_explained_variance"].iloc[0]) * 100
    ax.set_title("A. Eleven-metric morphology", loc="left", fontsize=12)
    ax.set_xlabel(f"Profile PCA 1 ({pca1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"Profile PCA 2 ({pca2:.1f}%)", fontsize=10)
    ax.grid(True, alpha=0.18)
    ax.legend(title="Morphology", title_fontsize=8, fontsize=7.5, frameon=False)

    ax = axes[1]
    cluster_summary = centroid_cluster_summary(centroid_assignments)
    colors = plt.get_cmap("tab10")
    for idx, row in enumerate(cluster_summary.itertuples(index=False)):
        subset = centroid_assignments.loc[
            centroid_assignments["centroid_cluster_id"] == row.centroid_cluster_id
        ]
        domain_label = (
            str(row.dominant_domain).replace(" Sciences", "")
            if float(row.dominant_domain_share) >= 0.5
            else "mixed"
        )
        label = (
            f"C{int(row.centroid_cluster_id)} ({int(row.n_subfields)}), "
            f"{domain_label}"
        )
        ax.scatter(
            subset["centroid_pca_1"],
            subset["centroid_pca_2"],
            s=32,
            c=[colors(idx % 10)],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
    pca1 = float(centroid_assignments["centroid_pca_1_explained_variance"].iloc[0]) * 100
    pca2 = float(centroid_assignments["centroid_pca_2_explained_variance"].iloc[0]) * 100
    selected_k = int(selected_solution["k"])
    selected_silhouette = float(selected_solution["silhouette"])
    method_label = str(selected_solution["algorithm"])
    if str(selected_solution.get("linkage", "")):
        method_label = f"{selected_solution['linkage']} {selected_solution['distance_metric']}"
    ax.set_title(f"B. Centroid embeddings ({method_label}, k={selected_k})", loc="left", fontsize=12)
    ax.set_xlabel(f"Centroid PCA 1 ({pca1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"Centroid PCA 2 ({pca2:.1f}%)", fontsize=10)
    ax.text(
        0.02,
        0.02,
        f"silhouette = {selected_silhouette:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.86, "pad": 3},
    )
    ax.grid(True, alpha=0.18)
    ax.legend(title="Centroid cluster", title_fontsize=8, fontsize=7.1, frameon=False)

    paths = figure_paths(output_dir, figure_dir)
    fig.savefig(paths["output_png"], dpi=240, bbox_inches="tight")
    fig.savefig(paths["output_pdf"], bbox_inches="tight")
    fig.savefig(paths["thesis_png"], dpi=240, bbox_inches="tight")
    plt.close(fig)


def render_summary_markdown(
    *,
    selected_solution: pd.Series,
    model_comparison: pd.DataFrame,
    assignments: pd.DataFrame,
    morphology_silhouette: float,
    root: Path,
    output_dir: Path,
    figure_dir: Path,
) -> str:
    ok = model_comparison.loc[model_comparison["status"] == "ok"].copy()
    ok = ok.sort_values("silhouette", ascending=False, kind="mergesort")
    top_rows = ok.head(8)
    selected_id = str(selected_solution["solution_id"])
    selected_silhouette = float(selected_solution["silhouette"])
    if selected_silhouette >= morphology_silhouette + 0.05:
        interpretation = (
            "Centroid embeddings cluster more clearly than the eleven-metric "
            "morphology profiles. This supports the narrower claim that semantic "
            "location has a stronger topic/location structure than morphology."
        )
    else:
        interpretation = (
            "Centroid embedding clustering is also weak. This supports a continuous "
            "reading of both semantic position and morphology."
        )
    selected_summary = centroid_cluster_summary(assignments)

    lines = [
        "# Centroid-Embedding Clustering Sensitivity",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "## Method",
        "",
        (
            "Each of the 241 analysis subfields is represented by the mean of its "
            "paper-level SPECTER2 vectors after L2 normalization. The resulting "
            "subfield centroid is L2-normalized again before clustering, so "
            "Euclidean and cosine distances are geometrically consistent on the "
            "unit sphere."
        ),
        "",
        "The exploration covers Ward and average-link hierarchical clustering, "
        "k-means, and nearest-neighbor spectral clustering for k=3,...,10. "
        "HDBSCAN is skipped unless the package is available in the environment.",
        "",
        "## Selected Diagnostic Solution",
        "",
        f"- Solution: `{selected_id}`",
        f"- Silhouette: {selected_silhouette:.3f}",
        f"- Cluster sizes: {selected_solution['cluster_sizes_sorted']}",
        f"- ARI vs morphology typology: {float(selected_solution['ari_vs_morphology']):.3f}",
        f"- AMI vs morphology typology: {float(selected_solution['ami_vs_morphology']):.3f}",
        f"- ARI vs OpenAlex domains: {float(selected_solution['ari_vs_domain']):.3f}",
        f"- AMI vs OpenAlex domains: {float(selected_solution['ami_vs_domain']):.3f}",
        f"- Domain purity: {float(selected_solution['domain_purity']):.3f}",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "The morphology typology remains a descriptive partition of profile shape; "
        "the centroid analysis asks a different question about semantic location.",
        "",
        "## Cluster Summary",
        "",
        dataframe_to_markdown(selected_summary),
        "",
        "## Top Completed Candidates",
        "",
        dataframe_to_markdown(
            top_rows[
            [
                "solution_id",
                "silhouette",
                "min_cluster_size",
                "max_cluster_size",
                "ari_vs_morphology",
                "ami_vs_domain",
                "domain_purity",
            ]
            ],
            float_digits=3,
        ),
        "",
        "## Outputs",
        "",
        f"- Output directory: `{display_path(output_dir, root=root)}`",
        f"- Thesis figure: `{display_path(figure_paths(output_dir, figure_dir)['thesis_png'], root=root)}`",
    ]
    return "\n".join(lines) + "\n"


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def build_centroid_embedding_clustering_outputs(
    *,
    analysis_index: pd.DataFrame,
    embedding_matrix: np.ndarray,
    typology_assignments_path: Path,
    morphology_pca_path: Path,
    output_dir: Path,
    figure_dir: Path,
    root: Path,
    random_seed: int,
    index_path_label: str,
    embedding_matrix_path_label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    typology_assignments = pd.read_csv(typology_assignments_path)
    morphology_pca = pd.read_csv(morphology_pca_path)

    centroid_metadata, centroid_matrix = compute_subfield_centroids(
        analysis_index,
        embedding_matrix,
        typology_assignments,
    )
    morph_labels = centroid_metadata["typology_id"].to_numpy()
    domain_labels = centroid_metadata["domain_display_name"].to_numpy()
    model_comparison, labels_by_solution = explore_centroid_cluster_models(
        centroid_matrix,
        morph_labels=morph_labels,
        domain_labels=domain_labels,
        random_seed=random_seed,
    )
    selected_solution = select_centroid_solution(model_comparison)
    selected_solution_id = str(selected_solution["solution_id"])
    selected_labels = labels_by_solution[selected_solution_id]

    centroid_scores, centroid_pca = centroid_pca_scores(centroid_metadata, centroid_matrix)
    assignments = centroid_scores.copy()
    assignments["centroid_cluster_id"] = selected_labels
    assignments["centroid_cluster_label"] = [
        f"C{int(value)}" if int(value) > 0 else "noise" for value in selected_labels
    ]

    paths = output_paths(output_dir)
    np.save(paths["centroid_embeddings"], centroid_matrix)
    centroid_metadata.to_csv(paths["centroid_metadata"], index=False)
    model_comparison.to_csv(paths["model_comparison"], index=False)
    pd.DataFrame([selected_solution]).to_csv(paths["selected_solution"], index=False)
    assignments.to_csv(paths["assignments"], index=False)

    domain_table, morphology_table = selected_pairwise_tables(assignments)
    domain_table.to_csv(paths["domain_alignment"], index=False)
    morphology_table.to_csv(paths["morphology_comparison"], index=False)

    plot_sensitivity_figure(
        morphology_pca=morphology_pca,
        centroid_assignments=assignments,
        selected_solution=selected_solution,
        output_dir=output_dir,
        figure_dir=figure_dir,
    )

    summary_md = render_summary_markdown(
        selected_solution=selected_solution,
        model_comparison=model_comparison,
        assignments=assignments,
        morphology_silhouette=MORPHOLOGICAL_SILHOUETTE,
        root=root,
        output_dir=output_dir,
        figure_dir=figure_dir,
    )
    paths["summary_md"].write_text(summary_md, encoding="utf-8")

    summary = {
        "generated_at": utc_now_iso(),
        "analysis_index_path": index_path_label,
        "embedding_matrix_path": embedding_matrix_path_label,
        "n_subfields": int(len(centroid_metadata)),
        "embedding_dim": int(centroid_matrix.shape[1]),
        "morphology_silhouette": MORPHOLOGICAL_SILHOUETTE,
        "selected_solution": selected_solution.to_dict(),
        "centroid_pca_explained_variance": [
            float(centroid_pca.explained_variance_ratio_[0]),
            float(centroid_pca.explained_variance_ratio_[1]),
        ],
        "output_paths": {
            key: display_path(value, root=root) for key, value in paths.items()
        },
        "figure_paths": {
            key: display_path(value, root=root)
            for key, value in figure_paths(output_dir, figure_dir).items()
        },
    }
    write_json(paths["summary_json"], summary)
    return summary
