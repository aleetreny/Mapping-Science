from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.clustering_sensitivity_common import (
    bootstrap_subsample_stability,
    canonicalize_cluster_labels,
    cluster_size_summary,
    dataframe_to_markdown,
    error_row,
    evaluate_labels,
    labels_hierarchical_features,
    labels_kmeans,
    labels_spectral_features,
    save_figure,
    standardize_matrix,
    write_json,
)
from src.morphological_typologies import TYPOLOGY_COLORS, TYPOLOGY_ORDER
from src.temporal_common import (
    WINDOW_STRUCTURAL_METRICS,
    display_path,
    ensure_outputs_do_not_exist,
    linear_slope,
    utc_now_iso,
)


STATIC_STRUCTURAL_METRICS = list(WINDOW_STRUCTURAL_METRICS)


TEMPORAL_OUTPUTS: dict[str, str] = {
    "features": "trajectory_feature_matrix.csv",
    "model_comparison": "trajectory_cluster_model_comparison.csv",
    "selected_solution": "trajectory_cluster_solution_selected.csv",
    "assignments": "trajectory_cluster_assignments.csv",
    "profile_summary": "trajectory_cluster_profile_summary.csv",
    "domain_composition": "trajectory_cluster_domain_composition.csv",
    "stability_summary": "trajectory_cluster_stability_summary.csv",
    "summary_md": "trajectory_clustering_summary.md",
    "summary_json": "summary.json",
}

TEMPORAL_FIGURE_STEMS = [
    "fig_c_temporal_cluster_quality",
    "fig_c_temporal_cluster_map",
    "fig_c_temporal_typology_profile_heatmap",
]


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {key: output_dir / filename for key, filename in TEMPORAL_OUTPUTS.items()}


def figure_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}
    for stem in TEMPORAL_FIGURE_STEMS:
        paths[f"{stem}_png"] = output_dir / f"{stem}.png"
        paths[f"{stem}_pdf"] = output_dir / f"{stem}.pdf"
    return paths


def existing_outputs(output_dir: str | Path) -> list[Path]:
    paths = list(output_paths(output_dir).values())
    paths.extend(figure_paths(output_dir).values())
    return [path for path in paths if path.exists()]


def ensure_temporal_trajectory_outputs(
    output_dir: str | Path,
    *,
    overwrite: bool,
    root: Path,
) -> None:
    ensure_outputs_do_not_exist(
        existing_outputs(output_dir),
        overwrite=overwrite,
        root=root,
        description="temporal trajectory clustering outputs",
    )


def load_complete_window_metrics(window_metrics_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(window_metrics_path)
    required = {
        "subfield_id",
        "subfield_display_name",
        "field_display_name",
        "domain_display_name",
        "window_index",
        "window_label",
        "metric_status",
        *STATIC_STRUCTURAL_METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Temporal window metrics are missing columns: {missing}")
    frame = frame.copy()
    frame["subfield_id"] = frame["subfield_id"].astype(str)
    completed = frame.loc[frame["metric_status"].eq("completed")].copy()
    for metric in STATIC_STRUCTURAL_METRICS:
        completed[f"z_{metric}"] = completed.groupby(lambda _: True)[metric].transform(
            lambda values: (values - values.mean()) / values.std(ddof=0)
        )
    window_count = completed.groupby("subfield_id")["window_label"].nunique()
    complete_ids = set(window_count.loc[window_count == 5].index)
    dropped = (
        frame.loc[~frame["subfield_id"].isin(complete_ids), ["subfield_id", "subfield_display_name"]]
        .drop_duplicates()
        .sort_values("subfield_id")
    )
    complete = completed.loc[completed["subfield_id"].isin(complete_ids)].copy()
    return complete, dropped


def temporal_feature_matrices(
    window_metrics: pd.DataFrame,
    *,
    centroid_path_metrics_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, list[str]]]:
    metadata = (
        window_metrics[
            [
                "subfield_id",
                "subfield_display_name",
                "field_display_name",
                "domain_display_name",
            ]
        ]
        .drop_duplicates("subfield_id")
        .sort_values("subfield_id")
        .reset_index(drop=True)
    )
    ids = metadata["subfield_id"].tolist()
    z_columns = [f"z_{metric}" for metric in STATIC_STRUCTURAL_METRICS]
    wide = {
        metric: window_metrics.pivot(index="subfield_id", columns="window_index", values=f"z_{metric}")
        .reindex(ids)
        .sort_index(axis=1)
        for metric in STATIC_STRUCTURAL_METRICS
    }

    delta_columns = [f"delta_{metric}" for metric in STATIC_STRUCTURAL_METRICS]
    delta = np.column_stack(
        [wide[metric][4].to_numpy(dtype=float) - wide[metric][0].to_numpy(dtype=float) for metric in STATIC_STRUCTURAL_METRICS]
    )

    slope_columns = [f"slope_{metric}" for metric in STATIC_STRUCTURAL_METRICS]
    x = np.arange(5, dtype=float)
    slopes = np.column_stack(
        [
            np.array([linear_slope(x, wide[metric].loc[subfield_id].to_numpy(dtype=float)) for subfield_id in ids])
            for metric in STATIC_STRUCTURAL_METRICS
        ]
    )

    trajectory_columns: list[str] = []
    trajectory_parts: list[np.ndarray] = []
    for metric in STATIC_STRUCTURAL_METRICS:
        for window_index in range(5):
            trajectory_columns.append(f"trajectory_{metric}_w{window_index}")
            trajectory_parts.append(wide[metric][window_index].to_numpy(dtype=float))
    trajectory = np.column_stack(trajectory_parts)

    volatility_columns = [f"volatility_{metric}" for metric in STATIC_STRUCTURAL_METRICS]
    volatility = np.column_stack(
        [wide[metric].std(axis=1, ddof=0).to_numpy(dtype=float) for metric in STATIC_STRUCTURAL_METRICS]
    )
    dynamic_summary = np.column_stack([delta, slopes, volatility])
    dynamic_columns = [*delta_columns, *slope_columns, *volatility_columns]
    if centroid_path_metrics_path is not None and centroid_path_metrics_path.exists():
        path_metrics = pd.read_parquet(centroid_path_metrics_path)
        path_metrics["subfield_id"] = path_metrics["subfield_id"].astype(str)
        extra_cols = [
            "total_path_length_euclidean_normalized",
            "early_late_drift_euclidean_normalized",
            "directionality_ratio_euclidean",
        ]
        available = [column for column in extra_cols if column in path_metrics.columns]
        if available:
            aligned = path_metrics.set_index("subfield_id").reindex(ids)[available]
            dynamic_summary = np.column_stack([dynamic_summary, aligned.to_numpy(dtype=float)])
            dynamic_columns.extend(available)

    raw = {
        "first_last_delta": delta,
        "linear_slope": slopes,
        "full_trajectory": trajectory,
        "dynamic_summary": dynamic_summary,
    }
    columns = {
        "first_last_delta": delta_columns,
        "linear_slope": slope_columns,
        "full_trajectory": trajectory_columns,
        "dynamic_summary": dynamic_columns,
    }
    matrices = {name: standardize_matrix(matrix) for name, matrix in raw.items()}
    for name, matrix in matrices.items():
        if not np.isfinite(matrix).all():
            raise ValueError(f"Temporal representation {name} contains non-finite values.")
    return metadata, matrices, columns


def attach_typology_metadata(metadata: pd.DataFrame, typology_assignments_path: Path) -> pd.DataFrame:
    assignments = pd.read_csv(typology_assignments_path)
    assignments["subfield_id"] = assignments["subfield_id"].astype(str)
    required = ["subfield_id", "typology_id", "typology_label"]
    missing = sorted(set(required) - set(assignments.columns))
    if missing:
        raise ValueError(f"Typology assignments are missing columns: {missing}")
    merged = metadata.merge(assignments[required], on="subfield_id", how="left")
    if merged["typology_id"].isna().any():
        raise ValueError("Some temporal subfields are missing morphology typology labels.")
    return merged


def explore_temporal_models(
    matrices: dict[str, np.ndarray],
    *,
    morph_labels: np.ndarray,
    domain_labels: np.ndarray,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    labels_by_solution: dict[str, np.ndarray] = {}

    def add_candidate(representation: str, algorithm: str, metric: str, k: int, labels: np.ndarray) -> None:
        solution_id = f"temporal_{representation}_{algorithm}_{metric}_k{k}"
        labels_by_solution[solution_id] = canonicalize_cluster_labels(labels)
        rows.append(
            evaluate_labels(
                solution_id=solution_id,
                representation=representation,
                algorithm=algorithm,
                distance_metric=metric,
                k=k,
                labels=labels,
                feature_matrix=matrices[representation],
                morph_labels=morph_labels,
                domain_labels=domain_labels,
                silhouette_metric="correlation" if metric == "correlation" else "euclidean",
            )
        )

    for representation, matrix in matrices.items():
        for k in range(3, 11):
            for method, metric in [
                ("ward", "euclidean"),
                ("average", "euclidean"),
                ("average", "correlation"),
            ]:
                try:
                    add_candidate(
                        representation,
                        f"hierarchical_{method}",
                        metric,
                        k,
                        labels_hierarchical_features(matrix, k=k, method=method, metric=metric),
                    )
                except Exception as exc:
                    rows.append(
                        error_row(
                            solution_id=f"temporal_{representation}_hierarchical_{method}_{metric}_k{k}",
                            representation=representation,
                            algorithm=f"hierarchical_{method}",
                            distance_metric=metric,
                            k=k,
                            exc=exc,
                        )
                    )
            try:
                add_candidate(
                    representation,
                    "kmeans",
                    "euclidean",
                    k,
                    labels_kmeans(matrix, k=k, random_seed=random_seed),
                )
            except Exception as exc:
                rows.append(
                    error_row(
                        solution_id=f"temporal_{representation}_kmeans_euclidean_k{k}",
                        representation=representation,
                        algorithm="kmeans",
                        distance_metric="euclidean",
                        k=k,
                        exc=exc,
                    )
                )
            try:
                add_candidate(
                    representation,
                    "spectral",
                    "nearest_neighbors",
                    k,
                    labels_spectral_features(matrix, k=k, random_seed=random_seed),
                )
            except Exception as exc:
                rows.append(
                    error_row(
                        solution_id=f"temporal_{representation}_spectral_nearest_neighbors_k{k}",
                        representation=representation,
                        algorithm="spectral",
                        distance_metric="nearest_neighbors",
                        k=k,
                        exc=exc,
                    )
                )

    if importlib.util.find_spec("sklearn_extra") is None:
        rows.append(_skipped_row("temporal_kmedoids_skipped", "all", "kmedoids", "euclidean"))
    if importlib.util.find_spec("hdbscan") is None:
        rows.append(_skipped_row("temporal_hdbscan_skipped", "all", "hdbscan", "euclidean"))
    return pd.DataFrame(rows), labels_by_solution


def _skipped_row(solution_id: str, representation: str, algorithm: str, metric: str) -> dict[str, Any]:
    return {
        "solution_id": solution_id,
        "representation": representation,
        "algorithm": algorithm,
        "distance_metric": metric,
        "k": np.nan,
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
        "status": f"skipped: {algorithm} is not installed",
    }


def select_temporal_solution(model_comparison: pd.DataFrame, *, n_subfields: int) -> pd.Series:
    ok = model_comparison.loc[model_comparison["status"].eq("ok")].copy()
    ok["silhouette"] = pd.to_numeric(ok["silhouette"], errors="coerce")
    balanced = ok.loc[
        (ok["silhouette"].notna())
        & (ok["min_cluster_size"] >= 8)
        & (ok["max_cluster_size"] <= max(30, int(round(0.70 * n_subfields))))
    ].copy()
    pool = balanced if not balanced.empty else ok.loc[ok["silhouette"].notna()].copy()
    pool["k_distance_from_5"] = (pd.to_numeric(pool["k"], errors="coerce") - 5).abs()
    return pool.sort_values(
        ["silhouette", "domain_purity", "k_distance_from_5"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]


def temporal_profile_summary(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    matrices: dict[str, np.ndarray],
    columns: dict[str, list[str]],
) -> pd.DataFrame:
    feature_names = [*columns["first_last_delta"], *columns["linear_slope"]]
    values = np.column_stack([matrices["first_last_delta"], matrices["linear_slope"]])
    feature_frame = pd.DataFrame(values, columns=feature_names)
    feature_frame["trajectory_cluster_id"] = labels
    rows: list[dict[str, Any]] = []
    for cluster_id, group in feature_frame.groupby("trajectory_cluster_id", sort=True):
        domain_counts = metadata.loc[group.index, "domain_display_name"].value_counts()
        row: dict[str, Any] = {
            "trajectory_cluster_id": int(cluster_id),
            "n_subfields": int(len(group)),
            "dominant_domain": str(domain_counts.index[0]),
            "dominant_domain_share": float(domain_counts.iloc[0] / len(group)),
        }
        means = group[feature_names].mean().sort_values(key=lambda values: values.abs(), ascending=False)
        row["largest_mean_features"] = "; ".join(f"{idx}={value:+.2f}" for idx, value in means.head(4).items())
        for feature in feature_names:
            row[feature] = float(group[feature].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def domain_composition(metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    frame = metadata.copy()
    frame["trajectory_cluster_id"] = labels
    counts = (
        frame.groupby(["trajectory_cluster_id", "domain_display_name"], as_index=False)
        .size()
        .rename(columns={"size": "n_subfields"})
    )
    totals = counts.groupby("trajectory_cluster_id")["n_subfields"].transform("sum")
    counts["share_within_cluster"] = counts["n_subfields"] / totals
    return counts


def plot_temporal_quality(model_comparison: pd.DataFrame, *, output_dir: Path) -> None:
    ok = model_comparison.loc[model_comparison["status"].eq("ok")].copy()
    ok["k"] = pd.to_numeric(ok["k"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for representation, group in ok.groupby("representation"):
        best = group.sort_values("silhouette", ascending=False).groupby("k", as_index=False).head(1)
        ax.plot(best["k"], best["silhouette"], marker="o", linewidth=1.6, label=representation.replace("_", " "))
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Best silhouette by representation")
    ax.set_title("Temporal trajectory clustering remains modest", loc="left", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_c_temporal_cluster_quality", output_dir=output_dir)


def plot_temporal_map(
    metadata: pd.DataFrame,
    selected_matrix: np.ndarray,
    labels: np.ndarray,
    selected_solution: pd.Series,
    *,
    output_dir: Path,
) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(selected_matrix)
    pca_frame = metadata.copy()
    pca_frame["trajectory_cluster_id"] = labels
    pca_frame["trajectory_pca_1"] = scores[:, 0]
    pca_frame["trajectory_pca_2"] = scores[:, 1]
    pca_frame["trajectory_pca_1_explained_variance"] = float(pca.explained_variance_ratio_[0])
    pca_frame["trajectory_pca_2_explained_variance"] = float(pca.explained_variance_ratio_[1])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cmap = plt.get_cmap("tab10")
    for idx, cluster_id in enumerate(sorted(np.unique(labels))):
        subset = pca_frame.loc[pca_frame["trajectory_cluster_id"] == cluster_id]
        ax.scatter(
            subset["trajectory_pca_1"],
            subset["trajectory_pca_2"],
            s=34,
            color=cmap(idx % 10),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            label=f"C{int(cluster_id)} ({len(subset)})",
        )
    ax.set_title(
        f"Temporal trajectory clusters ({selected_solution['representation']}, k={int(selected_solution['k'])})",
        loc="left",
        fontsize=12,
    )
    ax.set_xlabel(f"Trajectory PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"Trajectory PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.grid(True, alpha=0.18)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_c_temporal_cluster_map", output_dir=output_dir)
    return pca_frame


def plot_temporal_profile_heatmap(profile_summary: pd.DataFrame, *, output_dir: Path) -> None:
    compact_cols = [
        column
        for column in profile_summary.columns
        if column.startswith("delta_embedding_") or column.startswith("slope_embedding_")
    ]
    matrix = profile_summary[compact_cols].to_numpy(dtype=float)
    labels = [f"C{int(row.trajectory_cluster_id)} ({int(row.n_subfields)})" for row in profile_summary.itertuples()]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    vmax = max(1.0, float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    short_cols = [column.replace("embedding_", "").replace("_to_", "_").replace("distance_", "dist_") for column in compact_cols]
    ax.set_xticks(np.arange(len(short_cols)))
    ax.set_xticklabels(short_cols, rotation=55, ha="right", fontsize=7)
    ax.set_title("Selected temporal cluster mean change and slope profiles", loc="left", fontsize=12)
    ax.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Mean standardized feature")
    fig.tight_layout()
    save_figure(fig, "fig_c_temporal_typology_profile_heatmap", output_dir=output_dir)


def stability_for_selected(
    selected_solution: pd.Series,
    selected_labels: np.ndarray,
    matrices: dict[str, np.ndarray],
    *,
    random_seed: int,
) -> pd.DataFrame:
    representation = str(selected_solution["representation"])
    algorithm = str(selected_solution["algorithm"])
    metric = str(selected_solution["distance_metric"])
    k = int(selected_solution["k"])
    matrix = matrices[representation]

    def label_function(indices: np.ndarray) -> np.ndarray:
        subset = matrix[indices]
        if algorithm.startswith("hierarchical_"):
            method = algorithm.removeprefix("hierarchical_")
            return labels_hierarchical_features(subset, k=k, method=method, metric=metric)
        if algorithm == "kmeans":
            return labels_kmeans(subset, k=k, random_seed=random_seed)
        if algorithm == "spectral":
            return labels_spectral_features(subset, k=k, random_seed=random_seed)
        raise ValueError(f"Unsupported selected algorithm for stability: {algorithm}")

    return bootstrap_subsample_stability(
        matrix,
        selected_labels,
        label_function=label_function,
        random_seed=random_seed,
        n_iterations=100,
    )


def render_summary(
    *,
    selected_solution: pd.Series,
    model_comparison: pd.DataFrame,
    stability: pd.DataFrame,
    dropped: pd.DataFrame,
) -> str:
    top = (
        model_comparison.loc[model_comparison["status"].eq("ok")]
        .sort_values("silhouette", ascending=False)
        .head(8)
    )
    dropped_text = "None"
    if not dropped.empty:
        dropped_text = "; ".join(
            f"{row.subfield_id} {row.subfield_display_name}" for row in dropped.itertuples(index=False)
        )
    lines = [
        "# Temporal-Trajectory Clustering Sensitivity",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "## Inputs",
        "",
        "Windowed structural metrics use the eight static morphology metrics over five-year windows.",
        f"Subfields excluded for incomplete trajectories: {dropped_text}.",
        "",
        "## Selected Diagnostic Solution",
        "",
        f"- Solution: `{selected_solution['solution_id']}`",
        f"- Representation: {selected_solution['representation']}",
        f"- Silhouette: {float(selected_solution['silhouette']):.3f}",
        f"- Cluster sizes: {selected_solution['cluster_sizes_sorted']}",
        f"- AMI vs morphology typology: {float(selected_solution['ami_vs_morphology']):.3f}",
        f"- AMI vs OpenAlex domains: {float(selected_solution['ami_vs_domain']):.3f}",
        f"- Domain purity: {float(selected_solution['domain_purity']):.3f}",
        f"- Mean subsample ARI: {float(stability['adjusted_rand_index'].mean()):.3f}",
        "",
        "## Interpretation",
        "",
        "Temporal trajectory clustering adds a dynamic view but remains a weak-to-moderate diagnostic partition. It should be read as a sensitivity check rather than a replacement for the Chapter 9 morphology typology.",
        "",
        "## Top Candidates",
        "",
        dataframe_to_markdown(
            top[
                [
                    "solution_id",
                    "silhouette",
                    "min_cluster_size",
                    "max_cluster_size",
                    "ami_vs_morphology",
                    "ami_vs_domain",
                    "domain_purity",
                ]
            ]
        ),
    ]
    return "\n".join(lines) + "\n"


def build_temporal_trajectory_clustering_outputs(
    *,
    window_metrics_path: Path,
    centroid_path_metrics_path: Path | None,
    typology_assignments_path: Path,
    output_dir: Path,
    root: Path,
    random_seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    window_metrics, dropped = load_complete_window_metrics(window_metrics_path)
    metadata, matrices, columns = temporal_feature_matrices(
        window_metrics,
        centroid_path_metrics_path=centroid_path_metrics_path,
    )
    metadata = attach_typology_metadata(metadata, typology_assignments_path)
    morph_labels = metadata["typology_id"].to_numpy()
    domain_labels = metadata["domain_display_name"].to_numpy()
    model_comparison, labels_by_solution = explore_temporal_models(
        matrices,
        morph_labels=morph_labels,
        domain_labels=domain_labels,
        random_seed=random_seed,
    )
    selected_solution = select_temporal_solution(model_comparison, n_subfields=len(metadata))
    selected_labels = labels_by_solution[str(selected_solution["solution_id"])]
    selected_representation = str(selected_solution["representation"])

    pca_frame = plot_temporal_map(
        metadata,
        matrices[selected_representation],
        selected_labels,
        selected_solution,
        output_dir=output_dir,
    )
    profile_summary = temporal_profile_summary(metadata, selected_labels, matrices, columns)
    composition = domain_composition(metadata, selected_labels)
    stability = stability_for_selected(
        selected_solution,
        selected_labels,
        matrices,
        random_seed=random_seed,
    )
    plot_temporal_quality(model_comparison, output_dir=output_dir)
    plot_temporal_profile_heatmap(profile_summary, output_dir=output_dir)

    assignments = metadata.copy()
    assignments["trajectory_cluster_id"] = selected_labels
    assignments["trajectory_cluster_label"] = [f"D{int(value)}" for value in selected_labels]
    assignments = assignments.merge(
        pca_frame[
            [
                "subfield_id",
                "trajectory_pca_1",
                "trajectory_pca_2",
                "trajectory_pca_1_explained_variance",
                "trajectory_pca_2_explained_variance",
            ]
        ],
        on="subfield_id",
        how="left",
    )

    paths = output_paths(output_dir)
    feature_frame = metadata[["subfield_id", "subfield_display_name", "domain_display_name", "typology_id"]].copy()
    for representation, matrix in matrices.items():
        for idx in range(matrix.shape[1]):
            feature_frame[f"{representation}_{idx:02d}"] = matrix[:, idx]
    feature_frame.to_csv(paths["features"], index=False)
    model_comparison.to_csv(paths["model_comparison"], index=False)
    pd.DataFrame([selected_solution]).to_csv(paths["selected_solution"], index=False)
    assignments.to_csv(paths["assignments"], index=False)
    profile_summary.to_csv(paths["profile_summary"], index=False)
    composition.to_csv(paths["domain_composition"], index=False)
    stability.to_csv(paths["stability_summary"], index=False)
    paths["summary_md"].write_text(
        render_summary(
            selected_solution=selected_solution,
            model_comparison=model_comparison,
            stability=stability,
            dropped=dropped,
        ),
        encoding="utf-8",
    )
    summary = {
        "generated_at": utc_now_iso(),
        "window_metrics_path": display_path(window_metrics_path, root=root),
        "centroid_path_metrics_path": (
            display_path(centroid_path_metrics_path, root=root)
            if centroid_path_metrics_path is not None and centroid_path_metrics_path.exists()
            else None
        ),
        "n_subfields": int(len(metadata)),
        "dropped_incomplete_subfields": dropped.to_dict("records"),
        "representations": {name: int(matrix.shape[1]) for name, matrix in matrices.items()},
        "selected_solution": selected_solution.to_dict(),
        "stability_mean_ari": float(stability["adjusted_rand_index"].mean()),
        "stability_mean_ami": float(stability["adjusted_mutual_information"].mean()),
        "output_paths": {key: display_path(path, root=root) for key, path in paths.items()},
        "figure_paths": {key: display_path(path, root=root) for key, path in figure_paths(output_dir).items()},
    }
    write_json(paths["summary_json"], summary)
    return summary
