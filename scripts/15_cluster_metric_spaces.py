from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric_clustering import (
    ClusteringConfig,
    SUPPORTED_SPACES,
    build_metric_space,
    compare_cluster_partitions,
    comparison_summary_markdown,
    metric_space_umap_coordinates,
    metric_space_umap_output_frame,
    plot_agreement_heatmap,
    plot_cluster_profile_heatmap,
    plot_dendrogram,
    plot_metric_space_umap,
    plot_pca_scatter,
    run_metric_space_clustering,
    space_summary_markdown,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster OpenAlex subfields in UMAP, embedding, and combined metric spaces."
    )
    parser.add_argument(
        "--umap-metrics-path",
        default="data/processed/subfield_morphology_metrics.parquet",
    )
    parser.add_argument(
        "--embedding-metrics-path",
        default="data/processed/subfield_embedding_space_metrics.parquet",
    )
    parser.add_argument(
        "--low-information-path",
        default="outputs/analysis/metric_distributions/low_information_metrics.csv",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/metric_clustering",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        choices=sorted(SUPPORTED_SPACES),
        default=["embedding_only", "umap_only", "combined"],
    )
    parser.add_argument("--max-missing-share", type=float, default=0.30)
    parser.add_argument("--min-unique-values", type=int, default=4)
    parser.add_argument("--pca-variance-threshold", type=float, default=0.90)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--default-k", type=int, default=5)
    parser.add_argument("--scaler", choices=["standard", "robust"], default="standard")
    parser.add_argument(
        "--combined-strategy",
        choices=["block_pca", "raw"],
        default="block_pca",
    )
    parser.add_argument("--random-state", type=int, default=42)
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
        return str(path)


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing metric clustering outputs without "
            f"--overwrite: {display_path(output_root)}"
        )
    output_root.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_space_outputs(
    *,
    output_dir: Path,
    output_root: Path,
    space_data,
    result,
    config: ClusteringConfig,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_csv = output_dir / "cluster_assignments.csv"
    assignments_parquet = output_dir / "cluster_assignments.parquet"
    preprocessing_path = output_dir / "preprocessing_report.csv"
    pca_scores_path = output_dir / "pca_scores.csv"
    pca_loadings_path = output_dir / "pca_loadings.csv"
    pca_variance_path = output_dir / "pca_explained_variance.csv"
    quality_path = output_dir / "cluster_quality_by_k.csv"
    profiles_path = output_dir / "cluster_profiles.csv"
    representatives_path = output_dir / "cluster_representatives.csv"
    dendrogram_path = output_dir / "dendrogram.png"
    pca_scatter_path = output_dir / "pca_scatter_clusters.png"
    metric_umap_path = output_dir / "metric_space_umap_clusters.png"
    metric_umap_coordinates_csv = (
        output_root / f"{space_data.name}_metric_umap_coordinates.csv"
    )
    metric_umap_coordinates_parquet = (
        output_root / f"{space_data.name}_metric_umap_coordinates.parquet"
    )
    profile_heatmap_path = output_dir / "cluster_profile_heatmap.png"
    summary_md_path = output_dir / "summary.md"
    summary_json_path = output_dir / "summary.json"

    result.assignments.to_csv(assignments_csv, index=False)
    result.assignments.to_parquet(assignments_parquet, index=False)
    space_data.preprocessing_report.to_csv(preprocessing_path, index=False)
    pd.concat(
        [space_data.metadata.reset_index(drop=True), space_data.pca_scores],
        axis=1,
    ).to_csv(pca_scores_path, index=False)
    space_data.pca_loadings.to_csv(pca_loadings_path, index=False)
    space_data.pca_explained_variance.to_csv(pca_variance_path, index=False)
    result.quality_by_k.to_csv(quality_path, index=False)
    result.cluster_profiles.to_csv(profiles_path, index=False)
    result.representatives.to_csv(representatives_path, index=False)

    plot_dendrogram(
        space_data.feature_matrix,
        dendrogram_path,
        title=f"Ward Dendrogram: {space_data.name}",
    )
    plot_pca_scatter(
        space_data.pca_scores,
        result.assignments,
        pca_scatter_path,
        cluster_column=result.main_cluster_column,
        title=f"PCA Scores: {space_data.name}",
    )
    metric_umap_coordinates, coordinate_warning = metric_space_umap_coordinates(
        space_data.feature_matrix,
        random_state=config.random_state,
    )
    metric_umap_frame = metric_space_umap_output_frame(
        result.assignments,
        metric_umap_coordinates,
        cluster_column=result.main_cluster_column,
    )
    metric_umap_frame.to_csv(metric_umap_coordinates_csv, index=False)
    metric_umap_frame.to_parquet(metric_umap_coordinates_parquet, index=False)
    plot_warning = plot_metric_space_umap(
        space_data.feature_matrix,
        result.assignments,
        metric_umap_path,
        cluster_column=result.main_cluster_column,
        random_state=config.random_state,
        coordinates=metric_umap_coordinates,
    )
    umap_warning = coordinate_warning or plot_warning
    if umap_warning:
        space_data.warnings.append(umap_warning)
    plot_cluster_profile_heatmap(
        result.cluster_profiles,
        profile_heatmap_path,
        title=f"Cluster Profile Heatmap: {space_data.name}",
    )

    retained = int(space_data.preprocessing_report["retained"].sum())
    dropped = int((~space_data.preprocessing_report["retained"]).sum())
    summary_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "space": space_data.name,
        "n_subfields": int(len(space_data.metadata)),
        "n_retained_metrics": retained,
        "n_dropped_metrics": dropped,
        "main_cluster_column": result.main_cluster_column,
        "effective_default_k": int(result.effective_default_k),
        "component_blocks": space_data.component_blocks,
        "warnings": space_data.warnings,
        "config": {
            "max_missing_share": config.max_missing_share,
            "min_unique_values": config.min_unique_values,
            "pca_variance_threshold": config.pca_variance_threshold,
            "k_min": config.k_min,
            "k_max": config.k_max,
            "default_k": config.default_k,
            "scaler": config.scaler,
            "combined_strategy": config.combined_strategy,
            "random_state": config.random_state,
        },
    }
    write_json(summary_json_path, summary_payload)
    summary_md_path.write_text(
        space_summary_markdown(space_data, result, config=config),
        encoding="utf-8",
    )
    return umap_warning


def write_comparison_outputs(
    *,
    output_dir: Path,
    assignments_by_space: dict[str, pd.DataFrame],
    cluster_column: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    agreement, contingencies = compare_cluster_partitions(
        assignments_by_space,
        cluster_column=cluster_column,
    )
    agreement_path = output_dir / "cluster_partition_agreement.csv"
    agreement.to_csv(agreement_path, index=False)
    contingency_paths = {
        ("embedding_only", "umap_only"): output_dir / "cluster_contingency_embedding_vs_umap.csv",
        ("embedding_only", "combined"): output_dir / "cluster_contingency_embedding_vs_combined.csv",
        ("umap_only", "combined"): output_dir / "cluster_contingency_umap_vs_combined.csv",
    }
    for pair, path in contingency_paths.items():
        contingency = contingencies.get(pair, pd.DataFrame())
        contingency.to_csv(path)
    heatmap_path = output_dir / "cluster_agreement_heatmap.png"
    plot_agreement_heatmap(agreement, heatmap_path)
    summary_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cluster_column": cluster_column,
        "n_comparisons": int(len(agreement)),
        "agreement": agreement.to_dict("records"),
    }
    write_json(output_dir / "summary.json", summary_payload)
    (output_dir / "summary.md").write_text(
        comparison_summary_markdown(agreement),
        encoding="utf-8",
    )
    return summary_payload


def main() -> None:
    args = parse_args()
    config = ClusteringConfig(
        max_missing_share=args.max_missing_share,
        min_unique_values=args.min_unique_values,
        pca_variance_threshold=args.pca_variance_threshold,
        k_min=args.k_min,
        k_max=args.k_max,
        default_k=args.default_k,
        scaler=args.scaler,
        random_state=args.random_state,
        combined_strategy=args.combined_strategy,
    )
    validate_config(config)

    umap_path = resolve_path(args.umap_metrics_path)
    embedding_path = resolve_path(args.embedding_metrics_path)
    low_info_path = resolve_path(args.low_information_path)
    output_root = resolve_path(args.output_root)
    ensure_output_root(output_root, args.overwrite)

    if not umap_path.exists():
        raise FileNotFoundError(
            f"Missing UMAP metric table: {display_path(umap_path)}. "
            "Run scripts/11_compute_subfield_morphology_metrics.py first."
        )
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Missing embedding metric table: {display_path(embedding_path)}. "
            "Run scripts/12_compute_subfield_embedding_space_metrics.py first."
        )

    low_info_count = None
    if low_info_path.exists():
        low_info_count = int(len(pd.read_csv(low_info_path)))
        print(
            f"Found {low_info_count} low-information metric diagnostics at "
            f"{display_path(low_info_path)}. They are reported but not automatically dropped."
        )

    umap_metrics = pd.read_parquet(umap_path)
    embedding_metrics = pd.read_parquet(embedding_path)

    assignments_by_space: dict[str, pd.DataFrame] = {}
    result_columns: dict[str, str] = {}
    warnings: list[str] = []
    for space in args.spaces:
        print(f"Clustering metric space: {space}")
        space_data = build_metric_space(
            space,
            umap_metrics,
            embedding_metrics,
            config=config,
        )
        result = run_metric_space_clustering(space_data, config=config)
        warning = write_space_outputs(
            output_dir=output_root / space,
            output_root=output_root,
            space_data=space_data,
            result=result,
            config=config,
        )
        if warning:
            warnings.append(f"{space}: {warning}")
        assignments_by_space[space] = result.assignments
        result_columns[space] = result.main_cluster_column
        print(
            f"{space}: {len(result.assignments)} subfields, "
            f"{result.main_cluster_column}"
        )

    comparison_payload: dict[str, Any] | None = None
    requested_all_spaces = {"embedding_only", "umap_only", "combined"}.issubset(
        set(assignments_by_space)
    )
    if requested_all_spaces:
        unique_columns = set(result_columns.values())
        if len(unique_columns) == 1:
            comparison_payload = write_comparison_outputs(
                output_dir=output_root / "comparison",
                assignments_by_space=assignments_by_space,
                cluster_column=next(iter(unique_columns)),
            )
        else:
            warnings.append(
                "partition comparison skipped because spaces used different main cluster columns"
            )

    root_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "umap_metrics_path": display_path(umap_path),
            "embedding_metrics_path": display_path(embedding_path),
            "low_information_path": display_path(low_info_path)
            if low_info_path.exists()
            else None,
        },
        "low_information_metric_count": low_info_count,
        "spaces": list(args.spaces),
        "main_cluster_columns": result_columns,
        "comparison_written": comparison_payload is not None,
        "warnings": warnings,
        "config": {
            "max_missing_share": config.max_missing_share,
            "min_unique_values": config.min_unique_values,
            "pca_variance_threshold": config.pca_variance_threshold,
            "k_min": config.k_min,
            "k_max": config.k_max,
            "default_k": config.default_k,
            "scaler": config.scaler,
            "combined_strategy": config.combined_strategy,
            "random_state": config.random_state,
        },
    }
    write_json(output_root / "summary.json", root_summary)
    print(f"Wrote {display_path(output_root)}")


if __name__ == "__main__":
    main()
