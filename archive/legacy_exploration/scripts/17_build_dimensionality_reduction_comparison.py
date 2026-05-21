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

from src.cluster_interpretation import main_cluster_column, require_existing_paths
from src.dimensionality_reduction_comparison import (
    DimensionalityReductionConfig,
    METHOD_ORDER,
    comparison_summary_markdown,
    fit_projection,
    plot_projection_grid,
    projection_coordinate_frame,
    projection_quality_rows,
)
from src.metric_clustering import ClusteringConfig, build_metric_space, validate_config


SPACES = ["umap_only", "embedding_only", "combined"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare six dimensionality-reduction methods across metric feature sets."
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
        "--clustering-dir",
        default="outputs/analysis/metric_clustering",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/dimensionality_reduction_comparison",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--mds-n-init", type=int, default=4)
    parser.add_argument("--mds-max-iter", type=int, default=1000)
    parser.add_argument("--phate-decay", type=int, default=40)
    parser.add_argument("--max-missing-share", type=float, default=0.30)
    parser.add_argument("--min-unique-values", type=int, default=4)
    parser.add_argument("--pca-variance-threshold", type=float, default=0.90)
    parser.add_argument("--scaler", choices=["standard", "robust"], default="standard")
    parser.add_argument(
        "--combined-strategy",
        choices=["block_pca", "raw"],
        default="block_pca",
    )
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


def ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing dimensionality-reduction comparison "
            f"outputs without --overwrite: {display_path(path)}"
        )
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_cluster_assignments(
    clustering_dir: Path,
    *,
    space: str,
) -> tuple[pd.DataFrame, str]:
    assignments_path = clustering_dir / space / "cluster_assignments.csv"
    require_existing_paths({f"{space} cluster assignments": assignments_path})
    assignments = pd.read_csv(assignments_path)
    assignments["subfield_id"] = assignments["subfield_id"].astype(str)
    summary_path = clustering_dir / space / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    cluster_column = main_cluster_column(assignments, summary=summary)
    return assignments, cluster_column


def read_representatives(clustering_dir: Path, *, space: str) -> pd.DataFrame:
    path = clustering_dir / space / "cluster_representatives.csv"
    if not path.exists():
        return pd.DataFrame(columns=["cluster_id", "rank", "subfield_id"])
    frame = pd.read_csv(path)
    if "subfield_id" in frame.columns:
        frame["subfield_id"] = frame["subfield_id"].astype(str)
    return frame


def metadata_with_cluster(
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    cluster_column: str,
) -> pd.DataFrame:
    base = metadata.copy()
    base["subfield_id"] = base["subfield_id"].astype(str)
    joined = base.merge(
        assignments[["subfield_id", cluster_column]],
        on="subfield_id",
        how="left",
    )
    if joined[cluster_column].isna().any():
        missing = int(joined[cluster_column].isna().sum())
        raise ValueError(f"{missing} subfields have no cluster label for {cluster_column}")
    return joined


def main() -> None:
    args = parse_args()
    umap_path = resolve_path(args.umap_metrics_path)
    embedding_path = resolve_path(args.embedding_metrics_path)
    clustering_dir = resolve_path(args.clustering_dir)
    output_dir = resolve_path(args.output_dir)

    require_existing_paths(
        {
            "UMAP morphology metrics": umap_path,
            "embedding-space metrics": embedding_path,
            "metric clustering directory": clustering_dir,
        }
    )
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    clustering_config = ClusteringConfig(
        max_missing_share=args.max_missing_share,
        min_unique_values=args.min_unique_values,
        pca_variance_threshold=args.pca_variance_threshold,
        scaler=args.scaler,
        random_state=args.random_state,
        combined_strategy=args.combined_strategy,
    )
    validate_config(clustering_config)
    dr_config = DimensionalityReductionConfig(
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        tsne_perplexity=args.tsne_perplexity,
        mds_n_init=args.mds_n_init,
        mds_max_iter=args.mds_max_iter,
        phate_decay=args.phate_decay,
    )

    umap_metrics = pd.read_parquet(umap_path)
    embedding_metrics = pd.read_parquet(embedding_path)

    output_paths: list[str] = []
    quality_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    config_by_space: dict[str, Any] = {}

    for space in SPACES:
        print(f"Building dimensionality-reduction comparison: {space}")
        space_data = build_metric_space(
            space,
            umap_metrics,
            embedding_metrics,
            config=clustering_config,
        )
        assignments, cluster_column = read_cluster_assignments(
            clustering_dir,
            space=space,
        )
        metadata = metadata_with_cluster(
            space_data.metadata,
            assignments,
            cluster_column=cluster_column,
        )
        representatives = read_representatives(clustering_dir, space=space)
        coordinate_frames: list[pd.DataFrame] = []
        for method in METHOD_ORDER:
            coords, warning, params = fit_projection(
                space_data.feature_matrix,
                method=method,
                config=dr_config,
            )
            if warning:
                warnings.append(f"{space}/{method}: {warning}")
            coordinate_frames.append(
                projection_coordinate_frame(
                    space=space,
                    method=method,
                    metadata=metadata,
                    coordinates=coords,
                    cluster_column=cluster_column,
                    warning=warning,
                )
            )
            quality_rows.append(
                projection_quality_rows(
                    space=space,
                    method=method,
                    features=space_data.feature_matrix,
                    coordinates=coords,
                    labels=metadata[cluster_column],
                    warning=warning,
                    params=params,
                    config=dr_config,
                )
            )

        coords_long = pd.concat(coordinate_frames, ignore_index=True)
        coords_csv = output_dir / f"dr_coords_{space}.csv"
        coords_parquet = output_dir / f"dr_coords_{space}.parquet"
        coords_long.to_csv(coords_csv, index=False)
        coords_long.to_parquet(coords_parquet, index=False)
        output_paths.extend([display_path(coords_csv), display_path(coords_parquet)])

        figure_path = output_dir / f"dr_comparison_{space}_grid.png"
        plot_projection_grid(
            space=space,
            coordinates=coords_long,
            representatives=representatives,
            output_path=figure_path,
        )
        output_paths.append(display_path(figure_path))

        config_by_space[space] = {
            "n_subfields": int(len(space_data.metadata)),
            "n_features_in_clustering_feature_matrix": int(space_data.feature_matrix.shape[1]),
            "cluster_column": cluster_column,
            "component_blocks": space_data.component_blocks,
            "retained_metric_count": int(space_data.preprocessing_report["retained"].sum()),
            "dropped_metric_count": int((~space_data.preprocessing_report["retained"]).sum()),
        }

    quality = pd.DataFrame(quality_rows)
    if not quality.empty:
        quality["params_json"] = quality["params_json"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    quality_path = output_dir / "dr_projection_quality.csv"
    quality.to_csv(quality_path, index=False)
    output_paths.append(display_path(quality_path))

    config_path = output_dir / "dr_comparison_config.json"
    config_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "umap_metrics_path": display_path(umap_path),
            "embedding_metrics_path": display_path(embedding_path),
            "clustering_dir": display_path(clustering_dir),
        },
        "spaces": SPACES,
        "methods": METHOD_ORDER,
        "clustering_preprocessing_config": {
            "max_missing_share": clustering_config.max_missing_share,
            "min_unique_values": clustering_config.min_unique_values,
            "pca_variance_threshold": clustering_config.pca_variance_threshold,
            "scaler": clustering_config.scaler,
            "combined_strategy": clustering_config.combined_strategy,
            "random_state": clustering_config.random_state,
        },
        "dimensionality_reduction_config": {
            "random_state": dr_config.random_state,
            "n_neighbors": dr_config.n_neighbors,
            "tsne_perplexity": dr_config.tsne_perplexity,
            "mds_n_init": dr_config.mds_n_init,
            "mds_max_iter": dr_config.mds_max_iter,
            "phate_decay": dr_config.phate_decay,
        },
        "space_details": config_by_space,
        "warnings": warnings,
        "output_paths": output_paths,
    }
    write_json(config_path, config_payload)
    output_paths.append(display_path(config_path))

    summary_path = output_dir / "dr_comparison_summary.md"
    summary_path.write_text(
        comparison_summary_markdown(
            quality=quality,
            warnings=warnings,
            config=dr_config,
        ),
        encoding="utf-8",
    )
    output_paths.append(display_path(summary_path))

    summary_json_path = output_dir / "summary.json"
    write_json(
        summary_json_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_paths": output_paths + [display_path(summary_json_path)],
            "warnings": warnings,
            "spaces": SPACES,
            "methods": METHOD_ORDER,
        },
    )
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
