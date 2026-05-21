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
    UMAPTuningConfig,
    fit_umap_tuned_projection,
    plot_umap_tuning_grid,
    umap_tuning_coordinate_frame,
    umap_tuning_quality_row,
    umap_tuning_summary_markdown,
)
from src.metric_clustering import ClusteringConfig, build_metric_space, validate_config


SPACES = ["umap_only", "embedding_only", "combined"]


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
            "Refusing to overwrite existing UMAP tuning outputs without "
            f"--overwrite: {display_path(path)}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune UMAP hyperparameters across metric feature sets."
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
        default="outputs/analysis/dimensionality_reduction_comparison/umap_tuning",
    )
    parser.add_argument(
        "--n-neighbors-values",
        type=int,
        nargs="+",
        default=[5, 10, 15, 30, 50],
    )
    parser.add_argument(
        "--min-dist-values",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.3, 0.6],
    )
    parser.add_argument("--random-state", type=int, default=42)
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
    tuning_config = UMAPTuningConfig(
        random_state=args.random_state,
        n_neighbors_values=tuple(args.n_neighbors_values),
        min_dist_values=tuple(args.min_dist_values),
    )

    umap_metrics = pd.read_parquet(umap_path)
    embedding_metrics = pd.read_parquet(embedding_path)

    output_paths: list[str] = []
    quality_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    space_details: dict[str, Any] = {}

    for space in SPACES:
        print(f"Tuning UMAP projection: {space}")
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
        coordinate_frames: list[pd.DataFrame] = []
        for n_neighbors in tuning_config.n_neighbors_values:
            for min_dist in tuning_config.min_dist_values:
                coords, warning, _ = fit_umap_tuned_projection(
                    space_data.feature_matrix,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=tuning_config.random_state,
                )
                if warning:
                    warnings.append(f"{space}/k={n_neighbors}/min_dist={min_dist}: {warning}")
                coordinate_frames.append(
                    umap_tuning_coordinate_frame(
                        space=space,
                        metadata=metadata,
                        coordinates=coords,
                        cluster_column=cluster_column,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        warning=warning,
                    )
                )
                quality_rows.append(
                    umap_tuning_quality_row(
                        space=space,
                        features=space_data.feature_matrix,
                        coordinates=coords,
                        labels=metadata[cluster_column],
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        warning=warning,
                    )
                )

        coords_long = pd.concat(coordinate_frames, ignore_index=True)
        quality_space = pd.DataFrame(
            [row for row in quality_rows if row["feature_set"] == space]
        )
        coords_csv = output_dir / f"umap_tuning_coords_{space}.csv"
        coords_parquet = output_dir / f"umap_tuning_coords_{space}.parquet"
        coords_long.to_csv(coords_csv, index=False)
        coords_long.to_parquet(coords_parquet, index=False)
        output_paths.extend([display_path(coords_csv), display_path(coords_parquet)])

        figure_path = output_dir / f"umap_tuning_{space}_grid.png"
        plot_umap_tuning_grid(
            space=space,
            coordinates=coords_long,
            quality=quality_space,
            output_path=figure_path,
        )
        output_paths.append(display_path(figure_path))

        space_details[space] = {
            "n_subfields": int(len(space_data.metadata)),
            "n_features_in_clustering_feature_matrix": int(space_data.feature_matrix.shape[1]),
            "cluster_column": cluster_column,
            "component_blocks": space_data.component_blocks,
            "parameter_combinations": int(
                len(tuning_config.n_neighbors_values)
                * len(tuning_config.min_dist_values)
            ),
        }

    quality = pd.DataFrame(quality_rows)
    quality_path = output_dir / "umap_tuning_quality.csv"
    quality.to_csv(quality_path, index=False)
    output_paths.append(display_path(quality_path))

    summary_path = output_dir / "umap_tuning_summary.md"
    summary_path.write_text(
        umap_tuning_summary_markdown(
            quality=quality,
            warnings=warnings,
            config=tuning_config,
        ),
        encoding="utf-8",
    )
    output_paths.append(display_path(summary_path))

    config_path = output_dir / "umap_tuning_config.json"
    config_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "umap_metrics_path": display_path(umap_path),
            "embedding_metrics_path": display_path(embedding_path),
            "clustering_dir": display_path(clustering_dir),
        },
        "spaces": SPACES,
        "n_neighbors_values": list(tuning_config.n_neighbors_values),
        "min_dist_values": list(tuning_config.min_dist_values),
        "random_state": tuning_config.random_state,
        "clustering_preprocessing_config": {
            "max_missing_share": clustering_config.max_missing_share,
            "min_unique_values": clustering_config.min_unique_values,
            "pca_variance_threshold": clustering_config.pca_variance_threshold,
            "scaler": clustering_config.scaler,
            "combined_strategy": clustering_config.combined_strategy,
            "random_state": clustering_config.random_state,
        },
        "space_details": space_details,
        "warnings": warnings,
        "output_paths": output_paths,
    }
    write_json(config_path, config_payload)
    output_paths.append(display_path(config_path))

    summary_json_path = output_dir / "summary.json"
    write_json(
        summary_json_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_paths": output_paths + [display_path(summary_json_path)],
            "warnings": warnings,
            "spaces": SPACES,
            "n_parameter_combinations": int(
                len(tuning_config.n_neighbors_values)
                * len(tuning_config.min_dist_values)
            ),
        },
    )
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
