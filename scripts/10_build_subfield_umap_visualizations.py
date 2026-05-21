from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.per_subfield_umap_maps import (
    DEFAULT_DENSITY_GRID_SIZE,
    DEFAULT_DENSITY_METHOD,
    DEFAULT_DENSITY_SIGMA,
    DEFAULT_DENSITY_VMAX_PERCENTILE,
    VALID_DENSITY_METHODS,
    build_coordinate_frame,
    build_manifest_row,
    filter_input_window,
    main_analysis_subfields,
    manifest_frame,
    plot_subfield_panels,
    safe_subfield_stem,
    sample_subfield_rows,
    validate_index_columns,
    validate_year_window,
)
from src.storage import load_parquet, save_parquet


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Build one internal UMAP scatter+density map per OpenAlex subfield."
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/analysis_embedding_index.parquet",
        help="Parquet index aligned row-for-row with the main embedding matrix.",
    )
    parser.add_argument(
        "--embedding-dir",
        default=os.getenv(
            "LOCAL_EMBEDDINGS_DIR",
            "embeddings/specter2_v1_2000_2024_400py",
        ),
        help=(
            "Local folder containing SPECTER2 embedding artifacts. Used to derive "
            "--embeddings-path when that argument is not provided."
        ),
    )
    parser.add_argument(
        "--embeddings-path",
        default=None,
        help=(
            "Main SPECTER2 embedding matrix. Defaults to "
            "<embedding-dir>/analysis/main_embeddings.float16.npy."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/08_visualization/per_subfield_umap_smooth_density",
        help="Directory for per-subfield coordinates, figures, manifest, and summary.",
    )
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--min-papers", type=int, default=250)
    parser.add_argument("--max-papers-per-subfield", type=int, default=10000)
    parser.add_argument("--subfield-id", default=None)
    parser.add_argument("--limit-subfields", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--density-method",
        choices=sorted(VALID_DENSITY_METHODS),
        default=DEFAULT_DENSITY_METHOD,
        help="Density renderer for panel B. smooth_hist is the stable default for map figures.",
    )
    parser.add_argument("--density-grid-size", type=int, default=DEFAULT_DENSITY_GRID_SIZE)
    parser.add_argument("--density-sigma", type=float, default=DEFAULT_DENSITY_SIGMA)
    parser.add_argument(
        "--density-vmax-percentile",
        type=float,
        default=DEFAULT_DENSITY_VMAX_PERCENTILE,
        help=(
            "Positive-density percentile used as panel B color maximum. "
            "This reduces domination by tiny max-density hotspots."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def default_embeddings_path(embedding_dir: str | Path) -> Path:
    return Path(embedding_dir) / "analysis" / "main_embeddings.float16.npy"


def resolve_embeddings_path(args: argparse.Namespace) -> Path:
    path = args.embeddings_path or default_embeddings_path(args.embedding_dir)
    return resolve_path(path)


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_run_outputs(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path, Path]:
    coordinates_dir = output_dir / "coordinates"
    figures_dir = output_dir / "figures"
    manifest_path = output_dir / "per_subfield_umap_manifest.parquet"
    summary_path = output_dir / "per_subfield_umap_summary.json"

    existing_run_files = [
        path for path in [manifest_path, summary_path] if path.exists()
    ]
    if existing_run_files and not overwrite:
        formatted = ", ".join(display_path(path) for path in existing_run_files)
        raise FileExistsError(
            "Refusing to overwrite existing per-subfield UMAP run outputs without "
            f"--overwrite: {formatted}"
        )

    coordinates_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return coordinates_dir, figures_dir, manifest_path, summary_path


def select_attempted_subfields(
    subfields: pd.DataFrame,
    *,
    subfield_id: str | None,
    limit_subfields: int | None,
) -> pd.DataFrame:
    attempted = subfields.copy()
    if subfield_id is not None:
        attempted = attempted[
            attempted["subfield_id"].astype(str) == str(subfield_id)
        ].reset_index(drop=True)
        if attempted.empty:
            raise ValueError(f"No main-analysis subfield matched --subfield-id {subfield_id}")

    if limit_subfields is not None:
        if limit_subfields <= 0:
            raise ValueError("limit_subfields must be positive when provided")
        attempted = attempted.head(limit_subfields).reset_index(drop=True)

    return attempted


def make_subfield_group_lookup(window_index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(subfield_id): group.reset_index(drop=True)
        for subfield_id, group in window_index.groupby("subfield_id", sort=False)
    }


def load_matrix(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing embedding matrix: {display_path(embeddings_path)}. "
            "Run scripts/08_prepare_analysis_matrix.py first."
        )
    matrix = np.load(embeddings_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix at {display_path(embeddings_path)}, "
            f"got shape {matrix.shape}"
        )
    return matrix


def validate_matrix_alignment(
    index: pd.DataFrame,
    matrix: np.ndarray,
    *,
    index_path: Path,
    embeddings_path: Path,
) -> None:
    row_ids = pd.to_numeric(index["analysis_row_id"], errors="raise")
    if row_ids.empty:
        raise ValueError("analysis index is empty")
    min_row = int(row_ids.min())
    max_row = int(row_ids.max())
    if min_row < 0 or max_row >= matrix.shape[0]:
        raise ValueError(
            "analysis_embedding_index and embedding matrix appear to belong to "
            "different embedding versions. "
            f"index_path={display_path(index_path)}; "
            f"embeddings_path={display_path(embeddings_path)}; "
            f"analysis_row_id range {min_row}-{max_row}; "
            f"matrix rows {matrix.shape[0]}. Use a matching --embedding-dir or "
            "--embeddings-path."
        )


def main() -> None:
    args = parse_args()
    validate_year_window(args.year_min, args.year_max)
    if args.min_papers <= 0:
        raise ValueError("min_papers must be positive")
    if args.max_papers_per_subfield <= 0:
        raise ValueError("max_papers_per_subfield must be positive")
    if args.n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if args.density_grid_size < 10:
        raise ValueError("density_grid_size must be at least 10")
    if args.density_sigma <= 0:
        raise ValueError("density_sigma must be positive")
    if not 0 < args.density_vmax_percentile <= 100:
        raise ValueError("density_vmax_percentile must be in (0, 100]")

    index_path = resolve_path(args.index_path)
    embeddings_path = resolve_embeddings_path(args)
    output_dir = resolve_path(args.output_dir)
    coordinates_dir, figures_dir, manifest_path, summary_path = ensure_run_outputs(
        output_dir,
        args.overwrite,
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing analysis index: {display_path(index_path)}. "
            "Run scripts/08_prepare_analysis_matrix.py first."
        )

    print(f"Loading analysis index: {display_path(index_path)}")
    analysis_index = load_parquet(index_path)
    validate_index_columns(analysis_index)

    print(
        "Filtering to main-analysis rows in morphology input window "
        f"{args.year_min}-{args.year_max}"
    )
    subfields = main_analysis_subfields(analysis_index)
    attempted_subfields = select_attempted_subfields(
        subfields,
        subfield_id=args.subfield_id,
        limit_subfields=args.limit_subfields,
    )
    window_index = filter_input_window(
        analysis_index,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    groups = make_subfield_group_lookup(window_index)

    print(f"Memory-mapping embeddings: {display_path(embeddings_path)}")
    matrix = load_matrix(embeddings_path)
    validate_matrix_alignment(
        analysis_index,
        matrix,
        index_path=index_path,
        embeddings_path=embeddings_path,
    )

    import umap

    manifest_rows: list[dict[str, Any]] = []
    total = len(attempted_subfields)
    print(f"Attempting {total} subfields")

    for position, subfield in attempted_subfields.iterrows():
        subfield_id = subfield["subfield_id"]
        subfield_name = subfield["subfield_display_name"]
        subfield_label_short = subfield["subfield_label_short"]
        subfield_key = str(subfield_id)
        available = groups.get(subfield_key, window_index.head(0))
        n_available = int(len(available))
        progress_prefix = f"[{position + 1}/{total}] {subfield_label_short}"
        manifest_metadata = {
            "field_id": subfield.get("field_id", ""),
            "field_display_name": subfield.get("field_display_name", ""),
            "domain_id": subfield.get("domain_id", ""),
            "domain_display_name": subfield.get("domain_display_name", ""),
            "subfield_label_unique": subfield.get("subfield_label_unique", ""),
            "subfield_label_short": subfield.get("subfield_label_short", ""),
            "subfield_display_name_is_duplicated": bool(
                subfield.get("subfield_display_name_is_duplicated", False)
            ),
        }

        stem = safe_subfield_stem(subfield_id, subfield_name)
        coordinate_path = coordinates_dir / f"{stem}.parquet"
        figure_path = figures_dir / f"{stem}.png"

        if n_available < args.min_papers:
            print(f"{progress_prefix}: skipped ({n_available} rows available)")
            manifest_rows.append(
                build_manifest_row(
                    subfield_id=subfield_id,
                    subfield_name=subfield_name,
                    n_available=n_available,
                    n_used=0,
                    year_min=args.year_min,
                    year_max=args.year_max,
                    status="skipped",
                    error_message=(
                        f"Only {n_available} papers available; min_papers is "
                        f"{args.min_papers}"
                    ),
                    umap_n_neighbors=args.n_neighbors,
                    umap_min_dist=args.min_dist,
                    umap_metric=args.metric,
                    random_state=args.random_state,
                    max_papers_per_subfield=args.max_papers_per_subfield,
                    sampling_applied=False,
                    **manifest_metadata,
                )
            )
            continue

        try:
            if (
                (coordinate_path.exists() or figure_path.exists())
                and not args.overwrite
            ):
                raise FileExistsError(
                    "coordinate or figure output already exists; rerun with --overwrite"
                )

            sampled = sample_subfield_rows(
                available,
                max_papers=args.max_papers_per_subfield,
                random_state=args.random_state,
                subfield_id=subfield_id,
            )
            sampling_applied = len(sampled) < n_available
            row_ids = sampled["analysis_row_id"].astype(int).to_numpy()
            embeddings = np.asarray(matrix[row_ids], dtype=np.float32)

            effective_n_neighbors = min(args.n_neighbors, max(2, len(sampled) - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=effective_n_neighbors,
                min_dist=args.min_dist,
                metric=args.metric,
                random_state=args.random_state,
                low_memory=True,
                verbose=False,
            )
            coordinates = reducer.fit_transform(embeddings)

            coordinate_frame = build_coordinate_frame(sampled, coordinates)
            save_parquet(coordinate_frame, coordinate_path)
            density_method = plot_subfield_panels(
                coordinates,
                subfield_name=str(subfield_label_short),
                n_used=len(sampled),
                year_min=args.year_min,
                year_max=args.year_max,
                output_path=figure_path,
                dpi=args.dpi,
                density_method=args.density_method,
                density_grid_size=args.density_grid_size,
                density_sigma=args.density_sigma,
                density_vmax_percentile=args.density_vmax_percentile,
            )

            print(
                f"{progress_prefix}: completed "
                f"({len(sampled)} rows, density={density_method})"
            )
            manifest_rows.append(
                build_manifest_row(
                    subfield_id=subfield_id,
                    subfield_name=subfield_name,
                    n_available=n_available,
                    n_used=len(sampled),
                    year_min=args.year_min,
                    year_max=args.year_max,
                    status="completed",
                    coordinate_path=display_path(coordinate_path),
                    figure_path=display_path(figure_path),
                    umap_n_neighbors=effective_n_neighbors,
                    umap_min_dist=args.min_dist,
                    umap_metric=args.metric,
                    random_state=args.random_state,
                    max_papers_per_subfield=args.max_papers_per_subfield,
                    sampling_applied=sampling_applied,
                    **manifest_metadata,
                )
            )
        except Exception as exc:
            print(f"{progress_prefix}: failed ({exc})")
            manifest_rows.append(
                build_manifest_row(
                    subfield_id=subfield_id,
                    subfield_name=subfield_name,
                    n_available=n_available,
                    n_used=0,
                    year_min=args.year_min,
                    year_max=args.year_max,
                    status="failed",
                    coordinate_path=display_path(coordinate_path)
                    if coordinate_path.exists()
                    else None,
                    figure_path=display_path(figure_path) if figure_path.exists() else None,
                    error_message=str(exc),
                    umap_n_neighbors=args.n_neighbors,
                    umap_min_dist=args.min_dist,
                    umap_metric=args.metric,
                    random_state=args.random_state,
                    max_papers_per_subfield=args.max_papers_per_subfield,
                    sampling_applied=False,
                    **manifest_metadata,
                )
            )

    manifest = manifest_frame(manifest_rows)
    save_parquet(manifest, manifest_path)

    status_counts = manifest["status"].value_counts().to_dict()
    skipped_details = manifest.loc[
        manifest["status"] == "skipped",
        ["subfield_id", "subfield_name", "n_available", "error_message"],
    ].to_dict("records")
    failed_details = manifest.loc[
        manifest["status"] == "failed",
        ["subfield_id", "subfield_name", "n_available", "error_message"],
    ].to_dict("records")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "index_path": display_path(index_path),
            "embedding_dir": display_path(resolve_path(args.embedding_dir)),
            "embeddings_path": display_path(embeddings_path),
        },
        "output_paths": {
            "output_dir": display_path(output_dir),
            "coordinates_dir": display_path(coordinates_dir),
            "figures_dir": display_path(figures_dir),
            "manifest_path": display_path(manifest_path),
            "summary_path": display_path(summary_path),
        },
        "year_window": {
            "year_min": args.year_min,
            "year_max": args.year_max,
        },
        "n_subfields_attempted": int(len(manifest)),
        "n_completed": int(status_counts.get("completed", 0)),
        "n_skipped": int(status_counts.get("skipped", 0)),
        "n_failed": int(status_counts.get("failed", 0)),
        "total_papers_used": int(manifest["n_used"].sum()) if len(manifest) else 0,
        "skipped_subfields": skipped_details,
        "failed_subfields": failed_details,
        "umap": {
            "n_components": 2,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "low_memory": True,
        },
        "sampling": {
            "min_papers": args.min_papers,
            "max_papers_per_subfield": args.max_papers_per_subfield,
            "random_state": args.random_state,
            "subfield_id": args.subfield_id,
            "limit_subfields": args.limit_subfields,
        },
        "density": {
            "method": args.density_method,
            "grid_size": args.density_grid_size,
            "sigma": args.density_sigma,
            "vmax_percentile": args.density_vmax_percentile,
        },
    }
    write_json(summary_path, summary)

    print(f"Wrote {display_path(manifest_path)}")
    print(f"Wrote {display_path(summary_path)}")
    print(
        "Done: "
        f"{summary['n_completed']} completed, "
        f"{summary['n_skipped']} skipped, "
        f"{summary['n_failed']} failed"
    )


if __name__ == "__main__":
    main()
