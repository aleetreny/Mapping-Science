from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.per_category_umap_maps import (
    LEVEL_SPECS,
    build_category_coordinate_frame,
    category_manifest_frame,
    category_manifest_row,
    filter_category_input_window,
    level_spec,
    list_groups,
    make_group_lookup,
    plot_category_panels,
    safe_group_stem,
    sample_group_rows,
    select_attempted_groups,
    validate_category_index_columns,
)
from src.per_subfield_umap_maps import VALID_DENSITY_METHODS, validate_year_window
from src.storage import load_parquet, save_parquet


ROOT = Path(__file__).resolve().parents[1]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index-path",
        default="data/processed/analysis_embedding_index.parquet",
        help="Parquet index aligned row-for-row with the main embedding matrix.",
    )
    parser.add_argument(
        "--embedding-dir",
        default=os.getenv("LOCAL_EMBEDDINGS_DIR", "embeddings/specter2_v1"),
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--year-max", type=int, default=2025)
    parser.add_argument("--min-papers", type=int, default=250)
    parser.add_argument("--max-papers-per-group", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-legend-categories", type=int, default=25)
    parser.add_argument(
        "--density-method",
        choices=sorted(VALID_DENSITY_METHODS),
        default="smooth_hist",
        help="Density renderer for panel B. smooth_hist avoids hexbin artifacts on large groups.",
    )
    parser.add_argument("--density-grid-size", type=int, default=180)
    parser.add_argument("--density-sigma", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true")


def parse_category_args(
    argv: list[str] | None = None,
    *,
    fixed_level: str | None = None,
    deprecated: bool = False,
) -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    if fixed_level is None:
        description = (
            "Deprecated compatibility wrapper for grouped OpenAlex field/domain "
            "UMAP maps. Prefer scripts/10b_build_per_field_umap_maps.py or "
            "scripts/10c_build_per_domain_umap_maps.py."
            if deprecated
            else "Build one UMAP scatter+density map per OpenAlex field or domain."
        )
    else:
        spec = level_spec(fixed_level)
        description = f"Build one UMAP scatter+density map per OpenAlex {spec['group_label']}."

    parser = argparse.ArgumentParser(description=description)
    if fixed_level is None:
        parser.add_argument("--level", choices=sorted(LEVEL_SPECS), required=True)
        parser.add_argument("--group-id", default=None)
        parser.add_argument("--limit-groups", type=int, default=None)
    else:
        parser.set_defaults(level=fixed_level)
        label = level_spec(fixed_level)["group_label"]
        parser.add_argument(f"--{label}-id", "--group-id", dest="group_id", default=None)
        parser.add_argument(
            f"--limit-{label}s",
            "--limit-groups",
            dest="limit_groups",
            type=int,
            default=None,
        )
    add_common_arguments(parser)
    return parser.parse_args(argv)


def resolve_path(path: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def default_embeddings_path(embedding_dir: str | Path) -> Path:
    return Path(embedding_dir) / "analysis" / "main_embeddings.float16.npy"


def resolve_embeddings_path(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
) -> Path:
    path = args.embeddings_path or default_embeddings_path(args.embedding_dir)
    return resolve_path(path, root=root)


def display_path(path: str | Path, *, root: Path = ROOT) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_run_outputs(
    output_dir: Path,
    *,
    manifest_filename: str,
    summary_filename: str,
    overwrite: bool,
    root: Path = ROOT,
) -> tuple[Path, Path, Path, Path]:
    coordinates_dir = output_dir / "coordinates"
    figures_dir = output_dir / "figures"
    manifest_path = output_dir / manifest_filename
    summary_path = output_dir / summary_filename
    existing_run_files = [
        path for path in [manifest_path, summary_path] if path.exists()
    ]
    if existing_run_files and not overwrite:
        formatted = ", ".join(display_path(path, root=root) for path in existing_run_files)
        raise FileExistsError(
            "Refusing to overwrite existing grouped UMAP run outputs without "
            f"--overwrite: {formatted}"
        )
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return coordinates_dir, figures_dir, manifest_path, summary_path


def load_matrix(embeddings_path: Path, *, root: Path = ROOT) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing embedding matrix: {display_path(embeddings_path, root=root)}. "
            "Run scripts/08_prepare_analysis_matrix.py first."
        )
    matrix = np.load(embeddings_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix at {display_path(embeddings_path, root=root)}, "
            f"got shape {matrix.shape}"
        )
    return matrix


def validate_matrix_alignment(
    index: pd.DataFrame,
    matrix: np.ndarray,
    *,
    index_path: Path,
    embeddings_path: Path,
    root: Path = ROOT,
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
            f"index_path={display_path(index_path, root=root)}; "
            f"embeddings_path={display_path(embeddings_path, root=root)}; "
            f"analysis_row_id range {min_row}-{max_row}; "
            f"matrix rows {matrix.shape[0]}. Use a matching --embedding-dir or "
            "--embeddings-path."
        )


def validate_category_args(args: argparse.Namespace) -> None:
    validate_year_window(args.year_min, args.year_max)
    level_spec(args.level)
    if args.min_papers <= 0:
        raise ValueError("min_papers must be positive")
    if args.max_papers_per_group <= 0:
        raise ValueError("max_papers_per_group must be positive")
    if args.n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if args.max_legend_categories < 1:
        raise ValueError("max_legend_categories must be positive")
    if args.density_grid_size < 10:
        raise ValueError("density_grid_size must be at least 10")
    if args.density_sigma <= 0:
        raise ValueError("density_sigma must be positive")


def run_category_umap_maps(args: argparse.Namespace, *, root: Path = ROOT) -> None:
    validate_category_args(args)

    spec = level_spec(args.level)
    index_path = resolve_path(args.index_path, root=root)
    embeddings_path = resolve_embeddings_path(args, root=root)
    output_dir = resolve_path(args.output_dir or spec["output_dir"], root=root)
    coordinates_dir, figures_dir, manifest_path, summary_path = ensure_run_outputs(
        output_dir,
        manifest_filename=spec["manifest_filename"],
        summary_filename=spec["summary_filename"],
        overwrite=args.overwrite,
        root=root,
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing analysis index: {display_path(index_path, root=root)}. "
            "Run scripts/08_prepare_analysis_matrix.py first."
        )

    print(f"Loading analysis index: {display_path(index_path, root=root)}")
    analysis_index = load_parquet(index_path)
    validate_category_index_columns(analysis_index)
    window_index = filter_category_input_window(
        analysis_index,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    groups = list_groups(window_index, level=args.level)
    attempted_groups = select_attempted_groups(
        groups,
        group_id=args.group_id,
        limit_groups=args.limit_groups,
    )
    group_lookup = make_group_lookup(window_index, level=args.level)

    print(f"Memory-mapping embeddings: {display_path(embeddings_path, root=root)}")
    matrix = load_matrix(embeddings_path, root=root)
    validate_matrix_alignment(
        analysis_index,
        matrix,
        index_path=index_path,
        embeddings_path=embeddings_path,
        root=root,
    )

    import umap

    manifest_rows: list[dict[str, Any]] = []
    total = len(attempted_groups)
    print(f"Attempting {total} {args.level} groups")

    for position, group in attempted_groups.iterrows():
        group_id = group["group_id"]
        group_name = group["group_name"]
        group_key = str(group_id)
        available = group_lookup.get(group_key, window_index.head(0))
        n_available = int(len(available))
        progress_prefix = f"[{position + 1}/{total}] {args.level} {group_key} - {group_name}"

        stem = safe_group_stem(args.level, group_id, group_name)
        coordinate_path = coordinates_dir / f"{stem}.parquet"
        figure_path = figures_dir / f"{stem}.png"

        if n_available < args.min_papers:
            print(f"{progress_prefix}: skipped ({n_available} rows available)")
            manifest_rows.append(
                category_manifest_row(
                    level=args.level,
                    group_id=group_id,
                    group_name=group_name,
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
                    max_papers_per_group=args.max_papers_per_group,
                    sampling_applied=False,
                    color_by=spec["color_column"],
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
            sampled = sample_group_rows(
                available,
                max_papers=args.max_papers_per_group,
                random_state=args.random_state,
                group_id=group_id,
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
            coordinate_frame = build_category_coordinate_frame(sampled, coordinates)
            save_parquet(coordinate_frame, coordinate_path)
            plot_info = plot_category_panels(
                coordinate_frame,
                coordinates,
                level=args.level,
                group_name=str(group_name),
                n_used=len(sampled),
                year_min=args.year_min,
                year_max=args.year_max,
                color_column=spec["color_column"],
                output_path=figure_path,
                dpi=args.dpi,
                max_legend_categories=args.max_legend_categories,
                density_method=args.density_method,
                density_grid_size=args.density_grid_size,
                density_sigma=args.density_sigma,
            )
            print(
                f"{progress_prefix}: completed "
                f"({len(sampled)} rows, density={plot_info['density_method']})"
            )
            manifest_rows.append(
                category_manifest_row(
                    level=args.level,
                    group_id=group_id,
                    group_name=group_name,
                    n_available=n_available,
                    n_used=len(sampled),
                    year_min=args.year_min,
                    year_max=args.year_max,
                    status="completed",
                    coordinate_path=display_path(coordinate_path, root=root),
                    figure_path=display_path(figure_path, root=root),
                    umap_n_neighbors=effective_n_neighbors,
                    umap_min_dist=args.min_dist,
                    umap_metric=args.metric,
                    random_state=args.random_state,
                    max_papers_per_group=args.max_papers_per_group,
                    sampling_applied=sampling_applied,
                    color_by=plot_info["color_by"],
                    legend_included=plot_info["legend_included"],
                    density_method=plot_info["density_method"],
                )
            )
        except Exception as exc:
            print(f"{progress_prefix}: failed ({exc})")
            manifest_rows.append(
                category_manifest_row(
                    level=args.level,
                    group_id=group_id,
                    group_name=group_name,
                    n_available=n_available,
                    n_used=0,
                    year_min=args.year_min,
                    year_max=args.year_max,
                    status="failed",
                    coordinate_path=display_path(coordinate_path, root=root)
                    if coordinate_path.exists()
                    else None,
                    figure_path=display_path(figure_path, root=root)
                    if figure_path.exists()
                    else None,
                    error_message=str(exc),
                    umap_n_neighbors=args.n_neighbors,
                    umap_min_dist=args.min_dist,
                    umap_metric=args.metric,
                    random_state=args.random_state,
                    max_papers_per_group=args.max_papers_per_group,
                    sampling_applied=False,
                    color_by=spec["color_column"],
                )
            )

    manifest = category_manifest_frame(manifest_rows)
    save_parquet(manifest, manifest_path)
    status_counts = manifest["status"].value_counts().to_dict()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "level": args.level,
        "input_paths": {
            "index_path": display_path(index_path, root=root),
            "embedding_dir": display_path(
                resolve_path(args.embedding_dir, root=root),
                root=root,
            ),
            "embeddings_path": display_path(embeddings_path, root=root),
        },
        "output_paths": {
            "output_dir": display_path(output_dir, root=root),
            "coordinates_dir": display_path(coordinates_dir, root=root),
            "figures_dir": display_path(figures_dir, root=root),
            "manifest_path": display_path(manifest_path, root=root),
            "summary_path": display_path(summary_path, root=root),
        },
        "year_window": {
            "year_min": args.year_min,
            "year_max": args.year_max,
        },
        "n_groups_attempted": int(len(manifest)),
        "n_completed": int(status_counts.get("completed", 0)),
        "n_skipped": int(status_counts.get("skipped", 0)),
        "n_failed": int(status_counts.get("failed", 0)),
        "total_papers_used": int(manifest["n_used"].sum()) if len(manifest) else 0,
        "umap": {
            "n_components": 2,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "low_memory": True,
        },
        "sampling": {
            "min_papers": args.min_papers,
            "max_papers_per_group": args.max_papers_per_group,
            "random_state": args.random_state,
            "group_id": args.group_id,
            "limit_groups": args.limit_groups,
        },
        "density": {
            "method": args.density_method,
            "grid_size": args.density_grid_size,
            "sigma": args.density_sigma,
        },
    }
    write_json(summary_path, summary)
    print(f"Wrote {display_path(manifest_path, root=root)}")
    print(f"Wrote {display_path(summary_path, root=root)}")
    print(
        "Done: "
        f"{summary['n_completed']} completed, "
        f"{summary['n_skipped']} skipped, "
        f"{summary['n_failed']} failed"
    )


def main(
    argv: list[str] | None = None,
    *,
    fixed_level: str | None = None,
    deprecated: bool = False,
) -> None:
    args = parse_category_args(argv, fixed_level=fixed_level, deprecated=deprecated)
    if deprecated:
        print(
            "Deprecated: prefer scripts/10b_build_per_field_umap_maps.py or "
            "scripts/10c_build_per_domain_umap_maps.py."
        )
    run_category_umap_maps(args)
