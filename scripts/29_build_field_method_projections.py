from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.per_category_umap_maps import plot_category_panels
from src.storage import load_parquet, save_parquet


METHODS = ("UMAP", "PCA", "t-SNE", "PHATE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PCA, t-SNE, PHATE, and UMAP field comparison maps for the static web explorer."
    )
    parser.add_argument(
        "--field-umap-manifest",
        default="outputs/08_visualization/per_field_umap_smooth_density/per_field_umap_manifest.parquet",
    )
    parser.add_argument(
        "--embeddings-path",
        default="embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/08_visualization/per_field_method_comparison",
    )
    parser.add_argument("--methods", default="UMAP,PCA,t-SNE,PHATE")
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Reuse existing coordinate parquet files and only redraw PNG figures.",
    )
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


def parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {', '.join(invalid)}")
    return methods


def load_manifest(path: Path) -> pd.DataFrame:
    frame = load_parquet(path)
    completed = frame.loc[frame["status"] == "completed"].copy()
    if completed.empty:
        raise ValueError(f"No completed field UMAP rows found in {display_path(path)}")
    completed["group_id"] = completed["group_id"].astype(str)
    return completed.sort_values("group_id", kind="mergesort").reset_index(drop=True)


def output_paths(output_dir: Path, stem: str, method: str) -> tuple[Path, Path]:
    figure_path = output_dir / "figures" / "field_method_comparison" / stem / f"{method}.png"
    coordinate_path = output_dir / "coordinates" / "field_method_comparison" / stem / f"{method}.parquet"
    return figure_path, coordinate_path


def copy_umap_artifact(
    *,
    row: dict[str, Any],
    stem: str,
    output_dir: Path,
    year_min: int,
    year_max: int,
    random_state: int,
    dpi: int,
) -> dict[str, Any]:
    figure_path, coordinate_path = output_paths(output_dir, stem, "UMAP")
    source_coordinates = resolve_path(row["coordinate_path"])
    coordinate_frame = load_parquet(source_coordinates).copy()
    coordinate_frame["x"] = coordinate_frame["umap_x"].astype(np.float32)
    coordinate_frame["y"] = coordinate_frame["umap_y"].astype(np.float32)
    coordinate_frame["projection_method"] = "UMAP"
    coordinates = coordinate_frame[["x", "y"]].to_numpy(dtype=float)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    coordinate_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(coordinate_frame, coordinate_path)
    plot_category_panels(
        coordinate_frame,
        coordinates,
        level="field",
        group_name=str(row["group_name"]),
        n_used=len(coordinate_frame),
        year_min=year_min,
        year_max=year_max,
        color_column="subfield_display_name",
        output_path=figure_path,
        dpi=dpi,
        max_legend_categories=0,
        density_method="smooth_hist",
        axis_label="UMAP",
    )

    return {
        "field_id": str(row["group_id"]),
        "field_name": str(row["group_name"]),
        "method": "UMAP",
        "n_used": int(row["n_used"]),
        "year_min": year_min,
        "year_max": year_max,
        "status": "completed",
        "coordinate_path": display_path(coordinate_path),
        "figure_path": display_path(figure_path),
        "error_message": "",
        "random_state": random_state,
    }


def project_embeddings(method: str, embeddings: np.ndarray, *, random_state: int) -> np.ndarray:
    n_used = embeddings.shape[0]
    if method == "PCA":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(embeddings)
    if method == "t-SNE":
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(5.0, n_used / 3.0))
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=perplexity,
            n_jobs=1,
            init="pca" if n_used > 30 else "random",
        )
        return reducer.fit_transform(embeddings)
    if method == "PHATE":
        import phate

        reducer = phate.PHATE(
            n_components=2,
            random_state=random_state,
            n_jobs=1,
            verbose=False,
        )
        return reducer.fit_transform(embeddings)
    raise ValueError(f"Unsupported projection method: {method}")


def run_single_projection(
    *,
    field_id: str,
    field_name: str,
    stem: str,
    method: str,
    coordinate_path: str,
    embeddings_path: str,
    output_dir: str,
    year_min: int,
    year_max: int,
    random_state: int,
    dpi: int,
) -> dict[str, Any]:
    figure_path, out_coordinate_path = output_paths(Path(output_dir), stem, method)
    try:
        coordinate_frame = load_parquet(resolve_path(coordinate_path)).copy()
        row_ids = coordinate_frame["analysis_row_id"].astype(int).to_numpy()
        matrix = np.load(resolve_path(embeddings_path), mmap_mode="r")
        embeddings = np.asarray(matrix[row_ids], dtype=np.float32)
        coordinates = project_embeddings(method, embeddings, random_state=random_state)
        if not np.isfinite(coordinates).all():
            raise ValueError("Computed coordinates contain non-finite values.")

        figure_path.parent.mkdir(parents=True, exist_ok=True)
        out_coordinate_path.parent.mkdir(parents=True, exist_ok=True)
        coordinate_frame["x"] = coordinates[:, 0].astype(np.float32)
        coordinate_frame["y"] = coordinates[:, 1].astype(np.float32)
        coordinate_frame["projection_method"] = method
        save_parquet(coordinate_frame, out_coordinate_path)

        plot_category_panels(
            coordinate_frame,
            coordinates,
            level="field",
            group_name=field_name,
            n_used=len(coordinate_frame),
            year_min=year_min,
            year_max=year_max,
            color_column="subfield_display_name",
            output_path=figure_path,
            dpi=dpi,
            max_legend_categories=0,
            density_method="smooth_hist",
            axis_label=method,
        )
        status = "completed"
        error_message = ""
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        print(f"Error computing {method} for {field_name}: {exc}")

    return {
        "field_id": field_id,
        "field_name": field_name,
        "method": method,
        "n_used": 0 if status != "completed" else len(load_parquet(out_coordinate_path)),
        "year_min": year_min,
        "year_max": year_max,
        "status": status,
        "coordinate_path": display_path(out_coordinate_path) if status == "completed" else "",
        "figure_path": display_path(figure_path) if status == "completed" else "",
        "error_message": error_message,
        "random_state": random_state,
    }


def replot_existing_projection(
    *,
    field_id: str,
    field_name: str,
    stem: str,
    method: str,
    output_dir: Path,
    year_min: int,
    year_max: int,
    random_state: int,
    dpi: int,
) -> dict[str, Any]:
    figure_path, coordinate_path = output_paths(output_dir, stem, method)
    coordinate_frame = load_parquet(coordinate_path).copy()
    coordinates = coordinate_frame[["x", "y"]].to_numpy(dtype=float)
    plot_category_panels(
        coordinate_frame,
        coordinates,
        level="field",
        group_name=field_name,
        n_used=len(coordinate_frame),
        year_min=year_min,
        year_max=year_max,
        color_column="subfield_display_name",
        output_path=figure_path,
        dpi=dpi,
        max_legend_categories=0,
        density_method="smooth_hist",
        axis_label=method,
    )
    return {
        "field_id": field_id,
        "field_name": field_name,
        "method": method,
        "n_used": len(coordinate_frame),
        "year_min": year_min,
        "year_max": year_max,
        "status": "completed",
        "coordinate_path": display_path(coordinate_path),
        "figure_path": display_path(figure_path),
        "error_message": "",
        "random_state": random_state,
    }


def copy_web_figures(output_dir: Path) -> None:
    source_root = output_dir / "figures" / "field_method_comparison"
    targets = [
        ROOT / "docs" / "site-assets" / "figures" / "field_method_comparison",
        ROOT / "frontend" / "figures" / "field_method_comparison",
    ]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
    for source in source_root.rglob("*.png"):
        relative = source.relative_to(source_root)
        for target in targets:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    manifest = load_manifest(resolve_path(args.field_umap_manifest))
    output_dir = resolve_path(args.output_dir)
    embeddings_path = resolve_path(args.embeddings_path)

    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    skipped = 0

    for row in manifest.to_dict(orient="records"):
        stem = Path(str(row["figure_path"])).stem
        for method in methods:
            figure_path, coordinate_path = output_paths(output_dir, stem, method)
            if method == "UMAP":
                if args.force or args.replot_only or not (figure_path.exists() and coordinate_path.exists()):
                    rows.append(
                        copy_umap_artifact(
                            row=row,
                            stem=stem,
                            output_dir=output_dir,
                            year_min=args.year_min,
                            year_max=args.year_max,
                            random_state=args.random_state,
                            dpi=args.dpi,
                        )
                    )
                else:
                    skipped += 1
                continue

            if args.replot_only:
                if figure_path.exists() and coordinate_path.exists():
                    rows.append(
                        replot_existing_projection(
                            field_id=str(row["group_id"]),
                            field_name=str(row["group_name"]),
                            stem=stem,
                            method=method,
                            output_dir=output_dir,
                            year_min=args.year_min,
                            year_max=args.year_max,
                            random_state=args.random_state,
                            dpi=args.dpi,
                        )
                    )
                else:
                    print(f"Missing existing field projection for replot: {display_path(figure_path)}")
                continue

            if not args.force and figure_path.exists() and coordinate_path.exists():
                skipped += 1
                continue
            jobs.append(
                {
                    "field_id": str(row["group_id"]),
                    "field_name": str(row["group_name"]),
                    "stem": stem,
                    "method": method,
                    "coordinate_path": str(row["coordinate_path"]),
                    "embeddings_path": str(embeddings_path),
                    "output_dir": str(output_dir),
                    "year_min": args.year_min,
                    "year_max": args.year_max,
                    "random_state": args.random_state,
                    "dpi": args.dpi,
                }
            )

    print(f"Skipped completed runs: {skipped}")
    print(f"Projection jobs to run: {len(jobs)}")
    start = time.time()
    completed = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single_projection, **job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                rows.append(result)
                completed += 1
                if completed % 5 == 0 or completed == len(jobs):
                    elapsed = time.time() - start
                    print(f"Progress: {completed}/{len(jobs)} jobs in {elapsed:.1f}s")

    prior_manifest = output_dir / "manifests" / "all_fields_projections_manifest.csv"
    if prior_manifest.exists() and rows:
        prior = pd.read_csv(prior_manifest)
        prior["field_id"] = prior["field_id"].astype(str)
        new_rows = pd.DataFrame(rows)
        new_rows["field_id"] = new_rows["field_id"].astype(str)
        updated = pd.concat([prior, new_rows], ignore_index=True)
        updated = updated.drop_duplicates(["field_id", "method"], keep="last")
    else:
        updated = pd.DataFrame(rows)
        if not updated.empty:
            updated["field_id"] = updated["field_id"].astype(str)
    if not updated.empty:
        prior_manifest.parent.mkdir(parents=True, exist_ok=True)
        updated.sort_values(["field_id", "method"], kind="mergesort").to_csv(prior_manifest, index=False)
        shutil.copy2(prior_manifest, ROOT / "frontend" / "manifests" / "all_fields_projections_manifest.csv")
        (ROOT / "docs" / "site-assets" / "manifests").mkdir(parents=True, exist_ok=True)
        shutil.copy2(prior_manifest, ROOT / "docs" / "site-assets" / "manifests" / "all_fields_projections_manifest.csv")

    copy_web_figures(output_dir)
    print(f"Wrote {display_path(prior_manifest)}")
    print(f"Copied field method figures to docs and frontend assets.")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
