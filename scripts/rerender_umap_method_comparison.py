from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_all_projections import plot_projection_panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-render existing per-subfield UMAP coordinates in the same short-title "
            "style used by the method-comparison PCA, t-SNE, and PHATE figures."
        )
    )
    parser.add_argument(
        "--umap-coordinates-dir",
        default="outputs/08_visualization/per_subfield_umap_smooth_density/coordinates",
        help="Directory containing the existing per-subfield UMAP coordinate parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default="frontend",
        help="Frontend directory where method-comparison figures and coordinates are written.",
    )
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional debug limit for rendering only the first N coordinate files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = ROOT / args.umap_coordinates_dir
    output_dir = ROOT / args.output_dir
    figure_root = output_dir / "figures" / "method_comparison"
    coordinate_root = output_dir / "coordinates" / "method_comparison"

    if not source_dir.exists():
        raise FileNotFoundError(f"UMAP coordinates directory not found: {source_dir}")

    coordinate_files = sorted(source_dir.glob("*.parquet"))
    if args.limit:
        coordinate_files = coordinate_files[: args.limit]

    print(f"Found {len(coordinate_files)} UMAP coordinate files.")
    rendered = 0

    for coordinate_file in coordinate_files:
        stem = coordinate_file.stem
        frame = pd.read_parquet(coordinate_file)
        required = {"umap_x", "umap_y", "subfield_display_name"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{coordinate_file} is missing required columns: {sorted(missing)}")

        coordinates = frame[["umap_x", "umap_y"]].to_numpy(dtype=float)
        subfield_name = str(frame["subfield_display_name"].iloc[0])

        figure_path = figure_root / stem / "UMAP.png"
        coordinate_path = coordinate_root / stem / "UMAP.parquet"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        coordinate_path.parent.mkdir(parents=True, exist_ok=True)

        output_frame = frame.copy()
        output_frame["x"] = output_frame["umap_x"].astype("float32")
        output_frame["y"] = output_frame["umap_y"].astype("float32")
        output_frame.to_parquet(coordinate_path, index=False)

        plot_projection_panels(
            coordinates,
            subfield_name=subfield_name,
            method="UMAP",
            n_used=len(frame),
            year_min=args.year_min,
            year_max=args.year_max,
            output_path=figure_path,
            dpi=args.dpi,
        )

        rendered += 1
        if rendered % 25 == 0 or rendered == len(coordinate_files):
            print(f"Rendered {rendered}/{len(coordinate_files)} UMAP figures.")

    print(f"Done. Re-rendered {rendered} UMAP method-comparison figures in {figure_root}.")


if __name__ == "__main__":
    main()
