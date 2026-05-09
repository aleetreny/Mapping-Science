from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage import ensure_dirs, load_parquet, save_parquet
from src.per_subfield_umap_maps import validate_year_window
from src.umap_maps import (
    balanced_sample_by_subfield,
    build_umap_output_frame,
)


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Build the first balanced-sample UMAP map from main embeddings."
    )
    parser.add_argument(
        "--embedding-dir",
        default=os.getenv("LOCAL_EMBEDDINGS_DIR", "embeddings/specter2_v1"),
        help="Local folder containing SPECTER2 embedding artifacts.",
    )
    parser.add_argument("--sample-per-subfield", type=int, default=500)
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--year-max", type=int, default=2025)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_outputs_do_not_exist(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing outputs without --force: {formatted}"
        )


def plot_umap(output: pd.DataFrame, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    domains = pd.Categorical(output["domain_display_name"].fillna("Unknown"))
    codes = domains.codes

    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)
    scatter = ax.scatter(
        output["umap_x"],
        output["umap_y"],
        c=codes,
        cmap="tab20",
        s=2,
        alpha=0.65,
        linewidths=0,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("SPECTER2 UMAP Balanced Main-Analysis Sample")
    ax.grid(False)

    handles = []
    labels = [str(label) for label in domains.categories]
    color_values = np.linspace(0, 1, max(len(labels), 1))
    cmap = scatter.cmap
    for label, color_value in zip(labels, color_values):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=cmap(color_value),
                markersize=5,
                label=label,
            )
        )
    if handles:
        ax.legend(
            handles=handles,
            title="Domain",
            loc="best",
            fontsize=7,
            title_fontsize=8,
            frameon=False,
        )

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_year_window(args.year_min, args.year_max)
    config = load_config()
    ensure_dirs(config)

    processed_dir = ROOT / config["storage"]["processed_dir"]
    embedding_dir = ROOT / args.embedding_dir
    analysis_index_path = processed_dir / "analysis_embedding_index.parquet"
    matrix_path = embedding_dir / "analysis" / "main_embeddings.float16.npy"

    output_dir = ROOT / "outputs" / "maps"
    parquet_path = output_dir / "umap_global_sample.parquet"
    png_path = output_dir / "umap_global_sample.png"
    summary_path = output_dir / "umap_global_sample_summary.json"

    ensure_outputs_do_not_exist([parquet_path, png_path, summary_path], args.force)

    if not analysis_index_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/analysis_embedding_index.parquet. "
            "Run scripts/08_prepare_analysis_matrix.py first."
        )
    if not matrix_path.exists():
        raise FileNotFoundError(
            "Missing embeddings/specter2_v1/analysis/main_embeddings.float16.npy. "
            "Run scripts/08_prepare_analysis_matrix.py first."
        )

    analysis_index = load_parquet(analysis_index_path)
    if "publication_year" not in analysis_index.columns:
        raise ValueError("analysis_embedding_index.parquet must contain publication_year")
    years = pd.to_numeric(analysis_index["publication_year"], errors="coerce")
    analysis_index = analysis_index.loc[
        years.between(args.year_min, args.year_max, inclusive="both")
    ].copy()
    analysis_index["publication_year"] = years.loc[analysis_index.index].astype("int64")
    if analysis_index.empty:
        raise ValueError(
            f"No analysis rows remain in year window {args.year_min}-{args.year_max}"
        )
    sampled = balanced_sample_by_subfield(
        analysis_index=analysis_index,
        sample_per_subfield=args.sample_per_subfield,
        random_seed=args.random_seed,
    )

    matrix = np.load(matrix_path, mmap_mode="r")
    try:
        row_ids = sampled["analysis_row_id"].astype(int).to_numpy()
        sample_matrix = np.asarray(matrix[row_ids], dtype=np.float32)
    finally:
        del matrix

    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.05,
        metric="cosine",
        random_state=args.random_seed,
        low_memory=True,
        verbose=True,
    )
    coordinates = reducer.fit_transform(sample_matrix)

    output = build_umap_output_frame(sampled, coordinates)
    save_parquet(output, parquet_path)
    plot_umap(output, png_path)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "year_window": {
            "year_min": args.year_min,
            "year_max": args.year_max,
        },
        "sample_per_subfield": args.sample_per_subfield,
        "random_seed": args.random_seed,
        "n_points": int(len(output)),
        "n_subfields": int(output["subfield_id"].nunique()),
        "matrix_path": str(matrix_path.relative_to(ROOT)),
        "analysis_index_path": str(analysis_index_path.relative_to(ROOT)),
        "parquet_path": str(parquet_path.relative_to(ROOT)),
        "png_path": str(png_path.relative_to(ROOT)),
        "umap": {
            "n_components": 2,
            "n_neighbors": 30,
            "min_dist": 0.05,
            "metric": "cosine",
            "low_memory": True,
        },
        "color": "domain_display_name",
    }
    write_json(summary_path, summary)

    print(f"Wrote {parquet_path.relative_to(ROOT)}")
    print(f"Wrote {png_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
