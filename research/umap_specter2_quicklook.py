from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "research" / "results" / "specter2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a quick 2D UMAP projection from SPECTER2 embeddings."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-name", default="specter2_umap_2d")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--color-by", default="publication_year")
    return parser.parse_args()


def sample_rows(
    embeddings: np.ndarray, metadata: pd.DataFrame, sample_size: int | None, random_state: int
) -> tuple[np.ndarray, pd.DataFrame]:
    if sample_size is None or sample_size >= len(metadata):
        return embeddings, metadata.copy()

    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(metadata), size=sample_size, replace=False))
    return embeddings[indices], metadata.iloc[indices].reset_index(drop=True)


def add_plot_colors(ax, frame: pd.DataFrame, color_by: str):
    if color_by not in frame.columns:
        ax.scatter(frame["umap_x"], frame["umap_y"], s=4, alpha=0.65, linewidths=0)
        ax.set_title("SPECTER2 UMAP")
        return

    values = frame[color_by]
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.notna().sum() > 0.9 * len(values):
        scatter = ax.scatter(
            frame["umap_x"],
            frame["umap_y"],
            c=numeric_values,
            cmap="viridis",
            s=4,
            alpha=0.65,
            linewidths=0,
        )
        plt.colorbar(scatter, ax=ax, label=color_by)
        ax.set_title(f"SPECTER2 UMAP by {color_by}")
        return

    labels = values.fillna("missing").astype(str)
    top_labels = labels.value_counts().head(12).index
    plotted = labels.where(labels.isin(top_labels), other="other")
    categories = list(top_labels) + (["other"] if "other" in plotted.values else [])
    cmap = plt.get_cmap("tab20", len(categories))
    for idx, category in enumerate(categories):
        mask = plotted == category
        ax.scatter(
            frame.loc[mask, "umap_x"],
            frame.loc[mask, "umap_y"],
            s=4,
            alpha=0.65,
            linewidths=0,
            color=cmap(idx),
            label=category,
        )
    ax.legend(markerscale=3, fontsize=7, frameon=False, loc="best")
    ax.set_title(f"SPECTER2 UMAP by {color_by}")


def main() -> None:
    args = parse_args()
    embeddings_path = args.input_dir / "specter2_embeddings.npy"
    metadata_path = args.input_dir / "specter2_paper_metadata.csv"

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    embeddings, metadata = sample_rows(
        embeddings, metadata, args.sample_size, args.random_state
    )

    print(f"Running UMAP on {len(metadata):,} papers, embeddings={embeddings.shape}")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        low_memory=True,
    )
    coords = reducer.fit_transform(embeddings)

    output_frame = metadata.copy()
    output_frame.insert(1, "umap_x", coords[:, 0])
    output_frame.insert(2, "umap_y", coords[:, 1])

    csv_path = args.input_dir / f"{args.output_name}.csv"
    png_path = args.input_dir / f"{args.output_name}_{args.color_by}.png"
    output_frame.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    add_plot_colors(ax, output_frame, args.color_by)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.12, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved UMAP CSV: {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
