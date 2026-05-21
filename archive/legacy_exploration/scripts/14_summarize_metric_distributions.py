from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric_distribution_diagnostics import (
    EMBEDDING_METRIC_COLUMNS,
    UMAP_METRIC_COLUMNS,
    build_distribution_summary,
    low_information_metrics,
    markdown_distribution_summary,
    plot_metric_histograms,
    plot_zscore_boxplots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize distributions of projected and embedding-space metrics."
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
        "--output-dir",
        default="outputs/analysis/metric_distributions",
    )
    parser.add_argument("--high-missing-threshold", type=float, default=0.30)
    parser.add_argument("--low-unique-threshold", type=int, default=5)
    parser.add_argument("--low-unique-share-threshold", type=float, default=0.05)
    parser.add_argument("--zero-dominated-threshold", type=float, default=0.80)
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


def ensure_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing distribution outputs without "
            f"--overwrite: {formatted}"
        )


def main() -> None:
    args = parse_args()
    umap_path = resolve_path(args.umap_metrics_path)
    embedding_path = resolve_path(args.embedding_metrics_path)
    output_dir = resolve_path(args.output_dir)
    summary_csv = output_dir / "metric_distribution_summary.csv"
    low_info_csv = output_dir / "low_information_metrics.csv"
    summary_md = output_dir / "metric_distribution_summary.md"
    umap_hist_png = output_dir / "umap_metric_histograms.png"
    embedding_hist_png = output_dir / "embedding_metric_histograms.png"
    boxplot_png = output_dir / "all_metric_boxplots_zscore.png"
    ensure_outputs(
        [
            summary_csv,
            low_info_csv,
            summary_md,
            umap_hist_png,
            embedding_hist_png,
            boxplot_png,
        ],
        args.overwrite,
    )
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

    umap_metrics = pd.read_parquet(umap_path)
    embedding_metrics = pd.read_parquet(embedding_path)
    summary = build_distribution_summary(
        umap_metrics,
        embedding_metrics,
        high_missing_threshold=args.high_missing_threshold,
        low_unique_threshold=args.low_unique_threshold,
        low_unique_share_threshold=args.low_unique_share_threshold,
        zero_dominated_threshold=args.zero_dominated_threshold,
    )
    low_info = low_information_metrics(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    low_info.to_csv(low_info_csv, index=False)
    summary_md.write_text(
        markdown_distribution_summary(summary, low_info),
        encoding="utf-8",
    )
    plot_metric_histograms(
        umap_metrics,
        UMAP_METRIC_COLUMNS,
        umap_hist_png,
        title="Projected UMAP Metric Histograms",
    )
    plot_metric_histograms(
        embedding_metrics,
        EMBEDDING_METRIC_COLUMNS,
        embedding_hist_png,
        title="Embedding-Space Metric Histograms",
    )
    plot_zscore_boxplots(umap_metrics, embedding_metrics, boxplot_png)

    print(f"Wrote {display_path(summary_csv)}")
    print(f"Wrote {display_path(low_info_csv)}")
    print(f"Wrote {display_path(summary_md)}")
    print(f"Wrote {display_path(umap_hist_png)}")
    print(f"Wrote {display_path(embedding_hist_png)}")
    print(f"Wrote {display_path(boxplot_png)}")


if __name__ == "__main__":
    main()
