from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric_family_comparison import (
    analogue_pair_correlations,
    comparison_summary_payload,
    join_metric_families,
    markdown_summary,
    pairwise_correlations,
    plot_correlation_heatmap,
    top_absolute_correlations,
    weak_umap_relationships,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare projected UMAP and embedding-space metric families."
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
        default="outputs/analysis/metric_family_comparison",
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


def ensure_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing comparison outputs without --overwrite: {formatted}"
        )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    umap_path = resolve_path(args.umap_metrics_path)
    embedding_path = resolve_path(args.embedding_metrics_path)
    output_dir = resolve_path(args.output_dir)
    summary_json = output_dir / "metric_family_comparison_summary.json"
    summary_md = output_dir / "metric_family_comparison_summary.md"
    spearman_csv = output_dir / "cross_family_spearman_correlations.csv"
    pearson_csv = output_dir / "cross_family_pearson_correlations.csv"
    top_csv = output_dir / "top_absolute_spearman_correlations.csv"
    analogue_csv = output_dir / "analogue_metric_pair_correlations.csv"
    heatmap_png = output_dir / "cross_family_spearman_heatmap.png"
    ensure_outputs(
        [
            summary_json,
            summary_md,
            spearman_csv,
            pearson_csv,
            top_csv,
            analogue_csv,
            heatmap_png,
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
    joined = join_metric_families(umap_metrics, embedding_metrics)
    spearman = pairwise_correlations(joined, method="spearman")
    pearson = pairwise_correlations(joined, method="pearson")
    top = top_absolute_correlations(spearman, n=25)
    analogue = analogue_pair_correlations(joined, method="spearman")
    weak = weak_umap_relationships(spearman)

    output_dir.mkdir(parents=True, exist_ok=True)
    spearman.to_csv(spearman_csv, index=False)
    pearson.to_csv(pearson_csv, index=False)
    top.to_csv(top_csv, index=False)
    analogue.to_csv(analogue_csv, index=False)
    plot_correlation_heatmap(spearman, heatmap_png)

    payload = comparison_summary_payload(
        umap_rows=len(umap_metrics),
        embedding_rows=len(embedding_metrics),
        joined=joined,
        spearman=spearman,
        pearson=pearson,
        analogue=analogue,
    )
    payload.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_paths": {
                "umap_metrics_path": display_path(umap_path),
                "embedding_metrics_path": display_path(embedding_path),
            },
            "output_dir": display_path(output_dir),
        }
    )
    write_json(summary_json, payload)
    summary_text = markdown_summary(
        payload=payload,
        top_spearman=top,
        analogue=analogue,
        weak_umap=weak,
    )
    summary_md.write_text(summary_text, encoding="utf-8")

    print(f"Wrote {display_path(summary_json)}")
    print(f"Wrote {display_path(summary_md)}")
    print(f"Wrote {display_path(spearman_csv)}")
    print(f"Wrote {display_path(pearson_csv)}")
    print(f"Wrote {display_path(top_csv)}")
    print(f"Wrote {display_path(analogue_csv)}")
    print(f"Wrote {display_path(heatmap_png)}")


if __name__ == "__main__":
    main()
