from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clean_embedding_core_metrics import (
    build_clean_embedding_core_outputs,
    existing_outputs,
    output_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean-core embedding metric tables and diagnostics."
    )
    parser.add_argument(
        "--input-path",
        default="data/processed/subfield_embedding_space_metrics.parquet",
        help="Broad embedding-space metric table produced by script 12.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/clean_embedding_core_metrics",
        help="Output folder for clean core metric diagnostics.",
    )
    parser.add_argument(
        "--min-non-missing-share",
        type=float,
        default=0.70,
        help="Warn when a clean-core metric has lower non-missing share.",
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


def ensure_outputs_do_not_exist(output_dir: Path, overwrite: bool) -> None:
    existing = existing_outputs(output_dir)
    if existing and not overwrite:
        formatted = ", ".join(display_path(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing clean embedding core outputs without "
            f"--overwrite: {formatted}"
        )


def main() -> None:
    args = parse_args()
    if not 0 < args.min_non_missing_share <= 1:
        raise ValueError("--min-non-missing-share must be in (0, 1]")

    input_path = resolve_path(args.input_path)
    output_dir = resolve_path(args.output_dir)
    ensure_outputs_do_not_exist(output_dir, args.overwrite)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing embedding metric table: {display_path(input_path)}. "
            "Run scripts/12_compute_subfield_embedding_space_metrics.py first."
        )

    metrics = pd.read_parquet(input_path)
    summary = build_clean_embedding_core_outputs(
        metrics,
        input_path=display_path(input_path),
        output_dir=output_dir,
        min_non_missing_share=args.min_non_missing_share,
    )

    paths = output_paths(output_dir)
    print(f"Read {len(metrics)} rows from {display_path(input_path)}")
    print(f"Wrote {display_path(paths['clean_parquet'])}")
    print(f"Wrote {display_path(paths['spearman_heatmap'])}")
    print(f"Wrote {display_path(paths['pearson_heatmap'])}")
    print(f"Wrote {display_path(paths['summary_md'])}")
    if summary["missing_final_metrics"]:
        print(
            "Warning: missing final metrics: "
            + ", ".join(summary["missing_final_metrics"])
        )
    if summary["high_missing_final_metrics"]:
        print(
            "Warning: high-missing final metrics: "
            + ", ".join(summary["high_missing_final_metrics"])
        )


if __name__ == "__main__":
    main()
