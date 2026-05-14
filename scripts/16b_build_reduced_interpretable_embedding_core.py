from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reduced_interpretable_embedding_core import (
    build_reduced_interpretable_embedding_core_outputs,
    existing_outputs,
    output_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reduced interpretable embedding-core metrics and diagnostics."
    )
    parser.add_argument(
        "--input",
        default="data/processed/subfield_embedding_space_metrics.parquet",
        help="Full embedding-space metric table produced by script 12.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/reduced_interpretable_embedding_core",
        help="Output folder for reduced interpretable embedding-core diagnostics.",
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
            "Refusing to overwrite existing reduced embedding-core outputs without "
            f"--overwrite: {formatted}"
        )


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    ensure_outputs_do_not_exist(output_dir, args.overwrite)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing embedding metric table: {display_path(input_path)}. "
            "Run scripts/12_compute_subfield_embedding_space_metrics.py first."
        )

    metrics = pd.read_parquet(input_path)
    summary = build_reduced_interpretable_embedding_core_outputs(
        metrics,
        input_path=display_path(input_path),
        output_dir=output_dir,
    )
    paths = output_paths(output_dir)
    print(f"Read {summary['n_rows_input']} rows from {display_path(input_path)}")
    print(f"Eligible rows: {summary['n_rows_eligible']}")
    print(f"Wrote {display_path(paths['metrics_parquet'])}")
    print(f"Wrote {display_path(paths['spearman_heatmap'])}")
    print(f"Wrote {display_path(paths['pearson_heatmap'])}")
    print(f"Wrote {display_path(paths['histograms'])}")
    print(f"Wrote {display_path(paths['summary_md'])}")
    if summary["missing_core_metrics"]:
        print(
            "Warning: missing reduced-core metrics: "
            + ", ".join(summary["missing_core_metrics"])
        )
    if summary["distribution_flagged_metrics"]:
        print(
            "Warning: distribution flags: "
            + ", ".join(summary["distribution_flagged_metrics"])
        )


if __name__ == "__main__":
    main()
