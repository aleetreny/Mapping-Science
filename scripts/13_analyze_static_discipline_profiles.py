from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.static_discipline_profiles import (
    build_static_discipline_profile_outputs,
    ensure_static_profile_outputs,
    output_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze static discipline profiles from the reduced SPECTER2 "
            "embedding-space metric core."
        )
    )
    parser.add_argument(
        "--input",
        default="outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.parquet",
        help="Active 8-metric structural embedding-space core table.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/05_static_comparison",
        help="Output folder for static comparison profiles, rankings, and pairwise distances.",
    )
    parser.add_argument("--top-n", type=int, default=10)
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


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("top_n must be positive")

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    ensure_static_profile_outputs(output_dir, overwrite=args.overwrite, root=ROOT)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing reduced metric core: {display_path(input_path)}. "
            "Run scripts/12_build_reduced_metric_core.py first."
        )

    metrics = pd.read_parquet(input_path)
    summary = build_static_discipline_profile_outputs(
        metrics,
        input_path=display_path(input_path),
        output_dir=output_dir,
        top_n=args.top_n,
    )
    paths = output_paths(output_dir)
    print(f"Read {summary['n_rows_input']} rows from {display_path(input_path)}")
    print(f"Wrote {display_path(paths['profiles_parquet'])}")
    print(f"Wrote {display_path(paths['metric_rankings_csv'])}")
    print(f"Wrote {display_path(paths['top_profile_pairs_csv'])}")
    print(f"Wrote {display_path(paths['summary_md'])}")


if __name__ == "__main__":
    main()
