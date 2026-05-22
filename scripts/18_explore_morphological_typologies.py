from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphological_typologies import (
    build_morphological_typology_outputs,
    ensure_morphological_typology_outputs,
)
from src.temporal_common import display_path, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explore and write an exploratory typology of subfield morphology "
            "from the reduced eleven-metric embedding-space core."
        )
    )
    parser.add_argument(
        "--input-path",
        default="outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.csv",
        help="Reduced eleven-metric morphology table; CSV is preferred for portability.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/09_morphological_typologies",
        help="Output folder for model comparisons, assignments, and reports.",
    )
    parser.add_argument(
        "--figure-dir",
        default="memory/figures",
        help="Thesis figure folder for chapter-ready PNG exports.",
    )
    parser.add_argument(
        "--table-dir",
        default="memory/tables",
        help="Thesis table folder for chapter-ready LaTeX tables.",
    )
    parser.add_argument(
        "--docs-path",
        default="docs/morphological_typologies.md",
        help="Documentation note describing the typology pipeline and diagnostics.",
    )
    parser.add_argument("--selected-k", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selected_k < 2:
        raise ValueError("selected-k must be at least 2")
    if args.n_bootstrap < 0:
        raise ValueError("n-bootstrap must be non-negative")

    input_path = resolve_path(args.input_path, root=ROOT)
    output_dir = resolve_path(args.output_dir, root=ROOT)
    figure_dir = resolve_path(args.figure_dir, root=ROOT)
    table_dir = resolve_path(args.table_dir, root=ROOT)
    docs_path = resolve_path(args.docs_path, root=ROOT)

    ensure_morphological_typology_outputs(
        output_dir,
        figure_dir,
        table_dir,
        overwrite=args.overwrite,
        root=ROOT,
    )
    summary = build_morphological_typology_outputs(
        input_path=input_path,
        output_dir=output_dir,
        figure_dir=figure_dir,
        table_dir=table_dir,
        docs_path=docs_path,
        root=ROOT,
        selected_k=args.selected_k,
        random_seed=args.random_seed,
        n_bootstrap=args.n_bootstrap,
    )
    print(
        "Done: "
        f"{summary['n_subfields']} subfields, "
        f"{summary['n_metrics']} metrics, "
        f"selected k={int(summary['selected_solution']['k'])}."
    )
    print(f"Wrote {display_path(output_dir, root=ROOT)}")
    print(f"Wrote {display_path(docs_path, root=ROOT)}")


if __name__ == "__main__":
    main()

