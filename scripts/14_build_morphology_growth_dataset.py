from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.growth_targets import display_path, read_table
from src.morphology_growth_dataset import (
    build_data_dictionary,
    build_domain_target_summary,
    build_family_difference_rankings,
    build_family_score_target_summary,
    build_growth_rankings,
    build_morphology_growth_dataset,
    build_summary_payload,
    build_target_feature_correlations,
    save_outputs,
    write_stage14_figures,
)


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Build the Stage 14 morphology-growth modelling dataset."
    )
    parser.add_argument(
        "--morphology-path",
        default="outputs/metrics/morphology_analysis/tables/curated_model_features.parquet",
    )
    parser.add_argument(
        "--raw-morphology-path",
        default="data/processed/subfield_morphology_metrics.parquet",
    )
    parser.add_argument(
        "--family-scores-path",
        default="outputs/metrics/morphology_analysis/tables/family_scores.csv",
    )
    parser.add_argument(
        "--pca-scores-path",
        default="outputs/metrics/morphology_analysis/tables/metric_pca_scores.csv",
    )
    parser.add_argument(
        "--growth-path",
        default="data/processed/subfield_growth_targets.parquet",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/subfield_morphology_growth_dataset.parquet",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/subfield_morphology_growth_dataset.csv",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/modeling/stage14_morphology_growth_dataset_summary.json",
    )
    parser.add_argument(
        "--dictionary-path",
        default="outputs/modeling/stage14_morphology_growth_dataset_dictionary.csv",
    )
    parser.add_argument(
        "--feature-groups-path",
        default="outputs/modeling/stage14_feature_groups.json",
    )
    parser.add_argument(
        "--join-audit-path",
        default="outputs/modeling/stage14_join_audit.csv",
    )
    parser.add_argument(
        "--target-feature-correlations-path",
        default="outputs/modeling/stage14_target_feature_correlations.csv",
    )
    parser.add_argument(
        "--family-score-target-summary-path",
        default="outputs/modeling/stage14_family_score_target_summary.csv",
    )
    parser.add_argument(
        "--domain-target-summary-path",
        default="outputs/modeling/stage14_domain_target_summary.csv",
    )
    parser.add_argument(
        "--growth-rankings-path",
        default="outputs/modeling/stage14_growth_rankings.csv",
    )
    parser.add_argument(
        "--family-difference-rankings-path",
        default="outputs/modeling/stage14_family_difference_rankings.csv",
    )
    parser.add_argument("--figures-dir", default="outputs/modeling/stage14_figures")
    parser.add_argument("--expected-rows", type=int, default=240)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_optional_table(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Optional input missing and skipped: {display_path(path, ROOT)}")
        return None
    return read_table(path)


def ensure_outputs_do_not_exist(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path, ROOT) for path in existing[:12])
        raise FileExistsError(
            "Refusing to overwrite Stage 14 outputs without --overwrite: " + formatted
        )


def main() -> None:
    args = parse_args()
    if args.expected_rows < 0:
        raise ValueError("--expected-rows must be non-negative")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    morphology_path = resolve_path(args.morphology_path)
    raw_morphology_path = resolve_path(args.raw_morphology_path)
    family_scores_path = resolve_path(args.family_scores_path)
    pca_scores_path = resolve_path(args.pca_scores_path)
    growth_path = resolve_path(args.growth_path)
    figures_dir = resolve_path(args.figures_dir)
    paths = {
        "output_path": resolve_path(args.output_path),
        "output_csv": resolve_path(args.output_csv),
        "summary_path": resolve_path(args.summary_path),
        "dictionary_path": resolve_path(args.dictionary_path),
        "feature_groups_path": resolve_path(args.feature_groups_path),
        "join_audit_path": resolve_path(args.join_audit_path),
        "target_feature_correlations_path": resolve_path(
            args.target_feature_correlations_path
        ),
        "family_score_target_summary_path": resolve_path(
            args.family_score_target_summary_path
        ),
        "domain_target_summary_path": resolve_path(args.domain_target_summary_path),
        "growth_rankings_path": resolve_path(args.growth_rankings_path),
        "family_difference_rankings_path": resolve_path(
            args.family_difference_rankings_path
        ),
    }
    figure_paths = [
        figures_dir / "target_distribution.png",
        figures_dir / "domain_adjusted_target_distribution.png",
        figures_dir / "growth_by_domain_boxplot.png",
        figures_dir / "family_scores_vs_growth.png",
        figures_dir / "family_scores_by_growth_label.png",
        figures_dir / "core_metric_target_correlation_barplot.png",
        figures_dir / "morphology_growth_pca_or_family_scatter.png",
    ]
    ensure_outputs_do_not_exist(list(paths.values()) + figure_paths, args.overwrite)

    if not morphology_path.exists():
        raise FileNotFoundError(f"Missing morphology table: {display_path(morphology_path, ROOT)}")
    if not growth_path.exists():
        raise FileNotFoundError(f"Missing growth table: {display_path(growth_path, ROOT)}")

    warnings: list[str] = []
    assumptions = [
        "Stage 14 joins morphology and growth strictly by subfield_id.",
        "Morphology features are treated as 2010-2019 inputs.",
        "Growth columns are treated as 2020-2025 outcomes or audit controls, not predictors.",
        "No prediction model is trained in Stage 14.",
    ]

    print(f"Loading morphology features: {display_path(morphology_path, ROOT)}")
    morphology = read_table(morphology_path)
    print(f"Loading growth targets: {display_path(growth_path, ROOT)}")
    growth = read_table(growth_path)
    raw_morphology = read_optional_table(raw_morphology_path, warnings)
    family_scores = read_optional_table(family_scores_path, warnings)
    pca_scores = read_optional_table(pca_scores_path, warnings)

    dataset, feature_groups, join_audit, validation = build_morphology_growth_dataset(
        morphology=morphology,
        growth=growth,
        raw_morphology=raw_morphology,
        family_scores=family_scores,
        pca_scores=pca_scores,
        expected_rows=args.expected_rows,
    )

    dictionary = build_data_dictionary(dataset, feature_groups)
    domain_summary = build_domain_target_summary(dataset)
    family_summary = build_family_score_target_summary(
        dataset, feature_groups["family_score_columns"]
    )
    correlations = build_target_feature_correlations(
        dataset,
        core_columns=feature_groups["core_morphology_metric_columns"],
        family_columns=feature_groups["family_score_columns"],
    )
    growth_rankings = build_growth_rankings(dataset, n=args.top_n)
    family_difference_rankings = build_family_difference_rankings(
        family_summary,
        n=args.top_n,
    )
    written_figures = write_stage14_figures(
        dataset=dataset,
        feature_groups=feature_groups,
        correlations=correlations,
        figures_dir=figures_dir,
        dpi=args.dpi,
    )

    input_paths = {
        "morphology_path": display_path(morphology_path, ROOT),
        "raw_morphology_path": display_path(raw_morphology_path, ROOT)
        if raw_morphology is not None
        else "",
        "family_scores_path": display_path(family_scores_path, ROOT)
        if family_scores is not None
        else "",
        "pca_scores_path": display_path(pca_scores_path, ROOT)
        if pca_scores is not None
        else "",
        "growth_path": display_path(growth_path, ROOT),
    }
    output_paths = {
        key: display_path(path, ROOT)
        for key, path in paths.items()
    }
    output_paths.update(
        {
            f"figure_{name}": display_path(path, ROOT)
            for name, path in written_figures.items()
        }
    )
    summary = build_summary_payload(
        dataset=dataset,
        morphology_rows=len(morphology),
        growth_rows=len(growth),
        audit=join_audit,
        feature_groups=feature_groups,
        validation=validation,
        input_paths=input_paths,
        output_paths=output_paths,
        warnings=warnings,
        assumptions=assumptions,
    )

    save_outputs(
        dataset=dataset,
        feature_groups=feature_groups,
        audit=join_audit,
        dictionary=dictionary,
        domain_summary=domain_summary,
        family_summary=family_summary,
        correlations=correlations,
        growth_rankings=growth_rankings,
        family_difference_rankings=family_difference_rankings,
        summary=summary,
        paths=paths,
    )

    print(f"Wrote {display_path(paths['output_path'], ROOT)}")
    print(f"Wrote {display_path(paths['output_csv'], ROOT)}")
    print(f"Wrote {display_path(paths['summary_path'], ROOT)}")
    print(f"Wrote {display_path(paths['feature_groups_path'], ROOT)}")
    print(f"Wrote figures under {display_path(figures_dir, ROOT)}")
    print(
        f"Rows: morphology={len(morphology)}, growth={len(growth)}, joined={len(dataset)}"
    )
    print(
        "Global label balance: "
        + str(summary["class_balance_growth_above_median"])
    )
    print(
        "Domain label balance: "
        + str(summary["class_balance_growth_above_domain_median"])
    )
    print(
        "Feature groups: "
        + ", ".join(
            f"{key}={value}"
            for key, value in summary["feature_group_sizes"].items()
            if key.endswith("_columns")
        )
    )


if __name__ == "__main__":
    main()
