from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.growth_targets import (
    GrowthWindowConfig,
    build_balanced_year_panel,
    compute_growth_targets,
    prepare_main_analysis_subfields,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = GrowthWindowConfig()


def synthetic_subfields() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subfield_id": "1000",
                "subfield_display_name": "Shared Name",
                "field_id": "10",
                "field_display_name": "Field A",
                "domain_id": "1",
                "domain_display_name": "Domain One",
            },
            {
                "subfield_id": "1001",
                "subfield_display_name": "Growing Field",
                "field_id": "10",
                "field_display_name": "Field A",
                "domain_id": "1",
                "domain_display_name": "Domain One",
            },
            {
                "subfield_id": "2000",
                "subfield_display_name": "Shared Name",
                "field_id": "20",
                "field_display_name": "Field B",
                "domain_id": "2",
                "domain_display_name": "Domain Two",
            },
        ]
    )


def synthetic_counts() -> pd.DataFrame:
    annual_rates = {
        "1000": (10, 10),
        "1001": (10, 20),
        "2000": (10, 5),
    }
    rows = []
    for subfield_id, (input_rate, target_rate) in annual_rates.items():
        for year in range(CONFIG.input_year_min, CONFIG.input_year_max + 1):
            rows.append(
                {
                    "subfield_id": subfield_id,
                    "publication_year": year,
                    "n_works_article_preprint": input_rate,
                }
            )
        for year in range(CONFIG.target_year_min, CONFIG.target_year_max + 1):
            rows.append(
                {
                    "subfield_id": subfield_id,
                    "publication_year": year,
                    "n_works_article_preprint": target_rate,
                }
            )
    return pd.DataFrame(rows)


def synthetic_targets(counts: pd.DataFrame | None = None) -> pd.DataFrame:
    subfields = prepare_main_analysis_subfields(synthetic_subfields())
    panel, _, _ = build_balanced_year_panel(
        subfields,
        synthetic_counts() if counts is None else counts,
        config=CONFIG,
        count_column="n_works_article_preprint",
        count_source="synthetic",
        retrieved_at="2026-01-01T00:00:00+00:00",
        missing_count_policy="zero",
    )
    return compute_growth_targets(subfields, panel, config=CONFIG)


def test_annualized_rates_use_10_year_and_6_year_windows() -> None:
    targets = synthetic_targets()
    row = targets.set_index("subfield_id").loc["1000"]

    assert row["papers_2010_2019"] == 100
    assert row["papers_2020_2025"] == 60
    assert row["annual_rate_2010_2019"] == 10
    assert row["annual_rate_2020_2025"] == 10


def test_equal_annual_rates_produce_zero_log_growth() -> None:
    row = synthetic_targets().set_index("subfield_id").loc["1000"]

    assert math.isclose(row["annualized_log_growth"], 0.0, abs_tol=1e-12)


def test_higher_target_annual_rate_produces_positive_growth() -> None:
    row = synthetic_targets().set_index("subfield_id").loc["1001"]

    assert row["annualized_log_growth"] > 0


def test_lower_target_annual_rate_produces_negative_growth() -> None:
    row = synthetic_targets().set_index("subfield_id").loc["2000"]

    assert row["annualized_log_growth"] < 0


def test_growth_above_median_uses_strict_threshold() -> None:
    targets = synthetic_targets().set_index("subfield_id")

    assert math.isclose(targets.loc["1000", "global_growth_median"], 0.0, abs_tol=1e-12)
    assert bool(targets.loc["1000", "growth_above_median"]) is False
    assert bool(targets.loc["1001", "growth_above_median"]) is True
    assert bool(targets.loc["2000", "growth_above_median"]) is False


def test_domain_adjusted_growth_subtracts_within_domain_median() -> None:
    targets = synthetic_targets().set_index("subfield_id")
    positive_growth = targets.loc["1001", "annualized_log_growth"]
    expected_domain_one_median = positive_growth / 2

    assert math.isclose(
        targets.loc["1000", "domain_growth_median"],
        expected_domain_one_median,
        rel_tol=1e-12,
    )
    assert math.isclose(
        targets.loc["1001", "domain_adjusted_annualized_log_growth"],
        positive_growth - expected_domain_one_median,
        rel_tol=1e-12,
    )
    assert bool(targets.loc["2000", "growth_above_domain_median"]) is False


def test_duplicate_display_names_do_not_collapse_subfield_ids() -> None:
    targets = synthetic_targets()
    duplicated = targets[targets["subfield_display_name"] == "Shared Name"]

    assert len(duplicated) == 2
    assert set(duplicated["subfield_id"]) == {"1000", "2000"}
    assert duplicated["subfield_display_name_is_duplicated"].tolist() == [True, True]


def test_missing_subfield_year_combinations_are_completed_in_panel() -> None:
    subfields = prepare_main_analysis_subfields(synthetic_subfields())
    counts = synthetic_counts()
    counts = counts[
        ~(
            (counts["subfield_id"] == "1000")
            & (counts["publication_year"] == CONFIG.input_year_min)
        )
    ].copy()

    panel, warnings, assumptions = build_balanced_year_panel(
        subfields,
        counts,
        config=CONFIG,
        count_column="n_works_article_preprint",
        count_source="synthetic sparse",
        missing_count_policy="zero",
    )

    missing_row = panel[
        (panel["subfield_id"] == "1000")
        & (panel["publication_year"] == CONFIG.input_year_min)
    ].iloc[0]
    assert len(panel) == len(subfields) * len(CONFIG.all_years)
    assert missing_row["works_count"] == 0
    assert bool(missing_row["count_row_present"]) is False
    assert warnings
    assert any("interpreted as zero" in assumption for assumption in assumptions)


def test_output_has_exactly_one_row_per_input_subfield() -> None:
    subfields = prepare_main_analysis_subfields(synthetic_subfields())
    targets = synthetic_targets()

    assert len(targets) == len(subfields)
    assert targets["subfield_id"].is_unique


def test_cli_runs_on_tiny_synthetic_count_table(tmp_path: Path) -> None:
    subfields_path = tmp_path / "subfields.parquet"
    counts_path = tmp_path / "counts.csv"
    output_path = tmp_path / "targets.parquet"
    output_csv = tmp_path / "targets.csv"
    panel_path = tmp_path / "panel.parquet"
    panel_csv = tmp_path / "panel.csv"
    summary_path = tmp_path / "summary.json"
    rankings_path = tmp_path / "rankings.csv"
    domain_summary_path = tmp_path / "domain_summary.csv"
    figures_dir = tmp_path / "figures"

    synthetic_subfields().to_parquet(subfields_path, index=False)
    synthetic_counts().to_csv(counts_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "13_build_subfield_growth_targets.py"),
            "--subfields-path",
            str(subfields_path),
            "--counts-path",
            str(counts_path),
            "--output-path",
            str(output_path),
            "--output-csv",
            str(output_csv),
            "--yearly-panel-path",
            str(panel_path),
            "--yearly-panel-csv",
            str(panel_csv),
            "--summary-path",
            str(summary_path),
            "--rankings-path",
            str(rankings_path),
            "--domain-summary-path",
            str(domain_summary_path),
            "--figures-dir",
            str(figures_dir),
            "--expected-subfields",
            "3",
            "--dpi",
            "80",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    targets = pd.read_parquet(output_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(targets) == 3
    assert output_csv.exists()
    assert panel_path.exists()
    assert panel_csv.exists()
    assert rankings_path.exists()
    assert domain_summary_path.exists()
    assert summary["n_subfields_output"] == 3
    for figure_name in [
        "annualized_log_growth_histogram.png",
        "annual_rates_scatter.png",
        "growth_by_domain_boxplot.png",
        "domain_adjusted_growth_histogram.png",
        "top_bottom_growth_barplots.png",
        "top_bottom_domain_adjusted_growth_barplots.png",
        "yearly_counts_examples.png",
    ]:
        assert (figures_dir / figure_name).exists()
