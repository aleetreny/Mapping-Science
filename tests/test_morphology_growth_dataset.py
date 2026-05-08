from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.morphology_growth_dataset import (
    FAMILY_SCORE_COLUMNS,
    build_morphology_growth_dataset,
)
from src.morphology_metrics import CORE_METRIC_COLUMNS_V2, DIAGNOSTIC_METRIC_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


def synthetic_morphology() -> pd.DataFrame:
    rows = []
    base = [
        ("1000", "Shared Name", "1", "Domain One", False),
        ("1001", "Growing Field", "1", "Domain One", False),
        ("2000", "Shared Name", "2", "Domain Two", False),
    ]
    for row_index, (subfield_id, name, domain_id, domain_name, _) in enumerate(base):
        row = {
            "subfield_id": subfield_id,
            "subfield_display_name": name,
            "subfield_label_unique": f"{subfield_id} | {domain_name} / Field {domain_id} / {name}",
            "subfield_label_short": f"{subfield_id} | Field {domain_id} / {name}",
            "subfield_display_name_is_duplicated": name == "Shared Name",
            "field_id": f"F{domain_id}",
            "field_display_name": f"Field {domain_id}",
            "domain_id": domain_id,
            "domain_display_name": domain_name,
            "n_points": 3000,
        }
        for metric_index, column in enumerate(CORE_METRIC_COLUMNS_V2):
            row[column] = float(row_index + metric_index / 100.0)
        for score_index, column in enumerate(FAMILY_SCORE_COLUMNS):
            row[column] = float((row_index - 1) + score_index / 10.0)
        rows.append(row)
    return pd.DataFrame(rows)


def synthetic_raw_morphology() -> pd.DataFrame:
    raw = synthetic_morphology()[["subfield_id"]].copy()
    raw["year_min"] = 2010
    raw["year_max"] = 2019
    raw["metric_status"] = "completed"
    raw["n_years_available"] = 10
    raw["n_early_points"] = 900
    raw["n_late_points"] = 900
    raw["n_density_entropy_years"] = 10
    for index, column in enumerate(DIAGNOSTIC_METRIC_COLUMNS):
        raw[column] = index / 10.0
    return raw


def synthetic_growth() -> pd.DataFrame:
    rows = []
    values = [
        ("1000", 100, 60, 0.0, False, 0.0, False),
        ("1001", 100, 120, 0.6466, True, 0.3, True),
        ("2000", 100, 30, -0.6061, False, 0.0, False),
    ]
    for subfield_id, papers_in, papers_out, growth, above, adjusted, above_domain in values:
        rows.append(
            {
                "subfield_id": subfield_id,
                "papers_2010_2019": papers_in,
                "papers_2020_2025": papers_out,
                "annual_rate_2010_2019": papers_in / 10,
                "annual_rate_2020_2025": papers_out / 6,
                "annualized_growth_abs": papers_out / 6 - papers_in / 10,
                "annualized_growth_ratio": ((papers_out / 6) + 1) / ((papers_in / 10) + 1),
                "annualized_growth_pct": ((papers_out / 6) + 1) / ((papers_in / 10) + 1) - 1,
                "annualized_log_growth": growth,
                "global_growth_median": 0.0,
                "growth_above_median": above,
                "growth_rank": 1,
                "growth_percentile": 0.5,
                "domain_growth_median": 0.0,
                "domain_adjusted_annualized_log_growth": adjusted,
                "growth_above_domain_median": above_domain,
                "domain_growth_rank": 1,
                "domain_growth_percentile": 0.5,
                "input_year_min": 2010,
                "input_year_max": 2019,
                "target_year_min": 2020,
                "target_year_max": 2025,
                "input_years": 10,
                "target_years": 6,
                "n_years_with_counts_2010_2019": 10,
                "n_years_with_counts_2020_2025": 6,
                "has_zero_input_count": False,
                "has_zero_target_count": False,
                "count_source": "synthetic",
            }
        )
    return pd.DataFrame(rows)


def synthetic_pca() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subfield_id": ["1000", "1001", "2000"],
            "PC1": [-1.0, 0.0, 1.0],
            "PC2": [0.5, -0.5, 0.25],
        }
    )


def build_synthetic_dataset():
    return build_morphology_growth_dataset(
        morphology=synthetic_morphology(),
        growth=synthetic_growth(),
        raw_morphology=synthetic_raw_morphology(),
        pca_scores=synthetic_pca(),
        expected_rows=3,
    )


def test_one_to_one_join_by_subfield_id() -> None:
    dataset, _, audit, _ = build_synthetic_dataset()

    assert len(dataset) == 3
    assert dataset["subfield_id"].tolist() == ["1000", "1001", "2000"]
    assert audit["join_status"].eq("matched").all()


def test_duplicate_subfield_ids_raise_errors() -> None:
    morphology = pd.concat([synthetic_morphology(), synthetic_morphology().head(1)])

    with pytest.raises(ValueError, match="duplicate subfield_id"):
        build_morphology_growth_dataset(
            morphology=morphology,
            growth=synthetic_growth(),
            expected_rows=0,
        )


def test_missing_morphology_rows_raise_errors() -> None:
    morphology = synthetic_morphology().iloc[:2].copy()

    with pytest.raises(ValueError, match="growth rows missing morphology"):
        build_morphology_growth_dataset(
            morphology=morphology,
            growth=synthetic_growth(),
            expected_rows=0,
        )


def test_missing_growth_rows_raise_errors() -> None:
    growth = synthetic_growth().iloc[:2].copy()

    with pytest.raises(ValueError, match="morphology rows missing growth"):
        build_morphology_growth_dataset(
            morphology=synthetic_morphology(),
            growth=growth,
            expected_rows=0,
        )


def test_derived_log_controls_are_computed_correctly() -> None:
    dataset, _, _, _ = build_synthetic_dataset()
    row = dataset.set_index("subfield_id").loc["1000"]

    assert math.isclose(row["log_papers_2010_2019"], math.log1p(100))
    assert math.isclose(row["log_papers_2020_2025"], math.log1p(60))
    assert math.isclose(row["log_annual_rate_2010_2019"], math.log1p(10))
    assert math.isclose(row["log_annual_rate_2020_2025"], math.log1p(10))


def test_feature_groups_do_not_contain_target_columns() -> None:
    _, feature_groups, _, _ = build_synthetic_dataset()
    predictor_columns = set()
    for value in feature_groups["modeling_feature_sets"].values():
        predictor_columns.update(value)
    predictor_columns.update(feature_groups["recommended_primary_feature_columns"])

    assert "annualized_log_growth" not in predictor_columns
    assert "growth_above_median" not in predictor_columns
    assert "papers_2020_2025" not in predictor_columns
    assert "outlier_share_r_gt_1_5" not in feature_groups["core_morphology_metric_columns"]
    assert "outlier_share_r_gt_1_5" in feature_groups["diagnostic_morphology_metric_columns"]


def test_feature_groups_include_expected_core_family_and_pca_columns() -> None:
    _, feature_groups, _, _ = build_synthetic_dataset()

    assert set(CORE_METRIC_COLUMNS_V2).issubset(
        feature_groups["core_morphology_metric_columns"]
    )
    assert set(FAMILY_SCORE_COLUMNS).issubset(feature_groups["family_score_columns"])
    assert feature_groups["pca_score_columns"] == ["metric_pca_PC1", "metric_pca_PC2"]


def test_boolean_labels_are_converted_to_int_columns() -> None:
    dataset, _, _, _ = build_synthetic_dataset()

    assert dataset["growth_label_global_int"].tolist() == [0, 1, 0]
    assert dataset["growth_label_domain_int"].tolist() == [0, 1, 0]


def test_duplicate_display_names_do_not_collapse_rows() -> None:
    dataset, _, _, validation = build_synthetic_dataset()
    shared = dataset[dataset["subfield_display_name"] == "Shared Name"]

    assert len(shared) == 2
    assert set(shared["subfield_id"]) == {"1000", "2000"}
    assert validation["duplicated_display_name_rows"] == 2


def test_cli_runs_on_tiny_synthetic_files(tmp_path: Path) -> None:
    morphology_path = tmp_path / "morphology.parquet"
    raw_path = tmp_path / "raw.parquet"
    pca_path = tmp_path / "pca.csv"
    growth_path = tmp_path / "growth.parquet"
    output_path = tmp_path / "joined.parquet"
    output_csv = tmp_path / "joined.csv"
    summary_path = tmp_path / "summary.json"
    dictionary_path = tmp_path / "dictionary.csv"
    feature_groups_path = tmp_path / "feature_groups.json"
    figures_dir = tmp_path / "figures"

    synthetic_morphology().to_parquet(morphology_path, index=False)
    synthetic_raw_morphology().to_parquet(raw_path, index=False)
    synthetic_pca().to_csv(pca_path, index=False)
    synthetic_growth().to_parquet(growth_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "14_build_morphology_growth_dataset.py"),
            "--morphology-path",
            str(morphology_path),
            "--raw-morphology-path",
            str(raw_path),
            "--family-scores-path",
            str(tmp_path / "missing_family_scores.csv"),
            "--pca-scores-path",
            str(pca_path),
            "--growth-path",
            str(growth_path),
            "--output-path",
            str(output_path),
            "--output-csv",
            str(output_csv),
            "--summary-path",
            str(summary_path),
            "--dictionary-path",
            str(dictionary_path),
            "--feature-groups-path",
            str(feature_groups_path),
            "--join-audit-path",
            str(tmp_path / "audit.csv"),
            "--target-feature-correlations-path",
            str(tmp_path / "correlations.csv"),
            "--family-score-target-summary-path",
            str(tmp_path / "family_summary.csv"),
            "--domain-target-summary-path",
            str(tmp_path / "domain_summary.csv"),
            "--growth-rankings-path",
            str(tmp_path / "rankings.csv"),
            "--family-difference-rankings-path",
            str(tmp_path / "family_rankings.csv"),
            "--figures-dir",
            str(figures_dir),
            "--expected-rows",
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
    joined = pd.read_parquet(output_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    feature_groups = json.loads(feature_groups_path.read_text(encoding="utf-8"))

    assert len(joined) == 3
    assert output_csv.exists()
    assert dictionary_path.exists()
    assert summary["n_joined_rows"] == 3
    assert "recommended_primary_feature_columns" in feature_groups
    for figure_name in [
        "target_distribution.png",
        "domain_adjusted_target_distribution.png",
        "growth_by_domain_boxplot.png",
        "family_scores_vs_growth.png",
        "family_scores_by_growth_label.png",
        "core_metric_target_correlation_barplot.png",
        "morphology_growth_pca_or_family_scatter.png",
    ]:
        assert (figures_dir / figure_name).exists()
