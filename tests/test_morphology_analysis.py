from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.morphology_metrics import CORE_METRIC_COLUMNS_V2, DIAGNOSTIC_METRIC_COLUMNS
from src.subfield_labels import add_subfield_label_columns


ROOT = Path(__file__).resolve().parents[1]
STAGE12 = ROOT / "scripts" / "12_analyze_morphology_metrics.py"
FAMILY_SCORE_COLUMNS = [
    "diffuseness_score",
    "concentration_score",
    "fragmentation_score",
    "elongation_score",
    "temporal_dynamism_score",
    "directional_change_score",
    "expansion_score",
    "diversification_score",
]


def synthetic_metrics_table(n_rows: int = 18) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for idx in range(n_rows):
        duplicate_name = "Duplicated Name" if idx in {0, 1} else f"Subfield {idx:02d}"
        row = {
            "subfield_id": str(1000 + idx),
            "subfield_display_name": duplicate_name,
            "field_id": f"F{idx % 5}",
            "field_display_name": "Field A" if idx == 0 else f"Field {idx % 5}",
            "domain_id": f"D{idx % 4}",
            "domain_display_name": f"Domain {idx % 4}",
            "n_points": 2400 + idx * 17,
            "metric_status": "completed",
            "future_growth_2020_2025": 999.0 + idx,
        }
        if idx == 1:
            row["field_display_name"] = "Field B"
            row["domain_display_name"] = "Domain B"
        base = idx / max(n_rows - 1, 1)
        wave = np.sin(idx / 2.0)
        for metric_idx, metric_name in enumerate(CORE_METRIC_COLUMNS_V2):
            row[metric_name] = (
                base * (metric_idx + 1) * 0.12
                + wave * ((metric_idx % 4) - 1.5) * 0.08
                + rng.normal(scale=0.025)
                + (0.4 if metric_name == "density_entropy" else 0.0)
                - (0.15 if metric_name == "largest_component_mass_share" else 0.0)
            )
        for metric_name in DIAGNOSTIC_METRIC_COLUMNS:
            row[metric_name] = base + rng.normal(scale=0.02)
        rows.append(row)
    return add_subfield_label_columns(pd.DataFrame(rows))


def run_stage12(
    input_path: Path,
    output_dir: Path,
    *extra_args: str,
    timeout: int = 240,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(STAGE12),
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--top-n",
            "4",
            "--label-top-n",
            "3",
            "--dpi",
            "80",
            "--overwrite",
            *extra_args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_nonempty_file(path: Path) -> None:
    assert path.exists(), str(path)
    assert path.stat().st_size > 0, str(path)


def test_analysis_cli_writes_visual_suite_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "metrics.parquet"
    output_dir = tmp_path / "analysis"
    synthetic_metrics_table().to_parquet(input_path, index=False)

    result = run_stage12(input_path, output_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    for relative_dir in [
        "tables",
        "figures/00_quality",
        "figures/01_distributions",
        "figures/02_correlations",
        "figures/03_family_scores",
        "figures/04_domain_field_profiles",
        "figures/05_metric_space_pca",
        "figures/06_rankings_extremes",
        "figures/07_case_study_atlas",
    ]:
        assert (output_dir / relative_dir).is_dir(), relative_dir

    required_tables = [
        "tables/data_quality_summary.csv",
        "tables/metric_missingness.csv",
        "tables/low_variance_metrics.csv",
        "tables/duplicate_subfield_names_report.csv",
        "tables/domain_counts.csv",
        "tables/field_counts.csv",
        "tables/metric_descriptive_stats.csv",
        "tables/metric_quantiles_long.csv",
        "tables/metric_correlation_matrix.csv",
        "tables/metric_spearman_correlation_matrix.csv",
        "tables/high_correlation_pairs.csv",
        "tables/redundancy_clusters.csv",
        "tables/family_scores.csv",
        "tables/family_score_descriptive_stats.csv",
        "tables/family_score_correlation_matrix.csv",
        "tables/top_bottom_subfields_by_family_score.csv",
        "tables/domain_metric_summary.csv",
        "tables/field_metric_summary.csv",
        "tables/domain_family_score_summary.csv",
        "tables/field_family_score_summary.csv",
        "tables/metric_pca_scores.csv",
        "tables/metric_pca_loadings.csv",
        "tables/metric_pca_explained_variance.csv",
        "tables/top_pca_contributors.csv",
        "tables/top_bottom_subfields_by_metric.csv",
        "tables/case_study_candidates.csv",
        "tables/morphology_archetypes.csv",
        "tables/curated_model_features.parquet",
        "tables/curated_model_features.csv",
        "morphology_analysis_summary.json",
        "morphology_analysis_report.md",
        "morphology_analysis_figure_index.csv",
    ]
    for relative_path in required_tables:
        assert_nonempty_file(output_dir / relative_path)
    for stale_root_name in [
        "metric_descriptive_stats.csv",
        "metric_missingness.csv",
        "curated_model_features.csv",
        "curated_model_features.parquet",
    ]:
        assert not (output_dir / stale_root_name).exists()

    required_figures = [
        "figures/00_quality/data_quality_dashboard.png",
        "figures/01_distributions/core_metric_histograms_by_family.png",
        "figures/01_distributions/core_metric_boxplots_by_family.png",
        "figures/01_distributions/diagnostic_metric_histograms.png",
        "figures/02_correlations/core_metric_correlation_heatmap_clustered.png",
        "figures/02_correlations/core_metric_spearman_heatmap_clustered.png",
        "figures/02_correlations/high_correlation_network.png",
        "figures/02_correlations/redundancy_cluster_summary.png",
        "figures/03_family_scores/family_score_distributions.png",
        "figures/03_family_scores/family_score_correlation_heatmap.png",
        "figures/03_family_scores/family_score_pairplots_key_axes.png",
        "figures/03_family_scores/top_bottom_family_scores.png",
        "figures/04_domain_field_profiles/domain_family_score_heatmap.png",
        "figures/04_domain_field_profiles/domain_family_score_boxplots.png",
        "figures/04_domain_field_profiles/domain_metric_profile_heatmap.png",
        "figures/04_domain_field_profiles/top_fields_by_family_scores.png",
        "figures/05_metric_space_pca/pca_explained_variance.png",
        "figures/05_metric_space_pca/pca_scatter_by_domain.png",
        "figures/05_metric_space_pca/pca_scatter_labeled_extremes.png",
        "figures/05_metric_space_pca/pca_loadings_heatmap.png",
        "figures/05_metric_space_pca/pca_biplot_pc1_pc2.png",
        "figures/05_metric_space_pca/pca_biplot_pc2_pc3.png",
        "figures/06_rankings_extremes/top_bottom_by_selected_metrics.png",
        "figures/06_rankings_extremes/top_bottom_by_family_scores.png",
        "figures/07_case_study_atlas/case_study_candidate_matrix.png",
        "figures/07_case_study_atlas/archetype_counts.png",
        "figures/07_case_study_atlas/archetype_domain_distribution.png",
    ]
    for relative_path in required_figures:
        assert_nonempty_file(output_dir / relative_path)

    figure_index = pd.read_csv(output_dir / "morphology_analysis_figure_index.csv")
    assert set(
        [
            "figure_path",
            "figure_group",
            "title",
            "description",
            "primary_question_answered",
            "related_tables",
        ]
    ).issubset(figure_index.columns)
    assert len(figure_index) >= len(required_figures)
    for relative_path in figure_index["figure_path"]:
        assert_nonempty_file(output_dir / relative_path)

    family_scores = pd.read_csv(output_dir / "tables/family_scores.csv")
    archetypes = pd.read_csv(output_dir / "tables/morphology_archetypes.csv")
    curated = pd.read_parquet(output_dir / "tables/curated_model_features.parquet")
    duplicate_report = pd.read_csv(output_dir / "tables/duplicate_subfield_names_report.csv")
    summary = json.loads(
        (output_dir / "morphology_analysis_summary.json").read_text(encoding="utf-8")
    )
    report = (output_dir / "morphology_analysis_report.md").read_text(encoding="utf-8")

    assert set(FAMILY_SCORE_COLUMNS).issubset(family_scores.columns)
    assert len(archetypes) == len(family_scores)
    assert {"archetype_tags", "primary_archetype", "archetype_reason"}.issubset(
        archetypes.columns
    )
    assert set(CORE_METRIC_COLUMNS_V2).issubset(curated.columns)
    assert set(FAMILY_SCORE_COLUMNS).issubset(curated.columns)
    assert "future_growth_2020_2025" not in curated.columns
    assert len(duplicate_report) == 2
    assert summary["n_rows_analyzed"] == 18
    assert summary["n_family_scores"] == 8
    assert summary["n_duplicate_display_name_rows"] == 2
    assert summary["pca_explained_variance"]
    assert "No growth targets are joined in this stage." in report
    assert "No prediction model is estimated in this stage." in report


def test_analysis_cli_skip_pca_omits_pca_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "metrics.parquet"
    output_dir = tmp_path / "analysis_skip_pca"
    synthetic_metrics_table().to_parquet(input_path, index=False)

    result = run_stage12(
        input_path,
        output_dir,
        "--skip-pca",
        "--skip-heavy-plots",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (output_dir / "tables/metric_pca_scores.csv").exists()
    assert not (output_dir / "figures/05_metric_space_pca/pca_scatter_by_domain.png").exists()
    figure_index = pd.read_csv(output_dir / "morphology_analysis_figure_index.csv")
    assert "05_metric_space_pca" not in set(figure_index["figure_group"])
    summary = json.loads(
        (output_dir / "morphology_analysis_summary.json").read_text(encoding="utf-8")
    )
    assert summary["pca_explained_variance"] == {}


def test_analysis_cli_fails_clearly_when_required_core_metric_missing(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "metrics_missing.parquet"
    output_dir = tmp_path / "analysis_missing"
    frame = synthetic_metrics_table().drop(columns=[CORE_METRIC_COLUMNS_V2[0]])
    frame.to_parquet(input_path, index=False)

    result = run_stage12(input_path, output_dir, timeout=120)

    assert result.returncode != 0
    assert "morphology table missing core metrics" in result.stderr
    assert CORE_METRIC_COLUMNS_V2[0] in result.stderr
