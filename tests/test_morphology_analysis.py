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


def synthetic_metrics_table(n_rows: int = 16) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for idx in range(n_rows):
        duplicate_name = "Duplicated Name" if idx in {0, 1} else f"Subfield {idx:02d}"
        row = {
            "subfield_id": str(1000 + idx),
            "subfield_display_name": duplicate_name,
            "field_id": f"F{idx % 4}",
            "field_display_name": "Field A" if idx == 0 else f"Field {idx % 4}",
            "domain_id": f"D{idx % 3}",
            "domain_display_name": f"Domain {idx % 3}",
            "n_points": 2500 + idx,
            "metric_status": "completed",
        }
        if idx == 1:
            row["field_display_name"] = "Field B"
            row["domain_display_name"] = "Domain B"
        base = idx / max(n_rows - 1, 1)
        for metric_idx, metric_name in enumerate(CORE_METRIC_COLUMNS_V2):
            row[metric_name] = (
                base * (metric_idx + 1)
                + rng.normal(scale=0.03)
                + (0.2 if metric_name == "density_entropy" else 0.0)
            )
        for metric_name in DIAGNOSTIC_METRIC_COLUMNS:
            row[metric_name] = base + rng.normal(scale=0.02)
        rows.append(row)
    return add_subfield_label_columns(pd.DataFrame(rows))


def test_analysis_cli_writes_required_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "metrics.parquet"
    output_dir = tmp_path / "analysis"
    synthetic_metrics_table().to_parquet(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "12_analyze_morphology_metrics.py"),
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--limit-subfields",
            "12",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    required = [
        "metric_descriptive_stats.csv",
        "metric_missingness.csv",
        "metric_correlation_matrix.csv",
        "high_correlation_pairs.csv",
        "low_variance_metrics.csv",
        "top_bottom_subfields_by_metric.csv",
        "domain_metric_summary.csv",
        "field_metric_summary.csv",
        "duplicate_subfield_names_report.csv",
        "curated_model_features.parquet",
        "curated_model_features.csv",
        "family_scores.csv",
        "morphology_analysis_summary.json",
        "metric_pca_scores.csv",
        "metric_pca_loadings.csv",
        "figures/core_metric_histograms.png",
        "figures/core_metric_correlation_heatmap.png",
        "figures/domain_family_score_boxplots.png",
        "figures/metric_pca_scatter.png",
        "figures/metric_pca_loadings.png",
    ]
    for relative_path in required:
        path = output_dir / relative_path
        assert path.exists(), relative_path
        assert path.stat().st_size > 0, relative_path

    curated = pd.read_parquet(output_dir / "curated_model_features.parquet")
    duplicate_report = pd.read_csv(output_dir / "duplicate_subfield_names_report.csv")
    summary = json.loads(
        (output_dir / "morphology_analysis_summary.json").read_text(encoding="utf-8")
    )

    assert set(CORE_METRIC_COLUMNS_V2).issubset(curated.columns)
    assert "diffuseness_score" in curated.columns
    assert "diversification_score" in curated.columns
    assert len(duplicate_report) == 2
    assert summary["n_rows_analyzed"] == 12
    assert summary["n_duplicate_display_name_rows"] == 2

    descriptive = pd.read_csv(output_dir / "metric_descriptive_stats.csv")
    top_bottom = pd.read_csv(output_dir / "top_bottom_subfields_by_metric.csv")
    correlation = pd.read_csv(output_dir / "metric_correlation_matrix.csv", index_col=0)
    assert "density_entropy_slope_by_year" in set(descriptive["metric_name"])
    assert "outlier_share_r_gt_1_5" not in set(descriptive["metric_name"])
    assert "density_entropy_slope_by_year" in set(top_bottom["metric_name"])
    assert "density_entropy_slope_by_year" in correlation.columns
