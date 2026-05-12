from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.per_subfield_umap_maps import (
    MANIFEST_COLUMNS,
    build_manifest_row,
    filter_input_window,
    main_analysis_subfields,
    manifest_frame,
    plot_subfield_panels,
    safe_subfield_stem,
    sample_subfield_rows,
    validate_index_columns,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_index() -> pd.DataFrame:
    records = []
    row_id = 0
    subfields = [
        ("1100", "General Agricultural / Biological Sciences"),
        ("2200", "Biochemistry & Molecular Biology"),
    ]
    for subfield_id, subfield_name in subfields:
        for offset in range(10):
            records.append(
                {
                    "analysis_row_id": row_id,
                    "work_id": f"W{subfield_id}-{offset:02d}",
                    "subfield_id": subfield_id,
                    "subfield_display_name": subfield_name,
                    "field_id": f"F{subfield_id}",
                    "field_display_name": f"Field {subfield_id}",
                    "domain_id": "D1",
                    "domain_display_name": "Domain 1",
                    "primary_topic_id": f"T{row_id}",
                    "primary_topic_display_name": f"Topic {row_id}",
                    "publication_year": 2010 + (offset % 10),
                    "main_analysis_eligible_2500": True,
                }
            )
            row_id += 1

    records.append(
        {
            "analysis_row_id": row_id,
            "work_id": "W-outside",
            "subfield_id": "1100",
            "subfield_display_name": "General Agricultural / Biological Sciences",
            "field_id": "F1100",
            "field_display_name": "Field 1100",
            "domain_id": "D1",
            "domain_display_name": "Domain 1",
            "primary_topic_id": f"T{row_id}",
            "primary_topic_display_name": f"Topic {row_id}",
            "publication_year": 2021,
            "main_analysis_eligible_2500": True,
        }
    )
    return pd.DataFrame(records)


def test_safe_subfield_stem_removes_unsafe_characters() -> None:
    stem = safe_subfield_stem("https://openalex.org/subfields/11", "AI / ML: Test?")

    assert stem == "https_openalex.org_subfields_11__AI_ML_Test"
    assert "/" not in stem
    assert ":" not in stem
    assert "?" not in stem


def test_sample_subfield_rows_is_deterministic_and_ordered() -> None:
    rows = synthetic_index().query("subfield_id == '1100'").sample(
        frac=1,
        random_state=99,
    )

    first = sample_subfield_rows(
        rows,
        max_papers=5,
        random_state=42,
        subfield_id="1100",
    )
    second = sample_subfield_rows(
        rows.sample(frac=1, random_state=13),
        max_papers=5,
        random_state=42,
        subfield_id="1100",
    )

    assert first["analysis_row_id"].tolist() == second["analysis_row_id"].tolist()
    assert first["publication_year"].tolist() == sorted(first["publication_year"].tolist())


def test_manifest_rows_cover_statuses() -> None:
    rows = [
        build_manifest_row(
            subfield_id="1100",
            subfield_name="Completed",
            n_available=10,
            n_used=8,
            year_min=2010,
            year_max=2025,
            status="completed",
            coordinate_path="coordinates/1100.parquet",
            figure_path="figures/1100.png",
            umap_n_neighbors=3,
            umap_min_dist=0.05,
            umap_metric="cosine",
            random_state=42,
            max_papers_per_subfield=10,
            sampling_applied=True,
        ),
        build_manifest_row(
            subfield_id="2200",
            subfield_name="Skipped",
            n_available=2,
            n_used=0,
            year_min=2010,
            year_max=2025,
            status="skipped",
            error_message="too few papers",
            umap_n_neighbors=3,
            umap_min_dist=0.05,
            umap_metric="cosine",
            random_state=42,
            max_papers_per_subfield=10,
            sampling_applied=False,
        ),
        build_manifest_row(
            subfield_id="3300",
            subfield_name="Failed",
            n_available=10,
            n_used=0,
            year_min=2010,
            year_max=2025,
            status="failed",
            error_message="boom",
            umap_n_neighbors=3,
            umap_min_dist=0.05,
            umap_metric="cosine",
            random_state=42,
            max_papers_per_subfield=10,
            sampling_applied=False,
        ),
    ]

    manifest = manifest_frame(rows)

    assert manifest.columns.tolist() == MANIFEST_COLUMNS
    assert manifest["status"].tolist() == ["completed", "skipped", "failed"]
    assert manifest.loc[0, "coordinate_path"] == "coordinates/1100.parquet"
    assert manifest.loc[1, "error_message"] == "too few papers"


def test_filter_input_window_uses_main_analysis_and_year_bounds() -> None:
    index = synthetic_index()
    index.loc[index.index[0], "main_analysis_eligible_2500"] = False

    filtered = filter_input_window(index, year_min=2012, year_max=2014)

    assert filtered["publication_year"].min() >= 2012
    assert filtered["publication_year"].max() <= 2014
    assert filtered["main_analysis_eligible_2500"].all()
    assert index.loc[index.index[0], "analysis_row_id"] not in set(
        filtered["analysis_row_id"]
    )


def test_main_analysis_subfields_adds_unique_labels_for_duplicate_names() -> None:
    index = synthetic_index()
    index.loc[index["subfield_id"] == "2200", "subfield_display_name"] = "Shared Name"
    index.loc[index["subfield_id"] == "1100", "subfield_display_name"] = "Shared Name"

    subfields = main_analysis_subfields(index)

    assert subfields["subfield_display_name_is_duplicated"].all()
    labels = set(subfields["subfield_label_unique"])
    assert "1100 | Domain 1 / Field 1100 / Shared Name" in labels
    assert "2200 | Domain 1 / Field 2200 / Shared Name" in labels


def test_validate_index_columns_fails_clearly_without_publication_year() -> None:
    incomplete = pd.DataFrame(
        {
            "analysis_row_id": [0],
            "work_id": ["W1"],
            "subfield_id": ["1100"],
            "subfield_display_name": ["Subfield"],
            "main_analysis_eligible_2500": [True],
        }
    )

    with pytest.raises(ValueError, match="publication_year"):
        validate_index_columns(incomplete)


def test_density_plot_helper_writes_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    coordinates = rng.normal(size=(40, 2))
    output_path = tmp_path / "density.png"

    method = plot_subfield_panels(
        coordinates,
        subfield_name="Synthetic Subfield",
        n_used=len(coordinates),
        year_min=2010,
        year_max=2025,
        output_path=output_path,
        dpi=80,
    )
    plt.close("all")

    assert method in {"kde", "smooth_hist"}
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cli_runs_on_tiny_synthetic_embedding_matrix(tmp_path: Path) -> None:
    index = synthetic_index()
    rng = np.random.default_rng(123)
    matrix = rng.normal(size=(len(index), 6)).astype(np.float16)

    index_path = tmp_path / "index.parquet"
    embedding_dir = tmp_path / "embeddings"
    embeddings_path = embedding_dir / "analysis" / "main_embeddings.float16.npy"
    output_dir = tmp_path / "per_subfield_umap"
    embeddings_path.parent.mkdir(parents=True)
    index.to_parquet(index_path, index=False)
    np.save(embeddings_path, matrix)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "10_build_per_subfield_umap_maps.py"),
            "--index-path",
            str(index_path),
            "--embedding-dir",
            str(embedding_dir),
            "--output-dir",
            str(output_dir),
            "--subfield-id",
            "1100",
            "--limit-subfields",
            "1",
            "--min-papers",
            "5",
            "--max-papers-per-subfield",
            "6",
            "--n-neighbors",
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
    manifest = pd.read_parquet(output_dir / "per_subfield_umap_manifest.parquet")
    summary = json.loads((output_dir / "per_subfield_umap_summary.json").read_text())

    assert manifest["status"].tolist() == ["completed"]
    assert summary["n_completed"] == 1
    assert len(list((output_dir / "coordinates").glob("*.parquet"))) == 1
    assert len(list((output_dir / "figures").glob("*.png"))) == 1


def test_cli_fails_clearly_when_index_and_matrix_versions_differ(
    tmp_path: Path,
) -> None:
    index = synthetic_index()
    index_path = tmp_path / "index.parquet"
    embedding_dir = tmp_path / "embeddings"
    embeddings_path = embedding_dir / "analysis" / "main_embeddings.float16.npy"
    output_dir = tmp_path / "per_subfield_umap"
    embeddings_path.parent.mkdir(parents=True)
    index.to_parquet(index_path, index=False)
    np.save(embeddings_path, np.zeros((1, 6), dtype=np.float16))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "10_build_per_subfield_umap_maps.py"),
            "--index-path",
            str(index_path),
            "--embedding-dir",
            str(embedding_dir),
            "--output-dir",
            str(output_dir),
            "--limit-subfields",
            "1",
            "--min-papers",
            "1",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "different embedding versions" in output
    assert str(index_path) in output
    assert str(embeddings_path) in output
