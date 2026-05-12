from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis_matrix import (
    ensure_outputs_do_not_exist,
    prepare_analysis_embedding_index,
    validate_analysis_matrix,
    write_analysis_matrix,
)
from src.embeddings import MAIN_ANALYSIS_FLAG


def embedding_index_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "work_id": ["w3", "w1", "w2", "w4"],
            "subfield_id": ["s2", "s1", "s1", "s3"],
            "subfield_display_name": ["S2", "S1", "S1", "S3"],
            "field_id": ["f2", "f1", "f1", "f3"],
            "field_display_name": ["F2", "F1", "F1", "F3"],
            "domain_id": ["d2", "d1", "d1", "d3"],
            "domain_display_name": ["D2", "D1", "D1", "D3"],
            "primary_topic_id": ["t3", "t1", "t2", "t4"],
            "primary_topic_display_name": ["T3", "T1", "T2", "T4"],
            "publication_year": [2017, 2019, 2018, 2019],
            "main_analysis_eligible_2500": [True, True, True, False],
            "robustness_eligible_500": [True, True, True, True],
            "embedding_shard_file": [
                "shard_0001_embeddings.npy",
                "shard_0000_embeddings.npy",
                "shard_0000_embeddings.npy",
                "shard_0001_embeddings.npy",
            ],
            "embedding_row_in_shard": [0, 1, 0, 1],
            "embedding_shard_id": [1, 0, 0, 1],
            "metadata_shard_file": [
                "shard_0001_metadata.parquet",
                "shard_0000_metadata.parquet",
                "shard_0000_metadata.parquet",
                "shard_0001_metadata.parquet",
            ],
            "embedding_model": ["allenai/specter2"] * 4,
            "embedding_version": ["specter2_v1"] * 4,
            "embedding_dim": [2] * 4,
            "embedding_dtype": ["float16"] * 4,
        }
    )


def test_prepare_analysis_index_filters_main_and_orders_deterministically() -> None:
    index = prepare_analysis_embedding_index(embedding_index_fixture())

    assert index["work_id"].tolist() == ["w2", "w1", "w3"]
    assert index["analysis_row_id"].tolist() == [0, 1, 2]
    assert index[MAIN_ANALYSIS_FLAG].all()
    assert index["main_analysis_eligible_2500"].all()


def test_prepare_analysis_index_rejects_duplicate_work_ids() -> None:
    embedding_index = embedding_index_fixture()
    embedding_index.loc[1, "work_id"] = "w2"

    with pytest.raises(ValueError, match="duplicate work_id"):
        prepare_analysis_embedding_index(embedding_index)


def test_prepare_analysis_index_rejects_zero_selected_rows() -> None:
    embedding_index = embedding_index_fixture()
    embedding_index["main_analysis_eligible_2500"] = False

    with pytest.raises(ValueError, match="selected zero rows"):
        prepare_analysis_embedding_index(embedding_index)


def test_prepare_analysis_index_allows_zero_rows_only_for_debug() -> None:
    embedding_index = embedding_index_fixture()
    embedding_index["main_analysis_eligible_2500"] = False

    index = prepare_analysis_embedding_index(embedding_index, allow_empty=True)

    assert index.empty
    assert "analysis_row_id" in index.columns


def test_write_analysis_matrix_matches_index_rows(tmp_path: Path) -> None:
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    np.save(
        embedding_dir / "shard_0000_embeddings.npy",
        np.array([[20, 21], [10, 11]], dtype=np.float16),
    )
    np.save(
        embedding_dir / "shard_0001_embeddings.npy",
        np.array([[30, 31], [40, 41]], dtype=np.float16),
    )
    index = prepare_analysis_embedding_index(embedding_index_fixture())
    matrix_path = tmp_path / "main_embeddings.float16.npy"

    write_analysis_matrix(
        analysis_index=index,
        embedding_dir=embedding_dir,
        output_path=matrix_path,
        embedding_dim=2,
        embedding_dtype="float16",
    )

    matrix = np.load(matrix_path)
    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.float16
    assert matrix.tolist() == [[20, 21], [10, 11], [30, 31]]
    assert validate_analysis_matrix(index, matrix_path, 2, "float16") == []


def test_ensure_outputs_do_not_exist_fails_without_force(tmp_path: Path) -> None:
    output = tmp_path / "existing.npy"
    output.write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        ensure_outputs_do_not_exist([output], force=False)

    ensure_outputs_do_not_exist([output], force=True)
