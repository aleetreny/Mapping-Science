from pathlib import Path

import numpy as np
import pandas as pd

from src.embeddings import (
    build_embedding_index,
    parse_shard_id,
    read_embedding_metadata_index,
    validate_embedding_relationships,
    validate_embedding_shards,
)


def write_shard(
    root: Path,
    shard_id: int,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    np.save(root / f"shard_{shard_id:04d}_embeddings.npy", embeddings)
    metadata.to_parquet(root / f"shard_{shard_id:04d}_metadata.parquet", index=False)
    (root / f"shard_{shard_id:04d}_summary.json").write_text("{}", encoding="utf-8")


def works_text_for(work_ids: list[str], subfield_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "work_id": work_ids,
            "subfield_id": subfield_ids,
            "subfield_display_name": [f"Subfield {value}" for value in subfield_ids],
            "field_id": ["field-1"] * len(work_ids),
            "field_display_name": ["Field 1"] * len(work_ids),
            "domain_id": ["domain-1"] * len(work_ids),
            "domain_display_name": ["Domain 1"] * len(work_ids),
            "primary_topic_id": [f"topic-{index}" for index in range(len(work_ids))],
            "primary_topic_display_name": [
                f"Topic {index}" for index in range(len(work_ids))
            ],
            "publication_year": [2019] * len(work_ids),
        }
    )


def analysis_subfields_for() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subfield_id": ["s1", "s2"],
            "main_analysis_eligible_2500": [True, False],
            "robustness_eligible_500": [True, True],
        }
    )


def test_parse_shard_id_from_embedding_and_metadata_names() -> None:
    assert parse_shard_id("shard_0000_embeddings.npy") == 0
    assert parse_shard_id("shard_0036_metadata.parquet") == 36


def test_shape_validation_catches_wrong_dimension(tmp_path: Path) -> None:
    write_shard(
        tmp_path,
        0,
        np.zeros((2, 767), dtype=np.float16),
        pd.DataFrame(
            {
                "work_id": ["w1", "w2"],
                "subfield_id": ["s1", "s1"],
                "publication_year": [2018, 2019],
            }
        ),
    )

    result = validate_embedding_shards(tmp_path, expected_shards=1)

    assert any("embedding dimension 767" in error for error in result["errors"])


def test_validation_catches_row_count_mismatch(tmp_path: Path) -> None:
    write_shard(
        tmp_path,
        0,
        np.zeros((2, 768), dtype=np.float16),
        pd.DataFrame(
            {
                "work_id": ["w1"],
                "subfield_id": ["s1"],
                "publication_year": [2018],
            }
        ),
    )

    result = validate_embedding_shards(tmp_path, expected_shards=1)

    assert any("row-count mismatch" in error for error in result["errors"])


def test_embedding_index_has_zero_based_row_position_per_shard(tmp_path: Path) -> None:
    write_shard(
        tmp_path,
        0,
        np.zeros((2, 768), dtype=np.float16),
        pd.DataFrame(
            {
                "work_id": ["w1", "w2"],
                "subfield_id": ["s1", "s2"],
                "publication_year": [2018, 2019],
            }
        ),
    )
    write_shard(
        tmp_path,
        1,
        np.zeros((1, 768), dtype=np.float16),
        pd.DataFrame(
            {
                "work_id": ["w3"],
                "subfield_id": ["s1"],
                "publication_year": [2019],
            }
        ),
    )
    metadata_index = read_embedding_metadata_index(tmp_path)
    works_text = works_text_for(["w1", "w2", "w3"], ["s1", "s2", "s1"])

    index = build_embedding_index(
        metadata_index=metadata_index,
        works_text=works_text,
        analysis_subfields=analysis_subfields_for(),
    )

    assert index["embedding_row_in_shard"].tolist() == [0, 1, 0]
    assert index["embedding_shard_id"].tolist() == [0, 0, 1]


def test_duplicate_work_id_detection(tmp_path: Path) -> None:
    write_shard(
        tmp_path,
        0,
        np.zeros((2, 768), dtype=np.float16),
        pd.DataFrame(
            {
                "work_id": ["w1", "w1"],
                "subfield_id": ["s1", "s1"],
                "publication_year": [2018, 2018],
            }
        ),
    )
    metadata_index = read_embedding_metadata_index(tmp_path)
    works_text = works_text_for(["w1"], ["s1"])
    index = build_embedding_index(
        metadata_index=metadata_index,
        works_text=works_text,
        analysis_subfields=analysis_subfields_for(),
    )

    result = validate_embedding_relationships(
        metadata_index=metadata_index,
        works_text=works_text,
        analysis_subfields=analysis_subfields_for(),
        embedding_index=index,
    )

    assert result["duplicate_work_id_rows"] == 1
    assert any("duplicate work_id" in error for error in result["errors"])


def test_eligibility_flags_join_to_embedding_index(tmp_path: Path) -> None:
    write_shard(
        tmp_path,
        0,
        np.zeros((2, 768), dtype=np.float16),
        pd.DataFrame(
            {
                "work_id": ["w1", "w2"],
                "subfield_id": ["s1", "s2"],
                "publication_year": [2018, 2019],
            }
        ),
    )
    metadata_index = read_embedding_metadata_index(tmp_path)
    works_text = works_text_for(["w1", "w2"], ["s1", "s2"])

    index = build_embedding_index(
        metadata_index=metadata_index,
        works_text=works_text,
        analysis_subfields=analysis_subfields_for(),
    )

    by_work_id = index.set_index("work_id")
    assert bool(by_work_id.loc["w1", "main_analysis_eligible_2500"]) is True
    assert bool(by_work_id.loc["w2", "main_analysis_eligible_2500"]) is False
    assert bool(by_work_id.loc["w2", "robustness_eligible_500"]) is True
