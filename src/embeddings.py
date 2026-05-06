from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EMBEDDING_MODEL = "allenai/specter2"
EMBEDDING_VERSION = "specter2_v1"
EMBEDDING_DIM = 768
EMBEDDING_DTYPE = "float16"

REQUIRED_METADATA_COLUMNS = ["work_id", "subfield_id", "publication_year"]

WORKS_INDEX_COLUMNS = [
    "work_id",
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "primary_topic_id",
    "primary_topic_display_name",
    "publication_year",
]

INDEX_COLUMNS = [
    "work_id",
    "embedding_model",
    "embedding_version",
    "embedding_dim",
    "embedding_dtype",
    "embedding_shard_id",
    "embedding_shard_file",
    "embedding_row_in_shard",
    "metadata_shard_file",
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "primary_topic_id",
    "primary_topic_display_name",
    "publication_year",
    "main_analysis_eligible_2500",
    "robustness_eligible_500",
]


def parse_shard_id(path: str | Path) -> int:
    name = Path(path).name
    match = re.fullmatch(
        r"shard_(\d{4})_(embeddings|metadata|summary)\.(npy|parquet|json)", name
    )
    if not match:
        raise ValueError(f"Could not parse embedding shard id from {name!r}")
    return int(match.group(1))


def list_embedding_shards(embedding_dir: str | Path) -> dict[int, dict[str, Path]]:
    root = Path(embedding_dir)
    shards: dict[int, dict[str, Path]] = {}
    patterns = {
        "embedding": "shard_*_embeddings.npy",
        "metadata": "shard_*_metadata.parquet",
        "summary": "shard_*_summary.json",
    }
    for kind, pattern in patterns.items():
        for path in sorted(root.glob(pattern), key=parse_shard_id):
            shard_id = parse_shard_id(path)
            shards.setdefault(shard_id, {})[kind] = path
    return dict(sorted(shards.items()))


def count_embedding_files(embedding_dir: str | Path) -> dict[str, int]:
    root = Path(embedding_dir)
    return {
        "embedding_shards": len(list(root.glob("shard_*_embeddings.npy"))),
        "metadata_shards": len(list(root.glob("shard_*_metadata.parquet"))),
        "summary_files": len(list(root.glob("shard_*_summary.json"))),
    }


def load_embedding_shape(path: str | Path) -> tuple[tuple[int, ...], str]:
    array = np.load(Path(path), mmap_mode="r")
    try:
        return tuple(int(value) for value in array.shape), str(array.dtype)
    finally:
        del array


def validate_embedding_shards(
    embedding_dir: str | Path,
    expected_shards: int = 37,
    expected_dim: int = EMBEDDING_DIM,
    expected_dtype: str = EMBEDDING_DTYPE,
) -> dict[str, Any]:
    root = Path(embedding_dir)
    errors: list[str] = []
    counts = count_embedding_files(root)
    for key, label in [
        ("embedding_shards", "embedding shards"),
        ("metadata_shards", "metadata shards"),
        ("summary_files", "summary files"),
    ]:
        if counts[key] != expected_shards:
            errors.append(f"Expected {expected_shards} {label}, found {counts[key]}")

    shards = list_embedding_shards(root)
    expected_ids = set(range(expected_shards))
    all_ids = sorted(set(shards) | expected_ids)
    shard_summaries: list[dict[str, Any]] = []
    total_embedding_rows = 0
    total_metadata_rows = 0

    for shard_id in all_ids:
        files = shards.get(shard_id, {})
        record: dict[str, Any] = {
            "shard_id": shard_id,
            "embedding_file": files.get("embedding").name
            if files.get("embedding")
            else None,
            "metadata_file": files.get("metadata").name if files.get("metadata") else None,
            "summary_file": files.get("summary").name if files.get("summary") else None,
            "embedding_rows": None,
            "metadata_rows": None,
            "embedding_dim": None,
            "embedding_dtype": None,
        }

        for kind in ["embedding", "metadata", "summary"]:
            if kind not in files:
                errors.append(f"Missing {kind} file for shard {shard_id:04d}")

        if "embedding" in files:
            try:
                shape, dtype = load_embedding_shape(files["embedding"])
                record["embedding_dtype"] = dtype
                if len(shape) != 2:
                    errors.append(
                        f"Shard {shard_id:04d} has invalid embedding shape {shape}"
                    )
                else:
                    record["embedding_rows"] = shape[0]
                    record["embedding_dim"] = shape[1]
                    total_embedding_rows += shape[0]
                    if shape[1] != expected_dim:
                        errors.append(
                            f"Shard {shard_id:04d} has embedding dimension "
                            f"{shape[1]}, expected {expected_dim}"
                        )
                if dtype != expected_dtype:
                    errors.append(
                        f"Shard {shard_id:04d} has dtype {dtype}, expected {expected_dtype}"
                    )
            except Exception as exc:  # pragma: no cover - defensive path
                errors.append(f"Could not read embedding shard {shard_id:04d}: {exc}")

        if "metadata" in files:
            try:
                metadata = pd.read_parquet(files["metadata"])
                record["metadata_rows"] = int(len(metadata))
                record["metadata_columns"] = list(metadata.columns)
                total_metadata_rows += int(len(metadata))
                missing_columns = [
                    column
                    for column in REQUIRED_METADATA_COLUMNS
                    if column not in metadata.columns
                ]
                if missing_columns:
                    errors.append(
                        f"Shard {shard_id:04d} metadata missing columns: "
                        f"{', '.join(missing_columns)}"
                    )
            except Exception as exc:  # pragma: no cover - defensive path
                errors.append(f"Could not read metadata shard {shard_id:04d}: {exc}")

        if (
            record["embedding_rows"] is not None
            and record["metadata_rows"] is not None
            and record["embedding_rows"] != record["metadata_rows"]
        ):
            errors.append(
                f"Shard {shard_id:04d} row-count mismatch: "
                f"{record['embedding_rows']} embeddings vs {record['metadata_rows']} metadata"
            )

        shard_summaries.append(record)

    if total_embedding_rows != total_metadata_rows:
        errors.append(
            "Total row-count mismatch: "
            f"{total_embedding_rows} embeddings vs {total_metadata_rows} metadata"
        )

    return {
        "errors": errors,
        "file_counts": counts,
        "expected_shards": expected_shards,
        "total_embedding_rows": int(total_embedding_rows),
        "total_metadata_rows": int(total_metadata_rows),
        "shards": shard_summaries,
    }


def read_embedding_metadata_index(embedding_dir: str | Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for shard_id, files in list_embedding_shards(embedding_dir).items():
        if "metadata" not in files:
            continue
        metadata = pd.read_parquet(files["metadata"])
        missing_columns = [
            column for column in REQUIRED_METADATA_COLUMNS if column not in metadata.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Shard {shard_id:04d} metadata missing columns: "
                f"{', '.join(missing_columns)}"
            )
        shard_index = pd.DataFrame(
            {
                "work_id": metadata["work_id"].astype(str),
                "metadata_subfield_id": metadata["subfield_id"].astype(str),
                "metadata_publication_year": metadata["publication_year"],
                "embedding_shard_id": shard_id,
                "embedding_shard_file": files.get("embedding", Path()).name,
                "embedding_row_in_shard": range(len(metadata)),
                "metadata_shard_file": files["metadata"].name,
            }
        )
        records.append(shard_index)

    if not records:
        return pd.DataFrame(
            columns=[
                "work_id",
                "metadata_subfield_id",
                "metadata_publication_year",
                "embedding_shard_id",
                "embedding_shard_file",
                "embedding_row_in_shard",
                "metadata_shard_file",
            ]
        )
    return pd.concat(records, ignore_index=True)


def build_embedding_index(
    metadata_index: pd.DataFrame,
    works_text: pd.DataFrame,
    analysis_subfields: pd.DataFrame,
    embedding_model: str = EMBEDDING_MODEL,
    embedding_version: str = EMBEDDING_VERSION,
    embedding_dim: int = EMBEDDING_DIM,
    embedding_dtype: str = EMBEDDING_DTYPE,
) -> pd.DataFrame:
    base_columns = [
        "work_id",
        "embedding_shard_id",
        "embedding_shard_file",
        "embedding_row_in_shard",
        "metadata_shard_file",
    ]
    index = metadata_index[base_columns].copy()
    index["work_id"] = index["work_id"].astype(str)
    index.insert(1, "embedding_model", embedding_model)
    index.insert(2, "embedding_version", embedding_version)
    index.insert(3, "embedding_dim", int(embedding_dim))
    index.insert(4, "embedding_dtype", embedding_dtype)

    works_lookup = works_text.copy()
    for column in WORKS_INDEX_COLUMNS:
        if column not in works_lookup.columns:
            works_lookup[column] = pd.NA
    works_lookup = works_lookup[WORKS_INDEX_COLUMNS].drop_duplicates("work_id")
    works_lookup["work_id"] = works_lookup["work_id"].astype(str)
    works_lookup["subfield_id"] = works_lookup["subfield_id"].astype("string")

    index = index.merge(works_lookup, on="work_id", how="left", validate="many_to_one")

    flags = analysis_subfields.copy()
    for column in [
        "subfield_id",
        "main_analysis_eligible_2500",
        "robustness_eligible_500",
    ]:
        if column not in flags.columns:
            flags[column] = pd.NA
    flags = flags[
        [
            "subfield_id",
            "main_analysis_eligible_2500",
            "robustness_eligible_500",
        ]
    ].drop_duplicates("subfield_id")
    flags["subfield_id"] = flags["subfield_id"].astype("string")

    index = index.merge(flags, on="subfield_id", how="left", validate="many_to_one")
    return index[INDEX_COLUMNS]


def validate_embedding_relationships(
    metadata_index: pd.DataFrame,
    works_text: pd.DataFrame,
    analysis_subfields: pd.DataFrame,
    embedding_index: pd.DataFrame,
) -> dict[str, Any]:
    errors: list[str] = []

    metadata_work_ids = metadata_index["work_id"].astype(str)
    works_work_ids = works_text["work_id"].astype(str) if "work_id" in works_text else pd.Series(dtype=str)

    duplicate_work_ids = int(metadata_work_ids.duplicated().sum())
    if duplicate_work_ids:
        errors.append(f"Found {duplicate_work_ids} duplicate work_id rows in metadata shards")

    embedded_set = set(metadata_work_ids)
    works_set = set(works_work_ids)
    embedded_missing_from_works = len(embedded_set - works_set)
    works_missing_from_embeddings = len(works_set - embedded_set)
    if embedded_missing_from_works:
        errors.append(
            f"{embedded_missing_from_works} embedded work_id values are missing from works_text"
        )
    if works_missing_from_embeddings:
        errors.append(
            f"{works_missing_from_embeddings} works_text work_id values are missing from embeddings"
        )

    subfield_mismatches = 0
    if "subfield_id" in works_text.columns and not metadata_index.empty:
        works_subfields = works_text[["work_id", "subfield_id"]].copy()
        works_subfields["work_id"] = works_subfields["work_id"].astype(str)
        works_subfields["subfield_id"] = works_subfields["subfield_id"].astype(str)
        compare = metadata_index[["work_id", "metadata_subfield_id"]].merge(
            works_subfields, on="work_id", how="inner", validate="many_to_one"
        )
        subfield_mismatches = int(
            (compare["metadata_subfield_id"].astype(str) != compare["subfield_id"]).sum()
        )
        if subfield_mismatches:
            errors.append(
                f"{subfield_mismatches} embedded rows have subfield_id values "
                "inconsistent with works_text"
            )

    missing_flag_rows = 0
    if {
        "main_analysis_eligible_2500",
        "robustness_eligible_500",
    }.issubset(embedding_index.columns):
        missing_flag_rows = int(
            embedding_index[
                [
                    "main_analysis_eligible_2500",
                    "robustness_eligible_500",
                ]
            ]
            .isna()
            .any(axis=1)
            .sum()
        )
        if missing_flag_rows:
            errors.append(
                f"{missing_flag_rows} embedding_index rows could not join analysis flags"
            )
    else:
        errors.append("embedding_index is missing analysis eligibility flag columns")

    analysis_subfields_with_flags = 0
    if {
        "subfield_id",
        "main_analysis_eligible_2500",
        "robustness_eligible_500",
    }.issubset(analysis_subfields.columns):
        analysis_subfields_with_flags = int(
            analysis_subfields[
                [
                    "subfield_id",
                    "main_analysis_eligible_2500",
                    "robustness_eligible_500",
                ]
            ]
            .dropna()
            .drop_duplicates("subfield_id")
            .shape[0]
        )
    else:
        errors.append("analysis_subfields is missing eligibility flag columns")

    main_rows = int(
        embedding_index["main_analysis_eligible_2500"].fillna(False).astype(bool).sum()
    )
    robustness_rows = int(
        embedding_index["robustness_eligible_500"].fillna(False).astype(bool).sum()
    )

    return {
        "errors": errors,
        "duplicate_work_id_rows": duplicate_work_ids,
        "embedded_missing_from_works": embedded_missing_from_works,
        "works_missing_from_embeddings": works_missing_from_embeddings,
        "subfield_mismatch_rows": subfield_mismatches,
        "analysis_flag_missing_rows": missing_flag_rows,
        "analysis_subfields_with_flags": analysis_subfields_with_flags,
        "main_analysis_embedded_rows": main_rows,
        "robustness_embedded_rows": robustness_rows,
        "total_index_rows": int(len(embedding_index)),
    }
