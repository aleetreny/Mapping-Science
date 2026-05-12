from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_DTYPE,
    MAIN_ANALYSIS_FLAG,
    normalize_eligibility_flags,
)


ANALYSIS_SORT_COLUMNS = ["subfield_id", "publication_year", "work_id"]


def prepare_analysis_embedding_index(
    embedding_index: pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    required = {
        "work_id",
        "subfield_id",
        "publication_year",
        "embedding_shard_file",
        "embedding_row_in_shard",
    }
    missing = required - set(embedding_index.columns)
    if missing:
        raise ValueError(f"embedding_index missing columns: {', '.join(sorted(missing))}")

    try:
        normalized_index = normalize_eligibility_flags(
            embedding_index,
            require_robustness=False,
        )
    except ValueError as exc:
        raise ValueError(
            "Eligibility flags could not be resolved for the analysis matrix. "
            f"{exc}"
        ) from exc

    main = normalized_index[MAIN_ANALYSIS_FLAG].fillna(False).astype(bool)
    analysis_index = normalized_index.loc[main].copy()
    if analysis_index.empty and not allow_empty:
        raise ValueError(
            "No rows selected for the main-analysis embedding matrix. Eligibility "
            "flags could not be resolved or selected zero rows; check "
            "main_analysis_eligible and its compatible source columns."
        )

    duplicate_work_ids = int(analysis_index["work_id"].astype(str).duplicated().sum())
    if duplicate_work_ids:
        raise ValueError(
            f"analysis embedding index contains {duplicate_work_ids} duplicate work_id rows"
        )

    analysis_index["work_id"] = analysis_index["work_id"].astype(str)
    analysis_index["subfield_id"] = analysis_index["subfield_id"].astype(str)
    analysis_index = analysis_index.sort_values(
        ANALYSIS_SORT_COLUMNS,
        kind="mergesort",
    ).reset_index(drop=True)
    analysis_index.insert(0, "analysis_row_id", range(len(analysis_index)))
    return analysis_index


def ensure_outputs_do_not_exist(paths: Iterable[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing outputs without --force: {formatted}"
        )


def write_analysis_matrix(
    analysis_index: pd.DataFrame,
    embedding_dir: str | Path,
    output_path: str | Path,
    embedding_dim: int = EMBEDDING_DIM,
    embedding_dtype: str = EMBEDDING_DTYPE,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.dtype(embedding_dtype),
        shape=(len(analysis_index), embedding_dim),
    )

    root = Path(embedding_dir)
    for shard_file, group in analysis_index.groupby("embedding_shard_file", sort=True):
        shard_path = root / str(shard_file)
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing embedding shard: {shard_path}")
        shard = np.load(shard_path, mmap_mode="r")
        if shard.ndim != 2 or shard.shape[1] != embedding_dim:
            raise ValueError(
                f"{shard_path} has shape {shard.shape}, expected (*, {embedding_dim})"
            )
        if str(shard.dtype) != embedding_dtype:
            raise ValueError(
                f"{shard_path} has dtype {shard.dtype}, expected {embedding_dtype}"
            )

        source_rows = group["embedding_row_in_shard"].astype(int).to_numpy()
        target_rows = group["analysis_row_id"].astype(int).to_numpy()
        matrix[target_rows] = shard[source_rows]

    matrix.flush()
    del matrix


def validate_analysis_matrix(
    analysis_index: pd.DataFrame,
    matrix_path: str | Path,
    embedding_dim: int = EMBEDDING_DIM,
    embedding_dtype: str = EMBEDDING_DTYPE,
) -> list[str]:
    errors: list[str] = []
    matrix = np.load(Path(matrix_path), mmap_mode="r")
    try:
        if matrix.shape[0] != len(analysis_index):
            errors.append(
                f"Matrix rows {matrix.shape[0]} do not match index rows {len(analysis_index)}"
            )
        if matrix.ndim != 2 or matrix.shape[1] != embedding_dim:
            errors.append(f"Matrix shape {matrix.shape} does not have dim {embedding_dim}")
        if str(matrix.dtype) != embedding_dtype:
            errors.append(f"Matrix dtype {matrix.dtype} does not match {embedding_dtype}")
    finally:
        del matrix

    duplicate_work_ids = int(analysis_index["work_id"].astype(str).duplicated().sum())
    if duplicate_work_ids:
        errors.append(f"Found {duplicate_work_ids} duplicate work_id rows")

    try:
        normalized_index = normalize_eligibility_flags(
            analysis_index,
            require_robustness=False,
        )
    except ValueError as exc:
        errors.append(f"Analysis index eligibility flags could not be resolved: {exc}")
        normalized_index = analysis_index

    if (
        MAIN_ANALYSIS_FLAG in normalized_index
        and not normalized_index[MAIN_ANALYSIS_FLAG].fillna(False).astype(bool).all()
    ):
        errors.append("Analysis index contains rows outside main_analysis_eligible")

    expected_row_ids = list(range(len(analysis_index)))
    if analysis_index["analysis_row_id"].tolist() != expected_row_ids:
        errors.append("analysis_row_id is not zero-based and contiguous")

    return errors
