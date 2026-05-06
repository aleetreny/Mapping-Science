from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


DEFAULT_DIRS = [
    Path("data/raw"),
    Path("data/interim"),
    Path("data/processed"),
    Path("warehouse"),
]


def ensure_dirs(config: dict[str, Any] | None = None) -> None:
    dirs = list(DEFAULT_DIRS)
    if config:
        storage = config.get("storage", {})
        for key in ["raw_dir", "interim_dir", "processed_dir"]:
            if storage.get(key):
                dirs.append(Path(storage[key]))
        if storage.get("duckdb_path"):
            dirs.append(Path(storage["duckdb_path"]).parent)

    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def connect_duckdb(path: str | Path) -> duckdb.DuckDBPyConnection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def write_table(
    con: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame
) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(f"Unsafe table name: {table_name}")

    temp_name = f"tmp_{table_name}"
    con.register(temp_name, df)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM {temp_name}')
    finally:
        con.unregister(temp_name)
