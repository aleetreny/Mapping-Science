from __future__ import annotations

from typing import Any

import pandas as pd


COMPLETED_MANIFEST_STATUSES = {
    "completed_target_met",
    "completed_shortfall",
    "skipped_no_available_works",
}


def cell_key(subfield_id: Any, publication_year: Any) -> tuple[str, int]:
    return str(subfield_id), int(publication_year)


def completed_cell_keys_from_manifest(manifest: pd.DataFrame) -> set[tuple[str, int]]:
    """Return subfield-year cells that should be skipped on resume."""
    required = {"subfield_id", "publication_year", "status"}
    if manifest.empty or not required.issubset(manifest.columns):
        return set()

    completed = manifest[manifest["status"].isin(COMPLETED_MANIFEST_STATUSES)]
    return {
        cell_key(row.subfield_id, row.publication_year)
        for row in completed.itertuples(index=False)
    }
