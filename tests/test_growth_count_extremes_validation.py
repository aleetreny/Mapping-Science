from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "13b_validate_growth_count_extremes.py"
SPEC = importlib.util.spec_from_file_location("validate_growth_count_extremes", SCRIPT_PATH)
validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validation)


class FakeOpenAlexClient:
    def __init__(self, counts: dict[str, int], failing_filters: set[str] | None = None) -> None:
        self.counts = counts
        self.failing_filters = failing_filters or set()

    def get_count(self, endpoint: str, query_filter: str) -> int:
        assert endpoint == "works"
        if query_filter in self.failing_filters:
            raise RuntimeError("synthetic API failure")
        return self.counts[query_filter]


def test_query_filter_matches_stage01_count_filters() -> None:
    query_filter = validation.query_filter_for_subfield_year("https://openalex.org/subfields/2202", 2025)

    assert query_filter == (
        "primary_topic.subfield.id:2202,publication_year:2025,"
        "type:article|preprint,is_retracted:false,is_paratext:false"
    )


def test_prepare_local_counts_requires_unique_subfield_year_rows() -> None:
    counts = pd.DataFrame(
        [
            {"subfield_id": "2202", "publication_year": 2025, "n_works_article_preprint": 1},
            {"subfield_id": "2202", "publication_year": 2025, "n_works_article_preprint": 2},
        ]
    )

    with pytest.raises(ValueError, match="unique"):
        validation.prepare_local_counts(counts)


def test_validate_count_pairs_marks_matches_and_mismatches() -> None:
    query_2010 = validation.query_filter_for_subfield_year("2202", 2010)
    query_2025 = validation.query_filter_for_subfield_year("2202", 2025)
    local_counts = validation.prepare_local_counts(
        pd.DataFrame(
            [
                {"subfield_id": "2202", "publication_year": 2010, "n_works_article_preprint": 100},
                {"subfield_id": "2202", "publication_year": 2025, "n_works_article_preprint": 125},
            ]
        )
    )
    labels = pd.DataFrame(
        [
            {
                "subfield_id": "2202",
                "subfield_label_unique": "2202 | Physical Sciences / Engineering / Aerospace Engineering",
            }
        ]
    )
    client = FakeOpenAlexClient({query_2010: 100, query_2025: 120})

    result = validation.validate_count_pairs(
        local_counts=local_counts,
        labels=labels,
        client=client,
        subfield_ids=["2202"],
        years=[2010, 2025],
        sleep_seconds=0,
    )
    summary = validation.build_summary(result)

    assert result["status"].tolist() == ["matched", "mismatch"]
    assert result.loc[1, "absolute_difference"] == 5
    assert summary["n_matched"] == 1
    assert summary["n_mismatch"] == 1
    assert summary["max_absolute_difference"] == 5


def test_validate_count_pairs_marks_api_failures_and_missing_local_counts() -> None:
    query_2010 = validation.query_filter_for_subfield_year("2202", 2010)
    query_2025 = validation.query_filter_for_subfield_year("2202", 2025)
    local_counts = validation.prepare_local_counts(
        pd.DataFrame(
            [
                {"subfield_id": "2202", "publication_year": 2010, "n_works_article_preprint": 100},
            ]
        )
    )
    labels = pd.DataFrame(
        [{"subfield_id": "2202", "subfield_label_unique": "2202 | Aerospace Engineering"}]
    )
    client = FakeOpenAlexClient({query_2010: 100, query_2025: 120}, {query_2010})

    result = validation.validate_count_pairs(
        local_counts=local_counts,
        labels=labels,
        client=client,
        subfield_ids=["2202"],
        years=[2010, 2025],
        sleep_seconds=0,
    )
    summary = validation.build_summary(result)

    assert result["status"].tolist() == ["api_failed", "missing_local_count"]
    assert summary["n_api_failed"] == 1
    assert summary["n_missing_local_count"] == 1
