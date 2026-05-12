from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from src.download_state import cell_key
from src.openalex import OpenAlexRateLimitError


ROOT = Path(__file__).resolve().parents[1]


def load_downloader_module():
    path = ROOT / "scripts" / "04_download_sampled_corpus.py"
    spec = importlib.util.spec_from_file_location("download_sampled_corpus", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def minimal_config() -> dict:
    return {
        "filters": {
            "language": "en",
            "work_types": ["article", "preprint"],
            "min_title_tokens": 5,
            "min_abstract_tokens": 80,
        },
        "sampling": {
            "oversample_factor": 1.75,
            "backfill_seed_step": 100000,
            "max_sample_per_subfield": 10000,
        },
        "openalex": {
            "per_page": 200,
        },
    }


def sample_plan_row() -> pd.Series:
    return pd.Series(
        {
            "subfield_id": "S1",
            "subfield_display_name": "Subfield 1",
            "field_id": "F1",
            "field_display_name": "Field 1",
            "domain_id": "D1",
            "domain_display_name": "Domain 1",
            "publication_year": 2000,
            "sampling_method": "sample_api",
            "available_valid_works": 1000,
            "planned_sample_size": 400,
            "api_initial_sample_size": 700,
            "seed": 2043,
        }
    )


class RateLimitedClient:
    def get_json(self, endpoint, params):
        raise OpenAlexRateLimitError(
            "429 rate limit for https://api.openalex.org/works",
            retry_after_seconds=120,
            consecutive_429=2,
        )


def test_rate_limited_cell_gets_retryable_manifest_status() -> None:
    module = load_downloader_module()
    row = sample_plan_row()

    record = module.process_sample_plan_row(
        client=RateLimitedClient(),
        row=row,
        config=minimal_config(),
        seen_work_ids=set(),
        cell_counts={},
        subfield_counts={},
        new_records=[],
        max_backfill_rounds=0,
    )

    assert record["status"] == "rate_limited"
    assert "429" in record["error_message"]
    assert record["shortfall"] == 400


def test_resume_pending_plan_keeps_failed_and_rate_limited_cells() -> None:
    module = load_downloader_module()
    sample_plan = pd.DataFrame(
        [
            {"subfield_id": "S1", "publication_year": 2000, "planned_sample_size": 400},
            {"subfield_id": "S1", "publication_year": 2001, "planned_sample_size": 400},
            {"subfield_id": "S1", "publication_year": 2002, "planned_sample_size": 400},
        ]
    )
    completed_keys = {cell_key("S1", 2000)}

    pending, skipped = module.pending_sample_plan_for_resume(
        sample_plan,
        completed_keys,
        resume_enabled=True,
    )

    assert skipped == 1
    assert pending["publication_year"].tolist() == [2001, 2002]


def test_failed_cells_by_error_type_counts_429_separately() -> None:
    module = load_downloader_module()
    records = {
        ("S1", 2000): {"status": "rate_limited", "error_message": "429 rate limit"},
        ("S1", 2001): {"status": "failed", "error_message": "Unknown sampling method: x"},
        ("S1", 2002): {"status": "completed_shortfall", "error_message": ""},
    }

    summary = module.failed_cells_by_error_type(records)

    assert summary == {
        "rate_limit_429": 1,
        "unknown_sampling_method": 1,
    }
