from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml

from src.extraction_versions import (
    apply_dataset_version,
    extraction_paths,
    extraction_filter_summary,
)
from src.analysis import build_versioned_analysis_subfields
from src.openalex import DEFAULT_SELECT_FIELDS
from src.works import WORKS_TEXT_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
DATASET_VERSION = "2000_2024_400py"


def load_script_module(script_name: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def base_config() -> dict:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def version_config() -> dict:
    return apply_dataset_version(base_config(), DATASET_VERSION)


def test_version_config_resolves_2000_2024_400py() -> None:
    config = version_config()

    assert config["windows"]["morphology_start_year"] == 2000
    assert config["windows"]["morphology_end_year"] == 2024
    assert config["windows"]["exclude_years"] == [2025, 2026]
    assert config["sampling"]["target_per_year_per_subfield"] == 400
    assert config["sampling"]["target_sample_per_subfield"] == 10000
    assert config["sampling"]["max_sample_per_subfield"] == 10000
    assert config["sampling"]["redistribute_unused_annual_capacity"] is False


def test_versioned_paths_do_not_target_canonical_files() -> None:
    config = version_config()
    paths = extraction_paths(ROOT, config, DATASET_VERSION)

    assert paths.subfield_year_counts.name == "subfield_year_counts_2000_2024_400py.parquet"
    assert paths.sample_plan.name == "sample_plan_2000_2024_400py.parquet"
    assert paths.download_manifest.name == "download_manifest_2000_2024_400py.parquet"
    assert paths.works_text.name == "works_text_2000_2024_400py.parquet"
    assert paths.analysis_subfields.name == "analysis_subfields_2000_2024_400py.parquet"
    assert paths.works_text.name != "works_text.parquet"


def test_sample_plan_caps_each_year_at_400_and_uses_10000_total() -> None:
    module = load_script_module("03_build_sample_plan.py")
    config = version_config()
    years = list(range(2000, 2025))
    corpus_plan = pd.DataFrame(
        [
            {
                "subfield_id": "S1",
                "subfield_display_name": "Subfield 1",
                "field_id": "F1",
                "field_display_name": "Field 1",
                "domain_id": "D1",
                "domain_display_name": "Domain 1",
                "eligible_for_text_corpus": True,
            }
        ]
    )
    counts = pd.DataFrame(
        [
            {
                "subfield_id": "S1",
                "publication_year": year,
                "n_works_article_preprint_en_with_abstract": 1000
                if year != 2000
                else 250,
            }
            for year in years
        ]
    )

    sample_plan = module.build_sample_plan(corpus_plan, counts, config)

    assert sample_plan["planned_sample_size"].max() == 400
    assert sample_plan.loc[
        sample_plan["publication_year"] == 2000, "planned_sample_size"
    ].item() == 250
    assert sample_plan["planned_sample_size"].sum() == 9850
    assert sample_plan["planned_sample_size"].sum() > 3000
    assert sample_plan["expected_shortfall_risk"].any()


def test_validation_detects_versioned_extraction_problems() -> None:
    module = load_script_module("05_validate_database.py")
    config = version_config()
    sample_plan = pd.DataFrame(
        [
            {
                "subfield_id": "S1",
                "subfield_display_name": "Subfield 1",
                "field_id": "F1",
                "field_display_name": "Field 1",
                "domain_id": "D1",
                "domain_display_name": "Domain 1",
                "publication_year": 2000,
                "available_valid_works": 400,
                "planned_sample_size": 400,
                "sampling_method": "sample_api",
            }
        ]
    )
    works_text = pd.DataFrame(
        [
            {
                "work_id": "W1",
                "title": "A valid enough title here",
                "abstract": " ".join(["word"] * 100),
                "publication_year": 2000,
                "type": "article",
                "language": "en",
                "subfield_id": "S1",
                "field_id": "F1",
                "domain_id": "D1",
                "primary_topic_id": "T1",
                "primary_topic_display_name": "Topic",
                "topics_json": "[]",
                "title_token_count": 5,
                "abstract_token_count": 100,
            },
            {
                "work_id": "W1",
                "title": "",
                "abstract": "",
                "publication_year": 2025,
                "type": "book",
                "language": "es",
                "subfield_id": None,
                "field_id": None,
                "domain_id": None,
                "primary_topic_id": None,
                "primary_topic_display_name": None,
                "topics_json": "{bad",
                "title_token_count": 0,
                "abstract_token_count": 0,
            },
        ]
    )

    summary, coverage, _, _ = module.build_validation_summary(
        {
            "works_text": works_text,
            "sample_plan": sample_plan,
            "download_manifest": pd.DataFrame(),
        },
        config,
        DATASET_VERSION,
    )
    checks = summary["checks"]

    assert checks["duplicate_work_ids"] == 1
    assert checks["rows_outside_expected_year_window"] == 1
    assert checks["rows_in_2025_or_2026"] == 1
    assert checks["missing_subfield_id"] == 1
    assert checks["missing_field_id"] == 1
    assert checks["missing_domain_id"] == 1
    assert checks["missing_primary_topic_id"] == 1
    assert checks["unparsable_topics_json"] == 1
    assert summary["validation_failed"] is True
    assert coverage.loc[0, "shortfall"] == 399


def test_versioned_analysis_subfields_use_annual_coverage_columns() -> None:
    config = version_config()
    sample_plan = pd.DataFrame(
        [
            {
                "subfield_id": "S1",
                "subfield_display_name": "Subfield 1",
                "field_id": "F1",
                "field_display_name": "Field 1",
                "domain_id": "D1",
                "domain_display_name": "Domain 1",
                "publication_year": year,
                "planned_sample_size": 400,
            }
            for year in range(2000, 2025)
        ]
    )
    corpus_plan = pd.DataFrame(
        [
            {
                "subfield_id": "S1",
                "subfield_display_name": "Subfield 1",
                "field_id": "F1",
                "field_display_name": "Field 1",
                "domain_id": "D1",
                "domain_display_name": "Domain 1",
                "n_valid_works_2000_2024": 12000,
            }
        ]
    )
    works_text = pd.DataFrame(
        [
            {
                "work_id": f"W{year}_{idx}",
                "subfield_id": "S1",
                "publication_year": year,
            }
            for year in range(2000, 2025)
            for idx in range(400 if year < 2024 else 300)
        ]
    )

    analysis = build_versioned_analysis_subfields(
        works_text,
        sample_plan,
        corpus_plan,
        config,
    )

    row = analysis.iloc[0]
    assert row["n_valid_works_2000_2024"] == 12000
    assert row["n_downloaded_works_2000_2024"] == 9900
    assert row["n_years_reaching_400"] == 24
    assert row["n_years_below_400"] == 1
    assert bool(row["eligible_10000_full_period"]) is False
    assert bool(row["eligible_min_5000_full_period"]) is True
    assert bool(row["eligible_for_temporal_5year_exploration"]) is True


def test_filters_select_fields_and_output_columns_are_preserved() -> None:
    config = version_config()
    filters = extraction_filter_summary(config)

    assert filters == {
        "unit": "subfield",
        "use_primary_topic_subfield_only": True,
        "language": "en",
        "work_types": ["article", "preprint"],
        "require_abstract_for_text_corpus": True,
        "require_title_for_text_corpus": True,
        "min_title_tokens": 5,
        "min_abstract_tokens": 80,
        "exclude_retracted": True,
        "exclude_paratext": True,
    }
    for field in [
        "id",
        "doi",
        "title",
        "display_name",
        "abstract_inverted_index",
        "publication_year",
        "publication_date",
        "type",
        "language",
        "primary_topic",
        "topics",
        "cited_by_count",
        "referenced_works_count",
        "is_retracted",
        "is_paratext",
    ]:
        assert field in DEFAULT_SELECT_FIELDS
    for column in [
        "work_id",
        "doi",
        "title",
        "abstract",
        "text_for_embedding",
        "publication_year",
        "publication_date",
        "type",
        "language",
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        "primary_topic_id",
        "primary_topic_display_name",
        "topics_json",
        "title_token_count",
        "abstract_token_count",
        "text_token_count",
        "downloaded_at",
        "cited_by_count",
        "referenced_works_count",
    ]:
        assert column in WORKS_TEXT_COLUMNS
