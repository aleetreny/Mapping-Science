from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


VERSION_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class ExtractionPaths:
    dataset_version: str | None
    subfield_year_counts: Path
    field_year_counts: Path
    domain_year_counts: Path
    corpus_plan: Path
    sample_plan: Path
    download_manifest: Path
    works_text: Path
    analysis_subfields: Path
    validation_report: Path
    validation_summary: Path
    diagnostics_dir: Path
    subfield_year_download_coverage: Path
    subfield_coverage_summary: Path
    year_coverage_summary: Path
    extraction_summary: Path


def validate_dataset_version_name(dataset_version: str) -> None:
    if not VERSION_RE.fullmatch(dataset_version):
        raise ValueError(f"Unsafe dataset version: {dataset_version}")


def apply_dataset_version(
    config: dict[str, Any],
    dataset_version: str | None,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["active_dataset_version"] = dataset_version
    if dataset_version is None:
        return result

    validate_dataset_version_name(dataset_version)
    versions = result.get("dataset_versions", {})
    if dataset_version not in versions:
        available = ", ".join(sorted(versions)) or "(none configured)"
        raise ValueError(
            f"Unknown dataset version {dataset_version}. Available versions: {available}"
        )

    version_config = versions[dataset_version]
    result.setdefault("windows", {})
    result["windows"]["morphology_start_year"] = int(version_config["year_min"])
    result["windows"]["morphology_end_year"] = int(version_config["year_max"])
    result["windows"]["exclude_years"] = list(version_config.get("exclude_years", []))

    result.setdefault("sampling", {})
    for key, value in version_config.get("sampling", {}).items():
        result["sampling"][key] = value

    return result


def add_dataset_version_argument(parser: Any) -> None:
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Optional versioned extraction target, e.g. 2000_2024_400py.",
    )


def dataset_suffix(dataset_version: str | None) -> str:
    if dataset_version is None:
        return ""
    validate_dataset_version_name(dataset_version)
    return f"_{dataset_version}"


def versioned_filename(base: str, dataset_version: str | None, extension: str) -> str:
    return f"{base}{dataset_suffix(dataset_version)}.{extension.lstrip('.')}"


def versioned_table_name(base: str, dataset_version: str | None) -> str:
    return f"{base}{dataset_suffix(dataset_version)}"


def analysis_years(config: dict[str, Any]) -> list[int]:
    windows = config["windows"]
    return list(
        range(
            int(windows["morphology_start_year"]),
            int(windows["morphology_end_year"]) + 1,
        )
    )


def year_min_max(config: dict[str, Any]) -> tuple[int, int]:
    years = analysis_years(config)
    return min(years), max(years)


def period_label(config: dict[str, Any]) -> str:
    year_min, year_max = year_min_max(config)
    return f"{year_min}_{year_max}"


def n_valid_works_column(config: dict[str, Any]) -> str:
    return f"n_valid_works_{period_label(config)}"


def n_downloaded_works_column(config: dict[str, Any]) -> str:
    return f"n_downloaded_works_{period_label(config)}"


def extraction_paths(
    root: str | Path,
    config: dict[str, Any],
    dataset_version: str | None,
) -> ExtractionPaths:
    root = Path(root)
    storage = config["storage"]
    interim_dir = root / storage["interim_dir"]
    processed_dir = root / storage["processed_dir"]
    if dataset_version is None:
        validation_dir = interim_dir
        diagnostics_dir = root / "outputs" / "diagnostics" / "openalex_extraction"
        validation_report = validation_dir / "validation_report.md"
        validation_summary = validation_dir / "validation_summary.json"
    else:
        validation_dir = root / "outputs" / "validation"
        diagnostics_dir = (
            root
            / "outputs"
            / "diagnostics"
            / f"openalex_extraction_{dataset_version}"
        )
        validation_report = validation_dir / f"validation_report_{dataset_version}.md"
        validation_summary = validation_dir / f"validation_summary_{dataset_version}.json"

    return ExtractionPaths(
        dataset_version=dataset_version,
        subfield_year_counts=interim_dir
        / versioned_filename("subfield_year_counts", dataset_version, "parquet"),
        field_year_counts=interim_dir
        / versioned_filename("field_year_counts", dataset_version, "parquet"),
        domain_year_counts=interim_dir
        / versioned_filename("domain_year_counts", dataset_version, "parquet"),
        corpus_plan=interim_dir
        / versioned_filename("corpus_plan", dataset_version, "parquet"),
        sample_plan=interim_dir
        / versioned_filename("sample_plan", dataset_version, "parquet"),
        download_manifest=interim_dir
        / versioned_filename("download_manifest", dataset_version, "parquet"),
        works_text=processed_dir
        / versioned_filename("works_text", dataset_version, "parquet"),
        analysis_subfields=processed_dir
        / versioned_filename("analysis_subfields", dataset_version, "parquet"),
        validation_report=validation_report,
        validation_summary=validation_summary,
        diagnostics_dir=diagnostics_dir,
        subfield_year_download_coverage=diagnostics_dir
        / "subfield_year_download_coverage.csv",
        subfield_coverage_summary=diagnostics_dir / "subfield_coverage_summary.csv",
        year_coverage_summary=diagnostics_dir / "year_coverage_summary.csv",
        extraction_summary=diagnostics_dir / "extraction_summary.md",
    )


def output_path_display(path: str | Path, root: str | Path) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def count_table_paths(paths: ExtractionPaths) -> dict[str, Path]:
    return {
        "subfield": paths.subfield_year_counts,
        "field": paths.field_year_counts,
        "domain": paths.domain_year_counts,
    }


def target_total_from_config(config: dict[str, Any]) -> int:
    return int(config["sampling"]["target_sample_per_subfield"])


def target_per_year_from_config(config: dict[str, Any]) -> int:
    return int(config["sampling"]["target_per_year_per_subfield"])


def build_no_redistribution_allocations(
    available_by_year: dict[int, int],
    target_per_year: int,
) -> dict[int, int]:
    return {
        int(year): min(max(0, int(available)), int(target_per_year))
        for year, available in available_by_year.items()
    }


def safe_json_loads(value: Any) -> bool:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return False
    try:
        json.loads(str(value))
        return True
    except json.JSONDecodeError:
        return False


def extraction_filter_summary(config: dict[str, Any]) -> dict[str, Any]:
    filters = config["filters"]
    return {
        "unit": filters.get("unit"),
        "use_primary_topic_subfield_only": filters.get(
            "use_primary_topic_subfield_only"
        ),
        "language": filters.get("language"),
        "work_types": list(filters.get("work_types", [])),
        "require_abstract_for_text_corpus": filters.get(
            "require_abstract_for_text_corpus"
        ),
        "require_title_for_text_corpus": filters.get("require_title_for_text_corpus"),
        "min_title_tokens": filters.get("min_title_tokens"),
        "min_abstract_tokens": filters.get("min_abstract_tokens"),
        "exclude_retracted": filters.get("exclude_retracted"),
        "exclude_paratext": filters.get("exclude_paratext"),
    }
