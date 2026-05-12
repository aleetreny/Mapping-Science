from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.download_state import cell_key, completed_cell_keys_from_manifest
from src.openalex import (
    DEFAULT_SELECT_FIELDS,
    OpenAlexClient,
    OpenAlexRateLimitError,
    build_sampled_text_query_params,
    build_year_text_query_params,
    short_openalex_id,
)
from src.extraction_versions import (
    add_dataset_version_argument,
    apply_dataset_version,
    extraction_paths,
    output_path_display,
    versioned_table_name,
)
from src.sampling import api_initial_sample_size, stable_backfill_seed
from src.storage import connect_duckdb, ensure_dirs, load_parquet, save_parquet, write_table
from src.works import WORKS_TEXT_COLUMNS, validate_and_normalize_work


MANIFEST_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "publication_year",
    "sampling_method",
    "available_valid_works",
    "planned_sample_size",
    "api_initial_sample_size",
    "raw_returned_works",
    "valid_after_local_filter",
    "kept_works",
    "duplicate_or_already_seen",
    "discarded_local_validation",
    "shortfall",
    "backfill_rounds_used",
    "seeds_used",
    "status",
    "error_message",
]


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenAlex sampled text corpus.")
    add_dataset_version_argument(parser)
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads.")
    parser.add_argument("--resume", action="store_true", help="Resume existing outputs.")
    parser.add_argument("--force", action="store_true", help="Ignore existing outputs.")
    parser.add_argument(
        "--limit-subfields",
        type=int,
        default=None,
        help="Limit unique subfields for testing.",
    )
    parser.add_argument(
        "--only-subfield",
        default=None,
        help="Download one subfield id for debugging.",
    )
    parser.add_argument(
        "--max-backfill-rounds",
        type=int,
        default=None,
        help="Override configured max backfill rounds.",
    )
    parser.add_argument(
        "--cooldown-after-429",
        type=float,
        default=None,
        help="Seconds to sleep after OpenAlex 429 when Retry-After is absent.",
    )
    parser.add_argument(
        "--max-consecutive-429",
        type=int,
        default=None,
        help="Maximum consecutive 429 cooldown cycles before marking a cell rate_limited.",
    )
    parser.add_argument(
        "--write-every-n-subfields",
        type=int,
        default=None,
        help="Override checkpoint frequency. Use 1 for large/resumable downloads.",
    )
    return parser.parse_args()


def apply_download_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.cooldown_after_429 is not None:
        if args.cooldown_after_429 < 0:
            raise ValueError("--cooldown-after-429 must be non-negative")
        config.setdefault("openalex", {})["cooldown_after_429"] = float(
            args.cooldown_after_429
        )
    if args.max_consecutive_429 is not None:
        if args.max_consecutive_429 < 1:
            raise ValueError("--max-consecutive-429 must be at least 1")
        config.setdefault("openalex", {})["max_consecutive_429"] = int(
            args.max_consecutive_429
        )
    if args.write_every_n_subfields is not None:
        if args.write_every_n_subfields < 1:
            raise ValueError("--write-every-n-subfields must be at least 1")
        config.setdefault("sampling", {})["write_every_n_subfields"] = int(
            args.write_every_n_subfields
        )


def selected_sample_plan(
    sample_plan: pd.DataFrame,
    limit_subfields: int | None,
    only_subfield: str | None,
) -> pd.DataFrame:
    selected = sample_plan.copy()
    selected["subfield_id"] = selected["subfield_id"].astype(str)
    selected = selected.sort_values(["subfield_id", "publication_year"]).reset_index(
        drop=True
    )
    if only_subfield is not None:
        return selected[selected["subfield_id"] == str(only_subfield)].reset_index(
            drop=True
        )
    if limit_subfields is None:
        return selected
    keep_ids = selected["subfield_id"].drop_duplicates().head(limit_subfields)
    return selected[selected["subfield_id"].isin(set(keep_ids))].reset_index(drop=True)


def ensure_sample_plan_columns(sample_plan: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Support older sample_plan files by deriving new production columns."""
    result = sample_plan.copy()
    sampling = config["sampling"]
    oversample_factor = float(sampling.get("oversample_factor", 1.0))
    if "api_initial_sample_size" not in result.columns:
        result["api_initial_sample_size"] = result.apply(
            lambda row: api_initial_sample_size(
                available_valid_works=int(row["available_valid_works"]),
                planned_sample_size=int(row["planned_sample_size"]),
                sampling_method=str(row["sampling_method"]),
                oversample_factor=oversample_factor,
            ),
            axis=1,
        )
    if "expected_shortfall_risk" not in result.columns:
        result["expected_shortfall_risk"] = (
            (result["planned_sample_size"] > 0)
            & (result["api_initial_sample_size"] == result["planned_sample_size"])
        )
    return result


def sample_request_pages(sample_size: int, per_page: int) -> int:
    if sample_size <= 0:
        return 0
    return math.ceil(sample_size / min(sample_size, per_page, 200))


def estimate_initial_requests(sample_plan: pd.DataFrame, per_page: int) -> int:
    planned = sample_plan[sample_plan["planned_sample_size"] > 0]
    sample_rows = planned[planned["sampling_method"] == "sample_api"]
    all_rows = planned[planned["sampling_method"] == "download_all_available"]
    return int(
        sum(sample_request_pages(int(size), per_page) for size in sample_rows["api_initial_sample_size"])
    ) + int(
        sum(math.ceil(int(size) / per_page) for size in all_rows["planned_sample_size"])
    )


def query_params_for_row(
    row: pd.Series,
    config: dict[str, Any],
    sample_size: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    work_types = config["filters"]["work_types"]
    language = config["filters"]["language"]
    per_page = int(config["openalex"].get("per_page", 200))

    if row["sampling_method"] == "sample_api":
        return build_sampled_text_query_params(
            subfield_id=str(row["subfield_id"]),
            year=int(row["publication_year"]),
            sample_size=int(sample_size or row["api_initial_sample_size"]),
            seed=int(seed if seed is not None else row["seed"]),
            work_types=work_types,
            language=language,
            select_fields=DEFAULT_SELECT_FIELDS,
        )

    return build_year_text_query_params(
        subfield_id=str(row["subfield_id"]),
        year=int(row["publication_year"]),
        work_types=work_types,
        language=language,
        per_page=per_page,
        select_fields=DEFAULT_SELECT_FIELDS,
    )


def print_dry_run(sample_plan: pd.DataFrame, config: dict[str, Any], args: argparse.Namespace) -> None:
    per_page = int(config["openalex"].get("per_page", 200))
    planned = sample_plan[sample_plan["planned_sample_size"] > 0]
    sampling = config["sampling"]

    print(f"subfields selected: {sample_plan['subfield_id'].nunique()}")
    print(f"planned valid works: {int(sample_plan['planned_sample_size'].sum())}")
    print(f"initial raw API sample works: {int(sample_plan['api_initial_sample_size'].sum())}")
    print(f"estimated initial API requests: {estimate_initial_requests(sample_plan, per_page)}")
    print(f"oversample factor: {sampling.get('oversample_factor', 1.0)}")
    print(f"cooldown after 429: {config.get('openalex', {}).get('cooldown_after_429', 180)}")
    print(f"max consecutive 429: {config.get('openalex', {}).get('max_consecutive_429', 10)}")
    print(f"write every n subfields: {sampling.get('write_every_n_subfields', 5)}")
    print(
        "max backfill rounds: "
        f"{args.max_backfill_rounds if args.max_backfill_rounds is not None else sampling.get('max_backfill_rounds', 0)}"
    )
    print("planned valid works by sampling method:")
    if planned.empty:
        print("(none)")
    else:
        print(
            planned.groupby("sampling_method")["planned_sample_size"]
            .sum()
            .astype(int)
            .to_string()
        )
    print("initial raw works by sampling method:")
    if planned.empty:
        print("(none)")
    else:
        print(
            planned.groupby("sampling_method")["api_initial_sample_size"]
            .sum()
            .astype(int)
            .to_string()
        )

    examples = planned.head(3)
    if not examples.empty:
        print("example query params:")
        for _, row in examples.iterrows():
            print(json.dumps(query_params_for_row(row, config), indent=2))


def empty_works_df() -> pd.DataFrame:
    return pd.DataFrame(columns=WORKS_TEXT_COLUMNS)


def ensure_works_text_columns(works: pd.DataFrame) -> pd.DataFrame:
    result = works.copy()
    for column in WORKS_TEXT_COLUMNS:
        if column not in result.columns:
            result[column] = None
    return result[WORKS_TEXT_COLUMNS]


def empty_manifest_df() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def load_existing_outputs(
    works_path: Path,
    manifest_path: Path,
    resume_enabled: bool,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if force or not resume_enabled:
        return empty_works_df(), empty_manifest_df()

    works = load_parquet(works_path) if works_path.exists() else empty_works_df()
    works = ensure_works_text_columns(works)
    manifest = load_parquet(manifest_path) if manifest_path.exists() else empty_manifest_df()
    return works, manifest


def count_maps(works: pd.DataFrame) -> tuple[dict[tuple[str, int], int], dict[str, int]]:
    if works.empty:
        return {}, {}
    cell_counts = {
        cell_key(row.subfield_id, row.publication_year): int(row.count)
        for row in works.groupby(["subfield_id", "publication_year"])
        .size()
        .reset_index(name="count")
        .itertuples(index=False)
    }
    subfield_counts = {
        str(row.subfield_id): int(row.count)
        for row in works.groupby("subfield_id")
        .size()
        .reset_index(name="count")
        .itertuples(index=False)
    }
    return cell_counts, subfield_counts


def cap_works_per_subfield(works: pd.DataFrame, max_per_subfield: int) -> pd.DataFrame:
    if works.empty or "subfield_id" not in works.columns:
        return works
    ordered = works.sort_values(
        ["subfield_id", "publication_year", "work_id"], kind="mergesort"
    ).reset_index(drop=True)
    keep = ordered.groupby("subfield_id").cumcount() < int(max_per_subfield)
    return ordered.loc[keep, WORKS_TEXT_COLUMNS].reset_index(drop=True)


def current_works_dataframe(base_works: pd.DataFrame, new_records: list[dict[str, Any]]) -> pd.DataFrame:
    if not new_records:
        return base_works.copy()
    new_df = pd.DataFrame(new_records, columns=WORKS_TEXT_COLUMNS)
    if base_works.empty:
        return new_df
    return pd.concat([base_works, new_df], ignore_index=True)


def manifest_dict_from_df(manifest: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    records = {}
    if manifest.empty:
        return records
    for row in manifest.to_dict(orient="records"):
        records[cell_key(row["subfield_id"], row["publication_year"])] = row
    return records


def pending_sample_plan_for_resume(
    sample_plan: pd.DataFrame,
    completed_keys: set[tuple[str, int]],
    *,
    resume_enabled: bool,
) -> tuple[pd.DataFrame, int]:
    if not resume_enabled or not completed_keys or sample_plan.empty:
        return sample_plan.copy(), 0
    pending = sample_plan.copy()
    keys = [
        cell_key(row.subfield_id, row.publication_year)
        for row in pending[["subfield_id", "publication_year"]].itertuples(index=False)
    ]
    completed_mask = pd.Series(
        [key in completed_keys for key in keys],
        index=pending.index,
    )
    return pending.loc[~completed_mask].reset_index(drop=True), int(completed_mask.sum())


def make_manifest_record(
    row: pd.Series,
    raw_returned: int,
    valid_after_local_filter: int,
    kept_works: int,
    duplicate_or_already_seen: int,
    discarded_local_validation: int,
    backfill_rounds_used: int,
    seeds_used: list[int],
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    planned = int(row["planned_sample_size"])
    return {
        "subfield_id": str(row["subfield_id"]),
        "subfield_display_name": row.get("subfield_display_name"),
        "field_id": row.get("field_id"),
        "field_display_name": row.get("field_display_name"),
        "domain_id": row.get("domain_id"),
        "domain_display_name": row.get("domain_display_name"),
        "publication_year": int(row["publication_year"]),
        "sampling_method": row.get("sampling_method"),
        "available_valid_works": int(row["available_valid_works"]),
        "planned_sample_size": planned,
        "api_initial_sample_size": int(row.get("api_initial_sample_size", planned)),
        "raw_returned_works": int(raw_returned),
        "valid_after_local_filter": int(valid_after_local_filter),
        "kept_works": int(kept_works),
        "duplicate_or_already_seen": int(duplicate_or_already_seen),
        "discarded_local_validation": int(discarded_local_validation),
        "shortfall": max(0, planned - int(kept_works)),
        "backfill_rounds_used": int(backfill_rounds_used),
        "seeds_used": ",".join(str(seed) for seed in seeds_used),
        "status": status,
        "error_message": error_message,
    }


def update_manifest_kept_counts(
    manifest_records: dict[tuple[str, int], dict[str, Any]],
    works: pd.DataFrame,
) -> None:
    cell_counts, _ = count_maps(works)
    for key, record in manifest_records.items():
        kept = int(cell_counts.get(key, 0))
        planned = int(record.get("planned_sample_size", 0))
        record["kept_works"] = kept
        record["shortfall"] = max(0, planned - kept)
        if record.get("status") == "completed_target_met" and kept < planned:
            record["status"] = "completed_shortfall"


def error_type_from_manifest_record(record: dict[str, Any]) -> str:
    status = str(record.get("status") or "")
    error_message = str(record.get("error_message") or "").lower()
    if status == "rate_limited" or "429" in error_message or "rate limit" in error_message:
        return "rate_limit_429"
    if status == "failed" and "unknown sampling method" in error_message:
        return "unknown_sampling_method"
    if status == "failed" and error_message:
        return error_message.split(":", 1)[0][:80]
    if status == "failed":
        return "failed_unknown"
    return status or "unknown"


def failed_cells_by_error_type(
    manifest_records: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in manifest_records.values():
        if record.get("status") not in {"failed", "rate_limited"}:
            continue
        error_type = error_type_from_manifest_record(record)
        counts[error_type] = counts.get(error_type, 0) + 1
    return dict(sorted(counts.items()))


def write_outputs(
    base_works: pd.DataFrame,
    new_records: list[dict[str, Any]],
    manifest_records: dict[tuple[str, int], dict[str, Any]],
    config: dict[str, Any],
    dataset_version: str | None,
) -> pd.DataFrame:
    duckdb_path = ROOT / config["storage"]["duckdb_path"]
    max_per_subfield = int(config["sampling"]["max_sample_per_subfield"])
    paths = extraction_paths(ROOT, config, dataset_version)

    works = current_works_dataframe(base_works, new_records)
    works = works.drop_duplicates("work_id", keep="first").reset_index(drop=True)
    works = cap_works_per_subfield(works, max_per_subfield)
    update_manifest_kept_counts(manifest_records, works)

    manifest = pd.DataFrame(
        list(manifest_records.values()), columns=MANIFEST_COLUMNS
    ).sort_values(["subfield_id", "publication_year"])

    save_parquet(works, paths.works_text)
    save_parquet(manifest, paths.download_manifest)

    con = connect_duckdb(duckdb_path)
    try:
        write_table(con, versioned_table_name("works_text", dataset_version), works)
        write_table(
            con,
            versioned_table_name("download_manifest", dataset_version),
            manifest,
        )
    finally:
        con.close()

    return works


def process_raw_works(
    raw_works: list[dict[str, Any]],
    row: pd.Series,
    config: dict[str, Any],
    seen_work_ids: set[str],
    cell_counts: dict[tuple[str, int], int],
    subfield_counts: dict[str, int],
    max_per_subfield: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    subfield_id = str(row["subfield_id"])
    year = int(row["publication_year"])
    key = cell_key(subfield_id, year)
    target_keep = int(row["planned_sample_size"])

    accepted: list[dict[str, Any]] = []
    metrics = {
        "raw_returned": len(raw_works),
        "valid_after_local_filter": 0,
        "duplicate_or_already_seen": 0,
        "discarded_local_validation": 0,
    }

    for work in raw_works:
        normalized = validate_and_normalize_work(
            work,
            config=config,
            expected_subfield_id=subfield_id,
            expected_year=year,
        )
        if normalized is None:
            metrics["discarded_local_validation"] += 1
            continue

        metrics["valid_after_local_filter"] += 1
        work_id = normalized["work_id"]
        if work_id in seen_work_ids:
            metrics["duplicate_or_already_seen"] += 1
            continue
        if cell_counts.get(key, 0) >= target_keep:
            continue
        if subfield_counts.get(subfield_id, 0) >= max_per_subfield:
            continue

        seen_work_ids.add(work_id)
        cell_counts[key] = cell_counts.get(key, 0) + 1
        subfield_counts[subfield_id] = subfield_counts.get(subfield_id, 0) + 1
        accepted.append(normalized)

    return accepted, metrics


def fetch_sampled_raw_works(
    client: OpenAlexClient,
    row: pd.Series,
    config: dict[str, Any],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []

    params = query_params_for_row(row, config, sample_size=sample_size, seed=seed)
    raw_works: list[dict[str, Any]] = []
    page = 1
    while len(raw_works) < sample_size:
        page_params = dict(params)
        page_params["page"] = page
        data = client.get_json("works", page_params)
        works = data.get("results") or []
        if not works:
            break
        raw_works.extend(works)
        if len(works) < int(params.get("per-page", 200)):
            break
        page += 1
    return raw_works[:sample_size]


def fetch_all_available_raw_works(
    client: OpenAlexClient,
    row: pd.Series,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    planned = int(row["planned_sample_size"])
    if planned <= 0:
        return []
    params = query_params_for_row(row, config)
    raw_works = []
    for work in client.paginate("works", params):
        raw_works.append(work)
        if len(raw_works) >= planned:
            break
    return raw_works


def process_sample_plan_row(
    client: OpenAlexClient,
    row: pd.Series,
    config: dict[str, Any],
    seen_work_ids: set[str],
    cell_counts: dict[tuple[str, int], int],
    subfield_counts: dict[str, int],
    new_records: list[dict[str, Any]],
    max_backfill_rounds: int,
) -> dict[str, Any]:
    sampling = config["sampling"]
    oversample_factor = float(sampling.get("oversample_factor", 1.0))
    backfill_seed_step = int(sampling.get("backfill_seed_step", 100000))
    max_per_subfield = int(sampling["max_sample_per_subfield"])

    subfield_id = str(row["subfield_id"])
    year = int(row["publication_year"])
    key = cell_key(subfield_id, year)
    target_keep = int(row["planned_sample_size"])
    available = int(row["available_valid_works"])

    raw_returned = 0
    valid_after_local_filter = 0
    duplicate_or_already_seen = 0
    discarded_local_validation = 0
    backfill_rounds_used = 0
    seeds_used: list[int] = []
    raw_ids_seen: set[str] = set()

    if target_keep <= 0 or available <= 0:
        return make_manifest_record(
            row,
            raw_returned=0,
            valid_after_local_filter=0,
            kept_works=cell_counts.get(key, 0),
            duplicate_or_already_seen=0,
            discarded_local_validation=0,
            backfill_rounds_used=0,
            seeds_used=[],
            status="skipped_no_available_works",
        )

    if cell_counts.get(key, 0) >= target_keep:
        return make_manifest_record(
            row,
            raw_returned=0,
            valid_after_local_filter=0,
            kept_works=cell_counts.get(key, 0),
            duplicate_or_already_seen=0,
            discarded_local_validation=0,
            backfill_rounds_used=0,
            seeds_used=[],
            status="completed_target_met",
        )

    try:
        if row["sampling_method"] == "download_all_available":
            raw_works = fetch_all_available_raw_works(client, row, config)
            raw_ids_seen.update(
                work_id
                for work_id in (short_openalex_id(work.get("id")) for work in raw_works)
                if work_id
            )
            accepted, metrics = process_raw_works(
                raw_works,
                row,
                config,
                seen_work_ids,
                cell_counts,
                subfield_counts,
                max_per_subfield,
            )
            new_records.extend(accepted)
            raw_returned += metrics["raw_returned"]
            valid_after_local_filter += metrics["valid_after_local_filter"]
            duplicate_or_already_seen += metrics["duplicate_or_already_seen"]
            discarded_local_validation += metrics["discarded_local_validation"]
        elif row["sampling_method"] == "sample_api":
            current_round = 0
            while current_round <= max_backfill_rounds:
                current_count = cell_counts.get(key, 0)
                shortfall = target_keep - current_count
                if shortfall <= 0:
                    break
                if subfield_counts.get(subfield_id, 0) >= max_per_subfield:
                    break
                if len(raw_ids_seen) >= available:
                    break

                if current_round == 0:
                    sample_size = int(row["api_initial_sample_size"])
                    seed = int(row["seed"])
                else:
                    sample_size = min(
                        available,
                        math.ceil(shortfall * oversample_factor * 1.5),
                        10000,
                    )
                    seed = stable_backfill_seed(
                        int(row["seed"]), current_round, backfill_seed_step
                    )
                    backfill_rounds_used = current_round

                seeds_used.append(seed)
                raw_works = fetch_sampled_raw_works(
                    client, row, config, sample_size=sample_size, seed=seed
                )
                before_count = cell_counts.get(key, 0)
                raw_ids_seen.update(
                    work_id
                    for work_id in (short_openalex_id(work.get("id")) for work in raw_works)
                    if work_id
                )
                accepted, metrics = process_raw_works(
                    raw_works,
                    row,
                    config,
                    seen_work_ids,
                    cell_counts,
                    subfield_counts,
                    max_per_subfield,
                )
                new_records.extend(accepted)
                raw_returned += metrics["raw_returned"]
                valid_after_local_filter += metrics["valid_after_local_filter"]
                duplicate_or_already_seen += metrics["duplicate_or_already_seen"]
                discarded_local_validation += metrics["discarded_local_validation"]

                if cell_counts.get(key, 0) >= target_keep:
                    break
                if cell_counts.get(key, 0) == before_count:
                    break
                if not raw_works:
                    break
                current_round += 1
        else:
            return make_manifest_record(
                row,
                raw_returned=0,
                valid_after_local_filter=0,
                kept_works=cell_counts.get(key, 0),
                duplicate_or_already_seen=0,
                discarded_local_validation=0,
                backfill_rounds_used=0,
                seeds_used=[],
                status="failed",
                error_message=f"Unknown sampling method: {row['sampling_method']}",
            )
    except OpenAlexRateLimitError as exc:
        return make_manifest_record(
            row,
            raw_returned=raw_returned,
            valid_after_local_filter=valid_after_local_filter,
            kept_works=cell_counts.get(key, 0),
            duplicate_or_already_seen=duplicate_or_already_seen,
            discarded_local_validation=discarded_local_validation,
            backfill_rounds_used=backfill_rounds_used,
            seeds_used=seeds_used,
            status="rate_limited",
            error_message=str(exc),
        )
    except Exception as exc:
        return make_manifest_record(
            row,
            raw_returned=raw_returned,
            valid_after_local_filter=valid_after_local_filter,
            kept_works=cell_counts.get(key, 0),
            duplicate_or_already_seen=duplicate_or_already_seen,
            discarded_local_validation=discarded_local_validation,
            backfill_rounds_used=backfill_rounds_used,
            seeds_used=seeds_used,
            status="failed",
            error_message=str(exc),
        )

    kept = cell_counts.get(key, 0)
    status = "completed_target_met" if kept >= target_keep else "completed_shortfall"
    return make_manifest_record(
        row,
        raw_returned=raw_returned,
        valid_after_local_filter=valid_after_local_filter,
        kept_works=kept,
        duplicate_or_already_seen=duplicate_or_already_seen,
        discarded_local_validation=discarded_local_validation,
        backfill_rounds_used=backfill_rounds_used,
        seeds_used=seeds_used,
        status=status,
    )


def main() -> None:
    args = parse_args()
    config = apply_dataset_version(load_config(), args.dataset_version)
    apply_download_cli_overrides(config, args)
    ensure_dirs(config)

    paths = extraction_paths(ROOT, config, args.dataset_version)
    sample_plan_path = paths.sample_plan
    if not sample_plan_path.exists():
        raise FileNotFoundError(
            f"Missing {output_path_display(sample_plan_path, ROOT)}. "
            "Run scripts/03_build_sample_plan.py first."
        )

    selected_plan = ensure_sample_plan_columns(load_parquet(sample_plan_path), config)
    selected_plan = selected_sample_plan(
        selected_plan, args.limit_subfields, args.only_subfield
    )

    if args.dry_run:
        print_dry_run(selected_plan, config, args)
        return

    sampling = config["sampling"]
    resume_enabled = bool(sampling.get("resume_existing_download", True)) or args.resume
    if args.force:
        resume_enabled = False

    base_works, existing_manifest = load_existing_outputs(
        works_path=paths.works_text,
        manifest_path=paths.download_manifest,
        resume_enabled=resume_enabled,
        force=args.force,
    )
    base_works = cap_works_per_subfield(
        base_works.drop_duplicates("work_id", keep="first").reset_index(drop=True),
        int(sampling["max_sample_per_subfield"]),
    )
    manifest_records = manifest_dict_from_df(existing_manifest)
    completed_keys = (
        completed_cell_keys_from_manifest(existing_manifest) if resume_enabled else set()
    )
    sample_plan, skipped_completed_cells = pending_sample_plan_for_resume(
        selected_plan,
        completed_keys,
        resume_enabled=resume_enabled,
    )

    if skipped_completed_cells:
        print(
            f"Resume: skipping {skipped_completed_cells} completed cells; "
            f"retrying {len(sample_plan)} pending/failed cells."
        )

    max_backfill_rounds = (
        int(args.max_backfill_rounds)
        if args.max_backfill_rounds is not None
        else int(sampling.get("max_backfill_rounds", 0))
    )
    write_every_n_subfields = int(sampling.get("write_every_n_subfields", 5))

    cell_counts, subfield_counts = count_maps(base_works)
    seen_work_ids = set(base_works["work_id"].astype(str).tolist()) if not base_works.empty else set()
    new_records: list[dict[str, Any]] = []

    client = OpenAlexClient.from_config(config)
    subfield_ids = sample_plan["subfield_id"].drop_duplicates().tolist()
    progress_desc = "Downloading subfields"
    if resume_enabled:
        progress_desc = f"{progress_desc} (resume; {skipped_completed_cells} cells skipped)"

    for index, subfield_id in enumerate(tqdm(subfield_ids, desc=progress_desc), start=1):
        subfield_rows = sample_plan[sample_plan["subfield_id"] == subfield_id]
        for _, row in subfield_rows.iterrows():
            key = cell_key(row["subfield_id"], row["publication_year"])
            record = process_sample_plan_row(
                client=client,
                row=row,
                config=config,
                seen_work_ids=seen_work_ids,
                cell_counts=cell_counts,
                subfield_counts=subfield_counts,
                new_records=new_records,
                max_backfill_rounds=max_backfill_rounds,
            )
            manifest_records[key] = record

        if write_every_n_subfields > 0 and index % write_every_n_subfields == 0:
            base_works = write_outputs(
                base_works,
                new_records,
                manifest_records,
                config,
                args.dataset_version,
            )
            new_records = []
            cell_counts, subfield_counts = count_maps(base_works)
            seen_work_ids = set(base_works["work_id"].astype(str).tolist())

    final_works = write_outputs(
        base_works,
        new_records,
        manifest_records,
        config,
        args.dataset_version,
    )
    summary = {
        "selected_subfields": int(selected_plan["subfield_id"].nunique()),
        "remaining_subfields_processed": int(len(subfield_ids)),
        "skipped_completed_cells_on_resume": int(skipped_completed_cells),
        "planned_works": int(selected_plan["planned_sample_size"].sum()),
        "remaining_planned_works": int(sample_plan["planned_sample_size"].sum())
        if not sample_plan.empty
        else 0,
        "downloaded_works": len(final_works),
        "resume": resume_enabled,
        "force": args.force,
        "limit_subfields": args.limit_subfields,
        "only_subfield": args.only_subfield,
        "cooldown_after_429": float(config.get("openalex", {}).get("cooldown_after_429", 180)),
        "max_consecutive_429": int(config.get("openalex", {}).get("max_consecutive_429", 10)),
        "write_every_n_subfields": int(write_every_n_subfields),
        "failed_cells_by_error_type": failed_cells_by_error_type(manifest_records),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
