from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.growth_targets import display_path, read_table, write_table
from src.openalex import OpenAlexClient, build_filter, short_openalex_id


DEFAULT_SUBFIELD_IDS = ["2202", "2200", "3500", "2611", "2725", "3307"]
DEFAULT_YEARS = [2010, 2019, 2020, 2025]
COUNT_COLUMN = "n_works_article_preprint"


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Validate extreme Stage 13 growth counts against direct OpenAlex counts."
    )
    parser.add_argument(
        "--counts-path",
        default="data/interim/subfield_year_counts.parquet",
    )
    parser.add_argument(
        "--growth-targets-path",
        default="data/processed/subfield_growth_targets.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/growth/count_validation",
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--subfield-id",
        action="append",
        default=None,
        help="Subfield id to check. May be supplied more than once.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config() -> dict[str, Any]:
    path = ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_outputs_do_not_exist(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path, ROOT) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing count-validation outputs without "
            f"--overwrite: {formatted}"
        )


def checked_at_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def query_filter_for_subfield_year(subfield_id: str, year: int) -> str:
    short_id = short_openalex_id(subfield_id)
    if not short_id:
        raise ValueError("subfield_id must not be empty")
    return build_filter(
        [
            f"primary_topic.subfield.id:{short_id}",
            f"publication_year:{int(year)}",
            "type:article|preprint",
            "is_retracted:false",
            "is_paratext:false",
        ]
    )


def prepare_local_counts(counts: pd.DataFrame) -> pd.DataFrame:
    required = {"subfield_id", "publication_year", COUNT_COLUMN}
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(
            "Local count table is missing required columns: "
            + ", ".join(sorted(missing))
        )
    prepared = counts.copy()
    prepared["subfield_id"] = prepared["subfield_id"].map(short_openalex_id)
    prepared["publication_year"] = pd.to_numeric(
        prepared["publication_year"], errors="coerce"
    ).astype("Int64")
    prepared["local_count"] = pd.to_numeric(prepared[COUNT_COLUMN], errors="coerce")
    prepared = prepared[
        prepared["subfield_id"].notna()
        & prepared["publication_year"].notna()
        & prepared["local_count"].notna()
    ].copy()
    prepared["publication_year"] = prepared["publication_year"].astype(int)
    prepared["local_count"] = prepared["local_count"].round().astype("int64")
    duplicate_mask = prepared.duplicated(["subfield_id", "publication_year"], keep=False)
    if duplicate_mask.any():
        examples = prepared.loc[
            duplicate_mask, ["subfield_id", "publication_year"]
        ].drop_duplicates().head(10)
        raise ValueError(
            "Local count table must be unique by subfield_id and publication_year; "
            f"examples: {examples.to_dict('records')}"
        )
    return prepared[["subfield_id", "publication_year", "local_count"]]


def load_growth_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["subfield_id", "subfield_label_unique"])
    targets = read_table(path)
    if "subfield_id" not in targets.columns:
        return pd.DataFrame(columns=["subfield_id", "subfield_label_unique"])
    labels = targets.copy()
    labels["subfield_id"] = labels["subfield_id"].map(short_openalex_id)
    if "subfield_label_unique" not in labels.columns:
        display = (
            labels["subfield_display_name"]
            if "subfield_display_name" in labels.columns
            else labels["subfield_id"]
        )
        labels["subfield_label_unique"] = labels["subfield_id"].astype(str) + " | " + display.astype(str)
    return labels[["subfield_id", "subfield_label_unique"]].drop_duplicates(
        "subfield_id"
    )


def openalex_count_for_filter(client: OpenAlexClient, query_filter: str) -> int:
    return client.get_count("works", query_filter)


def validate_count_pairs(
    *,
    local_counts: pd.DataFrame,
    labels: pd.DataFrame,
    client: Any,
    subfield_ids: list[str],
    years: list[int],
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    local_lookup = {
        (str(row.subfield_id), int(row.publication_year)): int(row.local_count)
        for row in local_counts.itertuples(index=False)
    }
    label_lookup = {
        str(row.subfield_id): str(row.subfield_label_unique)
        for row in labels.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    for subfield_id in subfield_ids:
        short_id = short_openalex_id(subfield_id)
        if not short_id:
            raise ValueError(f"Invalid subfield id: {subfield_id!r}")
        for year in years:
            query_filter = query_filter_for_subfield_year(short_id, int(year))
            local_count = local_lookup.get((short_id, int(year)))
            api_count: int | None = None
            status = "matched"
            error = ""
            try:
                api_count = openalex_count_for_filter(client, query_filter)
            except Exception as exc:  # pragma: no cover - exercised through CLI behavior.
                status = "api_failed"
                error = str(exc)

            absolute_difference: int | None = None
            relative_difference: float | None = None
            matches_exactly = False
            if status != "api_failed":
                if local_count is None:
                    status = "missing_local_count"
                else:
                    absolute_difference = int(local_count - int(api_count))
                    denominator = max(abs(int(api_count)), 1)
                    relative_difference = float(absolute_difference / denominator)
                    matches_exactly = absolute_difference == 0
                    status = "matched" if matches_exactly else "mismatch"

            rows.append(
                {
                    "subfield_id": short_id,
                    "subfield_label_unique": label_lookup.get(short_id, f"{short_id} | Unknown"),
                    "publication_year": int(year),
                    "local_count": local_count,
                    "openalex_api_count": api_count,
                    "absolute_difference": absolute_difference,
                    "relative_difference": relative_difference,
                    "matches_exactly": matches_exactly,
                    "status": status,
                    "query_filter": query_filter,
                    "checked_at": checked_at_iso(),
                    "error_message": error,
                }
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame) -> dict[str, Any]:
    status_counts = results["status"].value_counts().to_dict()
    comparable = results[results["absolute_difference"].notna()].copy()
    mismatches = results[results["status"].isin(["mismatch", "missing_local_count", "api_failed"])]
    return {
        "n_checks": int(len(results)),
        "n_matched": int(status_counts.get("matched", 0)),
        "n_mismatch": int(status_counts.get("mismatch", 0)),
        "n_api_failed": int(status_counts.get("api_failed", 0)),
        "n_missing_local_count": int(status_counts.get("missing_local_count", 0)),
        "max_absolute_difference": (
            int(comparable["absolute_difference"].abs().max())
            if len(comparable)
            else None
        ),
        "max_relative_difference": (
            float(comparable["relative_difference"].abs().max())
            if len(comparable)
            else None
        ),
        "suspicious_cases": mismatches[
            [
                "subfield_id",
                "subfield_label_unique",
                "publication_year",
                "local_count",
                "openalex_api_count",
                "absolute_difference",
                "relative_difference",
                "status",
                "error_message",
            ]
        ].to_dict("records"),
        "notes": [
            "Direct API checks use the same article/preprint, non-retracted, non-paratext filters as Stage 01.",
            "The 2025 checks are included, but OpenAlex indexing lag can still affect interpretation.",
            "This validation script reads Stage 13 outputs for labels only and does not modify growth targets.",
        ],
    }


def print_report(results: pd.DataFrame, summary: dict[str, Any]) -> None:
    columns = [
        "subfield_id",
        "publication_year",
        "local_count",
        "openalex_api_count",
        "absolute_difference",
        "relative_difference",
        "status",
    ]
    print("\nGrowth count extremes validation")
    print(results[columns].to_string(index=False))
    print(
        "\nSummary: "
        f"{summary['n_matched']} matched, "
        f"{summary['n_mismatch']} mismatches, "
        f"{summary['n_missing_local_count']} missing local counts, "
        f"{summary['n_api_failed']} API failures"
    )
    if summary["n_api_failed"]:
        print("API failures are recorded in the CSV/JSON error_message fields.")
    if summary["n_mismatch"]:
        print("Mismatches are recorded in suspicious_cases in the JSON summary.")


def main() -> None:
    args = parse_args()
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be non-negative")
    years = sorted(set(int(year) for year in args.years))
    subfield_ids = (
        [short_openalex_id(value) for value in args.subfield_id]
        if args.subfield_id
        else DEFAULT_SUBFIELD_IDS
    )
    subfield_ids = [str(value) for value in subfield_ids if value]
    if not subfield_ids:
        raise ValueError("No valid subfield ids were provided")

    counts_path = resolve_path(args.counts_path)
    growth_targets_path = resolve_path(args.growth_targets_path)
    output_dir = resolve_path(args.output_dir)
    csv_path = output_dir / "growth_count_extremes_validation.csv"
    json_path = output_dir / "growth_count_extremes_validation.json"
    ensure_outputs_do_not_exist([csv_path, json_path], args.overwrite)

    if not counts_path.exists():
        raise FileNotFoundError(f"Missing local counts table: {display_path(counts_path, ROOT)}")

    local_counts = prepare_local_counts(read_table(counts_path))
    labels = load_growth_labels(growth_targets_path)
    client = OpenAlexClient.from_config(load_config())
    results = validate_count_pairs(
        local_counts=local_counts,
        labels=labels,
        client=client,
        subfield_ids=subfield_ids,
        years=years,
        sleep_seconds=args.sleep_seconds,
    )
    summary = build_summary(results)
    write_table(results, csv_path)
    write_json(json_path, summary)
    print_report(results, summary)
    print(f"\nWrote {display_path(csv_path, ROOT)}")
    print(f"Wrote {display_path(json_path, ROOT)}")


if __name__ == "__main__":
    main()
