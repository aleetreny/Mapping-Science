from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.growth_targets import (
    DEFAULT_COUNT_COLUMN,
    PREFERRED_SUBFIELD_PATHS,
    GrowthWindowConfig,
    build_balanced_year_panel,
    build_domain_growth_summary,
    build_growth_rankings,
    build_summary_payload,
    compute_growth_targets,
    display_path,
    fetch_openalex_subfield_year_counts,
    prepare_main_analysis_subfields,
    read_table,
    write_growth_figures,
    write_table,
)
from src.openalex import OpenAlexClient
from src.storage import ensure_dirs


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Build annualized subfield growth targets for main-analysis subfields."
    )
    parser.add_argument(
        "--subfields-path",
        default=None,
        help=(
            "Optional main-analysis subfield table. If omitted, discover in order: "
            "data/processed/subfield_morphology_metrics.parquet, "
            "outputs/metrics/morphology_analysis/tables/curated_model_features.parquet, "
            "data/processed/analysis_embedding_index.parquet."
        ),
    )
    parser.add_argument(
        "--counts-path",
        default=None,
        help=(
            "Optional full subfield-year count table. If omitted, the script checks "
            "data/interim/subfield_year_counts.parquet, a DuckDB subfield_year_counts "
            "table, then the raw OpenAlex cache."
        ),
    )
    parser.add_argument("--count-column", default=DEFAULT_COUNT_COLUMN)
    parser.add_argument(
        "--output-path",
        default="data/processed/subfield_growth_targets.parquet",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/subfield_growth_targets.csv",
    )
    parser.add_argument(
        "--yearly-panel-path",
        default="outputs/growth/subfield_year_counts_panel.parquet",
    )
    parser.add_argument(
        "--yearly-panel-csv",
        default="outputs/growth/subfield_year_counts_panel.csv",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/growth/subfield_growth_targets_summary.json",
    )
    parser.add_argument(
        "--rankings-path",
        default="outputs/growth/growth_rankings.csv",
    )
    parser.add_argument(
        "--domain-summary-path",
        default="outputs/growth/domain_growth_summary.csv",
    )
    parser.add_argument("--figures-dir", default="outputs/growth/figures")
    parser.add_argument("--input-year-min", type=int, default=2010)
    parser.add_argument("--input-year-max", type=int, default=2019)
    parser.add_argument("--target-year-min", type=int, default=2020)
    parser.add_argument("--target-year-max", type=int, default=2025)
    parser.add_argument(
        "--expected-subfields",
        type=int,
        default=240,
        help="Expected output row count. Use 0 to skip this check for synthetic runs.",
    )
    parser.add_argument(
        "--missing-counts-mean-zero",
        action="store_true",
        help="Treat missing subfield-year count rows as zeros for sparse count sources.",
    )
    parser.add_argument(
        "--fetch-counts-from-openalex",
        action="store_true",
        help="Fetch sparse OpenAlex group-by count rows if no local count table exists.",
    )
    parser.add_argument(
        "--raw-counts-cache",
        default="data/interim/openalex_subfield_year_counts_2010_2025.parquet",
    )
    parser.add_argument("--mailto", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--per-domain-top-n", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    path = ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_outputs_do_not_exist(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path, ROOT) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing growth outputs without --overwrite: "
            + formatted
        )


def file_modified_at_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def discover_subfields_path(path_arg: str | None) -> Path:
    if path_arg:
        path = resolve_path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Missing subfield table: {display_path(path, ROOT)}")
        return path

    for relative in PREFERRED_SUBFIELD_PATHS:
        path = resolve_path(relative)
        if path.exists():
            return path
    searched = ", ".join(str(path).replace("\\", "/") for path in PREFERRED_SUBFIELD_PATHS)
    raise FileNotFoundError(
        "No main-analysis subfield table found. Searched: " + searched
    )


def duckdb_counts_available(path: Path) -> bool:
    if not path.exists():
        return False
    con = duckdb.connect(str(path), read_only=True)
    try:
        rows = con.execute(
            "select count(*) from information_schema.tables where table_name = 'subfield_year_counts'"
        ).fetchone()
        return bool(rows and rows[0])
    finally:
        con.close()


def load_duckdb_counts(path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute("select * from subfield_year_counts").fetch_df()
    finally:
        con.close()


def discover_or_fetch_counts(
    args: argparse.Namespace,
    *,
    config: GrowthWindowConfig,
    project_config: dict[str, Any],
) -> tuple[pd.DataFrame, str, str | None, str, list[str], list[str]]:
    warnings: list[str] = []
    assumptions: list[str] = []

    if args.counts_path:
        counts_path = resolve_path(args.counts_path)
        if not counts_path.exists():
            raise FileNotFoundError(f"Missing count table: {display_path(counts_path, ROOT)}")
        counts = read_table(counts_path)
        return (
            counts,
            f"{display_path(counts_path, ROOT)}:{args.count_column}",
            file_modified_at_utc(counts_path),
            display_path(counts_path, ROOT),
            warnings,
            assumptions,
        )

    default_counts_path = ROOT / "data/interim/subfield_year_counts.parquet"
    if default_counts_path.exists():
        counts = read_table(default_counts_path)
        assumptions.append(
            "data/interim/subfield_year_counts.parquet is produced by scripts/01_build_counts.py from full OpenAlex grouped counts."
        )
        return (
            counts,
            f"{display_path(default_counts_path, ROOT)}:{args.count_column}",
            file_modified_at_utc(default_counts_path),
            display_path(default_counts_path, ROOT),
            warnings,
            assumptions,
        )

    duckdb_path = resolve_path(
        project_config.get("storage", {}).get("duckdb_path", "warehouse/tfm_openalex.duckdb")
    )
    if duckdb_counts_available(duckdb_path):
        counts = load_duckdb_counts(duckdb_path)
        assumptions.append(
            "DuckDB table subfield_year_counts is produced by scripts/01_build_counts.py from full OpenAlex grouped counts."
        )
        return (
            counts,
            f"{display_path(duckdb_path, ROOT)}:subfield_year_counts:{args.count_column}",
            file_modified_at_utc(duckdb_path),
            f"{display_path(duckdb_path, ROOT)}::subfield_year_counts",
            warnings,
            assumptions,
        )

    cache_path = resolve_path(args.raw_counts_cache)
    if cache_path.exists():
        counts = read_table(cache_path)
        assumptions.append(
            "Raw OpenAlex group-by cache is sparse, so missing subfield-year rows are interpreted as zero."
        )
        return (
            counts,
            f"{display_path(cache_path, ROOT)}:works_count",
            file_modified_at_utc(cache_path),
            display_path(cache_path, ROOT),
            warnings,
            assumptions,
        )

    if not args.fetch_counts_from_openalex:
        raise FileNotFoundError(
            "No local full count table found. Run scripts/01_build_counts.py, pass "
            "--counts-path, or pass --fetch-counts-from-openalex to create the raw count cache."
        )

    client = OpenAlexClient.from_config(project_config)
    if args.mailto:
        client.email = args.mailto
    elif not client.email:
        warnings.append(
            "OPENALEX_EMAIL/mailto is not set; OpenAlex requests will not include a polite email."
        )
    per_page = int(project_config.get("openalex", {}).get("per_page", 200))
    counts = fetch_openalex_subfield_year_counts(
        client=client,
        years=config.all_years,
        per_page=per_page,
        sleep_seconds=args.sleep_seconds,
    )
    if counts.empty:
        raise RuntimeError("OpenAlex count fetch returned no rows")
    write_table(counts, cache_path)
    assumptions.append(
        "Fetched OpenAlex group-by rows are sparse, so missing subfield-year rows are interpreted as zero."
    )
    return (
        counts,
        f"{display_path(cache_path, ROOT)}:works_count",
        counts["retrieved_at"].dropna().astype(str).max()
        if "retrieved_at" in counts.columns
        else None,
        display_path(cache_path, ROOT),
        warnings,
        assumptions,
    )


def add_coverage_warnings(
    warnings: list[str],
    *,
    retrieved_at: str | None,
    target_year_max: int,
) -> None:
    if target_year_max >= datetime.now(timezone.utc).year:
        warnings.append(
            f"Target window ends in {target_year_max}, which is not a fully closed historical year at run time."
        )
    if retrieved_at:
        try:
            retrieved_dt = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
        except ValueError:
            retrieved_dt = None
        if retrieved_dt is not None:
            end_of_target = datetime(target_year_max, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            if retrieved_dt < end_of_target:
                warnings.append(
                    "Count source appears to have been retrieved before the end of the target year; 2025 coverage may be partial."
                )
    warnings.append(
        "OpenAlex 2025 records may still reflect indexing lag; treat the target as an OpenAlex snapshot."
    )


def main() -> None:
    args = parse_args()
    project_config = load_config()
    ensure_dirs(project_config)

    window_config = GrowthWindowConfig(
        input_year_min=args.input_year_min,
        input_year_max=args.input_year_max,
        target_year_min=args.target_year_min,
        target_year_max=args.target_year_max,
    )
    window_config.validate()

    output_path = resolve_path(args.output_path)
    output_csv = resolve_path(args.output_csv)
    yearly_panel_path = resolve_path(args.yearly_panel_path)
    yearly_panel_csv = resolve_path(args.yearly_panel_csv)
    summary_path = resolve_path(args.summary_path)
    rankings_path = resolve_path(args.rankings_path)
    domain_summary_path = resolve_path(args.domain_summary_path)
    figures_dir = resolve_path(args.figures_dir)
    ensure_outputs_do_not_exist(
        [
            output_path,
            output_csv,
            yearly_panel_path,
            yearly_panel_csv,
            summary_path,
            rankings_path,
            domain_summary_path,
        ],
        args.overwrite,
    )

    subfields_path = discover_subfields_path(args.subfields_path)
    print(f"Loading main-analysis subfields: {display_path(subfields_path, ROOT)}")
    subfields = prepare_main_analysis_subfields(read_table(subfields_path))
    if args.expected_subfields > 0 and len(subfields) != args.expected_subfields:
        raise ValueError(
            f"Expected {args.expected_subfields} main-analysis subfields, found {len(subfields)}"
        )

    counts, count_source, retrieved_at, counts_input_path, source_warnings, source_assumptions = (
        discover_or_fetch_counts(
            args,
            config=window_config,
            project_config=project_config,
        )
    )
    warnings = list(source_warnings)
    assumptions = list(source_assumptions)
    add_coverage_warnings(
        warnings,
        retrieved_at=retrieved_at,
        target_year_max=window_config.target_year_max,
    )

    sparse_source = (
        args.missing_counts_mean_zero
        or args.fetch_counts_from_openalex
        or Path(counts_input_path).name.startswith("openalex_subfield_year_counts")
    )
    missing_policy = "zero" if sparse_source else "fail"

    effective_count_column = args.count_column
    if (
        args.count_column == DEFAULT_COUNT_COLUMN
        and DEFAULT_COUNT_COLUMN not in counts.columns
        and "works_count" in counts.columns
    ):
        effective_count_column = "works_count"

    print(f"Using full count source: {count_source}")
    panel, panel_warnings, panel_assumptions = build_balanced_year_panel(
        subfields,
        counts,
        config=window_config,
        count_column=effective_count_column,
        count_source=count_source,
        retrieved_at=retrieved_at,
        missing_count_policy=missing_policy,
    )
    warnings.extend(panel_warnings)
    assumptions.extend(panel_assumptions)

    targets = compute_growth_targets(subfields, panel, config=window_config)
    if args.expected_subfields > 0 and len(targets) != args.expected_subfields:
        raise ValueError(
            f"Expected {args.expected_subfields} target rows, found {len(targets)}"
        )

    rankings = build_growth_rankings(
        targets,
        top_n=args.top_n,
        per_domain_n=args.per_domain_top_n,
    )
    domain_summary = build_domain_growth_summary(targets)

    write_table(targets, output_path)
    write_table(targets, output_csv)
    write_table(panel, yearly_panel_path)
    write_table(panel, yearly_panel_csv)
    write_table(rankings, rankings_path)
    write_table(domain_summary, domain_summary_path)
    figure_paths = write_growth_figures(
        targets=targets,
        panel=panel,
        figures_dir=figures_dir,
        dpi=args.dpi,
    )

    input_paths = {
        "subfields_path": display_path(subfields_path, ROOT),
        "counts_path": counts_input_path,
    }
    output_paths = {
        "targets_parquet": display_path(output_path, ROOT),
        "targets_csv": display_path(output_csv, ROOT),
        "yearly_panel_parquet": display_path(yearly_panel_path, ROOT),
        "yearly_panel_csv": display_path(yearly_panel_csv, ROOT),
        "summary_json": display_path(summary_path, ROOT),
        "rankings_csv": display_path(rankings_path, ROOT),
        "domain_summary_csv": display_path(domain_summary_path, ROOT),
        "figures_dir": display_path(figures_dir, ROOT),
    }
    output_paths.update(
        {
            f"figure_{name}": display_path(path, ROOT)
            for name, path in figure_paths.items()
        }
    )
    summary = build_summary_payload(
        targets=targets,
        panel=panel,
        config=window_config,
        input_paths=input_paths,
        output_paths=output_paths,
        count_source=count_source,
        warnings=warnings,
        assumptions=assumptions,
        n_subfields_expected=args.expected_subfields if args.expected_subfields > 0 else len(subfields),
    )
    write_json(summary_path, summary)

    print(f"Wrote {display_path(output_path, ROOT)}")
    print(f"Wrote {display_path(output_csv, ROOT)}")
    print(f"Wrote {display_path(yearly_panel_path, ROOT)}")
    print(f"Wrote {display_path(summary_path, ROOT)}")
    print(f"Wrote figures under {display_path(figures_dir, ROOT)}")
    print(f"Subfields: {len(targets)}")
    print(f"Global annualized log-growth median: {summary['global_growth_median']:.6f}")
    print(
        "Above/below global median: "
        f"{summary['n_growth_above_median']} / {summary['n_growth_below_or_equal_median']}"
    )
    print(
        "Above/below domain median: "
        f"{summary['n_growth_above_domain_median']} / "
        f"{summary['n_growth_below_or_equal_domain_median']}"
    )


if __name__ == "__main__":
    main()
