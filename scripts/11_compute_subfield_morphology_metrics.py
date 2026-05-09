from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morphology_metrics import (
    CONTROL_COLUMNS,
    CORE_METRIC_COLUMNS_V2,
    DIAGNOSTIC_METRIC_COLUMNS,
    METRIC_COLUMNS,
    compute_subfield_metric_row,
    failed_metric_row,
    metric_dictionary_frame,
    metric_row_frame,
)
from src.subfield_labels import add_subfield_label_columns, duplicate_subfield_names_report
from src.storage import load_parquet, save_parquet


REQUIRED_MANIFEST_COLUMNS = {
    "subfield_id",
    "subfield_name",
    "status",
    "coordinate_path",
    "year_min",
    "year_max",
}


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Compute projected morphology metrics per completed subfield UMAP."
    )
    parser.add_argument(
        "--manifest-path",
        default="outputs/maps/per_subfield_umap/per_subfield_umap_manifest.parquet",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/processed/subfield_morphology_metrics.parquet",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/subfield_morphology_metrics.csv",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/metrics/subfield_morphology_metrics_summary.json",
    )
    parser.add_argument(
        "--dictionary-path",
        default="outputs/metrics/subfield_morphology_metrics_dictionary.csv",
    )
    parser.add_argument(
        "--duplicate-report-path",
        default="outputs/metrics/duplicate_subfield_names_report.csv",
    )
    parser.add_argument("--limit-subfields", type=int, default=None)
    parser.add_argument("--subfield-id", default=None)
    parser.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="Override the manifest year_min when filtering coordinate rows.",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="Override the manifest year_max when filtering coordinate rows.",
    )
    parser.add_argument("--grid-size", type=int, default=160)
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--mst-max-points", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_outputs_do_not_exist(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing metric outputs without --overwrite: {formatted}"
        )


def validate_manifest(manifest: pd.DataFrame) -> None:
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(
            "per-subfield UMAP manifest is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )


def select_completed_manifest_rows(
    manifest: pd.DataFrame,
    *,
    subfield_id: str | None,
    limit_subfields: int | None,
) -> pd.DataFrame:
    validate_manifest(manifest)
    completed = manifest.loc[manifest["status"] == "completed"].copy()
    completed["_subfield_sort_key"] = completed["subfield_id"].astype(str)
    completed = completed.sort_values("_subfield_sort_key", kind="mergesort").drop(
        columns="_subfield_sort_key"
    )

    if subfield_id is not None:
        completed = completed[
            completed["subfield_id"].astype(str) == str(subfield_id)
        ].copy()
        if completed.empty:
            raise ValueError(
                f"No completed manifest row matched --subfield-id {subfield_id}"
            )

    if limit_subfields is not None:
        if limit_subfields <= 0:
            raise ValueError("limit_subfields must be positive when provided")
        completed = completed.head(limit_subfields).copy()

    if completed.empty:
        raise ValueError("No completed subfield UMAP rows are available to process")
    return completed.reset_index(drop=True)


def resolve_coordinate_path(path_value: object) -> Path:
    if path_value is None or pd.isna(path_value) or str(path_value).strip() == "":
        raise ValueError("manifest row has an empty coordinate_path")
    return resolve_path(str(path_value))


def main() -> None:
    args = parse_args()
    if args.grid_size < 10:
        raise ValueError("grid_size must be at least 10")
    if args.k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    if args.mst_max_points < 2:
        raise ValueError("mst_max_points must be at least 2")

    manifest_path = resolve_path(args.manifest_path)
    output_parquet = resolve_path(args.output_parquet)
    output_csv = resolve_path(args.output_csv)
    summary_path = resolve_path(args.summary_path)
    dictionary_path = resolve_path(args.dictionary_path)
    duplicate_report_path = resolve_path(args.duplicate_report_path)
    ensure_outputs_do_not_exist(
        [output_parquet, output_csv, summary_path, dictionary_path, duplicate_report_path],
        args.overwrite,
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing per-subfield UMAP manifest: {display_path(manifest_path)}. "
            "Run scripts/10_build_per_subfield_umap_maps.py first."
        )

    print(f"Loading manifest: {display_path(manifest_path)}")
    manifest = load_parquet(manifest_path)
    validate_manifest(manifest)
    attempted = select_completed_manifest_rows(
        manifest,
        subfield_id=args.subfield_id,
        limit_subfields=args.limit_subfields,
    )

    rows: list[dict[str, Any]] = []
    total = len(attempted)
    print(f"Computing morphology metrics for {total} completed subfield maps")

    for position, manifest_row in attempted.iterrows():
        subfield_id = manifest_row["subfield_id"]
        subfield_name = manifest_row["subfield_name"]
        percent = ((position + 1) / total) * 100
        prefix = f"[{position + 1}/{total} | {percent:.1f}%] {subfield_id} - {subfield_name}"
        coordinate_path = None
        year_min = int(args.year_min if args.year_min is not None else manifest_row["year_min"])
        year_max = int(args.year_max if args.year_max is not None else manifest_row["year_max"])

        try:
            coordinate_path = resolve_coordinate_path(manifest_row["coordinate_path"])
            if not coordinate_path.exists():
                raise FileNotFoundError(
                    f"coordinate parquet does not exist: {display_path(coordinate_path)}"
                )
            coordinate_frame = load_parquet(coordinate_path)
            row = compute_subfield_metric_row(
                coordinate_frame,
                coordinate_path=display_path(coordinate_path),
                year_min=year_min,
                year_max=year_max,
                grid_size=args.grid_size,
                k_neighbors=args.k_neighbors,
                mst_max_points=args.mst_max_points,
                random_state=args.random_state,
            )
            rows.append(row)
            print(f"{prefix}: {row['metric_status']}")
        except Exception as exc:
            failed_path = (
                display_path(coordinate_path)
                if coordinate_path is not None
                else str(manifest_row.get("coordinate_path", ""))
            )
            rows.append(
                failed_metric_row(
                    manifest_row=manifest_row,
                    coordinate_path=failed_path,
                    year_min=year_min,
                    year_max=year_max,
                    grid_size=args.grid_size,
                    k_neighbors=args.k_neighbors,
                    error_message=str(exc),
                )
            )
            print(f"{prefix}: failed ({exc})")

    metrics = add_subfield_label_columns(metric_row_frame(rows))
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(metrics, output_parquet)
    metrics.to_csv(output_csv, index=False)

    dictionary = metric_dictionary_frame()
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(dictionary_path, index=False)
    duplicate_report = duplicate_subfield_names_report(metrics)
    duplicate_report_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_report.to_csv(duplicate_report_path, index=False)

    status_counts = metrics["metric_status"].value_counts().to_dict()
    failed = metrics.loc[
        metrics["metric_status"] == "failed",
        ["subfield_id", "subfield_display_name", "metric_error_message"],
    ].to_dict("records")
    warnings = metrics.loc[
        metrics["metric_warning_message"].fillna("") != "",
        ["subfield_id", "subfield_display_name", "metric_warning_message"],
    ].to_dict("records")
    n_failed = int(status_counts.get("failed", 0))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": display_path(manifest_path),
        "output_parquet": display_path(output_parquet),
        "output_csv": display_path(output_csv),
        "dictionary_path": display_path(dictionary_path),
        "duplicate_report_path": display_path(duplicate_report_path),
        "n_subfields_in_manifest": int(len(manifest)),
        "n_subfields_attempted": int(len(metrics)),
        "n_completed": int(len(metrics) - n_failed),
        "n_failed": n_failed,
        "metric_columns": METRIC_COLUMNS,
        "core_metric_columns_v2": CORE_METRIC_COLUMNS_V2,
        "diagnostic_metric_columns": DIAGNOSTIC_METRIC_COLUMNS,
        "control_columns": CONTROL_COLUMNS,
        "n_duplicate_display_name_rows": int(len(duplicate_report)),
        "n_duplicated_display_names": int(
            duplicate_report["subfield_display_name"].nunique()
        )
        if len(duplicate_report)
        else 0,
        "grid_size": int(args.grid_size),
        "k_neighbors": int(args.k_neighbors),
        "mst_max_points": int(args.mst_max_points),
        "random_state": int(args.random_state),
        "year_override": {
            "year_min": args.year_min,
            "year_max": args.year_max,
        },
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "failed_subfields": failed,
        "warnings": warnings,
    }
    write_json(summary_path, summary)

    print(f"Wrote {display_path(output_parquet)}")
    print(f"Wrote {display_path(output_csv)}")
    print(f"Wrote {display_path(summary_path)}")
    print(f"Wrote {display_path(dictionary_path)}")
    print(f"Wrote {display_path(duplicate_report_path)}")
    print(
        "Done: "
        f"{summary['n_completed']} completed, "
        f"{summary['n_failed']} failed"
    )


if __name__ == "__main__":
    main()
