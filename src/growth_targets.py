from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.openalex import OpenAlexClient, build_count_query_params, short_openalex_id
from src.subfield_labels import add_subfield_label_columns


CONTEXT_COLUMNS = [
    "subfield_id",
    "subfield_display_name",
    "subfield_label_unique",
    "subfield_label_short",
    "subfield_display_name_is_duplicated",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
]

PREFERRED_SUBFIELD_PATHS = [
    Path("data/processed/subfield_morphology_metrics.parquet"),
    Path("outputs/metrics/morphology_analysis/tables/curated_model_features.parquet"),
    Path("data/processed/analysis_embedding_index.parquet"),
]

DEFAULT_COUNT_COLUMN = "n_works_article_preprint"
COUNT_COLUMN_CANDIDATES = [
    DEFAULT_COUNT_COLUMN,
    "works_count",
    "n_works_total",
    "n_works_article_preprint_en",
    "n_works_article_preprint_en_with_abstract",
]

SUMMARY_QUANTILES = {
    "p05": 0.05,
    "p25": 0.25,
    "median": 0.50,
    "p75": 0.75,
    "p95": 0.95,
}


@dataclass(frozen=True)
class GrowthWindowConfig:
    input_year_min: int = 2010
    input_year_max: int = 2019
    target_year_min: int = 2020
    target_year_max: int = 2025

    @property
    def input_years(self) -> int:
        return self.input_year_max - self.input_year_min + 1

    @property
    def target_years(self) -> int:
        return self.target_year_max - self.target_year_min + 1

    @property
    def all_years(self) -> list[int]:
        return list(range(self.input_year_min, self.target_year_max + 1))

    def validate(self) -> None:
        if self.input_year_min > self.input_year_max:
            raise ValueError("input_year_min must be <= input_year_max")
        if self.target_year_min > self.target_year_max:
            raise ValueError("target_year_min must be <= target_year_max")
        if self.input_year_max >= self.target_year_min:
            raise ValueError("input and target windows must not overlap")
        if self.input_years <= 0 or self.target_years <= 0:
            raise ValueError("window lengths must be positive")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: str | Path, root: Path | None = None) -> str:
    path = Path(path)
    if root is not None:
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            pass
    return str(path).replace("\\", "/")


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path)
    raise ValueError(f"Unsupported table format for {table_path}: {suffix}")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(table_path, index=False)
        return
    if suffix == ".csv":
        frame.to_csv(table_path, index=False)
        return
    raise ValueError(f"Unsupported table format for {table_path}: {suffix}")


def normalize_subfield_id(value: Any) -> str | None:
    return short_openalex_id(value)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def prepare_main_analysis_subfields(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one labelled row per main-analysis subfield."""
    if "subfield_id" not in frame.columns:
        raise ValueError("subfield table must contain subfield_id")

    prepared = frame.copy()
    prepared["subfield_id"] = prepared["subfield_id"].map(normalize_subfield_id)
    prepared = prepared[prepared["subfield_id"].notna()].copy()

    if "main_analysis_eligible_2500" in prepared.columns:
        prepared = prepared[_coerce_bool_series(prepared["main_analysis_eligible_2500"])].copy()

    if "subfield_name" in prepared.columns and "subfield_display_name" not in prepared.columns:
        prepared = prepared.rename(columns={"subfield_name": "subfield_display_name"})

    if "subfield_display_name" not in prepared.columns:
        prepared["subfield_display_name"] = prepared["subfield_id"].map(
            lambda value: f"Subfield {value}"
        )

    for column in ["field_id", "field_display_name", "domain_id", "domain_display_name"]:
        if column not in prepared.columns:
            prepared[column] = pd.NA

    sort_cols = [column for column in ["subfield_id", "publication_year"] if column in prepared.columns]
    if sort_cols:
        prepared = prepared.sort_values(sort_cols, kind="mergesort")

    duplicated = prepared["subfield_id"].duplicated(keep=False)
    if duplicated.any():
        context_cols = [
            "subfield_id",
            "subfield_display_name",
            "field_id",
            "field_display_name",
            "domain_id",
            "domain_display_name",
        ]
        prepared = prepared[context_cols].drop_duplicates("subfield_id", keep="first")

    prepared = add_subfield_label_columns(prepared)
    for column in CONTEXT_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA

    result = prepared[CONTEXT_COLUMNS].drop_duplicates("subfield_id", keep="first")
    result = result.sort_values("subfield_id", kind="mergesort").reset_index(drop=True)
    if result["subfield_id"].duplicated().any():
        raise ValueError("subfield table must have exactly one row per subfield_id")
    return result


def infer_count_column(counts: pd.DataFrame, requested: str | None = DEFAULT_COUNT_COLUMN) -> str:
    if requested and requested.lower() != "auto":
        if requested not in counts.columns:
            raise ValueError(
                f"count table does not contain requested count column {requested!r}"
            )
        return requested

    for column in COUNT_COLUMN_CANDIDATES:
        if column in counts.columns:
            return column
    raise ValueError(
        "count table must contain one count column; checked "
        + ", ".join(COUNT_COLUMN_CANDIDATES)
    )


def prepare_counts(
    counts: pd.DataFrame,
    *,
    count_column: str | None = DEFAULT_COUNT_COLUMN,
) -> tuple[pd.DataFrame, str]:
    if "subfield_id" not in counts.columns:
        raise ValueError("count table must contain subfield_id")
    if "publication_year" not in counts.columns:
        raise ValueError("count table must contain publication_year")

    selected_count_column = infer_count_column(counts, count_column)
    prepared = counts.copy()
    prepared["subfield_id"] = prepared["subfield_id"].map(normalize_subfield_id)
    prepared["publication_year"] = pd.to_numeric(
        prepared["publication_year"], errors="coerce"
    ).astype("Int64")
    prepared["works_count"] = pd.to_numeric(
        prepared[selected_count_column], errors="coerce"
    )

    prepared = prepared[
        prepared["subfield_id"].notna()
        & prepared["publication_year"].notna()
        & prepared["works_count"].notna()
    ].copy()
    prepared["publication_year"] = prepared["publication_year"].astype(int)
    prepared["works_count"] = prepared["works_count"].clip(lower=0).round().astype("int64")

    duplicate_mask = prepared.duplicated(["subfield_id", "publication_year"], keep=False)
    if duplicate_mask.any():
        examples = prepared.loc[
            duplicate_mask, ["subfield_id", "publication_year"]
        ].drop_duplicates().head(10)
        raise ValueError(
            "count table must be unique by subfield_id and publication_year; "
            f"examples: {examples.to_dict('records')}"
        )

    keep_cols = ["subfield_id", "publication_year", "works_count"]
    for column in ["count_source", "retrieved_at"]:
        if column in prepared.columns:
            keep_cols.append(column)
    return prepared[keep_cols].reset_index(drop=True), selected_count_column


def build_balanced_year_panel(
    subfields: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    config: GrowthWindowConfig,
    count_column: str | None = DEFAULT_COUNT_COLUMN,
    count_source: str,
    retrieved_at: str | None = None,
    missing_count_policy: str = "fail",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a complete subfield-year panel for all requested years."""
    config.validate()
    if missing_count_policy not in {"fail", "zero"}:
        raise ValueError("missing_count_policy must be 'fail' or 'zero'")

    subfield_ids = subfields["subfield_id"].astype(str).tolist()
    prepared_counts, selected_count_column = prepare_counts(
        counts,
        count_column=count_column,
    )
    prepared_counts = prepared_counts[
        prepared_counts["publication_year"].isin(config.all_years)
    ].copy()
    prepared_counts = prepared_counts[
        prepared_counts["subfield_id"].isin(set(subfield_ids))
    ].copy()

    grid = pd.MultiIndex.from_product(
        [subfield_ids, config.all_years],
        names=["subfield_id", "publication_year"],
    ).to_frame(index=False)
    panel = grid.merge(prepared_counts, on=["subfield_id", "publication_year"], how="left")
    panel["count_row_present"] = panel["works_count"].notna()

    missing_rows = panel.loc[
        ~panel["count_row_present"], ["subfield_id", "publication_year"]
    ].copy()
    missing_subfields = sorted(missing_rows["subfield_id"].astype(str).unique().tolist())
    warnings: list[str] = []
    assumptions: list[str] = [
        f"Growth counts use {selected_count_column} as the yearly works count.",
        "Counts are joined by subfield_id, not by display name.",
    ]

    if len(missing_rows):
        message = (
            f"Missing {len(missing_rows)} subfield-year count rows across "
            f"{len(missing_subfields)} subfields."
        )
        if missing_count_policy == "fail":
            examples = missing_rows.head(10).to_dict("records")
            raise ValueError(
                message
                + " Missing rows are not being treated as zero. "
                + f"Examples: {examples}"
            )
        warnings.append(message + " Filled with zero because the count source is sparse.")
        assumptions.append(
            "Missing subfield-year combinations in the count source are interpreted as zero."
        )

    panel["works_count"] = panel["works_count"].fillna(0).astype("int64")
    panel = panel.merge(subfields[CONTEXT_COLUMNS], on="subfield_id", how="left")
    panel["count_source"] = count_source
    panel["retrieved_at"] = retrieved_at or ""
    ordered = [
        "subfield_id",
        "subfield_display_name",
        "subfield_label_unique",
        "subfield_label_short",
        "subfield_display_name_is_duplicated",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        "publication_year",
        "works_count",
        "count_row_present",
        "count_source",
        "retrieved_at",
    ]
    return panel[ordered].sort_values(
        ["subfield_id", "publication_year"], kind="mergesort"
    ).reset_index(drop=True), warnings, assumptions


def _window_aggregate(
    panel: pd.DataFrame,
    *,
    year_min: int,
    year_max: int,
    count_output: str,
    years_output: str,
) -> pd.DataFrame:
    mask = panel["publication_year"].between(year_min, year_max)
    return (
        panel.loc[mask]
        .groupby("subfield_id", dropna=False)
        .agg(
            **{
                count_output: ("works_count", "sum"),
                years_output: ("count_row_present", "sum"),
            }
        )
        .reset_index()
    )


def compute_growth_targets(
    subfields: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    config: GrowthWindowConfig,
) -> pd.DataFrame:
    config.validate()
    expected_panel_rows = len(subfields) * len(config.all_years)
    if len(panel) != expected_panel_rows:
        raise ValueError(
            f"panel has {len(panel)} rows, expected {expected_panel_rows} "
            "for a balanced subfield-year panel"
        )
    if panel.duplicated(["subfield_id", "publication_year"]).any():
        raise ValueError("panel must be unique by subfield_id and publication_year")

    input_counts = _window_aggregate(
        panel,
        year_min=config.input_year_min,
        year_max=config.input_year_max,
        count_output="papers_2010_2019",
        years_output="n_years_with_counts_2010_2019",
    )
    target_counts = _window_aggregate(
        panel,
        year_min=config.target_year_min,
        year_max=config.target_year_max,
        count_output="papers_2020_2025",
        years_output="n_years_with_counts_2020_2025",
    )

    targets = subfields[CONTEXT_COLUMNS].merge(input_counts, on="subfield_id", how="left")
    targets = targets.merge(target_counts, on="subfield_id", how="left")
    for column in [
        "papers_2010_2019",
        "papers_2020_2025",
        "n_years_with_counts_2010_2019",
        "n_years_with_counts_2020_2025",
    ]:
        targets[column] = targets[column].fillna(0).astype("int64")

    targets["annual_rate_2010_2019"] = targets["papers_2010_2019"] / float(
        config.input_years
    )
    targets["annual_rate_2020_2025"] = targets["papers_2020_2025"] / float(
        config.target_years
    )

    rx = targets["annual_rate_2010_2019"].astype(float)
    ry = targets["annual_rate_2020_2025"].astype(float)
    targets["annualized_growth_abs"] = ry - rx
    targets["annualized_growth_ratio"] = (ry + 1.0) / (rx + 1.0)
    targets["annualized_growth_pct"] = targets["annualized_growth_ratio"] - 1.0
    targets["annualized_log_growth"] = np.log1p(ry) - np.log1p(rx)

    global_median = float(targets["annualized_log_growth"].median())
    targets["global_growth_median"] = global_median
    targets["growth_above_median"] = targets["annualized_log_growth"] > global_median
    targets["growth_rank"] = targets["annualized_log_growth"].rank(
        ascending=False, method="min"
    )
    targets["growth_percentile"] = targets["annualized_log_growth"].rank(
        ascending=True, pct=True, method="average"
    )

    targets["domain_growth_median"] = targets.groupby("domain_id", dropna=False)[
        "annualized_log_growth"
    ].transform("median")
    targets["domain_adjusted_annualized_log_growth"] = (
        targets["annualized_log_growth"] - targets["domain_growth_median"]
    )
    targets["growth_above_domain_median"] = (
        targets["domain_adjusted_annualized_log_growth"] > 0
    )
    targets["domain_growth_rank"] = targets.groupby("domain_id", dropna=False)[
        "annualized_log_growth"
    ].rank(ascending=False, method="min")
    targets["domain_growth_percentile"] = targets.groupby("domain_id", dropna=False)[
        "annualized_log_growth"
    ].rank(ascending=True, pct=True, method="average")

    targets["input_year_min"] = config.input_year_min
    targets["input_year_max"] = config.input_year_max
    targets["target_year_min"] = config.target_year_min
    targets["target_year_max"] = config.target_year_max
    targets["input_years"] = config.input_years
    targets["target_years"] = config.target_years
    targets["has_zero_input_count"] = targets["papers_2010_2019"] == 0
    targets["has_zero_target_count"] = targets["papers_2020_2025"] == 0

    count_sources = (
        panel.groupby("subfield_id", dropna=False)["count_source"]
        .agg(lambda values: "; ".join(sorted(set(str(value) for value in values))))
        .reset_index()
    )
    targets = targets.merge(count_sources, on="subfield_id", how="left")

    ordered = CONTEXT_COLUMNS + [
        "papers_2010_2019",
        "papers_2020_2025",
        "annual_rate_2010_2019",
        "annual_rate_2020_2025",
        "annualized_growth_abs",
        "annualized_growth_ratio",
        "annualized_growth_pct",
        "annualized_log_growth",
        "global_growth_median",
        "growth_above_median",
        "growth_rank",
        "growth_percentile",
        "domain_growth_median",
        "domain_adjusted_annualized_log_growth",
        "growth_above_domain_median",
        "domain_growth_rank",
        "domain_growth_percentile",
        "input_year_min",
        "input_year_max",
        "target_year_min",
        "target_year_max",
        "input_years",
        "target_years",
        "n_years_with_counts_2010_2019",
        "n_years_with_counts_2020_2025",
        "has_zero_input_count",
        "has_zero_target_count",
        "count_source",
    ]
    return targets[ordered].sort_values("subfield_id", kind="mergesort").reset_index(
        drop=True
    )


def summary_stats(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
            "std": None,
        }
    def clean(value: float) -> float | None:
        return float(value) if math.isfinite(float(value)) else None

    stats: dict[str, float | None] = {
        "min": clean(values.min()),
        "mean": clean(values.mean()),
        "max": clean(values.max()),
        "std": clean(values.std()),
    }
    for name, quantile in SUMMARY_QUANTILES.items():
        stats[name] = clean(values.quantile(quantile))
    return {
        "min": stats["min"],
        "p05": stats["p05"],
        "p25": stats["p25"],
        "median": stats["median"],
        "mean": stats["mean"],
        "p75": stats["p75"],
        "p95": stats["p95"],
        "max": stats["max"],
        "std": stats["std"],
    }


def build_domain_growth_summary(targets: pd.DataFrame) -> pd.DataFrame:
    grouped = targets.groupby(["domain_id", "domain_display_name"], dropna=False)
    summary = grouped.agg(
        n_subfields=("subfield_id", "nunique"),
        total_papers_2010_2019=("papers_2010_2019", "sum"),
        total_papers_2020_2025=("papers_2020_2025", "sum"),
        mean_annual_rate_2010_2019=("annual_rate_2010_2019", "mean"),
        mean_annual_rate_2020_2025=("annual_rate_2020_2025", "mean"),
        median_annualized_log_growth=("annualized_log_growth", "median"),
        mean_annualized_log_growth=("annualized_log_growth", "mean"),
        std_annualized_log_growth=("annualized_log_growth", "std"),
        share_growth_above_global_median=("growth_above_median", "mean"),
    )
    return summary.reset_index().sort_values("domain_id", kind="mergesort")


def _ranking_section(
    frame: pd.DataFrame,
    *,
    section: str,
    metric: str,
    ascending: bool,
    n: int,
) -> pd.DataFrame:
    selected = frame.sort_values(
        [metric, "subfield_id"],
        ascending=[ascending, True],
        kind="mergesort",
    ).head(n)
    result = selected[
        CONTEXT_COLUMNS
        + [
            "annualized_log_growth",
            "domain_adjusted_annualized_log_growth",
            "annual_rate_2010_2019",
            "annual_rate_2020_2025",
            "papers_2010_2019",
            "papers_2020_2025",
            "growth_rank",
            "domain_growth_rank",
        ]
    ].copy()
    result.insert(0, "ranking_section", section)
    result.insert(1, "ranking_metric", metric)
    result.insert(2, "rank_within_section", range(1, len(result) + 1))
    return result


def build_growth_rankings(
    targets: pd.DataFrame,
    *,
    top_n: int = 20,
    per_domain_n: int = 5,
) -> pd.DataFrame:
    sections = [
        _ranking_section(
            targets,
            section="global_top_annualized_log_growth",
            metric="annualized_log_growth",
            ascending=False,
            n=top_n,
        ),
        _ranking_section(
            targets,
            section="global_bottom_annualized_log_growth",
            metric="annualized_log_growth",
            ascending=True,
            n=top_n,
        ),
        _ranking_section(
            targets,
            section="global_top_domain_adjusted_annualized_log_growth",
            metric="domain_adjusted_annualized_log_growth",
            ascending=False,
            n=top_n,
        ),
        _ranking_section(
            targets,
            section="global_bottom_domain_adjusted_annualized_log_growth",
            metric="domain_adjusted_annualized_log_growth",
            ascending=True,
            n=top_n,
        ),
    ]
    if per_domain_n > 0:
        for domain_id, domain_rows in targets.groupby("domain_id", dropna=False):
            domain_label = str(domain_id)
            sections.append(
                _ranking_section(
                    domain_rows,
                    section=f"domain_{domain_label}_top_annualized_log_growth",
                    metric="annualized_log_growth",
                    ascending=False,
                    n=per_domain_n,
                )
            )
            sections.append(
                _ranking_section(
                    domain_rows,
                    section=f"domain_{domain_label}_bottom_annualized_log_growth",
                    metric="annualized_log_growth",
                    ascending=True,
                    n=per_domain_n,
                )
            )
    return pd.concat(sections, ignore_index=True)


def build_summary_payload(
    *,
    targets: pd.DataFrame,
    panel: pd.DataFrame,
    config: GrowthWindowConfig,
    input_paths: dict[str, str],
    output_paths: dict[str, str],
    count_source: str,
    warnings: list[str],
    assumptions: list[str],
    n_subfields_expected: int,
) -> dict[str, Any]:
    global_median = float(targets["global_growth_median"].iloc[0])
    global_ties = int((targets["annualized_log_growth"] == global_median).sum())
    domain_ties = int((targets["domain_adjusted_annualized_log_growth"] == 0).sum())
    domain_counts = {
        str(key): int(value)
        for key, value in targets.groupby("domain_id", dropna=False)["subfield_id"]
        .nunique()
        .sort_index()
        .items()
    }
    above_global = int(targets["growth_above_median"].sum())
    above_domain = int(targets["growth_above_domain_median"].sum())
    retrieved_values = sorted(
        {str(value) for value in panel.get("retrieved_at", pd.Series(dtype=str)).dropna()}
    )
    return {
        "created_at": utc_now_iso(),
        "input_paths": input_paths,
        "output_paths": output_paths,
        "count_source": count_source,
        "count_retrieved_at_values": retrieved_values,
        "n_subfields_expected": int(n_subfields_expected),
        "n_subfields_output": int(len(targets)),
        "input_year_min": int(config.input_year_min),
        "input_year_max": int(config.input_year_max),
        "target_year_min": int(config.target_year_min),
        "target_year_max": int(config.target_year_max),
        "input_years": int(config.input_years),
        "target_years": int(config.target_years),
        "global_growth_median": global_median,
        "n_growth_above_median": above_global,
        "n_growth_below_or_equal_median": int(len(targets) - above_global),
        "n_global_median_ties": global_ties,
        "n_growth_above_domain_median": above_domain,
        "n_growth_below_or_equal_domain_median": int(len(targets) - above_domain),
        "n_domain_median_ties": domain_ties,
        "n_domains": int(targets["domain_id"].nunique(dropna=False)),
        "domain_counts": domain_counts,
        "n_subfields_with_zero_input_count": int(targets["has_zero_input_count"].sum()),
        "n_subfields_with_zero_target_count": int(targets["has_zero_target_count"].sum()),
        "annualized_log_growth_summary": summary_stats(targets["annualized_log_growth"]),
        "domain_adjusted_annualized_log_growth_summary": summary_stats(
            targets["domain_adjusted_annualized_log_growth"]
        ),
        "warnings": warnings,
        "assumptions": assumptions,
    }


def fetch_openalex_subfield_year_counts(
    *,
    client: OpenAlexClient,
    years: Iterable[int],
    per_page: int = 200,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    retrieved_at = utc_now_iso()
    for year in years:
        params = build_count_query_params(
            int(year),
            group_by="primary_topic.subfield.id",
            extra_filters=["type:article|preprint"],
            per_page=per_page,
        )
        for group in client.paginate("works", params, results_key="group_by"):
            subfield_id = short_openalex_id(group.get("key"))
            if not subfield_id or subfield_id.lower() == "unknown":
                continue
            rows.append(
                {
                    "subfield_id": subfield_id,
                    "publication_year": int(year),
                    "works_count": int(group.get("count") or 0),
                    "count_source": "openalex_works_group_by_primary_topic_subfield",
                    "retrieved_at": retrieved_at,
                }
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return pd.DataFrame(rows)


def short_text(value: object, max_len: int = 48) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_annualized_log_growth_histogram(
    targets: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    values = targets["annualized_log_growth"].astype(float)
    ax.hist(values, bins=min(30, max(8, int(math.sqrt(len(values)) * 2))), color="#4C78A8", edgecolor="white")
    median = float(values.median())
    ax.axvline(median, color="#C44E52", linewidth=2, label=f"Global median: {median:.3f}")
    ax.axvline(0, color="#444444", linewidth=1.2, linestyle="--", label="Zero growth")
    ax.set_title("Annualized Log Growth Across Main-Analysis Subfields")
    ax.set_xlabel("Annualized log growth")
    ax.set_ylabel("Subfields")
    ax.legend(frameon=False)
    _save_figure(fig, path, dpi)


def plot_annual_rates_scatter(
    targets: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    domains = targets["domain_display_name"].fillna("Unknown").astype(str)
    unique_domains = sorted(domains.unique())
    cmap = plt.get_cmap("tab10")
    for index, domain in enumerate(unique_domains):
        mask = domains == domain
        ax.scatter(
            targets.loc[mask, "annual_rate_2010_2019"],
            targets.loc[mask, "annual_rate_2020_2025"],
            s=28,
            alpha=0.75,
            label=short_text(domain, 28),
            color=cmap(index % 10),
        )
    max_rate = float(
        max(
            targets["annual_rate_2010_2019"].max(),
            targets["annual_rate_2020_2025"].max(),
            1.0,
        )
    )
    ax.plot([0, max_rate], [0, max_rate], color="#333333", linewidth=1.2, linestyle="--")
    if max_rate > 100:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
    ax.set_title("Annual Publication Rates Before And After 2020")
    ax.set_xlabel("Annual rate, 2010-2019")
    ax.set_ylabel("Annual rate, 2020-2025")
    if len(unique_domains) <= 12:
        ax.legend(frameon=False, fontsize=8, loc="best")
    _save_figure(fig, path, dpi)


def plot_growth_by_domain_boxplot(
    targets: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    plot_frame = targets.copy()
    plot_frame["_domain_plot_label"] = (
        plot_frame["domain_display_name"].fillna("Unknown").astype(str)
    )
    grouped = []
    labels = []
    order = (
        plot_frame.groupby("_domain_plot_label", dropna=False)["annualized_log_growth"]
        .median()
        .sort_values()
        .index
    )
    for domain in order:
        values = plot_frame.loc[
            plot_frame["_domain_plot_label"] == domain,
            "annualized_log_growth",
        ].astype(float)
        grouped.append(values.to_numpy())
        labels.append(short_text(domain, 24))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.boxplot(grouped, labels=labels, vert=True, patch_artist=True)
    ax.axhline(
        float(targets["annualized_log_growth"].median()),
        color="#C44E52",
        linestyle="--",
        linewidth=1.2,
        label="Global median",
    )
    ax.set_title("Annualized Log Growth By Domain")
    ax.set_ylabel("Annualized log growth")
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(frameon=False)
    _save_figure(fig, path, dpi)


def plot_domain_adjusted_growth_histogram(
    targets: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    values = targets["domain_adjusted_annualized_log_growth"].astype(float)
    ax.hist(values, bins=min(30, max(8, int(math.sqrt(len(values)) * 2))), color="#59A14F", edgecolor="white")
    ax.axvline(0, color="#C44E52", linewidth=2, label="Domain median")
    ax.set_title("Domain-Adjusted Annualized Log Growth")
    ax.set_xlabel("Annualized log growth minus domain median")
    ax.set_ylabel("Subfields")
    ax.legend(frameon=False)
    _save_figure(fig, path, dpi)


def _plot_top_bottom_barplot(
    targets: pd.DataFrame,
    *,
    metric: str,
    title: str,
    path: Path,
    dpi: int = 150,
    n: int = 15,
) -> None:
    top = targets.sort_values([metric, "subfield_id"], ascending=[False, True]).head(n)
    bottom = targets.sort_values([metric, "subfield_id"], ascending=[True, True]).head(n)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=False)
    for ax, frame, subtitle, color in [
        (axes[0], bottom.sort_values(metric), "Bottom", "#C44E52"),
        (axes[1], top.sort_values(metric), "Top", "#4C78A8"),
    ]:
        labels = [short_text(value, 42) for value in frame["subfield_label_short"]]
        ax.barh(labels, frame[metric].astype(float), color=color)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(subtitle)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(title)
    axes[0].set_xlabel(metric)
    axes[1].set_xlabel(metric)
    _save_figure(fig, path, dpi)


def plot_top_bottom_growth_barplots(
    targets: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    _plot_top_bottom_barplot(
        targets,
        metric="annualized_log_growth",
        title="Top And Bottom Subfields By Annualized Log Growth",
        path=path,
        dpi=dpi,
    )


def plot_top_bottom_domain_adjusted_growth_barplots(
    targets: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    _plot_top_bottom_barplot(
        targets,
        metric="domain_adjusted_annualized_log_growth",
        title="Top And Bottom Subfields By Domain-Adjusted Growth",
        path=path,
        dpi=dpi,
    )


def _example_subfield_ids(targets: pd.DataFrame, max_examples: int = 6) -> list[str]:
    selected: list[str] = []

    def add(ids: Iterable[str]) -> None:
        for subfield_id in ids:
            if subfield_id not in selected:
                selected.append(str(subfield_id))
            if len(selected) >= max_examples:
                return

    add(targets.nlargest(1, "annualized_log_growth")["subfield_id"].astype(str))
    add(targets.nsmallest(1, "annualized_log_growth")["subfield_id"].astype(str))
    add(targets.nlargest(1, "domain_adjusted_annualized_log_growth")["subfield_id"].astype(str))
    add(targets.nsmallest(1, "domain_adjusted_annualized_log_growth")["subfield_id"].astype(str))
    stable = targets.assign(
        abs_growth=targets["annualized_log_growth"].abs(),
        size_rank=targets["annual_rate_2010_2019"].rank(ascending=False, method="min"),
    ).sort_values(["abs_growth", "size_rank"], kind="mergesort")
    add(stable["subfield_id"].astype(str))
    return selected[:max_examples]


def plot_yearly_counts_examples(
    targets: pd.DataFrame,
    panel: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    ids = _example_subfield_ids(targets)
    labels = targets.set_index("subfield_id")["subfield_label_short"].to_dict()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for subfield_id in ids:
        rows = panel[panel["subfield_id"].astype(str) == str(subfield_id)].sort_values(
            "publication_year"
        )
        ax.plot(
            rows["publication_year"],
            rows["works_count"],
            marker="o",
            linewidth=1.7,
            label=short_text(labels.get(subfield_id, subfield_id), 38),
        )
    ax.axvline(2019.5, color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Yearly Count Examples")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Works count")
    ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, path, dpi)


def plot_growth_vs_initial_size(
    targets: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        targets["annual_rate_2010_2019"],
        targets["annualized_log_growth"],
        s=28,
        alpha=0.75,
        color="#4C78A8",
    )
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    if float(targets["annual_rate_2010_2019"].max()) > 100:
        ax.set_xscale("symlog", linthresh=1)
    ax.set_title("Growth Versus Initial Annual Rate")
    ax.set_xlabel("Annual rate, 2010-2019")
    ax.set_ylabel("Annualized log growth")
    _save_figure(fig, path, dpi)


def plot_growth_percentile_by_domain(
    targets: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    domains = targets["domain_display_name"].fillna("Unknown").astype(str)
    order = sorted(domains.unique())
    positions = {domain: index for index, domain in enumerate(order)}
    x = domains.map(positions)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(x, targets["growth_percentile"], s=28, alpha=0.75, color="#F28E2B")
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([short_text(domain, 24) for domain in order], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Global Growth Percentile By Domain")
    ax.set_ylabel("Global growth percentile")
    _save_figure(fig, path, dpi)


def write_growth_figures(
    *,
    targets: pd.DataFrame,
    panel: pd.DataFrame,
    figures_dir: str | Path,
    dpi: int = 150,
) -> dict[str, str]:
    output_dir = Path(figures_dir)
    paths = {
        "annualized_log_growth_histogram": output_dir
        / "annualized_log_growth_histogram.png",
        "annual_rates_scatter": output_dir / "annual_rates_scatter.png",
        "growth_by_domain_boxplot": output_dir / "growth_by_domain_boxplot.png",
        "domain_adjusted_growth_histogram": output_dir
        / "domain_adjusted_growth_histogram.png",
        "top_bottom_growth_barplots": output_dir / "top_bottom_growth_barplots.png",
        "top_bottom_domain_adjusted_growth_barplots": output_dir
        / "top_bottom_domain_adjusted_growth_barplots.png",
        "yearly_counts_examples": output_dir / "yearly_counts_examples.png",
        "growth_vs_initial_size": output_dir / "growth_vs_initial_size.png",
        "growth_percentile_by_domain": output_dir / "growth_percentile_by_domain.png",
    }
    plot_annualized_log_growth_histogram(
        targets, paths["annualized_log_growth_histogram"], dpi=dpi
    )
    plot_annual_rates_scatter(targets, paths["annual_rates_scatter"], dpi=dpi)
    plot_growth_by_domain_boxplot(targets, paths["growth_by_domain_boxplot"], dpi=dpi)
    plot_domain_adjusted_growth_histogram(
        targets, paths["domain_adjusted_growth_histogram"], dpi=dpi
    )
    plot_top_bottom_growth_barplots(
        targets, paths["top_bottom_growth_barplots"], dpi=dpi
    )
    plot_top_bottom_domain_adjusted_growth_barplots(
        targets, paths["top_bottom_domain_adjusted_growth_barplots"], dpi=dpi
    )
    plot_yearly_counts_examples(targets, panel, paths["yearly_counts_examples"], dpi=dpi)
    plot_growth_vs_initial_size(targets, paths["growth_vs_initial_size"], dpi=dpi)
    plot_growth_percentile_by_domain(
        targets, paths["growth_percentile_by_domain"], dpi=dpi
    )
    return {name: str(path) for name, path in paths.items()}
