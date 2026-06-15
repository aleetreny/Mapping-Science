from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.storage import save_parquet
from src.temporal_common import (
    REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS,
    display_path,
    standardization_parameters,
    utc_now_iso,
)


VALID_METRIC_STATUSES = {"completed", "completed_with_warnings"}

OUTPUT_FILENAMES = {
    "profiles_parquet": "static_discipline_profiles.parquet",
    "profiles_csv": "static_discipline_profiles.csv",
    "profile_distances_parquet": "static_profile_distances.parquet",
    "profile_distances_csv": "static_profile_distances.csv",
    "top_profile_pairs_csv": "top_static_profile_pairs.csv",
    "metric_rankings_csv": "static_metric_rankings.csv",
    "scaling_parameters_csv": "static_profile_scaling_parameters.csv",
    "summary_md": "summary.md",
    "summary_json": "summary.json",
}


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}


def existing_outputs(output_dir: str | Path) -> list[Path]:
    return [path for path in output_paths(output_dir).values() if path.exists()]


def ensure_static_profile_outputs(output_dir: str | Path, *, overwrite: bool, root: Path) -> None:
    existing = existing_outputs(output_dir)
    if existing and not overwrite:
        formatted = ", ".join(display_path(path, root=root) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing static comparison outputs without "
            f"--overwrite: {formatted}"
        )


def filter_valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "metric_status" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["metric_status"].isin(VALID_METRIC_STATUSES)].copy()


def require_metrics(frame: pd.DataFrame, metrics: list[str]) -> None:
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        raise ValueError("Missing reduced-core metrics: " + ", ".join(missing))


def _entity_columns_for_level(level: str) -> tuple[str, str]:
    if level == "subfield":
        return "subfield_id", "subfield_display_name"
    if level == "field":
        return "field_id", "field_display_name"
    if level == "domain":
        return "domain_id", "domain_display_name"
    raise ValueError("level must be subfield, field, or domain")


def profile_table(
    reduced_metrics: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    metrics = list(metrics or REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)
    frame = filter_valid_rows(reduced_metrics)
    require_metrics(frame, metrics)
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    rows: list[pd.DataFrame] = []
    for level in ["subfield", "field", "domain"]:
        id_column, name_column = _entity_columns_for_level(level)
        required = {id_column, name_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns for {level} profiles: {', '.join(sorted(missing))}")

        if level == "subfield":
            profile = frame[[id_column, name_column, *metrics]].copy()
            profile["n_subfields"] = 1
        else:
            profile = (
                frame.groupby([id_column, name_column], dropna=False, sort=True)[metrics]
                .mean()
                .reset_index()
            )
            counts = (
                frame.groupby([id_column, name_column], dropna=False, sort=True)["subfield_id"]
                .nunique()
                .reset_index(name="n_subfields")
            )
            profile = profile.merge(counts, on=[id_column, name_column], how="left")

        profile = profile.rename(
            columns={id_column: "entity_id", name_column: "entity_display_name"}
        )
        profile.insert(0, "level", level)
        rows.append(profile)

    result = pd.concat(rows, ignore_index=True)
    result["entity_id"] = result["entity_id"].astype(str)
    result["entity_display_name"] = result["entity_display_name"].astype(str)
    return result.sort_values(["level", "entity_id"], kind="mergesort").reset_index(drop=True)


def robust_scale_profiles(
    profiles: pd.DataFrame,
    *,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaled = profiles.copy()
    parameter_rows: list[dict[str, Any]] = []
    for level, level_frame in profiles.groupby("level", sort=False):
        level_index = level_frame.index
        for metric in metrics:
            values = pd.to_numeric(level_frame[metric], errors="coerce")
            params = standardization_parameters(values)
            scale = params["iqr"]
            fallback = ""
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = params["std"]
                fallback = "std"
            if not np.isfinite(scale) or scale <= 1e-12:
                scaled.loc[level_index, metric] = np.nan
                scale = np.nan
                fallback = "unavailable"
            else:
                scaled.loc[level_index, metric] = (values - params["median"]) / scale
            parameter_rows.append(
                {
                    "level": level,
                    "metric": metric,
                    "scaler": "robust",
                    "center": params["median"],
                    "scale": scale,
                    "fallback_scale": fallback,
                    **params,
                }
            )
    return scaled, pd.DataFrame(parameter_rows)


def _distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    distance_metric: str,
) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return float("nan")
    a = left[valid]
    b = right[valid]
    if distance_metric == "euclidean":
        return float(np.linalg.norm(a - b))
    if distance_metric == "correlation":
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            return float("nan")
        return float(1.0 - np.corrcoef(a, b)[0, 1])
    raise ValueError("distance_metric must be euclidean or correlation")


def pairwise_profile_distances(
    profiles: pd.DataFrame,
    *,
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scaled, _params = robust_scale_profiles(profiles, metrics=metrics)
    for level, level_frame in scaled.groupby("level", sort=False):
        level_frame = level_frame.sort_values(["entity_id"], kind="mergesort").reset_index(drop=True)
        for left_idx, right_idx in combinations(range(len(level_frame)), 2):
            left = level_frame.iloc[left_idx]
            right = level_frame.iloc[right_idx]
            left_values = left[metrics].to_numpy(dtype=float)
            right_values = right[metrics].to_numpy(dtype=float)
            for distance_metric in ["euclidean", "correlation"]:
                rows.append(
                    {
                        "level": level,
                        "distance_metric": distance_metric,
                        "pair_id": f"{left.entity_id}__{right.entity_id}",
                        "entity_id_a": left.entity_id,
                        "entity_display_name_a": left.entity_display_name,
                        "entity_id_b": right.entity_id,
                        "entity_display_name_b": right.entity_display_name,
                        "profile_distance": _distance(
                            left_values,
                            right_values,
                            distance_metric=distance_metric,
                        ),
                        "n_metrics": int(
                            (np.isfinite(left_values) & np.isfinite(right_values)).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def metric_rankings(
    profiles: pd.DataFrame,
    *,
    metrics: list[str],
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level, level_frame in profiles.groupby("level", sort=False):
        for metric in metrics:
            values = level_frame.copy()
            values[metric] = pd.to_numeric(values[metric], errors="coerce")
            values = values.dropna(subset=[metric])
            for direction, ascending in [("lowest", True), ("highest", False)]:
                ranked = values.sort_values(
                    [metric, "entity_display_name"],
                    ascending=[ascending, True],
                    kind="mergesort",
                ).head(top_n)
                for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                    rows.append(
                        {
                            "level": level,
                            "metric": metric,
                            "direction": direction,
                            "rank": rank,
                            "entity_id": row.entity_id,
                            "entity_display_name": row.entity_display_name,
                            "value": float(getattr(row, metric)),
                            "n_subfields": int(row.n_subfields),
                        }
                    )
    return pd.DataFrame(rows)


def top_profile_pairs(
    distances: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    valid = distances.dropna(subset=["profile_distance"]).copy()
    for (level, distance_metric), group in valid.groupby(["level", "distance_metric"], sort=True):
        closest = group.sort_values(
            ["profile_distance", "pair_id"],
            ascending=[True, True],
            kind="mergesort",
        ).head(top_n)
        farthest = group.sort_values(
            ["profile_distance", "pair_id"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top_n)
        rows.append(closest.assign(direction="closest"))
        rows.append(farthest.assign(direction="farthest"))
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    order = [
        "level",
        "distance_metric",
        "direction",
        "pair_id",
        "entity_id_a",
        "entity_display_name_a",
        "entity_id_b",
        "entity_display_name_b",
        "profile_distance",
        "n_metrics",
    ]
    return result[order]


def markdown_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Static Discipline Profiles",
            "",
            "This stage compares OpenAlex subfields, fields, and domains using the "
            "active 8-metric structural SPECTER2 embedding-space core.",
            "",
            f"- Input rows: {summary['n_rows_input']}",
            f"- Profile rows: {summary['n_profile_rows']}",
            f"- Pairwise rows: {summary['n_pairwise_rows']}",
            f"- Metric rankings: {summary['n_ranking_rows']}",
            "",
            "UMAP-derived metrics are not used in this static comparison.",
            "",
        ]
    )


def build_static_discipline_profile_outputs(
    reduced_metrics: pd.DataFrame,
    *,
    input_path: str,
    output_dir: str | Path,
    top_n: int = 10,
) -> dict[str, Any]:
    metrics = list(REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS)
    if len(metrics) != 8:
        raise ValueError(
            "Static discipline profiles require the active 8-metric structural core."
        )
    paths = output_paths(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    profiles = profile_table(reduced_metrics, metrics=metrics)
    distances = pairwise_profile_distances(profiles, metrics=metrics)
    rankings = metric_rankings(profiles, metrics=metrics, top_n=top_n)
    top_pairs = top_profile_pairs(distances, top_n=top_n)
    _scaled, scaling = robust_scale_profiles(profiles, metrics=metrics)

    save_parquet(profiles, paths["profiles_parquet"])
    profiles.to_csv(paths["profiles_csv"], index=False)
    save_parquet(distances, paths["profile_distances_parquet"])
    distances.to_csv(paths["profile_distances_csv"], index=False)
    rankings.to_csv(paths["metric_rankings_csv"], index=False)
    top_pairs.to_csv(paths["top_profile_pairs_csv"], index=False)
    scaling.to_csv(paths["scaling_parameters_csv"], index=False)

    summary = {
        "created_at": utc_now_iso(),
        "input_path": input_path,
        "output_dir": str(output_dir),
        "metrics": metrics,
        "n_rows_input": int(len(reduced_metrics)),
        "n_profile_rows": int(len(profiles)),
        "n_pairwise_rows": int(len(distances)),
        "n_ranking_rows": int(len(rankings)),
        "top_n": int(top_n),
        "output_paths": {key: str(path) for key, path in paths.items()},
    }
    paths["summary_md"].write_text(markdown_summary(summary), encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
