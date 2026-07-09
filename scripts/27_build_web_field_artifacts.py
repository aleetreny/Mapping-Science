from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding_space_metrics import compute_embedding_metrics
from src.per_category_umap_maps import plot_category_panels
from src.storage import load_parquet, save_parquet


WEB_METRIC_COLUMNS = [
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_iqr",
    "embedding_distance_to_centroid_p90",
    "embedding_knn_median_distance",
    "embedding_knn_distance_cv",
    "embedding_knn_indegree_gini",
    "embedding_pca_dim_80",
    "embedding_pca_spectral_entropy",
    "embedding_centroid_drift_early_late",
]


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Build field-level map and metric artifacts for the static web explorer."
    )
    parser.add_argument(
        "--field-umap-dir",
        default="outputs/08_visualization/per_field_umap_smooth_density",
    )
    parser.add_argument(
        "--embedding-dir",
        default=os.getenv(
            "LOCAL_EMBEDDINGS_DIR",
            "embeddings/specter2_v1_2000_2024_400py",
        ),
    )
    parser.add_argument("--embeddings-path", default=None)
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--metrics-csv",
        default="outputs/03_embedding_metrics/field_embedding_space_metrics_for_web.csv",
    )
    parser.add_argument(
        "--metrics-parquet",
        default="outputs/03_embedding_metrics/field_embedding_space_metrics_for_web.parquet",
    )
    parser.add_argument(
        "--payload-json",
        default="outputs/03_embedding_metrics/field_web_payload.json",
    )
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-replot", action="store_true")
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


def default_embeddings_path(embedding_dir: str | Path) -> Path:
    return Path(embedding_dir) / "analysis" / "main_embeddings.float16.npy"


def resolve_embeddings_path(args: argparse.Namespace) -> Path:
    path = args.embeddings_path or default_embeddings_path(args.embedding_dir)
    return resolve_path(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_js_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(f"window.fieldWebPayload = {json_payload};\n", encoding="utf-8")


def safe_number(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def clean_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("/", " & ")


def load_manifest(field_umap_dir: Path) -> pd.DataFrame:
    manifest_path = field_umap_dir / "per_field_umap_manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing field UMAP manifest: {display_path(manifest_path)}")
    manifest = load_parquet(manifest_path)
    completed = manifest.loc[manifest["status"] == "completed"].copy()
    if completed.empty:
        raise ValueError("Field UMAP manifest contains no completed rows")
    return completed.sort_values(["group_id"], kind="mergesort").reset_index(drop=True)


def replot_field_figures(
    manifest: pd.DataFrame,
    *,
    field_umap_dir: Path,
    dpi: int,
) -> None:
    for row in manifest.itertuples(index=False):
        coordinate_path = resolve_path(row.coordinate_path)
        figure_path = resolve_path(row.figure_path)
        coordinates_frame = load_parquet(coordinate_path)
        coordinates = coordinates_frame[["umap_x", "umap_y"]].to_numpy(dtype=float)
        plot_info = plot_category_panels(
            coordinates_frame,
            coordinates,
            level="field",
            group_name=str(row.group_name),
            n_used=int(row.n_used),
            year_min=int(row.year_min),
            year_max=int(row.year_max),
            color_column="subfield_display_name",
            output_path=figure_path,
            dpi=dpi,
            max_legend_categories=0,
            density_method="smooth_hist",
        )
        print(
            f"Re-rendered {display_path(figure_path)} "
            f"(legend={plot_info['legend_included']}, density={plot_info['density_method']})"
        )


def copy_field_figures(manifest: pd.DataFrame) -> None:
    targets = [
        ROOT / "docs" / "site-assets" / "figures" / "field_umap",
        ROOT / "frontend" / "figures" / "field_umap",
    ]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)

    for row in manifest.itertuples(index=False):
        source = resolve_path(row.figure_path)
        if not source.exists():
            raise FileNotFoundError(f"Missing field figure: {display_path(source)}")
        for target in targets:
            shutil.copy2(source, target / source.name)
    print(f"Copied {len(manifest)} field figures to docs and frontend assets.")


def load_matrix(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embedding matrix: {display_path(embeddings_path)}")
    matrix = np.load(embeddings_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got {matrix.shape}")
    return matrix


def compute_field_metrics(
    manifest: pd.DataFrame,
    *,
    embeddings_path: Path,
    year_min: int,
    year_max: int,
    k_neighbors: int,
) -> pd.DataFrame:
    matrix = load_matrix(embeddings_path)
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        coordinate_path = resolve_path(row.coordinate_path)
        frame = load_parquet(coordinate_path)
        row_ids = frame["analysis_row_id"].astype(int).to_numpy()
        years = pd.to_numeric(frame["publication_year"], errors="raise").to_numpy()
        embeddings = np.asarray(matrix[row_ids], dtype=np.float32)
        metrics, controls, warnings = compute_embedding_metrics(
            embeddings,
            years,
            year_min=year_min,
            year_max=year_max,
            k_neighbors=k_neighbors,
        )
        rows.append(
            {
                "field_id": str(row.group_id),
                "field_display_name": clean_name(row.group_name),
                "domain_id": str(frame["domain_id"].dropna().iloc[0]),
                "domain_display_name": clean_name(frame["domain_display_name"].dropna().iloc[0]),
                "n_available": int(row.n_available),
                "n_used": int(len(frame)),
                "year_min": int(year_min),
                "year_max": int(year_max),
                "metric_status": "completed_with_warnings" if warnings else "completed",
                "metric_warning_message": "; ".join(warnings),
                "embedding_matrix_path": display_path(embeddings_path),
                "coordinate_path": display_path(coordinate_path),
                "k_neighbors": int(k_neighbors),
                **metrics,
                **controls,
            }
        )
        print(f"[{position}/{len(manifest)}] metrics field {row.group_id} - {row.group_name}")
    return pd.DataFrame(rows)


def percentile_rank(values: pd.Series, value: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().sort_values().to_numpy()
    if len(clean) == 0 or not np.isfinite(value):
        return 0.5
    lower = int(np.count_nonzero(clean < value))
    equal = int(np.count_nonzero(clean == value))
    return float((lower + equal / 2) / len(clean))


def band_label(score: float, low: str, mid: str, high: str) -> str:
    if score < 0.33:
        return low
    if score < 0.67:
        return mid
    return high


def complexity_label(metrics: pd.DataFrame, row: pd.Series) -> str:
    score = percentile_rank(metrics["embedding_pca_dim_80"], row["embedding_pca_dim_80"])
    if score < 0.25:
        return "Low Complexity (High Focus)"
    if score < 0.50:
        return "Medium Complexity"
    if score < 0.75:
        return "Medium-High Complexity"
    return "Very High Complexity (Broad)"


def drift_label(metrics: pd.DataFrame, row: pd.Series) -> str:
    score = percentile_rank(
        metrics["embedding_centroid_drift_early_late"],
        row["embedding_centroid_drift_early_late"],
    )
    if score < 0.25:
        return "Static (Highly Stable)"
    if score < 0.50:
        return "Low Historical Shift"
    if score < 0.75:
        return "Moderate Historical Shift"
    return "Highly Dynamic Shift"


def structural_label(metrics: pd.DataFrame, row: pd.Series) -> str:
    dispersion = np.mean(
        [
            percentile_rank(metrics["embedding_distance_to_centroid_median"], row["embedding_distance_to_centroid_median"]),
            percentile_rank(metrics["embedding_distance_to_centroid_p90"], row["embedding_distance_to_centroid_p90"]),
        ]
    )
    density = percentile_rank(metrics["embedding_knn_median_distance"], row["embedding_knn_median_distance"])
    hubness = np.mean(
        [
            percentile_rank(metrics["embedding_knn_distance_cv"], row["embedding_knn_distance_cv"]),
            percentile_rank(metrics["embedding_knn_indegree_gini"], row["embedding_knn_indegree_gini"]),
        ]
    )
    spectral = np.mean(
        [
            percentile_rank(metrics["embedding_pca_dim_80"], row["embedding_pca_dim_80"]),
            percentile_rank(metrics["embedding_pca_spectral_entropy"], row["embedding_pca_spectral_entropy"]),
        ]
    )

    if spectral >= 0.67:
        return "Spectrally complex profiles"
    if dispersion >= 0.67:
        return "Broad dispersed profiles"
    if dispersion < 0.33 and density < 0.33:
        return "Compact dense profiles"
    if hubness >= 0.67:
        return "Hub-concentrated uneven profiles"
    return "Balanced field profiles"


def closest_fields(metrics: pd.DataFrame) -> dict[str, str]:
    values = metrics[WEB_METRIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    medians = values.median(axis=0)
    scales = values.quantile(0.75, axis=0) - values.quantile(0.25, axis=0)
    scales = scales.mask(scales <= 1e-12, values.std(axis=0)).replace(0, np.nan)
    scaled = (values - medians) / scales
    scaled = scaled.fillna(0.0)
    result: dict[str, str] = {}
    ids = metrics["field_id"].astype(str).tolist()
    names = metrics["field_display_name"].astype(str).tolist()
    matrix = scaled.to_numpy(dtype=float)
    for index, field_id in enumerate(ids):
        distances = np.linalg.norm(matrix - matrix[index], axis=1)
        order = np.argsort(distances)
        neighbors = [names[pos] for pos in order if pos != index][:2]
        result[field_id] = " & ".join(neighbors)
    return result


def build_payload(metrics: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    metrics = metrics.copy()
    metrics["field_id"] = metrics["field_id"].astype(str)
    manifest = manifest.copy()
    manifest["group_id"] = manifest["group_id"].astype(str)
    stem_lookup = {
        str(row.group_id): Path(row.figure_path).stem
        for row in manifest.itertuples(index=False)
    }
    neighbor_lookup = closest_fields(metrics)
    raw: dict[str, dict[str, Any]] = {}
    field_data: dict[str, dict[str, str]] = {}
    field_names: dict[str, str] = {}
    field_stems: dict[str, str] = {}
    for row in metrics.sort_values("field_display_name", kind="mergesort").itertuples(index=False):
        field_id = str(row.field_id)
        field_name = str(row.field_display_name)
        domain_name = str(row.domain_display_name)
        row_series = metrics.loc[metrics["field_id"] == field_id].iloc[0]
        raw[field_id] = {
            "field": field_name,
            "domain": domain_name,
            **{
                metric: safe_number(getattr(row, metric))
                for metric in WEB_METRIC_COLUMNS
            },
        }
        field_data[field_id] = {
            "parent": domain_name,
            "topology": structural_label(metrics, row_series),
            "overlap": neighbor_lookup.get(field_id, ""),
            "n": f"{int(row.n_used):,} papers",
            "complexity": complexity_label(metrics, row_series),
            "drift": drift_label(metrics, row_series),
        }
        field_names[field_id] = field_name
        field_stems[field_id] = stem_lookup[field_id]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rawFieldMetricsData": raw,
        "fieldData": field_data,
        "fieldNames": field_names,
        "fieldImageStems": field_stems,
    }


def main() -> None:
    args = parse_args()
    field_umap_dir = resolve_path(args.field_umap_dir)
    manifest = load_manifest(field_umap_dir)

    if not args.skip_replot:
        replot_field_figures(manifest, field_umap_dir=field_umap_dir, dpi=args.dpi)
    copy_field_figures(manifest)

    metrics_csv = resolve_path(args.metrics_csv)
    metrics_parquet = resolve_path(args.metrics_parquet)
    if args.skip_metrics and metrics_csv.exists():
        metrics = pd.read_csv(metrics_csv)
    else:
        metrics = compute_field_metrics(
            manifest,
            embeddings_path=resolve_embeddings_path(args),
            year_min=args.year_min,
            year_max=args.year_max,
            k_neighbors=args.k_neighbors,
        )
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_csv, index=False)
        save_parquet(metrics, metrics_parquet)
        print(f"Wrote {display_path(metrics_csv)}")
        print(f"Wrote {display_path(metrics_parquet)}")

    payload = build_payload(metrics, manifest)
    payload_json = resolve_path(args.payload_json)
    write_json(payload_json, payload)
    print(f"Wrote {display_path(payload_json)}")

    for payload_js in [
        ROOT / "docs" / "site-assets" / "data" / "field_payload.js",
        ROOT / "frontend" / "site-assets" / "data" / "field_payload.js",
    ]:
        write_js_payload(payload_js, payload)
        print(f"Wrote {display_path(payload_js)}")


if __name__ == "__main__":
    main()
