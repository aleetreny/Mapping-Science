from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EPS = 1e-12

DEFAULT_WINDOWS: list[tuple[int, int]] = [
    (2000, 2004),
    (2005, 2009),
    (2010, 2014),
    (2015, 2019),
    (2020, 2024),
]

STRUCTURAL_MORPHOLOGY_METRICS: list[str] = [
    "embedding_distance_to_centroid_median",
    "embedding_distance_to_centroid_iqr",
    "embedding_distance_to_centroid_p90",
    "embedding_knn_median_distance",
    "embedding_knn_distance_cv",
    "embedding_knn_indegree_gini",
    "embedding_pca_dim_80",
    "embedding_pca_spectral_entropy",
]

WINDOW_STRUCTURAL_METRICS: list[str] = list(STRUCTURAL_MORPHOLOGY_METRICS)

REDUCED_INTERPRETABLE_EMBEDDING_CORE_METRICS: list[str] = list(
    STRUCTURAL_MORPHOLOGY_METRICS
)

CENTROID_DRIFT_METRIC = "embedding_centroid_drift_early_late"


@dataclass(frozen=True)
class TemporalWindow:
    window_start: int
    window_end: int
    window_index: int

    @property
    def window_label(self) -> str:
        return f"{self.window_start}-{self.window_end}"

    @property
    def window_mid_year(self) -> float:
        return (self.window_start + self.window_end) / 2.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: str | Path, *, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_path(path: str | Path, *, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def default_embeddings_path(embedding_dir: str | Path) -> Path:
    return Path(embedding_dir) / "analysis" / "main_embeddings.float16.npy"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def ensure_outputs_do_not_exist(
    paths: Iterable[str | Path],
    *,
    overwrite: bool,
    root: Path,
    description: str,
) -> None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        formatted = ", ".join(display_path(path, root=root) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing {description} without --overwrite: "
            f"{formatted}"
        )


def parse_multi_value(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    parsed: list[str] = []
    for item in values:
        parsed.extend(part.strip() for part in str(item).split(",") if part.strip())
    return parsed


def make_windows(
    *,
    year_min: int,
    year_max: int,
    window_size: int,
) -> list[TemporalWindow]:
    if year_min > year_max:
        raise ValueError("year_min must be <= year_max")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    windows: list[TemporalWindow] = []
    start = year_min
    index = 0
    while start <= year_max:
        end = min(start + window_size - 1, year_max)
        windows.append(TemporalWindow(start, end, index))
        start = end + 1
        index += 1
    return windows


def default_or_computed_windows(
    *,
    year_min: int,
    year_max: int,
    window_size: int,
) -> list[TemporalWindow]:
    if (
        year_min == DEFAULT_WINDOWS[0][0]
        and year_max == DEFAULT_WINDOWS[-1][1]
        and window_size == 5
    ):
        return [
            TemporalWindow(start, end, index)
            for index, (start, end) in enumerate(DEFAULT_WINDOWS)
        ]
    return make_windows(year_min=year_min, year_max=year_max, window_size=window_size)


def add_window_columns(frame: pd.DataFrame, windows: list[TemporalWindow]) -> pd.DataFrame:
    rows = [
        {
            "window_start": window.window_start,
            "window_end": window.window_end,
            "window_index": window.window_index,
            "window_label": window.window_label,
            "window_mid_year": window.window_mid_year,
        }
        for window in windows
    ]
    window_frame = pd.DataFrame(rows)
    return frame.merge(window_frame, on=["window_start", "window_end"], how="left")


def load_embedding_matrix(path: str | Path) -> np.ndarray:
    embeddings_path = Path(path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embedding matrix: {embeddings_path}")
    matrix = np.load(embeddings_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {matrix.shape}")
    return matrix


def validate_matrix_alignment(
    index: pd.DataFrame,
    matrix: np.ndarray,
    *,
    index_path: str | Path,
    embeddings_path: str | Path,
) -> None:
    if "analysis_row_id" not in index.columns:
        raise ValueError("analysis index is missing analysis_row_id")
    row_ids = pd.to_numeric(index["analysis_row_id"], errors="raise")
    if row_ids.empty:
        raise ValueError("analysis index is empty")
    min_row = int(row_ids.min())
    max_row = int(row_ids.max())
    if min_row < 0 or max_row >= matrix.shape[0]:
        raise ValueError(
            "analysis_embedding_index and embedding matrix appear to belong to "
            "different embedding versions. "
            f"index_path={index_path}; embeddings_path={embeddings_path}; "
            f"analysis_row_id range {min_row}-{max_row}; matrix rows {matrix.shape[0]}"
        )


def vector_columns(frame: pd.DataFrame, prefix: str = "centroid_dim_") -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith(prefix)],
        key=lambda value: int(value.removeprefix(prefix)),
    )


def vectors_to_columns(vectors: np.ndarray, prefix: str = "centroid_dim_") -> pd.DataFrame:
    vectors = np.asarray(vectors, dtype=np.float32)
    columns = [f"{prefix}{idx:03d}" for idx in range(vectors.shape[1])]
    return pd.DataFrame(vectors, columns=columns)


def row_norms(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(matrix, dtype=float), axis=1)


def l2_normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    valid = np.isfinite(norms) & (norms > EPS)
    output = np.zeros_like(values, dtype=np.float32)
    output[valid] = values[valid] / norms[valid, None]
    return output


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= EPS or second_norm <= EPS:
        return np.nan
    value = 1.0 - float(np.dot(first / first_norm, second / second_norm))
    return float(np.clip(value, 0.0, 2.0))


def zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.mean(skipna=True)
    std = numeric.std(skipna=True, ddof=0)
    if not np.isfinite(std) or std <= EPS:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - mean) / std


def robust_zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median(skipna=True)
    q25 = numeric.quantile(0.25)
    q75 = numeric.quantile(0.75)
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= EPS:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - median) / iqr


def standardization_parameters(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    q25 = numeric.quantile(0.25)
    q75 = numeric.quantile(0.75)
    return {
        "mean": float(numeric.mean(skipna=True)),
        "std": float(numeric.std(skipna=True, ddof=0)),
        "median": float(numeric.median(skipna=True)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "n_non_missing": int(numeric.notna().sum()),
    }


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.nan
    x = x[finite]
    y = y[finite]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0 if np.allclose(y, y[0]) else np.nan
    centered = x - x.mean()
    return float(np.polyfit(centered, y, deg=1)[0])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.nan
    x = x[finite]
    y = y[finite]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0 if np.allclose(y, y[0]) else np.nan
    from scipy.stats import spearmanr

    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else np.nan


def direction_label(delta: float, *, stable_threshold: float = 0.25) -> str:
    if not np.isfinite(delta):
        return "unknown"
    if abs(delta) < stable_threshold:
        return "stable"
    return "increase" if delta > 0 else "decrease"


def dataframe_to_records(frame: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    return frame.head(limit).replace({np.nan: None}).to_dict("records")


def write_placeholder_figure(path: str | Path, message: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=220)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
