from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "openalex_stats_probability" / "corpus.sqlite"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "research" / "results"
SNAPSHOT_DATE = pd.Timestamp("2026-04-24")
RANDOM_STATE = 42


SQL_QUERY = """
WITH author_features AS (
    SELECT
        work_id,
        COUNT(*) AS author_count,
        SUM(COALESCE(is_corresponding, 0)) AS corresponding_author_count
    FROM work_authorships
    GROUP BY work_id
),
topic_features AS (
    SELECT
        work_id,
        COUNT(*) AS topic_count,
        AVG(score) AS avg_topic_score,
        MAX(score) AS max_topic_score
    FROM work_topics
    GROUP BY work_id
)
SELECT
    w.work_id,
    COALESCE(w.display_name, w.title) AS title,
    w.abstract_text,
    w.cited_by_count,
    w.publication_year,
    w.publication_date,
    w.referenced_works_count,
    w.locations_count,
    w.has_abstract,
    w.has_references,
    w.has_pdf_url,
    w.has_fulltext,
    w.has_content_pdf,
    w.has_content_grobid_xml,
    w.is_oa,
    w.any_repository_has_fulltext,
    w.primary_topic_name,
    w.primary_field_name,
    w.primary_domain_name,
    w.primary_source_name,
    w.primary_source_type,
    w.primary_source_host_org_name,
    w.primary_source_is_oa,
    w.primary_source_is_in_doaj,
    w.primary_source_is_core,
    w.oa_status,
    w.license,
    w.version,
    COALESCE(a.author_count, 0) AS author_count,
    COALESCE(a.corresponding_author_count, 0) AS corresponding_author_count,
    COALESCE(t.topic_count, 0) AS topic_count,
    COALESCE(t.avg_topic_score, 0.0) AS avg_topic_score,
    COALESCE(t.max_topic_score, 0.0) AS max_topic_score
FROM works AS w
LEFT JOIN author_features AS a ON a.work_id = w.work_id
LEFT JOIN topic_features AS t ON t.work_id = w.work_id
WHERE w.cited_by_count IS NOT NULL
"""


STRUCTURED_NUMERIC_COLUMNS = [
    "paper_age_days",
    "publication_year",
    "referenced_works_count",
    "locations_count",
    "has_abstract",
    "has_references",
    "has_pdf_url",
    "has_fulltext",
    "has_content_pdf",
    "has_content_grobid_xml",
    "is_oa",
    "any_repository_has_fulltext",
    "primary_source_is_oa",
    "primary_source_is_in_doaj",
    "primary_source_is_core",
    "author_count",
    "corresponding_author_count",
    "topic_count",
    "avg_topic_score",
    "max_topic_score",
    "title_char_count",
    "title_word_count",
    "abstract_char_count",
    "abstract_word_count",
]

LINEAR_CATEGORICAL_COLUMNS = [
    "primary_topic_name",
    "primary_field_name",
    "primary_domain_name",
    "primary_source_name",
    "primary_source_type",
    "primary_source_host_org_name",
    "oa_status",
    "license",
    "version",
]

TREE_CATEGORICAL_COLUMNS = [
    "primary_topic_name",
    "primary_field_name",
    "primary_domain_name",
    "primary_source_type",
    "oa_status",
    "license",
    "version",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark citation count prediction models on the OpenAlex paper corpus."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--max-text-features", type=int, default=20000)
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


def load_dataset(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(SQL_QUERY, conn)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    publication_date = pd.to_datetime(result["publication_date"], errors="coerce")
    publication_year = pd.to_numeric(result["publication_year"], errors="coerce").fillna(0).astype(int)
    fallback_date = pd.to_datetime(publication_year.astype(str) + "-07-01", errors="coerce")
    resolved_date = publication_date.fillna(fallback_date)

    result["publication_year"] = publication_year
    result["paper_age_days"] = (SNAPSHOT_DATE - resolved_date).dt.days.clip(lower=0)

    result["title"] = result["title"].fillna("")
    result["abstract_text"] = result["abstract_text"].fillna("")
    result["text_for_model"] = (result["title"] + ". " + result["abstract_text"]).str.strip()

    result["title_char_count"] = result["title"].str.len()
    result["title_word_count"] = result["title"].str.count(r"\S+")
    result["abstract_char_count"] = result["abstract_text"].str.len()
    result["abstract_word_count"] = result["abstract_text"].str.count(r"\S+")

    for column in STRUCTURED_NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def build_text_selector() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "squeeze",
                FunctionTransformer(
                    lambda frame: frame.iloc[:, 0].fillna("").astype(str),
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            )
        ]
    )


def build_model_specs(max_text_features: int) -> list[ModelSpec]:
    age_only_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["paper_age_days"],
            )
        ]
    )

    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                STRUCTURED_NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                LINEAR_CATEGORICAL_COLUMNS,
            ),
        ]
    )

    text_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                STRUCTURED_NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                LINEAR_CATEGORICAL_COLUMNS,
            ),
            (
                "text",
                Pipeline(
                    steps=[
                        ("select", build_text_selector()),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                max_features=max_text_features,
                                min_df=5,
                                stop_words="english",
                            ),
                        ),
                    ]
                ),
                ["text_for_model"],
            ),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]),
                STRUCTURED_NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                TREE_CATEGORICAL_COLUMNS,
            ),
        ]
    )

    specs = [
        ModelSpec(
            name="dummy_median",
            estimator=Pipeline(
                steps=[
                    ("preprocess", age_only_preprocessor),
                    ("model", DummyRegressor(strategy="median")),
                ]
            ),
            feature_columns=["paper_age_days"],
        ),
        ModelSpec(
            name="age_only_linear",
            estimator=Pipeline(
                steps=[
                    ("preprocess", age_only_preprocessor),
                    ("model", LinearRegression()),
                ]
            ),
            feature_columns=["paper_age_days"],
        ),
        ModelSpec(
            name="structured_ridge",
            estimator=Pipeline(
                steps=[
                    ("preprocess", linear_preprocessor),
                    ("model", Ridge(alpha=50.0)),
                ]
            ),
            feature_columns=STRUCTURED_NUMERIC_COLUMNS + LINEAR_CATEGORICAL_COLUMNS,
        ),
        ModelSpec(
            name="structured_random_forest",
            estimator=Pipeline(
                steps=[
                    ("preprocess", tree_preprocessor),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_features="sqrt",
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            feature_columns=STRUCTURED_NUMERIC_COLUMNS + TREE_CATEGORICAL_COLUMNS,
        ),
        ModelSpec(
            name="structured_hist_gradient_boosting",
            estimator=Pipeline(
                steps=[
                    ("preprocess", tree_preprocessor),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            learning_rate=0.05,
                            max_iter=400,
                            max_depth=8,
                            min_samples_leaf=20,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            feature_columns=STRUCTURED_NUMERIC_COLUMNS + TREE_CATEGORICAL_COLUMNS,
        ),
        ModelSpec(
            name="text_plus_metadata_ridge",
            estimator=Pipeline(
                steps=[
                    ("preprocess", text_preprocessor),
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
            feature_columns=STRUCTURED_NUMERIC_COLUMNS + LINEAR_CATEGORICAL_COLUMNS + ["text_for_model"],
        ),
    ]
    return specs


def build_wrapped_estimator(spec: ModelSpec) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=clone(spec.estimator),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def precision_at_top_decile(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    k = max(1, int(round(len(y_true) * 0.1)))
    top_true = np.argsort(y_true)[-k:]
    top_pred = np.argsort(y_pred)[-k:]
    return float(len(set(top_true) & set(top_pred)) / k)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    upper_clip: float | None = None,
) -> dict[str, float]:
    clipped = np.clip(y_pred, a_min=0, a_max=upper_clip)
    if np.allclose(clipped, clipped[0]) or np.allclose(y_true, y_true[0]):
        spearman_value = 0.0
    else:
        spearman_value = spearmanr(y_true, clipped).statistic
        if np.isnan(spearman_value):
            spearman_value = 0.0
    return {
        "mae_raw": float(mean_absolute_error(y_true, clipped)),
        "rmse_raw": float(np.sqrt(mean_squared_error(y_true, clipped))),
        "r2_raw": float(r2_score(y_true, clipped)),
        "r2_log": float(r2_score(np.log1p(y_true), np.log1p(clipped))),
        "spearman": float(spearman_value),
        "top_decile_precision": precision_at_top_decile(y_true, clipped),
    }


def benchmark_models(
    df: pd.DataFrame,
    model_specs: list[ModelSpec],
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    y_train = train_df["cited_by_count"].to_numpy(dtype=float)
    y_test = test_df["cited_by_count"].to_numpy(dtype=float)

    metrics_rows: list[dict[str, float | str | int]] = []
    prediction_frames: list[pd.DataFrame] = []

    for spec in model_specs:
        estimator = build_wrapped_estimator(spec)
        X_train = train_df[spec.feature_columns]
        X_test = test_df[spec.feature_columns]
        upper_clip = float(y_train.max())

        started_at = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started_at

        y_pred = estimator.predict(X_test)
        clipped_predictions = np.clip(y_pred, a_min=0, a_max=upper_clip)
        metrics = evaluate_predictions(y_test, y_pred, upper_clip=upper_clip)
        metrics_rows.append(
            {
                "model": spec.name,
                "fit_seconds": round(fit_seconds, 3),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                **metrics,
            }
        )

        prediction_frame = pd.DataFrame(
            {
                "model": spec.name,
                "work_id": test_df["work_id"].to_numpy(),
                "title": test_df["title"].to_numpy(),
                "actual_citations": y_test,
                "predicted_citations": clipped_predictions,
            }
        )
        prediction_frames.append(prediction_frame)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["r2_log", "spearman", "top_decile_precision"],
        ascending=[False, False, False],
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return metrics_df, predictions_df


def build_report(
    metrics_df: pd.DataFrame,
    dataset: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> str:
    best = metrics_df.iloc[0]
    summary = {
        "snapshot_date": str(SNAPSHOT_DATE.date()),
        "database_path": str(args.db_path.resolve()),
        "papers": int(len(dataset)),
        "citation_mean": float(dataset["cited_by_count"].mean()),
        "citation_median": float(dataset["cited_by_count"].median()),
        "citation_90p": float(dataset["cited_by_count"].quantile(0.9)),
        "citation_99p": float(dataset["cited_by_count"].quantile(0.99)),
        "best_model": best["model"],
        "best_r2_log": float(best["r2_log"]),
        "best_spearman": float(best["spearman"]),
        "best_top_decile_precision": float(best["top_decile_precision"]),
    }

    report_lines = [
        "# Citation Prediction Benchmark",
        "",
        "## Dataset",
        f"- Papers: {summary['papers']:,}",
        f"- Snapshot date for citation counts: {summary['snapshot_date']}",
        f"- Mean citations: {summary['citation_mean']:.2f}",
        f"- Median citations: {summary['citation_median']:.2f}",
        f"- 90th percentile citations: {summary['citation_90p']:.2f}",
        f"- 99th percentile citations: {summary['citation_99p']:.2f}",
        "",
        "## Best Quick Model",
        f"- Model: `{summary['best_model']}`",
        f"- Test R2 on log1p(citations): {summary['best_r2_log']:.4f}",
        f"- Test Spearman rank correlation: {summary['best_spearman']:.4f}",
        f"- Top-decile precision: {summary['best_top_decile_precision']:.4f}",
        "",
        "## Notes",
        "- This is a quick held-out benchmark, not an exhaustive hyperparameter search.",
        "- Age and venue metadata carry strong signal because the target is current cumulative citations.",
        "- A stricter next step would be a temporal split that predicts future citations for newer papers.",
        "",
        "## Artifacts",
        f"- Metrics CSV: `{(output_dir / 'citation_prediction_metrics.csv').resolve()}`",
        f"- Predictions CSV: `{(output_dir / 'citation_prediction_predictions.csv').resolve()}`",
        f"- Summary JSON: `{(output_dir / 'citation_prediction_summary.json').resolve()}`",
    ]

    (output_dir / "citation_prediction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return "\n".join(report_lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = engineer_features(load_dataset(args.db_path.resolve()))
    if args.sample_size:
        dataset = dataset.sample(n=min(args.sample_size, len(dataset)), random_state=args.random_state)

    model_specs = build_model_specs(max_text_features=args.max_text_features)
    metrics_df, predictions_df = benchmark_models(
        df=dataset,
        model_specs=model_specs,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    metrics_path = output_dir / "citation_prediction_metrics.csv"
    predictions_path = output_dir / "citation_prediction_predictions.csv"
    report_path = output_dir / "citation_prediction_report.md"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    report_path.write_text(build_report(metrics_df, dataset, output_dir, args), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print()
    print(f"Wrote metrics to: {metrics_path}")
    print(f"Wrote predictions to: {predictions_path}")
    print(f"Wrote report to: {report_path}")


if __name__ == "__main__":
    main()
