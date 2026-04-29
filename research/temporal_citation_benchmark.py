from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "openalex_stats_probability" / "corpus.sqlite"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "research" / "results"
SNAPSHOT_DATE = pd.Timestamp("2026-04-24")
RANDOM_STATE = 42


WORKS_SQL = """
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


AUTHORSHIPS_SQL = """
SELECT
    work_id,
    author_id,
    countries_json,
    institutions_json
FROM work_authorships
"""


COUNT_COLUMNS = [
    "referenced_works_count",
    "locations_count",
    "author_count",
    "corresponding_author_count",
    "topic_count",
    "title_char_count",
    "title_word_count",
    "abstract_char_count",
    "abstract_word_count",
    "known_author_count",
    "unique_country_count",
    "unique_institution_count",
]

BASE_NUMERIC_COLUMNS = [
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
    "known_author_count",
    "unique_country_count",
    "unique_institution_count",
    "has_international_collaboration",
    "author_id_coverage",
    "references_per_author",
    "institutions_per_author",
    "countries_per_author",
]

CALENDAR_COLUMNS = [
    "paper_age_days",
    "publication_year",
]

LOG_COUNT_COLUMNS = [f"log_{column}" for column in COUNT_COLUMNS]

PRIOR_COLUMNS = [
    "source_prior_mean_log",
    "source_prior_count",
    "host_org_prior_mean_log",
    "host_org_prior_count",
    "topic_prior_mean_log",
    "topic_prior_count",
    "field_prior_mean_log",
    "field_prior_count",
    "author_prior_mean_log_mean",
    "author_prior_mean_log_max",
    "author_prior_count_mean",
    "author_prior_count_max",
    "author_prior_seen_fraction",
]

CATEGORICAL_COLUMNS = [
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
        description="Temporal citation prediction benchmark with skew-aware diagnostics."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--test-end-year", type=int, default=2023)
    parser.add_argument("--max-text-features", type=int, default=15000)
    parser.add_argument("--text-components", type=int, default=64)
    return parser.parse_args()


def load_data(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        works = pd.read_sql_query(WORKS_SQL, conn)
        authorships = pd.read_sql_query(AUTHORSHIPS_SQL, conn)
    return works, authorships


def safe_json_loads(raw_value: object) -> list:
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and np.isnan(raw_value):
        return []
    if not isinstance(raw_value, (str, bytes, bytearray)):
        return []
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def build_authorship_features(authorships: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    work_authors: dict[str, list[str]] = defaultdict(list)
    feature_state: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "known_author_count": 0,
            "countries": set(),
            "institutions": set(),
        }
    )

    for row in authorships.itertuples(index=False):
        state = feature_state[row.work_id]

        if row.author_id:
            author_id = str(row.author_id)
            work_authors[row.work_id].append(author_id)
            state["known_author_count"] = int(state["known_author_count"]) + 1

        for country in safe_json_loads(row.countries_json):
            if country:
                state["countries"].add(str(country))

        for institution in safe_json_loads(row.institutions_json):
            if isinstance(institution, dict):
                institution_id = institution.get("id") or institution.get("display_name")
            else:
                institution_id = institution
            if institution_id:
                state["institutions"].add(str(institution_id))

    records = []
    for work_id, state in feature_state.items():
        unique_country_count = len(state["countries"])
        unique_institution_count = len(state["institutions"])
        records.append(
            {
                "work_id": work_id,
                "known_author_count": int(state["known_author_count"]),
                "unique_country_count": unique_country_count,
                "unique_institution_count": unique_institution_count,
                "has_international_collaboration": int(unique_country_count > 1),
            }
        )

    authorship_features = pd.DataFrame.from_records(records)
    return authorship_features, work_authors


def engineer_features(works: pd.DataFrame, authorship_features: pd.DataFrame) -> pd.DataFrame:
    df = works.merge(authorship_features, on="work_id", how="left")
    df["title"] = df["title"].fillna("")
    df["abstract_text"] = df["abstract_text"].fillna("")
    df["text_for_model"] = (df["title"] + ". " + df["abstract_text"]).str.strip()

    publication_date = pd.to_datetime(df["publication_date"], errors="coerce")
    publication_year = pd.to_numeric(df["publication_year"], errors="coerce").fillna(0).astype(int)
    fallback_date = pd.to_datetime(publication_year.astype(str) + "-07-01", errors="coerce")
    resolved_date = publication_date.fillna(fallback_date)
    df["publication_year"] = publication_year
    df["paper_age_days"] = (SNAPSHOT_DATE - resolved_date).dt.days.clip(lower=0)

    df["title_char_count"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.count(r"\S+")
    df["abstract_char_count"] = df["abstract_text"].str.len()
    df["abstract_word_count"] = df["abstract_text"].str.count(r"\S+")

    fill_zero_columns = [
        "known_author_count",
        "unique_country_count",
        "unique_institution_count",
        "has_international_collaboration",
    ]
    for column in fill_zero_columns:
        df[column] = df[column].fillna(0)

    numeric_columns = list(
        dict.fromkeys(
            COUNT_COLUMNS
            + [
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
                "avg_topic_score",
                "max_topic_score",
            ]
            + CALENDAR_COLUMNS
        )
    )
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    author_denominator = df["author_count"].clip(lower=1.0)
    df["author_id_coverage"] = df["known_author_count"] / author_denominator
    df["references_per_author"] = df["referenced_works_count"] / author_denominator
    df["institutions_per_author"] = df["unique_institution_count"] / author_denominator
    df["countries_per_author"] = df["unique_country_count"] / author_denominator

    for column in COUNT_COLUMNS:
        df[f"log_{column}"] = np.log1p(df[column].clip(lower=0.0))

    return df


def split_temporal(
    df: pd.DataFrame,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = df["publication_year"] <= train_end_year
    test_mask = (df["publication_year"] >= test_start_year) & (df["publication_year"] <= test_end_year)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split produced an empty train or test set.")

    return train_df, test_df


def apply_group_prior_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column_name: str,
    prefix: str,
    global_mean: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_stats = train_df.groupby(column_name, dropna=False)["target_log"].agg(["sum", "count"]).reset_index()
    temp_sum_column = f"__{prefix}_sum"
    temp_count_column = f"__{prefix}_count"
    group_stats = group_stats.rename(columns={"sum": temp_sum_column, "count": temp_count_column})

    train_df = train_df.merge(group_stats, on=column_name, how="left")
    test_df = test_df.merge(group_stats, on=column_name, how="left")

    train_sum = train_df[temp_sum_column].fillna(0.0)
    train_count = train_df[temp_count_column].fillna(0.0)
    test_sum = test_df[temp_sum_column].fillna(0.0)
    test_count = test_df[temp_count_column].fillna(0.0)

    train_df[f"{prefix}_prior_mean_log"] = np.where(
        train_count > 1,
        (train_sum - train_df["target_log"]) / (train_count - 1),
        global_mean,
    )
    train_df[f"{prefix}_prior_count"] = np.maximum(train_count - 1, 0)

    test_df[f"{prefix}_prior_mean_log"] = np.where(test_count > 0, test_sum / test_count, global_mean)
    test_df[f"{prefix}_prior_count"] = test_count

    drop_columns = [temp_sum_column, temp_count_column]
    train_df = train_df.drop(columns=drop_columns)
    test_df = test_df.drop(columns=drop_columns)
    return train_df, test_df


def add_author_prior_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    work_authors: dict[str, list[str]],
    global_mean: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    author_sum: dict[str, float] = defaultdict(float)
    author_count: dict[str, int] = defaultdict(int)
    train_target_map = train_df.set_index("work_id")["target_log"].to_dict()

    for work_id, target_log in train_target_map.items():
        for author_id in work_authors.get(work_id, []):
            author_sum[author_id] += float(target_log)
            author_count[author_id] += 1

    def build_features(frame: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        records = []
        for row in frame.itertuples(index=False):
            target_log = float(row.target_log) if is_train else None
            prior_means = []
            prior_counts = []
            authors = work_authors.get(row.work_id, [])

            for author_id in authors:
                total_sum = author_sum.get(author_id, 0.0)
                total_count = author_count.get(author_id, 0)
                if is_train and target_log is not None:
                    total_sum -= target_log
                    total_count -= 1
                if total_count > 0:
                    prior_means.append(total_sum / total_count)
                    prior_counts.append(total_count)

            if prior_means:
                mean_value = float(np.mean(prior_means))
                max_value = float(np.max(prior_means))
                count_mean = float(np.mean(prior_counts))
                count_max = float(np.max(prior_counts))
                seen_fraction = float(len(prior_means) / max(len(authors), 1))
            else:
                mean_value = global_mean
                max_value = global_mean
                count_mean = 0.0
                count_max = 0.0
                seen_fraction = 0.0

            records.append(
                {
                    "work_id": row.work_id,
                    "author_prior_mean_log_mean": mean_value,
                    "author_prior_mean_log_max": max_value,
                    "author_prior_count_mean": count_mean,
                    "author_prior_count_max": count_max,
                    "author_prior_seen_fraction": seen_fraction,
                }
            )

        return pd.DataFrame.from_records(records)

    train_features = build_features(train_df, is_train=True)
    test_features = build_features(test_df, is_train=False)
    train_df = train_df.merge(train_features, on="work_id", how="left")
    test_df = test_df.merge(test_features, on="work_id", how="left")
    return train_df, test_df


def add_historical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    work_authors: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["target_log"] = np.log1p(train_df["cited_by_count"].astype(float))
    test_df["target_log"] = np.log1p(test_df["cited_by_count"].astype(float))
    global_mean = float(train_df["target_log"].mean())

    group_columns = [
        ("primary_source_name", "source"),
        ("primary_source_host_org_name", "host_org"),
        ("primary_topic_name", "topic"),
        ("primary_field_name", "field"),
    ]

    for column_name, prefix in group_columns:
        train_df, test_df = apply_group_prior_features(train_df, test_df, column_name, prefix, global_mean)

    train_df, test_df = add_author_prior_features(train_df, test_df, work_authors, global_mean)
    return train_df, test_df


def text_selector() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "select",
                FunctionTransformer(
                    lambda frame: frame.iloc[:, 0].fillna("").astype(str),
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            )
        ]
    )


def build_model_specs(max_text_features: int, text_components: int) -> list[ModelSpec]:
    numeric_with_age = BASE_NUMERIC_COLUMNS + LOG_COUNT_COLUMNS + PRIOR_COLUMNS + CALENDAR_COLUMNS
    numeric_no_age = BASE_NUMERIC_COLUMNS + LOG_COUNT_COLUMNS + PRIOR_COLUMNS
    base_numeric_no_age = BASE_NUMERIC_COLUMNS + LOG_COUNT_COLUMNS

    def make_preprocessor(
        numeric_columns: list[str],
        categorical_columns: list[str],
        include_text: bool,
    ) -> ColumnTransformer:
        transformers: list[tuple[str, Pipeline, list[str]]] = [
            (
                "numeric",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]),
                numeric_columns,
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
                categorical_columns,
            ),
        ]

        if include_text:
            transformers.append(
                (
                    "text",
                    Pipeline(
                        steps=[
                            ("select", text_selector()),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    max_features=max_text_features,
                                    min_df=5,
                                    stop_words="english",
                                ),
                            ),
                            ("svd", TruncatedSVD(n_components=text_components, random_state=RANDOM_STATE)),
                        ]
                    ),
                    ["text_for_model"],
                )
            )

        return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)

    def make_hgb(preprocessor: ColumnTransformer) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=8,
                        max_iter=450,
                        min_samples_leaf=20,
                        l2_regularization=0.25,
                        early_stopping=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    minimal_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]),
                ["paper_age_days", "publication_year"],
            )
        ],
        sparse_threshold=0.0,
    )

    specs = [
        ModelSpec(
            name="constant_zero",
            estimator=Pipeline(
                steps=[
                    ("preprocess", minimal_preprocessor),
                    ("model", DummyRegressor(strategy="constant", constant=0.0)),
                ]
            ),
            feature_columns=["paper_age_days", "publication_year"],
        ),
        ModelSpec(
            name="dummy_median",
            estimator=Pipeline(
                steps=[
                    ("preprocess", minimal_preprocessor),
                    ("model", DummyRegressor(strategy="median")),
                ]
            ),
            feature_columns=["paper_age_days", "publication_year"],
        ),
        ModelSpec(
            name="age_only_linear",
            estimator=Pipeline(
                steps=[
                    ("preprocess", minimal_preprocessor),
                    ("model", LinearRegression()),
                ]
            ),
            feature_columns=["paper_age_days", "publication_year"],
        ),
        ModelSpec(
            name="metadata_hgb_no_age",
            estimator=make_hgb(make_preprocessor(base_numeric_no_age, CATEGORICAL_COLUMNS, include_text=False)),
            feature_columns=base_numeric_no_age + CATEGORICAL_COLUMNS,
        ),
        ModelSpec(
            name="metadata_hgb_with_age",
            estimator=make_hgb(
                make_preprocessor(base_numeric_no_age + CALENDAR_COLUMNS, CATEGORICAL_COLUMNS, include_text=False)
            ),
            feature_columns=base_numeric_no_age + CALENDAR_COLUMNS + CATEGORICAL_COLUMNS,
        ),
        ModelSpec(
            name="metadata_hgb_text_no_age",
            estimator=make_hgb(make_preprocessor(base_numeric_no_age, CATEGORICAL_COLUMNS, include_text=True)),
            feature_columns=base_numeric_no_age + CATEGORICAL_COLUMNS + ["text_for_model"],
        ),
        ModelSpec(
            name="metadata_hgb_text_with_age",
            estimator=make_hgb(
                make_preprocessor(base_numeric_no_age + CALENDAR_COLUMNS, CATEGORICAL_COLUMNS, include_text=True)
            ),
            feature_columns=base_numeric_no_age + CALENDAR_COLUMNS + CATEGORICAL_COLUMNS + ["text_for_model"],
        ),
        ModelSpec(
            name="enhanced_hgb_priors_no_age",
            estimator=make_hgb(make_preprocessor(numeric_no_age, CATEGORICAL_COLUMNS, include_text=False)),
            feature_columns=numeric_no_age + CATEGORICAL_COLUMNS,
        ),
        ModelSpec(
            name="enhanced_hgb_priors_text_no_age",
            estimator=make_hgb(make_preprocessor(numeric_no_age, CATEGORICAL_COLUMNS, include_text=True)),
            feature_columns=numeric_no_age + CATEGORICAL_COLUMNS + ["text_for_model"],
        ),
        ModelSpec(
            name="enhanced_hgb_priors_text_with_age",
            estimator=make_hgb(make_preprocessor(numeric_with_age, CATEGORICAL_COLUMNS, include_text=True)),
            feature_columns=numeric_with_age + CATEGORICAL_COLUMNS + ["text_for_model"],
        ),
    ]
    return specs


def wrap_estimator(estimator: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=clone(estimator),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def top_k_metrics(y_true: np.ndarray, y_pred: np.ndarray, fraction: float = 0.1) -> tuple[float, float]:
    k = max(1, int(round(len(y_true) * fraction)))
    top_true = set(np.argsort(y_true)[-k:])
    top_pred = set(np.argsort(y_pred)[-k:])
    overlap = len(top_true & top_pred)
    precision = overlap / k
    recall = overlap / k
    return float(precision), float(recall)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0
    value = spearmanr(y_true, y_pred).statistic
    return 0.0 if np.isnan(value) else float(value)


def safe_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y_true_binary))
    negatives = int(len(y_true_binary) - positives)
    if positives == 0 or negatives == 0:
        return 0.0
    return float(roc_auc_score(y_true_binary, scores))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    clipped = np.clip(y_pred, a_min=0.0, a_max=float(np.max(y_true)) if len(y_true) else None)
    top10_precision, top10_recall = top_k_metrics(y_true, clipped, fraction=0.1)

    high_impact_threshold = float(np.quantile(y_true, 0.9))
    high_impact_binary = (y_true >= high_impact_threshold).astype(int)
    gt10_binary = (y_true > 10).astype(int)

    gt10_mask = y_true > 10
    gt40_mask = y_true > 40

    metrics = {
        "mae_raw": float(mean_absolute_error(y_true, clipped)),
        "rmse_raw": float(np.sqrt(mean_squared_error(y_true, clipped))),
        "r2_raw": float(r2_score(y_true, clipped)),
        "r2_log": float(r2_score(np.log1p(y_true), np.log1p(clipped))),
        "spearman": safe_spearman(y_true, clipped),
        "within_10": float(np.mean(np.abs(y_true - clipped) <= 10)),
        "within_20": float(np.mean(np.abs(y_true - clipped) <= 20)),
        "top10_precision": top10_precision,
        "top10_recall": top10_recall,
        "top10_average_precision": float(average_precision_score(high_impact_binary, clipped)),
        "top10_auc": safe_auc(high_impact_binary, clipped),
        "gt10_average_precision": float(average_precision_score(gt10_binary, clipped)),
        "gt10_auc": safe_auc(gt10_binary, clipped),
        "pred_share_le_10": float(np.mean(clipped <= 10)),
        "pred_median": float(np.median(clipped)),
    }

    metrics["mae_actual_gt_10"] = float(mean_absolute_error(y_true[gt10_mask], clipped[gt10_mask])) if np.any(gt10_mask) else 0.0
    metrics["mae_actual_gt_40"] = float(mean_absolute_error(y_true[gt40_mask], clipped[gt40_mask])) if np.any(gt40_mask) else 0.0
    return metrics


def benchmark_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_specs: list[ModelSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = train_df["cited_by_count"].to_numpy(dtype=float)
    y_test = test_df["cited_by_count"].to_numpy(dtype=float)

    metrics_rows: list[dict[str, float | str | int]] = []
    prediction_frames: list[pd.DataFrame] = []

    for spec in model_specs:
        model = wrap_estimator(spec.estimator)
        X_train = train_df[spec.feature_columns]
        X_test = test_df[spec.feature_columns]

        started_at = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started_at

        predictions = np.asarray(model.predict(X_test), dtype=float)
        predictions = np.clip(predictions, a_min=0.0, a_max=float(np.max(y_train)))
        metrics = evaluate_predictions(y_test, predictions)

        metrics_rows.append(
            {
                "model": spec.name,
                "fit_seconds": round(fit_seconds, 3),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                **metrics,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": spec.name,
                    "work_id": test_df["work_id"].to_numpy(),
                    "publication_year": test_df["publication_year"].to_numpy(),
                    "title": test_df["title"].to_numpy(),
                    "actual_citations": y_test,
                    "predicted_citations": predictions,
                }
            )
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["r2_log", "spearman", "top10_average_precision"],
        ascending=[False, False, False],
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return metrics_df, predictions_df


def make_distribution_summary(frame: pd.DataFrame) -> dict[str, float]:
    citations = frame["cited_by_count"].astype(float)
    return {
        "papers": int(len(frame)),
        "mean": float(citations.mean()),
        "median": float(citations.median()),
        "p90": float(citations.quantile(0.9)),
        "p99": float(citations.quantile(0.99)),
        "share_le_0": float((citations <= 0).mean()),
        "share_le_5": float((citations <= 5).mean()),
        "share_le_10": float((citations <= 10).mean()),
        "share_le_20": float((citations <= 20).mean()),
        "share_gt_40": float((citations > 40).mean()),
    }


def build_report(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    train_summary = make_distribution_summary(train_df)
    test_summary = make_distribution_summary(test_df)
    best_exact = metrics_df.sort_values(by=["r2_log", "spearman"], ascending=[False, False]).iloc[0]
    best_rank = metrics_df.sort_values(
        by=["top10_average_precision", "spearman", "gt10_average_precision"],
        ascending=[False, False, False],
    ).iloc[0]
    no_age_mask = metrics_df["model"].str.contains("no_age", regex=False)
    best_no_age = metrics_df.loc[no_age_mask].sort_values(
        by=["top10_average_precision", "spearman", "gt10_average_precision"],
        ascending=[False, False, False],
    ).iloc[0]
    zero_baseline = metrics_df.loc[metrics_df["model"] == "constant_zero"].iloc[0]

    lines = [
        "# Temporal Citation Benchmark",
        "",
        "## Split",
        f"- Train papers: publication_year <= {args.train_end_year}",
        f"- Test papers: publication_year in [{args.test_start_year}, {args.test_end_year}]",
        "",
        "## Train Distribution",
        f"- Papers: {train_summary['papers']:,}",
        f"- Mean citations: {train_summary['mean']:.2f}",
        f"- Median citations: {train_summary['median']:.2f}",
        f"- Share <= 10 citations: {train_summary['share_le_10']:.3f}",
        "",
        "## Test Distribution",
        f"- Papers: {test_summary['papers']:,}",
        f"- Mean citations: {test_summary['mean']:.2f}",
        f"- Median citations: {test_summary['median']:.2f}",
        f"- 90th percentile citations: {test_summary['p90']:.2f}",
        f"- Share <= 10 citations: {test_summary['share_le_10']:.3f}",
        f"- Share > 40 citations: {test_summary['share_gt_40']:.3f}",
        "",
        "## Best Quick Models",
        f"- Best exact-count model: `{best_exact['model']}` with `R2_log = {best_exact['r2_log']:.4f}` and `Spearman = {best_exact['spearman']:.4f}`.",
        f"- Best high-impact ranking model: `{best_rank['model']}` with `top10 AP = {best_rank['top10_average_precision']:.4f}`, `top10 precision = {best_rank['top10_precision']:.4f}`, and `gt10 AP = {best_rank['gt10_average_precision']:.4f}`.",
        f"- Best no-age model: `{best_no_age['model']}` with `Spearman = {best_no_age['spearman']:.4f}` and `top10 precision = {best_no_age['top10_precision']:.4f}`.",
        "",
        "## Why Raw Accuracy Can Mislead",
        f"- Constant-zero baseline predicts 0 citations for every paper.",
        f"- It still gets `within_10 = {zero_baseline['within_10']:.4f}` because the test set is skewed low.",
        f"- But its `r2_log = {zero_baseline['r2_log']:.4f}`, `spearman = {zero_baseline['spearman']:.4f}`, and `top10_precision = {zero_baseline['top10_precision']:.4f}`.",
        "- That is why ranking and high-impact retrieval metrics matter more here than a naive accuracy-style threshold.",
        "",
        "## Artifacts",
        f"- Metrics CSV: `{(output_dir / 'temporal_citation_metrics.csv').resolve()}`",
        f"- Predictions CSV: `{(output_dir / 'temporal_citation_predictions.csv').resolve()}`",
        f"- Summary JSON: `{(output_dir / 'temporal_citation_summary.json').resolve()}`",
    ]

    summary = {
        "snapshot_date": str(SNAPSHOT_DATE.date()),
        "train_end_year": args.train_end_year,
        "test_start_year": args.test_start_year,
        "test_end_year": args.test_end_year,
        "train_distribution": train_summary,
        "test_distribution": test_summary,
        "best_exact_model": best_exact["model"],
        "best_exact_r2_log": float(best_exact["r2_log"]),
        "best_exact_spearman": float(best_exact["spearman"]),
        "best_rank_model": best_rank["model"],
        "best_rank_top10_average_precision": float(best_rank["top10_average_precision"]),
        "best_rank_top10_precision": float(best_rank["top10_precision"]),
        "best_rank_gt10_average_precision": float(best_rank["gt10_average_precision"]),
        "best_no_age_model": best_no_age["model"],
        "best_no_age_spearman": float(best_no_age["spearman"]),
        "best_no_age_top10_precision": float(best_no_age["top10_precision"]),
        "constant_zero_within_10": float(zero_baseline["within_10"]),
        "constant_zero_top10_precision": float(zero_baseline["top10_precision"]),
    }
    (output_dir / "temporal_citation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    works, authorships = load_data(args.db_path.resolve())
    authorship_features, work_authors = build_authorship_features(authorships)
    dataset = engineer_features(works, authorship_features)
    train_df, test_df = split_temporal(
        dataset,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )
    train_df, test_df = add_historical_features(train_df, test_df, work_authors)

    model_specs = build_model_specs(
        max_text_features=args.max_text_features,
        text_components=args.text_components,
    )
    metrics_df, predictions_df = benchmark_models(train_df, test_df, model_specs)

    metrics_path = output_dir / "temporal_citation_metrics.csv"
    predictions_path = output_dir / "temporal_citation_predictions.csv"
    report_path = output_dir / "temporal_citation_report.md"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    report_path.write_text(build_report(args, train_df, test_df, metrics_df, output_dir), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print()
    print(f"Wrote metrics to: {metrics_path}")
    print(f"Wrote predictions to: {predictions_path}")
    print(f"Wrote report to: {report_path}")


if __name__ == "__main__":
    main()
