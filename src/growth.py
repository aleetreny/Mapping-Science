from __future__ import annotations

import math

import pandas as pd


def compute_log_relative_growth(
    past_share: float, future_share: float, epsilon: float = 1e-9
) -> float:
    """Compute log growth in share with a small epsilon for zero shares."""
    return math.log((future_share + epsilon) / (past_share + epsilon))


def assign_above_median_within_group(
    df: pd.DataFrame,
    growth_col: str,
    group_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Add a boolean column marking values above their group median."""
    result = df.copy()
    if output_col is None:
        output_col = f"{growth_col}_above_{group_col}_median"

    medians = result.groupby(group_col, dropna=False)[growth_col].transform("median")
    result[output_col] = result[growth_col] > medians
    return result
