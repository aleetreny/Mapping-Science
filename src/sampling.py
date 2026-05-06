from __future__ import annotations

import math

import pandas as pd


def stratified_sample_by_year(
    df: pd.DataFrame,
    year_col: str,
    target_total: int,
    target_per_year: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample deterministically by year, redistributing unused yearly capacity."""
    if df.empty or target_total <= 0:
        return df.iloc[0:0].copy()

    if len(df) <= target_total:
        return df.copy()

    years = sorted(df[year_col].dropna().unique().tolist())
    counts = df.groupby(year_col).size().to_dict()
    quotas: dict[int, int] = {}

    for year in years:
        quotas[year] = min(int(counts.get(year, 0)), target_per_year)

    allocated = sum(quotas.values())
    remaining = min(target_total, len(df)) - allocated

    while remaining > 0:
        years_with_capacity = [
            year for year in years if quotas[year] < int(counts.get(year, 0))
        ]
        if not years_with_capacity:
            break

        step = max(1, math.floor(remaining / len(years_with_capacity)))
        for year in years_with_capacity:
            if remaining <= 0:
                break
            capacity = int(counts.get(year, 0)) - quotas[year]
            add = min(capacity, step, remaining)
            quotas[year] += add
            remaining -= add

    sampled_parts = []
    for position, year in enumerate(years):
        quota = quotas.get(year, 0)
        if quota <= 0:
            continue
        group = df[df[year_col] == year]
        if len(group) <= quota:
            sampled_parts.append(group)
        else:
            sampled_parts.append(
                group.sample(n=quota, random_state=random_seed + position)
            )

    if not sampled_parts:
        return df.iloc[0:0].copy()

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > target_total:
        sampled = sampled.sample(n=target_total, random_state=random_seed)
    return sampled.reset_index(drop=True)
