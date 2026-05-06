import math

import pandas as pd

from src.growth import assign_above_median_within_group, compute_log_relative_growth


def test_compute_log_relative_growth() -> None:
    result = compute_log_relative_growth(0.10, 0.20)

    assert math.isclose(result, math.log(2.0), rel_tol=1e-6)


def test_compute_log_relative_growth_handles_zero_share() -> None:
    result = compute_log_relative_growth(0.0, 0.10)

    assert result > 0
    assert math.isfinite(result)


def test_assign_above_median_within_group() -> None:
    df = pd.DataFrame(
        {
            "field_id": ["A", "A", "A", "B", "B"],
            "growth": [0.0, 1.0, 2.0, -1.0, 1.0],
        }
    )

    result = assign_above_median_within_group(
        df, "growth", "field_id", "above_median"
    )

    assert result["above_median"].tolist() == [False, False, True, False, True]
