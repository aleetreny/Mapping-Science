import pytest

from src.embedding_space_metrics import validate_year_window
from src.metric_year_windows import resolve_metric_year_window
from src.morphology_metrics import validate_morphology_year_window


def test_2000_2024_metric_window_is_supported() -> None:
    validate_morphology_year_window(2000, 2024)
    validate_year_window(2000, 2024)


def test_2000_2024_temporal_edge_windows_are_first_and_last_five_years() -> None:
    window = resolve_metric_year_window(2000, 2024)

    assert (window.early_year_min, window.early_year_max) == (2000, 2004)
    assert (window.late_year_min, window.late_year_max) == (2020, 2024)


def test_legacy_metric_window_still_resolves() -> None:
    window = resolve_metric_year_window(2010, 2025)

    assert (window.early_year_min, window.early_year_max) == (2010, 2014)
    assert (window.late_year_min, window.late_year_max) == (2021, 2025)


def test_unsupported_metric_window_fails_clearly() -> None:
    with pytest.raises(ValueError, match="supported artifact window"):
        resolve_metric_year_window(1999, 2024)

    with pytest.raises(ValueError, match="year_min"):
        resolve_metric_year_window(2024, 2000)
