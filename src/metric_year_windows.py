from __future__ import annotations

from dataclasses import asdict, dataclass


TEMPORAL_EDGE_YEARS = 5


@dataclass(frozen=True)
class MetricYearWindow:
    year_min: int
    year_max: int
    early_year_min: int
    early_year_max: int
    late_year_min: int
    late_year_max: int
    supported_window: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


SUPPORTED_METRIC_WINDOWS: tuple[tuple[int, int, str], ...] = (
    (2000, 2024, "2000_2024_400py"),
    (2010, 2025, "legacy_2010_2025"),
)


def supported_metric_window_labels() -> str:
    return ", ".join(
        f"{year_min}-{year_max}" for year_min, year_max, _ in SUPPORTED_METRIC_WINDOWS
    )


def resolve_metric_year_window(year_min: int, year_max: int) -> MetricYearWindow:
    start = int(year_min)
    end = int(year_max)
    if start > end:
        raise ValueError(f"year_min ({start}) must be <= year_max ({end})")

    matching_windows = [
        (supported_min, supported_max, label)
        for supported_min, supported_max, label in SUPPORTED_METRIC_WINDOWS
        if start >= supported_min and end <= supported_max
    ]
    if not matching_windows:
        raise ValueError(
            "metric year window must fall inside a supported artifact window "
            f"({supported_metric_window_labels()}); got {start}-{end}"
        )

    supported_min, supported_max, label = min(
        matching_windows,
        key=lambda item: (item[1] - item[0], item[0], item[1]),
    )
    del supported_min, supported_max
    early_end = min(start + TEMPORAL_EDGE_YEARS - 1, end)
    late_start = max(end - TEMPORAL_EDGE_YEARS + 1, start)
    return MetricYearWindow(
        year_min=start,
        year_max=end,
        early_year_min=start,
        early_year_max=early_end,
        late_year_min=late_start,
        late_year_max=end,
        supported_window=label,
    )


def validate_metric_year_window(year_min: int, year_max: int) -> None:
    resolve_metric_year_window(year_min, year_max)
