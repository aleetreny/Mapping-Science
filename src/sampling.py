from __future__ import annotations

import re
from typing import Any


def numeric_part_of_subfield_id(subfield_id: Any) -> int:
    """Extract a deterministic numeric value from an OpenAlex subfield id."""
    digits = re.findall(r"\d+", str(subfield_id))
    if not digits:
        return 0
    return int("".join(digits))


def stable_sample_seed(subfield_id: Any, publication_year: int, random_seed: int) -> int:
    """Build a stable seed for one subfield-year cell."""
    return int(random_seed) + int(publication_year) + numeric_part_of_subfield_id(subfield_id)


def allocate_yearly_sample_sizes(
    available_by_year: dict[int, int],
    target_total: int,
    target_per_year: int,
) -> dict[int, int]:
    """Allocate yearly sample sizes with simple redistribution of unused capacity."""
    years = sorted(available_by_year)
    allocations: dict[int, int] = {}

    for year in years:
        available = max(0, int(available_by_year.get(year, 0)))
        allocations[year] = min(available, target_per_year)

    remaining = min(target_total, sum(max(0, int(v)) for v in available_by_year.values()))
    remaining -= sum(allocations.values())

    while remaining > 0:
        years_with_capacity = [
            year
            for year in years
            if allocations[year] < max(0, int(available_by_year.get(year, 0)))
        ]
        if not years_with_capacity:
            break

        for year in years_with_capacity:
            if remaining <= 0:
                break
            available = max(0, int(available_by_year.get(year, 0)))
            capacity = available - allocations[year]
            if capacity <= 0:
                continue
            allocations[year] += 1
            remaining -= 1

    total_allocated = sum(allocations.values())
    if total_allocated > target_total:
        overflow = total_allocated - target_total
        for year in sorted(years, reverse=True):
            if overflow <= 0:
                break
            remove = min(allocations[year], overflow)
            allocations[year] -= remove
            overflow -= remove

    return allocations


def sampling_method_for_cell(
    available_valid_works: int,
    planned_sample_size: int,
    target_per_year: int,
    use_openalex_sample_api: bool = True,
) -> str:
    """Classify how one subfield-year cell should be downloaded."""
    if available_valid_works <= 0 or planned_sample_size <= 0:
        return "skip_no_available_works"
    if use_openalex_sample_api and available_valid_works >= target_per_year:
        return "sample_api"
    return "download_all_available"
