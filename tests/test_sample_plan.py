from src.sampling import (
    allocate_yearly_sample_sizes,
    api_initial_sample_size,
    sampling_method_for_cell,
    stable_backfill_seed,
    stable_sample_seed,
)


def test_sample_plan_redistribution_never_exceeds_subfield_target() -> None:
    available = {
        2010: 0,
        2011: 100,
        2012: 1000,
        2013: 1000,
        2014: 1000,
        2015: 1000,
        2016: 1000,
        2017: 1000,
        2018: 1000,
        2019: 1000,
    }

    allocations = allocate_yearly_sample_sizes(
        available_by_year=available,
        target_total=3000,
        target_per_year=300,
    )

    assert sum(allocations.values()) == 3000
    assert all(allocations[year] <= available[year] for year in available)
    assert allocations[2010] == 0
    assert allocations[2011] == 100


def test_sample_allocation_never_exceeds_available_works() -> None:
    available = {year: 50 for year in range(2010, 2020)}

    allocations = allocate_yearly_sample_sizes(
        available_by_year=available,
        target_total=3000,
        target_per_year=300,
    )

    assert sum(allocations.values()) == 500
    assert all(allocations[year] <= available[year] for year in available)


def test_api_initial_sample_size_oversamples_when_possible() -> None:
    initial = api_initial_sample_size(
        available_valid_works=1000,
        planned_sample_size=300,
        sampling_method="sample_api",
        oversample_factor=1.75,
    )

    assert initial == 525
    assert initial > 300
    assert (
        api_initial_sample_size(400, 300, "sample_api", 1.75) == 400
    )
    assert (
        api_initial_sample_size(120, 120, "download_all_available", 1.75) == 120
    )


def test_sampling_method_uses_sample_api_for_large_cells() -> None:
    assert (
        sampling_method_for_cell(
            available_valid_works=300,
            planned_sample_size=300,
            target_per_year=300,
            use_openalex_sample_api=True,
        )
        == "sample_api"
    )
    assert (
        sampling_method_for_cell(
            available_valid_works=120,
            planned_sample_size=120,
            target_per_year=300,
            use_openalex_sample_api=True,
        )
        == "download_all_available"
    )


def test_stable_sample_seed_is_deterministic() -> None:
    assert stable_sample_seed("https://openalex.org/subfields/2613", 2019, 42) == 4674
    assert stable_sample_seed("2613", 2019, 42) == 4674


def test_backfill_seed_increment_is_deterministic() -> None:
    assert stable_backfill_seed(4674, 1, 100000) == 104674
    assert stable_backfill_seed(4674, 4, 100000) == 404674
