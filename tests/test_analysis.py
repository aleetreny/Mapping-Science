from src.analysis import eligibility_flags


def test_eligibility_flags_include_3000_work_subfields_in_main_analysis() -> None:
    flags = eligibility_flags(3000)

    assert flags["main_analysis_eligible_2500"] is True
    assert flags["robustness_eligible_500"] is True
    assert flags["is_low_sample"] is False
    assert flags["exclusion_reason"] == "main_analysis_included"


def test_eligibility_flags_include_2500_work_subfields_in_main_analysis() -> None:
    flags = eligibility_flags(2500)

    assert flags["main_analysis_eligible_2500"] is True
    assert flags["robustness_eligible_500"] is True
    assert flags["is_low_sample"] is False
    assert flags["exclusion_reason"] == "main_analysis_included"


def test_eligibility_flags_keep_2499_work_subfields_for_robustness_only() -> None:
    flags = eligibility_flags(2499)

    assert flags["main_analysis_eligible_2500"] is False
    assert flags["robustness_eligible_500"] is True
    assert flags["is_low_sample"] is True
    assert flags["exclusion_reason"] == "below_2500_valid_works"


def test_eligibility_flags_keep_500_work_subfields_for_robustness() -> None:
    flags = eligibility_flags(500)

    assert flags["main_analysis_eligible_2500"] is False
    assert flags["robustness_eligible_500"] is True
    assert flags["is_low_sample"] is True
    assert flags["exclusion_reason"] == "below_2500_valid_works"


def test_eligibility_flags_exclude_499_work_subfields_from_robustness() -> None:
    flags = eligibility_flags(499)

    assert flags["main_analysis_eligible_2500"] is False
    assert flags["robustness_eligible_500"] is False
    assert flags["is_low_sample"] is True
    assert flags["exclusion_reason"] == "below_500_valid_works"
