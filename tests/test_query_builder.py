from src.openalex import (
    build_count_query_params,
    build_text_query_params,
    safe_url_for_logging,
)


def test_text_query_uses_primary_subfield_not_topics_subfield() -> None:
    params = build_text_query_params(
        subfield_id="2613",
        start_year=2010,
        end_year=2019,
        work_types=["article", "preprint"],
    )

    assert "primary_topic.subfield.id:2613" in params["filter"]
    assert "topics.subfield.id" not in params["filter"]


def test_text_query_includes_select() -> None:
    params = build_text_query_params(
        subfield_id="2613",
        start_year=2010,
        end_year=2019,
        work_types=["article", "preprint"],
    )

    assert "select" in params
    assert "abstract_inverted_index" in params["select"]
    assert "primary_topic" in params["select"]


def test_count_query_can_group_by_primary_subfield() -> None:
    params = build_count_query_params(
        year=2019,
        group_by="primary_topic.subfield.id",
        extra_filters=["type:article|preprint"],
    )

    assert params["group_by"] == "primary_topic.subfield.id"
    assert "publication_year:2019" in params["filter"]


def test_safe_url_for_logging_redacts_api_key() -> None:
    redacted = safe_url_for_logging(
        "https://api.openalex.org/works?api_key=secret-value&mailto=a@example.com&filter=x"
    )

    assert "secret-value" not in redacted
    assert "a@example.com" not in redacted
    assert "api_key=REDACTED" in redacted
