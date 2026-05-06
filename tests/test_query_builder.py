from src.openalex import (
    build_count_query_params,
    build_sampled_text_query_params,
    build_text_query_params,
    load_repo_env,
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
        "https://api.openalex.org/works?api_key=secret-value&mailto=a@example.com&contact_email=b@example.com&filter=x"
    )

    assert "secret-value" not in redacted
    assert "a@example.com" not in redacted
    assert "b@example.com" not in redacted
    assert "api_key=REDACTED" in redacted


def test_sampled_text_query_includes_sample_seed_and_select() -> None:
    params = build_sampled_text_query_params(
        subfield_id="https://openalex.org/subfields/2613",
        year=2019,
        sample_size=300,
        seed=4974,
        work_types=["article", "preprint"],
    )

    assert params["sample"] == 300
    assert params["seed"] == 4974
    assert params["per-page"] == 200
    assert "select" in params
    assert "primary_topic.subfield.id:2613" in params["filter"]
    assert "publication_year:2019" in params["filter"]
    assert "topics.subfield.id" not in params["filter"]


def test_env_loading_reads_dotenv_without_exposing_credentials(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENALEX_EMAIL=agent@example.com\nOPENALEX_API_KEY=secret-from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENALEX_EMAIL", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    assert load_repo_env(env_file)
    redacted = safe_url_for_logging(
        "https://api.openalex.org/works?api_key=secret-from-dotenv&mailto=agent@example.com"
    )

    assert "secret-from-dotenv" not in redacted
    assert "agent@example.com" not in redacted
