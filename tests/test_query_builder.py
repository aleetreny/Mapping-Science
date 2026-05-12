import pytest

from src.openalex import (
    DEFAULT_SELECT_FIELDS,
    OpenAlexClient,
    OpenAlexRateLimitError,
    build_count_query_params,
    build_sampled_text_query_params,
    build_text_query_params,
    load_repo_env,
    retry_after_seconds,
    safe_url_for_logging,
)


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None, url="https://api.openalex.org/works"):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.url = url

    def json(self):
        return self._payload


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
    assert "topics" in params["select"]


def test_default_select_fields_include_topics() -> None:
    assert "primary_topic" in DEFAULT_SELECT_FIELDS
    assert "topics" in DEFAULT_SELECT_FIELDS


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


def test_retry_after_seconds_prefers_header_value() -> None:
    response = FakeResponse(429, headers={"Retry-After": "7"})

    assert retry_after_seconds(response, fallback_seconds=180) == 7


def test_client_honors_retry_after_on_429_then_succeeds(monkeypatch) -> None:
    responses = [
        FakeResponse(429, headers={"Retry-After": "3"}),
        FakeResponse(200, payload={"results": [{"id": "W1"}]}),
    ]
    sleeps = []

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("src.openalex.requests.get", fake_get)
    monkeypatch.setattr("src.openalex.time.sleep", lambda seconds: sleeps.append(seconds))
    client = OpenAlexClient(
        max_requests_per_second=0,
        cooldown_after_429=120,
        max_consecutive_429=2,
    )

    data = client.get_json("works", {"filter": "x"})

    assert data["results"][0]["id"] == "W1"
    assert sleeps == [3.0]


def test_client_uses_cooldown_and_raises_after_repeated_429(monkeypatch) -> None:
    def fake_get(*args, **kwargs):
        return FakeResponse(429, headers={})

    sleeps = []
    monkeypatch.setattr("src.openalex.requests.get", fake_get)
    monkeypatch.setattr("src.openalex.time.sleep", lambda seconds: sleeps.append(seconds))
    client = OpenAlexClient(
        max_requests_per_second=0,
        cooldown_after_429=120,
        max_consecutive_429=1,
    )

    with pytest.raises(OpenAlexRateLimitError) as excinfo:
        client.get_json("works", {"filter": "x"})

    assert sleeps == [120.0]
    assert excinfo.value.consecutive_429 == 2
