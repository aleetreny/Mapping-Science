from src.works import validate_and_normalize_work


def make_work(title: str, abstract_words: int) -> dict:
    abstract_index = {f"word{i}": [i] for i in range(abstract_words)}
    return {
        "id": "https://openalex.org/W123",
        "doi": None,
        "title": title,
        "display_name": title,
        "abstract_inverted_index": abstract_index,
        "publication_year": 2019,
        "publication_date": "2019-01-01",
        "type": "article",
        "language": "en",
        "primary_topic": {
            "subfield": {"id": "https://openalex.org/subfields/2613", "display_name": "Statistics"},
            "field": {"id": "https://openalex.org/fields/26", "display_name": "Mathematics"},
            "domain": {"id": "https://openalex.org/domains/4", "display_name": "Physical Sciences"},
        },
        "cited_by_count": 0,
        "referenced_works_count": 0,
        "is_retracted": False,
        "is_paratext": False,
    }


def config() -> dict:
    return {
        "filters": {
            "language": "en",
            "work_types": ["article", "preprint"],
            "min_title_tokens": 5,
            "min_abstract_tokens": 80,
        }
    }


def test_validate_rejects_short_title() -> None:
    work = make_work("Too short", 100)

    assert validate_and_normalize_work(work, config(), "2613", 2019) is None


def test_validate_rejects_short_abstract() -> None:
    work = make_work("A long enough title for validation", 20)

    assert validate_and_normalize_work(work, config(), "2613", 2019) is None


def test_validate_accepts_valid_work() -> None:
    work = make_work("A long enough title for validation", 100)

    normalized = validate_and_normalize_work(work, config(), "2613", 2019)

    assert normalized is not None
    assert normalized["work_id"] == "W123"
    assert normalized["subfield_id"] == "2613"
