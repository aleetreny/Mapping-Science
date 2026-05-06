from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from src.abstracts import reconstruct_abstract
from src.openalex import short_openalex_id


TOKEN_RE = re.compile(r"\b\w+\b")

WORKS_TEXT_COLUMNS = [
    "work_id",
    "doi",
    "title",
    "abstract",
    "text_for_embedding",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "primary_topic_id",
    "primary_topic_display_name",
    "topics_json",
    "title_token_count",
    "abstract_token_count",
    "text_token_count",
    "downloaded_at",
    "cited_by_count",
    "referenced_works_count",
]


def token_count(text: str | None) -> int:
    if not text:
        return 0
    return len(TOKEN_RE.findall(text))


def topic_ref(topic: dict[str, Any], name: str) -> tuple[str | None, str | None]:
    entity = topic.get(name)
    if not isinstance(entity, dict):
        return None, None
    return short_openalex_id(entity.get("id")), entity.get("display_name")


def validate_and_normalize_work(
    work: dict[str, Any],
    config: dict[str, Any],
    expected_subfield_id: str,
    expected_year: int,
) -> dict[str, Any] | None:
    filters = config["filters"]
    title = work.get("title") or work.get("display_name")
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    publication_year = work.get("publication_year")
    work_type = work.get("type")
    language = work.get("language")
    primary_topic = work.get("primary_topic") or {}
    title_tokens = token_count(title)
    abstract_tokens = token_count(abstract)

    if publication_year != int(expected_year):
        return None
    if language != filters["language"]:
        return None
    if work_type not in set(filters["work_types"]):
        return None
    if bool(work.get("is_retracted")):
        return None
    if bool(work.get("is_paratext")):
        return None
    if title_tokens < int(filters["min_title_tokens"]):
        return None
    if abstract_tokens < int(filters["min_abstract_tokens"]):
        return None

    subfield_id, subfield_name = topic_ref(primary_topic, "subfield")
    field_id, field_name = topic_ref(primary_topic, "field")
    domain_id, domain_name = topic_ref(primary_topic, "domain")
    primary_topic_id = short_openalex_id(primary_topic.get("id"))
    primary_topic_display_name = primary_topic.get("display_name")
    if not subfield_id or subfield_id != str(expected_subfield_id):
        return None

    work_id = short_openalex_id(work.get("id"))
    if not work_id:
        return None
    text_for_embedding = f"{title}\n\n{abstract}"

    return {
        "work_id": work_id,
        "doi": work.get("doi"),
        "title": title,
        "abstract": abstract,
        "text_for_embedding": text_for_embedding,
        "publication_year": publication_year,
        "publication_date": work.get("publication_date"),
        "type": work_type,
        "language": language,
        "subfield_id": subfield_id,
        "subfield_display_name": subfield_name,
        "field_id": field_id,
        "field_display_name": field_name,
        "domain_id": domain_id,
        "domain_display_name": domain_name,
        "primary_topic_id": primary_topic_id,
        "primary_topic_display_name": primary_topic_display_name,
        "topics_json": json.dumps(
            work.get("topics") or [], ensure_ascii=False, separators=(",", ":")
        ),
        "title_token_count": title_tokens,
        "abstract_token_count": abstract_tokens,
        "text_token_count": token_count(text_for_embedding),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "cited_by_count": work.get("cited_by_count"),
        "referenced_works_count": work.get("referenced_works_count"),
    }
