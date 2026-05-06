from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.openalex import OpenAlexClient, short_openalex_id
from src.storage import connect_duckdb, ensure_dirs, save_parquet, write_table


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def entity_ref(entity: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not isinstance(entity, dict):
        return None, None
    return short_openalex_id(entity.get("id")), entity.get("display_name")


def fetch_all(client: OpenAlexClient, endpoint: str, per_page: int) -> list[dict[str, Any]]:
    params = {"per-page": per_page}
    return list(tqdm(client.paginate(endpoint, params), desc=f"Fetching {endpoint}"))


def normalize_domains(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "domain_id": short_openalex_id(row.get("id")),
                "domain_display_name": row.get("display_name"),
                "description": row.get("description"),
                "works_count": row.get("works_count"),
                "cited_by_count": row.get("cited_by_count"),
            }
        )
    return pd.DataFrame(records).dropna(subset=["domain_id"]).drop_duplicates("domain_id")


def normalize_fields(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        domain_id, domain_name = entity_ref(row.get("domain"))
        records.append(
            {
                "field_id": short_openalex_id(row.get("id")),
                "field_display_name": row.get("display_name"),
                "description": row.get("description"),
                "domain_id": domain_id,
                "domain_display_name": domain_name,
                "works_count": row.get("works_count"),
                "cited_by_count": row.get("cited_by_count"),
            }
        )
    return pd.DataFrame(records).dropna(subset=["field_id"]).drop_duplicates("field_id")


def normalize_subfields(
    rows: list[dict[str, Any]], fields_df: pd.DataFrame
) -> pd.DataFrame:
    records = []
    for row in rows:
        field_id, field_name = entity_ref(row.get("field"))
        domain_id, domain_name = entity_ref(row.get("domain"))
        records.append(
            {
                "subfield_id": short_openalex_id(row.get("id")),
                "subfield_display_name": row.get("display_name"),
                "description": row.get("description"),
                "field_id": field_id,
                "field_display_name": field_name,
                "domain_id": domain_id,
                "domain_display_name": domain_name,
                "works_count": row.get("works_count"),
                "cited_by_count": row.get("cited_by_count"),
            }
        )

    subfields = (
        pd.DataFrame(records)
        .dropna(subset=["subfield_id"])
        .drop_duplicates("subfield_id")
    )

    if not fields_df.empty:
        field_lookup = fields_df[
            ["field_id", "field_display_name", "domain_id", "domain_display_name"]
        ].copy()
        subfields = subfields.merge(
            field_lookup,
            on="field_id",
            how="left",
            suffixes=("", "_from_field"),
        )
        for column in ["field_display_name", "domain_id", "domain_display_name"]:
            fallback = f"{column}_from_field"
            if fallback in subfields.columns:
                subfields[column] = subfields[column].fillna(subfields[fallback])
                subfields = subfields.drop(columns=[fallback])

    return subfields


def main() -> None:
    config = load_config()
    ensure_dirs(config)
    per_page = int(config["openalex"].get("per_page", 200))
    interim_dir = ROOT / config["storage"]["interim_dir"]
    duckdb_path = ROOT / config["storage"]["duckdb_path"]

    client = OpenAlexClient.from_config(config)

    domains_raw = fetch_all(client, "domains", per_page)
    fields_raw = fetch_all(client, "fields", per_page)
    subfields_raw = fetch_all(client, "subfields", per_page)

    domains = normalize_domains(domains_raw)
    fields = normalize_fields(fields_raw)
    subfields = normalize_subfields(subfields_raw, fields)

    save_parquet(domains, interim_dir / "domains.parquet")
    save_parquet(fields, interim_dir / "fields.parquet")
    save_parquet(subfields, interim_dir / "subfields.parquet")

    con = connect_duckdb(duckdb_path)
    try:
        write_table(con, "domains", domains)
        write_table(con, "fields", fields)
        write_table(con, "subfields", subfields)
    finally:
        con.close()

    print(f"Wrote {len(domains)} domains, {len(fields)} fields, {len(subfields)} subfields.")


if __name__ == "__main__":
    main()
