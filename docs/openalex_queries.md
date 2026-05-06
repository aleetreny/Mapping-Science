# OpenAlex Queries

The active pipeline uses `primary_topic.subfield.id` as the subfield assignment.

## Taxonomy

```text
GET https://api.openalex.org/domains?per-page=200
GET https://api.openalex.org/fields?per-page=200
GET https://api.openalex.org/subfields?per-page=200
```

## Count-Only Subfield/Year Query

```text
GET https://api.openalex.org/works?filter=publication_year:2019,is_retracted:false,is_paratext:false,type:article|preprint,primary_topic.subfield.id:2613&per-page=1
```

The count is read from `meta.count`.

For efficient full count tables, `scripts/01_build_counts.py` uses OpenAlex group-by queries such as:

```text
GET https://api.openalex.org/works?filter=publication_year:2019,is_retracted:false,is_paratext:false,type:article|preprint&group_by=primary_topic.subfield.id
```

## Text Corpus Query

```text
GET https://api.openalex.org/works?filter=primary_topic.subfield.id:2613,from_publication_date:2010-01-01,to_publication_date:2019-12-31,has_abstract:true,language:en,type:article|preprint,is_retracted:false,is_paratext:false&select=id,doi,title,display_name,abstract_inverted_index,publication_year,publication_date,type,language,primary_topic,cited_by_count,referenced_works_count,is_retracted,is_paratext
```

The abstract is reconstructed locally from `abstract_inverted_index`.
