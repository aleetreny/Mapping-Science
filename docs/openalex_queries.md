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

Large subfield-year cells use OpenAlex API sampling. The requested sample can be larger than the planned kept count because the downloader oversamples raw results before local validation:

```text
GET https://api.openalex.org/works?filter=primary_topic.subfield.id:2613,publication_year:2019,has_abstract:true,language:en,type:article|preprint,is_retracted:false,is_paratext:false&sample=525&seed=4674&select=id,doi,title,display_name,abstract_inverted_index,publication_year,publication_date,type,language,primary_topic,cited_by_count,referenced_works_count,is_retracted,is_paratext
```

If too few valid works remain after local validation, the downloader repeats sampled requests with deterministic backfill seeds.

Small cells are downloaded in full with cursor pagination:

```text
GET https://api.openalex.org/works?filter=primary_topic.subfield.id:2613,publication_year:2019,has_abstract:true,language:en,type:article|preprint,is_retracted:false,is_paratext:false&cursor=*&select=id,doi,title,display_name,abstract_inverted_index,publication_year,publication_date,type,language,primary_topic,cited_by_count,referenced_works_count,is_retracted,is_paratext
```

The abstract is reconstructed locally from `abstract_inverted_index`.
