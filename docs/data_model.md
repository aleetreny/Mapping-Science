# Data Model

The pipeline writes one DuckDB database at `warehouse/tfm_openalex.duckdb` and mirrors important tables as Parquet files.

## `domains`

- `domain_id`
- `domain_display_name`
- `description`
- `works_count`
- `cited_by_count`

## `fields`

- `field_id`
- `field_display_name`
- `description`
- `domain_id`
- `domain_display_name`
- `works_count`
- `cited_by_count`

## `subfields`

- `subfield_id`
- `subfield_display_name`
- `description`
- `field_id`
- `field_display_name`
- `domain_id`
- `domain_display_name`
- `works_count`
- `cited_by_count`

## `subfield_year_counts`

- `subfield_id`
- `field_id`
- `domain_id`
- `publication_year`
- `n_works_total`
- `n_works_article_preprint`
- `n_works_article_preprint_en`
- `n_works_article_preprint_en_with_abstract`

## `field_year_counts`

- `field_id`
- `domain_id`
- `publication_year`
- `n_works_total`
- `n_works_article_preprint`
- `n_works_article_preprint_en`
- `n_works_article_preprint_en_with_abstract`

## `domain_year_counts`

- `domain_id`
- `publication_year`
- `n_works_total`
- `n_works_article_preprint`
- `n_works_article_preprint_en`
- `n_works_article_preprint_en_with_abstract`

## `corpus_plan`

- `subfield_id`
- `subfield_display_name`
- `field_id`
- `field_display_name`
- `domain_id`
- `domain_display_name`
- `past_count_2010_2019`
- `future_count_2020_2025`
- `past_text_count_2010_2019`
- `field_past_count_2010_2019`
- `field_future_count_2020_2025`
- `domain_past_count_2010_2019`
- `domain_future_count_2020_2025`
- `past_share_within_field`
- `future_share_within_field`
- `past_share_within_domain`
- `future_share_within_domain`
- `log_growth_within_field`
- `log_growth_within_domain`
- `growth_above_field_median`
- `growth_above_domain_median`
- `eligible_for_text_corpus`
- `planned_sample_size`

## `sample_plan`

One row per eligible subfield-year cell in the 2010-2019 morphology window.

- `subfield_id`
- `subfield_display_name`
- `field_id`
- `field_display_name`
- `domain_id`
- `domain_display_name`
- `publication_year`
- `available_valid_works`
- `planned_sample_size`
- `sampling_method`
- `seed`

`sampling_method` is one of:

- `sample_api`
- `download_all_available`
- `skip_no_available_works`

## `works_text`

- `work_id`
- `doi`
- `title`
- `abstract`
- `text_for_embedding`
- `publication_year`
- `publication_date`
- `type`
- `language`
- `subfield_id`
- `subfield_display_name`
- `field_id`
- `field_display_name`
- `domain_id`
- `domain_display_name`
- `cited_by_count`
- `referenced_works_count`
