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
- `analysis_count_2010_2025`
- `analysis_text_count_2010_2025`
- `field_analysis_count_2010_2025`
- `domain_analysis_count_2010_2025`
- `analysis_share_within_field`
- `analysis_share_within_domain`
- `eligible_for_text_corpus`
- `planned_sample_size`

## `sample_plan`

One row per eligible subfield-year cell in the 2010-2025 analysis window.

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
- `api_initial_sample_size`
- `expected_shortfall_risk`

`sampling_method` is one of:

- `sample_api`
- `download_all_available`
- `skip_no_available_works`

## `download_manifest`

One row per processed subfield-year cell.

- `subfield_id`
- `subfield_display_name`
- `field_id`
- `field_display_name`
- `domain_id`
- `domain_display_name`
- `publication_year`
- `sampling_method`
- `available_valid_works`
- `planned_sample_size`
- `api_initial_sample_size`
- `raw_returned_works`
- `valid_after_local_filter`
- `kept_works`
- `duplicate_or_already_seen`
- `discarded_local_validation`
- `shortfall`
- `backfill_rounds_used`
- `seeds_used`
- `status`
- `error_message`

`status` is one of:

- `completed_target_met`
- `completed_shortfall`
- `skipped_no_available_works`
- `failed`

Versioned extraction `2000_2024_400py` writes the same logical tables with a
suffix instead of replacing canonical files:

```text
data/interim/subfield_year_counts_2000_2024_400py.parquet
data/interim/field_year_counts_2000_2024_400py.parquet
data/interim/domain_year_counts_2000_2024_400py.parquet
data/interim/corpus_plan_2000_2024_400py.parquet
data/interim/sample_plan_2000_2024_400py.parquet
data/interim/download_manifest_2000_2024_400py.parquet
data/processed/works_text_2000_2024_400py.parquet
data/processed/analysis_subfields_2000_2024_400py.parquet
```

See
[openalex_extraction_2000_2024_400py.md](openalex_extraction_2000_2024_400py.md).

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
- `primary_topic_id`
- `primary_topic_display_name`
- `topics_json`
- `title_token_count`
- `abstract_token_count`
- `text_token_count`
- `downloaded_at`
- `cited_by_count`
- `referenced_works_count`

`topics_json` is stored as compact JSON for later interpretation of subfield morphology. It does not make OpenAlex topics the unit of analysis.

## `analysis_subfields`

One row per planned subfield. This table is an analysis eligibility layer; it does not remove papers from `works_text`.

- `subfield_id`
- `subfield_display_name`
- `field_id`
- `field_display_name`
- `domain_id`
- `domain_display_name`
- `n_valid_works`
- `planned_works`
- `shortfall`
- `main_analysis_eligible_2500`
- `robustness_eligible_500`
- `is_low_sample`
- `exclusion_reason`

`main_analysis_eligible_2500` is true when `n_valid_works >= 2500`. `robustness_eligible_500` is true when `n_valid_works >= 500`. The main threshold avoids unstable morphology metrics in very small semantic clouds while keeping all downloaded works available for embeddings and sensitivity checks.

`exclusion_reason` is one of:

- `main_analysis_included`
- `below_2500_valid_works`
- `below_500_valid_works`

For versioned extraction `2000_2024_400py`, `analysis_subfields` instead uses
annual/full-period coverage flags:

- `eligible_10000_full_period`
- `eligible_min_5000_full_period`
- `eligible_for_temporal_5year_exploration`

Embedding validation normalizes those to canonical downstream flags:

- `main_analysis_eligible = eligible_for_temporal_5year_exploration`
- `robustness_eligible = eligible_min_5000_full_period`
- `strict_full_period_eligible = eligible_10000_full_period`

## `embedding_index`

One row per embedded work. This table is a lightweight pointer from `work_id` to a SPECTER2 shard and row position. It does not store embedding vectors.

- `work_id`
- `embedding_model`
- `embedding_version`
- `embedding_dim`
- `embedding_dtype`
- `embedding_shard_id`
- `embedding_shard_file`
- `embedding_row_in_shard`
- `metadata_shard_file`
- `subfield_id`
- `subfield_display_name`
- `field_id`
- `field_display_name`
- `domain_id`
- `domain_display_name`
- `primary_topic_id`
- `primary_topic_display_name`
- `publication_year`
- `main_analysis_eligible`
- `robustness_eligible`
- `strict_full_period_eligible`
- `main_analysis_eligible_2500`
- `robustness_eligible_500`

`main_analysis_eligible`, `robustness_eligible`, and
`strict_full_period_eligible` are the canonical columns going forward. The
`*_2500` and `*_500` columns are backward-compatible aliases for older scripts
and notebooks.

The SPECTER2 shard files live in `embeddings/specter2_v1/`, which is ignored by Git. See [embedding_data_model.md](embedding_data_model.md) for the Drive path, download commands, validation checks, and examples for loading one shard with NumPy.

## `analysis_embedding_index`

One row per main-analysis embedded work. This table has the same row order as:

```text
embeddings/specter2_v1/analysis/main_embeddings.float16.npy
```

It includes all `embedding_index` columns plus:

- `analysis_row_id`

`analysis_row_id` is zero-based and points to the corresponding row in the main-analysis matrix.

Rows are filtered to `main_analysis_eligible == true` and sorted by:

```text
subfield_id, publication_year, work_id
```

## First UMAP Sample Output

The first visual-inspection map writes:

```text
outputs/maps/umap_global_sample.parquet
```

Columns:

- `work_id`
- `analysis_row_id`
- `subfield_id`
- `subfield_display_name`
- `field_id`
- `field_display_name`
- `domain_id`
- `domain_display_name`
- `primary_topic_id`
- `primary_topic_display_name`
- `publication_year`
- `umap_x`
- `umap_y`

The PNG companion is:

```text
outputs/maps/umap_global_sample.png
```

## Higher-Level UMAP Outputs

Field and domain UMAP maps are supporting inspection outputs. Subfield remains
the main analysis unit.

Field outputs:

```text
outputs/maps/per_field_umap/coordinates/*.parquet
outputs/maps/per_field_umap/figures/*.png
outputs/maps/per_field_umap/per_field_umap_manifest.parquet
outputs/maps/per_field_umap/per_field_umap_summary.json
```

Domain outputs:

```text
outputs/maps/per_domain_umap/coordinates/*.parquet
outputs/maps/per_domain_umap/figures/*.png
outputs/maps/per_domain_umap/per_domain_umap_manifest.parquet
outputs/maps/per_domain_umap/per_domain_umap_summary.json
```

## Metric Tables

Projected UMAP morphology metrics:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_morphology_metrics.csv
```

These contain one row per completed subfield UMAP map, with control columns,
25 core projected morphology metrics, and diagnostic columns.

Embedding-space structure metrics:

```text
data/processed/subfield_embedding_space_metrics.parquet
data/processed/subfield_embedding_space_metrics.csv
```

These contain one row per attempted main-analysis subfield, with control
columns, 25 core embedding-space metrics, and diagnostic columns. Joins should
use `subfield_id`; display names are included for readability but are not
unique.

Metric diagnostics are written under:

```text
outputs/analysis/metric_family_comparison/
outputs/analysis/metric_distributions/
outputs/analysis/metric_clustering/
```
