# Agent Prompt — Add future-proof metadata before full download

Before launching the full OpenAlex production download, make one small future-proofing update.

The current pipeline works and the 5-subfield test reached 15,000/15,000 valid works. Do not rewrite the pipeline.

## Goal

Add a few cheap metadata columns to `works_text` so we do not need to re-query hundreds of thousands of works later.

Do **not** add embeddings, UMAP, clustering, models, plots, dashboards, authors, institutions, PDFs, fulltext, OA metadata, or heavy ML dependencies.

## Required changes

### 1. Update selected OpenAlex fields

In `src/openalex.py`, add `topics` to `DEFAULT_SELECT_FIELDS`.

Keep `primary_topic`.

Do not add heavy fields like `authorships`, `locations`, `concepts`, `open_access`, `primary_location`, or fulltext/PDF fields.

### 2. Extend `works_text`

In `src/works.py`, add these output columns:

```text
primary_topic_id
primary_topic_display_name
topics_json
title_token_count
abstract_token_count
text_token_count
downloaded_at
```

Keep all existing columns.

### 3. Populate the new fields

In `validate_and_normalize_work()`:

- Extract `primary_topic_id` from `primary_topic.id`.
- Extract `primary_topic_display_name` from `primary_topic.display_name`.
- Store `topics_json` as compact JSON from `work["topics"]`, using `ensure_ascii=False`.
- Compute:
  - `title_token_count`
  - `abstract_token_count`
  - `text_token_count`
- Set `downloaded_at` as current UTC timestamp in ISO format.

Keep the current validation rules unchanged.

### 4. Update validation

In `scripts/05_validate_database.py`, report:

```text
missing primary_topic_id
missing topics_json
title_token_count distribution
abstract_token_count distribution
text_token_count distribution
```

### 5. Update docs

Briefly update:

```text
README.md
docs/data_model.md
docs/full_download_runbook.md
```

Explain that topic metadata is stored only for later interpretation of subfield morphology, not as the main unit of analysis.

### 6. Tests

Add/update tests to check:

- `topics` is included in `DEFAULT_SELECT_FIELDS`
- `works_text` includes the new columns
- `primary_topic_id` and `primary_topic_display_name` are extracted correctly
- `topics_json` is valid JSON
- token counts are computed correctly

Run:

```bash
pytest
```

## Final check before full download

After changes, run:

```bash
python scripts/04_download_sampled_corpus.py --force --limit-subfields 5
python scripts/05_validate_database.py
```

Expected result: still close to or exactly 15,000 valid works, with the new metadata columns populated.

Then we can launch:

```bash
python scripts/04_download_sampled_corpus.py --resume
python scripts/05_validate_database.py
```

Keep this update small and surgical.
