# Research Context

Project title:

```text
OpenAlex Subfield Morphology and Scientific Growth Prediction
```

This file is the compact project brief for future agents. Keep it current and simple.

## Core Question

Can the semantic morphology of an OpenAlex subfield before 2020 help predict whether that subfield grows during 2020-2025?

The project studies scientific growth at the level of subfields. The intuition is that subfields may differ not only in size and citation activity, but also in the shape of their semantic space: some may be compact and coherent, some fragmented, some expanding around multiple fronts, and some concentrated around a narrow conceptual core.

Later phases will test whether those pre-2020 morphology patterns are associated with later growth.

## Unit Of Analysis

The unit is the OpenAlex subfield, assigned through:

```text
primary_topic.subfield.id
```

Do not use OpenAlex topics as the main unit of analysis. Topics are too numerous for the first stable thesis pipeline. Subfields are broad enough to form meaningful semantic spaces and numerous enough to support comparative analysis.

## Time Design

- Morphology window: 2010-2019
- Growth window: 2020-2025
- Excluded year: 2026, because it is incomplete
- Text source: title plus abstract
- Sample target: 3,000 works per eligible subfield
- Main work types: article and preprint
- Main language for text corpus: English

The morphology window must precede the growth window. Future models should use only information observed before 2020 to explain or predict growth in 2020-2025.

## Growth Target

Growth is relative, not just raw publication count growth.

For subfield `i` inside field `f`:

```text
past_share_i = N_i_2010_2019 / N_f_2010_2019
future_share_i = N_i_2020_2025 / N_f_2020_2025
log_growth_within_field_i = log((future_share_i + 1e-9) / (past_share_i + 1e-9))
```

The same idea is also computed relative to the containing domain.

Growth targets should primarily use article/preprint production counts, not only abstract-available counts. Abstract availability is a data feasibility filter, not the scientific growth outcome.

## Whole Project Roadmap

Phase 1, active now: data infrastructure.

- fetch OpenAlex taxonomy
- build yearly count tables
- compute growth-ready corpus plan
- build a subfield-year sample plan
- download sampled 2010-2019 title and abstract corpus with oversampling, backfill, resume, and a manifest
- validate the local DuckDB database and Parquet files

Phase 2, later: semantic representation.

- choose or compare text embedding methods
- embed `title + abstract`
- keep embeddings tied to `works_text` and subfield IDs
- avoid changing the growth target while experimenting with representations

Phase 3, later: morphology metrics.

Possible metrics may include semantic dispersion, compactness, density, local neighborhood structure, fragmentation, centrality/concentration, or temporal spread within each subfield. Do not assume these are final; choose metrics that can be explained clearly in a thesis.

Phase 4, later: growth modeling.

- predict or explain 2020-2025 relative growth using only 2010-2019 morphology
- compare against simple baselines such as past size and past share
- evaluate within field/domain where possible
- keep the modeling interpretable enough for the thesis

Phase 5, optional later: visualization.

Maps or dashboards are optional communication tools, not the core pipeline. Add them only after the data, morphology, and modeling design are stable.

## Current Code Boundary

The current repository should implement only Phase 1.

Do not add embeddings, dimensionality reduction, clustering, morphology metrics, regression models, prediction models, visual maps, dashboards, notebooks, or ML infrastructure yet.

When future phases are added, keep them separated from the data pipeline and avoid reintroducing unrelated previous workflows as active code.

## Active Workflow

```bash
python scripts/00_fetch_taxonomy.py
python scripts/01_build_counts.py
python scripts/02_build_corpus_plan.py
python scripts/03_build_sample_plan.py
python scripts/04_download_sampled_corpus.py --limit-subfields 5
python scripts/05_validate_database.py
```

Use dry runs before long API work:

```bash
python scripts/01_build_counts.py --dry-run
python scripts/04_download_sampled_corpus.py --dry-run
```
