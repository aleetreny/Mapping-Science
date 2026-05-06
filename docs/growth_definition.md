# Growth Definition

The unit of analysis is the OpenAlex subfield. Subfields are broad enough to support stable text samples, but numerous enough to compare different areas of science in a master thesis.

## Time Windows

Morphology uses works from 2010-2019. This gives a ten-year pre-2020 text window for estimating the semantic structure of each subfield later.

Growth uses works from 2020-2025. This gives a later outcome window that is separated from the morphology window.

The year 2026 is excluded from the main analysis because it is the current incomplete year and would bias publication counts downward.

## Relative Growth

Growth is measured as change in publication share, not just raw counts. Larger fields and domains produce more papers, so subfield growth is compared against its containing field and domain.

For subfield `i` in field `f`:

```text
past_share_i = N_i_2010_2019 / N_f_2010_2019
future_share_i = N_i_2020_2025 / N_f_2020_2025
log_growth_within_field_i = log((future_share_i + 1e-9) / (past_share_i + 1e-9))
```

The same calculation is also made relative to the domain.

## Text Corpus

The morphology corpus uses title plus abstract because those fields are widely available through OpenAlex and give a compact representation of a paper's semantic content.

The corpus is planned at the subfield-year level. Large cells are sampled directly through the OpenAlex API with deterministic seeds, while small cells are downloaded in full. This keeps the corpus comparable without downloading every candidate work first.

Growth targets are based primarily on article and preprint counts, not only works with available abstracts. This avoids turning abstract availability into the growth outcome.
