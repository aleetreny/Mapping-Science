# Agent Prompt — Add Higher-Level UMAP Maps and Simple Metric Diagnostics

## Context

You are working inside the `aleetreny/TFM` repository.

Important: the previous refactor has already been implemented, but the user has **not yet rerun the full local pipeline**. Do not assume that fresh 2010–2025 PNGs, UMAP coordinates, projected metrics, or embedding-space metrics already exist locally.

Your task is to extend the active pipeline scripts and documentation so that, once the user runs the pipeline, it can also produce:

1. UMAP PNGs/coordinates for higher OpenAlex levels:
   - subfields remain the main analysis unit;
   - fields should also be generated for backup/inspection;
   - domains should also be generated for backup/inspection;
   - global UMAP PNG should be easy to regenerate.

2. A simple, readable comparison between:
   - projected UMAP morphology metrics;
   - original embedding-space metrics.

3. A simple, readable distribution diagnostic for all metric columns:
   - missingness;
   - number of unique values;
   - low-variance / almost-constant metrics;
   - basic quantiles;
   - simple plots.

Prioritize simplicity and understanding. Do not start clustering, predictive modeling, dashboards, or advanced interpretive analysis yet.

---

## Current repo state to account for

The active README already frames the project as:

```text
OpenAlex Subfield Morphology Metrics
analysis period: 2010–2025
excluded year: 2026
descriptive/comparative, not predictive
```

The active pipeline currently includes approximately:

```bash
python scripts/07_validate_embeddings.py
python scripts/08_prepare_analysis_matrix.py --force

python scripts/09_build_first_umap_maps.py --sample-per-subfield 500 --year-min 2010 --year-max 2025 --force
python scripts/10_build_per_subfield_umap_maps.py --year-min 2010 --year-max 2025 --overwrite

python scripts/11_compute_subfield_morphology_metrics.py --year-min 2010 --year-max 2025 --overwrite
python scripts/12_compute_subfield_embedding_space_metrics.py --year-min 2010 --year-max 2025 --overwrite
```

Relevant files likely include:

```text
README.md
config.yaml

scripts/09_build_first_umap_maps.py
scripts/10_build_per_subfield_umap_maps.py
scripts/11_compute_subfield_morphology_metrics.py
scripts/12_compute_subfield_embedding_space_metrics.py

src/per_subfield_umap_maps.py
src/morphology_metrics.py
src/embedding_space_metrics.py
src/umap_maps.py

docs/analysis_matrix_and_first_umap.md
docs/per_subfield_umap_maps.md
docs/subfield_morphology_metrics.md
docs/subfield_embedding_space_metrics.md

tests/
```

Before editing, audit the repo yourself. There may already be old or partially removed field-level code. Reuse existing helpers where reasonable, but do not force awkward abstractions if a small clear module/script is simpler.

---

## Key methodological rule

The main analysis unit remains:

```text
OpenAlex subfield
```

Fields and domains are **supporting inspection levels**, not the main analytical unit. Make that clear in docs and README.

Do not rename the thesis pipeline away from subfields. Instead, add field/domain/global maps as convenient visual outputs.

---

## Part 1 — Add scripts for higher-level UMAP PNGs

### Need

The user wants to regenerate:

1. PNGs for the ~240 subfields — already handled by the current subfield script.
2. PNGs for fields — about 20–30 categories.
3. PNGs for domains — about 4 broad branches.
4. A general/global UMAP PNG.

The existing `scripts/10_build_per_subfield_umap_maps.py` handles subfields. The existing `scripts/09_build_first_umap_maps.py` handles a global balanced sample. Extend or add scripts so the user can regenerate all relevant PNGs from the active 2010–2025 period.

### Recommended design

Do not duplicate too much code. Prefer one general reusable script if it stays readable.

Possible designs:

#### Option A — New generic grouped UMAP script

Create:

```text
scripts/10b_build_per_category_umap_maps.py
src/per_category_umap_maps.py
```

with CLI:

```bash
python scripts/10b_build_per_category_umap_maps.py --level field --year-min 2010 --year-max 2025 --overwrite
python scripts/10b_build_per_category_umap_maps.py --level domain --year-min 2010 --year-max 2025 --overwrite
```

Supported levels:

```text
field
domain
```

Maybe support `subfield` too only if it does not disrupt the existing script. Otherwise keep subfield script separate.

Output directories:

```text
outputs/maps/per_field_umap/coordinates/*.parquet
outputs/maps/per_field_umap/figures/*.png
outputs/maps/per_field_umap/per_field_umap_manifest.parquet
outputs/maps/per_field_umap/per_field_umap_summary.json

outputs/maps/per_domain_umap/coordinates/*.parquet
outputs/maps/per_domain_umap/figures/*.png
outputs/maps/per_domain_umap/per_domain_umap_manifest.parquet
outputs/maps/per_domain_umap/per_domain_umap_summary.json
```

#### Option B — Separate scripts

Create:

```text
scripts/10b_build_per_field_umap_maps.py
scripts/10c_build_per_domain_umap_maps.py
```

Only do this if it is much clearer than a generic script.

### Grouping columns

Use the existing metadata in `analysis_embedding_index.parquet`.

Expected group columns:

```text
field_id
field_display_name
domain_id
domain_display_name
```

For field-level maps:

```text
group_id = field_id
group_name = field_display_name
```

For domain-level maps:

```text
group_id = domain_id
group_name = domain_display_name
```

Keep subfield metadata in coordinate outputs when available, because it is useful for color/interpretation later.

### UMAP fitting

For field and domain maps, fit one separate UMAP per group, analogous to the current per-subfield script.

Use active default period:

```text
year_min = 2010
year_max = 2025
```

Keep 2026 excluded.

Use memory mapping for embeddings:

```python
np.load(embeddings_path, mmap_mode="r")
```

Use deterministic sampling if a group has too many papers.

Expose CLI args:

```text
--year-min
--year-max
--min-papers
--max-papers-per-group
--group-id
--limit-groups
--random-state
--n-neighbors
--min-dist
--metric
--dpi
--overwrite
```

Use simple defaults, likely:

```text
--min-papers 250
--max-papers-per-group 10000
--n-neighbors 30
--min-dist 0.05
--metric cosine
```

If fields/domains are too large, the user can lower `--max-papers-per-group`.

### PNG design

Keep the same simple two-panel layout as subfield maps if possible:

```text
A. Scatter
B. Density
```

For field and domain maps, consider coloring the scatter by subfield or field when this is still readable:

- Field-level map: color by `subfield_display_name` only if the number of subfields is not too high; otherwise use a neutral scatter and density.
- Domain-level map: color by `field_display_name` if readable.
- Do not create unreadable legends with hundreds of labels. If labels are too many, skip legend and document the color handling.

The goal is simple visual inspection, not a publication-quality atlas yet.

### Global map

The existing `scripts/09_build_first_umap_maps.py` already creates:

```text
outputs/maps/umap_global_sample.parquet
outputs/maps/umap_global_sample.png
outputs/maps/umap_global_sample_summary.json
```

Audit it and improve it only if needed.

Recommended improvements:

- Make output names clearly include active period or store the period in summary.
- Allow easy color choice:

```bash
--color-by domain
--color-by field
--color-by subfield
```

or alternatively produce multiple PNGs in one run:

```text
outputs/maps/global_umap/umap_global_sample_color_domain.png
outputs/maps/global_umap/umap_global_sample_color_field.png
outputs/maps/global_umap/umap_global_sample_color_subfield.png
```

Do not overcomplicate this. At minimum, make sure one global PNG is reproducible and documented.

### README commands to add

Add clear commands such as:

```bash
# Global map
python scripts/09_build_first_umap_maps.py --sample-per-subfield 500 --year-min 2010 --year-max 2025 --force

# Main unit: subfields
python scripts/10_build_per_subfield_umap_maps.py --year-min 2010 --year-max 2025 --overwrite

# Supporting levels
python scripts/10b_build_per_category_umap_maps.py --level field --year-min 2010 --year-max 2025 --overwrite
python scripts/10b_build_per_category_umap_maps.py --level domain --year-min 2010 --year-max 2025 --overwrite
```

Adjust names to match your implementation.

---

## Part 2 — Add simple UMAP vs embedding metric comparison script

### Need

Create a simple script to compare the two metric families:

```text
projected UMAP morphology metrics
vs
original embedding-space metrics
```

This is not supposed to “prove” UMAP is correct. It should give a simple first diagnostic of whether projected morphology and high-dimensional embedding structure tell compatible stories.

### Recommended script

Create:

```text
scripts/13_compare_metric_families.py
src/metric_family_comparison.py
docs/metric_family_comparison.md
```

If a separate `src` module feels unnecessary, keep logic in the script, but keep the code readable and tested.

### Inputs

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_embedding_space_metrics.parquet
```

Join on:

```text
subfield_id
```

Use only rows with completed/valid metric status, but do not silently drop everything. Record counts.

### Outputs

Recommended output directory:

```text
outputs/analysis/metric_family_comparison/
```

Write:

```text
outputs/analysis/metric_family_comparison/metric_family_comparison_summary.json
outputs/analysis/metric_family_comparison/metric_family_comparison_summary.md
outputs/analysis/metric_family_comparison/cross_family_spearman_correlations.csv
outputs/analysis/metric_family_comparison/cross_family_pearson_correlations.csv
outputs/analysis/metric_family_comparison/top_absolute_spearman_correlations.csv
outputs/analysis/metric_family_comparison/analogue_metric_pair_correlations.csv
outputs/analysis/metric_family_comparison/cross_family_spearman_heatmap.png
```

Keep outputs easy to inspect.

### Correlations

Compute:

```text
Spearman correlation
Pearson correlation
n_valid_pairwise
p-value optional, not necessary
```

Spearman is more important here because morphology metrics may be non-normal and nonlinear.

The heatmap should be readable. If 25×25 is too dense, still acceptable, but use sensible font sizes and a large figure. No new dependencies if avoidable. Use matplotlib.

### Suggested analogue metric pairs

Create a small dictionary of conceptually related pairs. Do not force one-to-one equivalence if it is not meaningful. Suggested pairs:

```text
knn_median_distance ↔ embedding_knn_median_distance
knn_distance_cv ↔ embedding_knn_distance_cv
radial_tail_index ↔ embedding_tail_index_p90_median
centroid_drift_early_late ↔ embedding_centroid_drift_early_late
annual_centroid_path_length ↔ embedding_annual_centroid_path_length
directionality_ratio ↔ embedding_directionality_ratio
radial_expansion_slope ↔ embedding_radial_expansion_slope
mst_gap_index ↔ embedding_graph_edge_distance_p90
dense_component_count ↔ embedding_graph_connected_component_count
largest_component_mass_share ↔ embedding_graph_largest_component_share
anisotropy_ratio ↔ embedding_pca_first_component_share
```

These are imperfect analogues. Say so in the docs and summary.

### Interpretive summary

The generated `.md` summary should explain in plain language:

- how many subfields were compared;
- which metric pairs agree most strongly;
- which metric pairs disagree;
- which UMAP metrics have weak relationship with embedding-space metrics;
- that weak correlation is not automatically failure, because projected morphology and high-dimensional structure measure different aspects.

Do not write grand conclusions. Keep it diagnostic.

---

## Part 3 — Add simple metric distribution diagnostic script

### Need

Create a simple script that summarizes the distribution of all metric columns so the user can quickly spot:

- metrics with too many missing values;
- metrics with very few unique values;
- almost-constant metrics;
- metrics dominated by zero;
- extreme ranges/outliers.

This should be very easy to understand.

### Recommended script

Create:

```text
scripts/14_summarize_metric_distributions.py
src/metric_distribution_diagnostics.py
docs/metric_distribution_diagnostics.md
```

Again, keep it simple. If a `src` module is overkill, implement cleanly in the script.

### Inputs

By default use:

```text
data/processed/subfield_morphology_metrics.parquet
data/processed/subfield_embedding_space_metrics.parquet
```

Optionally support:

```text
--input-parquet
--metric-prefix
```

But do not overcomplicate.

### Outputs

Recommended output directory:

```text
outputs/analysis/metric_distributions/
```

Write:

```text
outputs/analysis/metric_distributions/metric_distribution_summary.csv
outputs/analysis/metric_distributions/low_information_metrics.csv
outputs/analysis/metric_distributions/metric_distribution_summary.md
outputs/analysis/metric_distributions/umap_metric_histograms.png
outputs/analysis/metric_distributions/embedding_metric_histograms.png
outputs/analysis/metric_distributions/all_metric_boxplots_zscore.png
```

### Summary columns

For each metric:

```text
metric_name
metric_family              # umap/projected or embedding
n_rows
n_non_missing
missing_share
n_unique
unique_share
zero_share
mean
std
min
p01
p05
p25
median
p75
p95
p99
max
iqr
is_constant
is_low_unique
is_high_missing
is_zero_dominated
recommended_review_flag
```

Simple flag thresholds:

```text
is_constant: n_unique <= 1
is_low_unique: n_unique <= 5 OR unique_share <= 0.05
is_high_missing: missing_share >= 0.30
is_zero_dominated: zero_share >= 0.80
recommended_review_flag: any of the above
```

Make thresholds CLI options if easy:

```text
--high-missing-threshold 0.30
--low-unique-threshold 5
--zero-dominated-threshold 0.80
```

### Plots

Keep plots simple and readable.

- Histograms for UMAP metrics in a grid.
- Histograms for embedding metrics in a grid.
- One z-scored boxplot figure for all metrics, grouped/faceted if possible.

Do not spend too much time making them beautiful. These are diagnostics.

No seaborn. Use matplotlib and existing dependencies only.

### Interpretive `.md` summary

The `.md` summary should say:

- number of UMAP metrics summarized;
- number of embedding metrics summarized;
- metrics flagged for review;
- why they were flagged;
- reminder that flagged does not mean “delete automatically”.

---

## Documentation update

Update active docs and README to include the new scripts.

At minimum:

```text
README.md
docs/analysis_matrix_and_first_umap.md
docs/per_subfield_umap_maps.md
docs/subfield_morphology_metrics.md
docs/subfield_embedding_space_metrics.md
```

Add new docs:

```text
docs/higher_level_umap_maps.md
docs/metric_family_comparison.md
docs/metric_distribution_diagnostics.md
```

If fewer docs are cleaner, combine some, but make sure the active pipeline is understandable.

---

## Tests

Add or update tests.

### Higher-level UMAP grouping tests

Test helper functions without running heavy UMAP when possible:

- grouping by field/domain works;
- required columns are validated;
- deterministic sampling is deterministic;
- output manifest has expected columns;
- too-small groups are skipped with a clear status.

### Metric comparison tests

Use tiny synthetic metric tables.

Check:

- join on `subfield_id`;
- Spearman/Pearson outputs have expected shape;
- analogue-pair table works even if some metrics are missing;
- output does not crash on NaNs.

### Distribution diagnostic tests

Use synthetic tables.

Check:

- constant metrics are flagged;
- low-unique metrics are flagged;
- high-missing metrics are flagged;
- zero-dominated metrics are flagged;
- normal continuous metrics are not flagged.

Run:

```bash
python -m pytest
python -m compileall scripts src
```

Do not leave failing tests.

---

## Important constraints

- The user has not rerun the full pipeline yet. Do not depend on local generated outputs existing for tests.
- Avoid new heavy dependencies.
- Do not add clustering.
- Do not add regression.
- Do not add dashboards.
- Do not interpret scientific findings yet.
- Keep command names and output paths simple.
- Keep the active pipeline coherent and non-obsolete.
- If you discover older field/domain outputs in docs or code, either revive them coherently or remove obsolete references.

---

## Final response requested from the agent

When finished, summarize:

1. Files changed.
2. New scripts added.
3. New outputs added.
4. Commands the user should run, in order:
   - global map;
   - subfield maps;
   - field maps;
   - domain maps;
   - UMAP morphology metrics;
   - embedding-space metrics;
   - metric family comparison;
   - metric distribution diagnostics.
5. Whether any scripts require the full outputs to already exist.
6. Tests run and result.
7. Remaining risks, especially runtime/memory risks for field/domain/domain-level UMAPs.

Do not run full expensive map generation. It is enough to ensure scripts, docs, and tests are correct.
