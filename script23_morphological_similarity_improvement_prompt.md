# Prompt for Agent — Improve Script 23 Morphological Similarity Evolution

You are working in the `TFM` repository.

The current script:

```text
scripts/23_compute_morphological_similarity_evolution.py
```

already computes morphological distances between subfields/fields/domains using internal semantic-structure metric profiles. This is conceptually important for the thesis because it studies **shape similarity**, not topic/content similarity.

The current outputs are useful but still too matrix-heavy and not thesis-ready. This task is to improve interpretability, QA, figures, and top-pair diagnostics.

---

## High-level goal

Improve script 23 so it clearly answers:

1. Which disciplines/subfields are morphologically similar or dissimilar in the static full-period view?
2. Which pairs become more morphologically similar between 2000–2004 and 2020–2024?
3. Which pairs become more morphologically different?
4. Are the results driven by Euclidean magnitude differences or correlation-profile differences?
5. Are the top pair results dominated by a few outlier subfields/fields?
6. Which individual entities are driving the strongest pairwise morphological changes?

The analysis must remain clearly distinct from script 22:

- Script 22: semantic distance = distance between centroid positions/content location.
- Script 23: morphological distance = distance between internal structural metric profiles.

---

## Existing method to preserve

Keep the conceptual setup:

- Morphological distance is computed between robust-scaled metric profiles.
- Temporal morphology uses the eight non-temporal structural metrics recomputed per five-year window.
- Static morphology uses the reduced interpretable embedding core when available, plus the 8-metric structural sensitivity set.
- Negative final-minus-initial delta means morphological convergence.
- Positive final-minus-initial delta means morphological divergence.
- Field/domain profiles are averages of robust-scaled subfield profiles.

Do **not** introduce publication count as a substantive feature. Counts can only appear as QA metadata.

---

# Required improvements

## 1. Add explicit entity retention QA

There is a possible mismatch between the expected number of subfields and the number retained in morphological matrices. The project usually has around 241 subfields, but some matrices appear to have around 234 subfields.

Add explicit QA outputs.

### New output

```text
outputs/analysis/morphological_similarity_evolution/entity_retention_diagnostics.csv
outputs/analysis/morphological_similarity_evolution/dropped_entities.csv
```

For each level (`subfield`, `field`, `domain`), report:

```text
level
n_input_entities
n_retained_entities
n_dropped_entities
reason
```

For dropped entities, include:

```text
level
entity_id
entity_display_name
field_display_name
domain_display_name
reason_dropped
missing_metrics
n_windows_available
```

Also add this to `summary.md` and `summary.json`.

The final chat response should explicitly say how many subfields were retained and why any were dropped.

---

## 2. Separate Euclidean and correlation interpretations

Euclidean and correlation morphological distances are not just minor sensitivity checks. They answer different questions:

- **Euclidean morphological distance**: difference in magnitude of structural metrics.
- **Correlation morphological distance**: difference in relative metric profile/pattern.

Update code comments, figure titles, summary text, and output filenames so this distinction is clear.

In `summary.md`, add:

```markdown
## Interpretation of Distance Types

- Euclidean morphological distance compares the magnitude of robust-scaled metric profiles.
- Correlation morphological distance compares the relative pattern of metrics, ignoring much of the absolute magnitude.
- The two may disagree; disagreement is substantively meaningful rather than automatically an error.
```

---

## 3. Generate thesis-ready delta heatmaps

The script currently saves many matrices but does not provide enough readable figures. Add clean heatmaps for temporal change.

### New required figures

```text
outputs/analysis/morphological_similarity_evolution/domain_euclidean_delta_heatmap.png
outputs/analysis/morphological_similarity_evolution/domain_correlation_delta_heatmap.png
outputs/analysis/morphological_similarity_evolution/field_euclidean_delta_heatmap.png
outputs/analysis/morphological_similarity_evolution/field_correlation_delta_heatmap.png
```

Requirements:

- Use a diverging colormap centered at zero.
- Negative values = morphological convergence.
- Positive values = morphological divergence.
- Title must explicitly say:
  - `Morphological Distance Change`
  - distance type: Euclidean or Correlation
  - level: domain or field
- Use a symmetric color scale around zero for each figure, unless there is a strong reason not to.
- Include colorbar label:
  - `Final minus initial morphological distance`
- Use readable axis labels.
- For field-level heatmaps, use clustering/order only if the same ordering is used consistently across Euclidean/correlation comparisons. Otherwise order by domain/field name.

Optional but useful:

```text
outputs/analysis/morphological_similarity_evolution/domain_euclidean_vs_correlation_delta_heatmaps.png
```

A two-panel figure comparing Euclidean and correlation domain-level deltas.

---

## 4. Add initial/final/delta triptych figures

For domain and field levels, create triptych figures showing:

```text
initial distance | final distance | final - initial delta
```

### New figures

```text
outputs/analysis/morphological_similarity_evolution/domain_euclidean_initial_final_delta_triptych.png
outputs/analysis/morphological_similarity_evolution/domain_correlation_initial_final_delta_triptych.png
outputs/analysis/morphological_similarity_evolution/field_euclidean_initial_final_delta_triptych.png
outputs/analysis/morphological_similarity_evolution/field_correlation_initial_final_delta_triptych.png
```

Requirements:

- Same row/column order across the three panels.
- First two panels use sequential colormap.
- Delta panel uses diverging colormap centered at zero.
- Titles should be clear:
  - `2000–2004`
  - `2020–2024`
  - `Change`
- Mention in caption/summary that the delta panel is the key temporal result.

---

## 5. Strictly separate converging and diverging top-pair tables

Ensure top-pair files use the sign of `delta_distance` correctly.

### Required logic

```python
converging = delta_distance < 0
diverging = delta_distance > 0
```

Do not include positive deltas in converging tables or negative deltas in diverging tables.

If a level has fewer than top N pairs of one sign, return fewer rows rather than mixing signs.

### Required outputs

For each level and distance type:

```text
outputs/analysis/morphological_similarity_evolution/top_pairs/<level>_<distance_type>_top_converging_pairs.csv
outputs/analysis/morphological_similarity_evolution/top_pairs/<level>_<distance_type>_top_diverging_pairs.csv
```

Columns should include:

```text
level
distance_type
entity_a_id
entity_a_name
entity_a_field
entity_a_domain
entity_b_id
entity_b_name
entity_b_field
entity_b_domain
initial_distance
final_distance
delta_distance
slope
rank
```

Also keep backwards-compatible aggregate files if already expected:

```text
outputs/analysis/morphological_similarity_evolution/top_morphological_converging_pairs.csv
outputs/analysis/morphological_similarity_evolution/top_morphological_diverging_pairs.csv
```

but make sure they are not misleading.

---

## 6. Improve top-pair plots

Create clean bar/dot plots for convergence/divergence.

### New figures

```text
outputs/analysis/morphological_similarity_evolution/field_euclidean_top_converging_diverging_pairs.png
outputs/analysis/morphological_similarity_evolution/field_correlation_top_converging_diverging_pairs.png
outputs/analysis/morphological_similarity_evolution/domain_euclidean_top_converging_diverging_pairs.png
outputs/analysis/morphological_similarity_evolution/domain_correlation_top_converging_diverging_pairs.png
```

Optional for subfield:

```text
outputs/analysis/morphological_similarity_evolution/subfield_euclidean_top_converging_diverging_pairs.png
outputs/analysis/morphological_similarity_evolution/subfield_correlation_top_converging_diverging_pairs.png
```

For subfield plots, limit to top 10 convergence and top 10 divergence because labels are long.

Figure requirements:

- Negative values left = convergence.
- Positive values right = divergence.
- Vertical zero line.
- Top 10 each side by default.
- Abbreviate long labels.
- Title should include level and distance type.
- X-axis label:
  - `Final minus initial morphological distance`
- Use clear legend/caption:
  - `Negative = convergence; positive = divergence`.

---

## 7. Add frequency diagnostics for top pairs

At subfield and field level, top pair lists can be dominated by a few outlier entities such as Philosophy, Molecular Medicine, Computer Science, etc.

Add diagnostics that count how often each entity appears in top converging/diverging pairs.

### New outputs

```text
outputs/analysis/morphological_similarity_evolution/top_pair_entity_frequency.csv
outputs/analysis/morphological_similarity_evolution/subfield_top_pair_entity_frequency.png
outputs/analysis/morphological_similarity_evolution/field_top_pair_entity_frequency.png
```

For each entity, include:

```text
level
distance_type
entity_id
entity_name
field_display_name
domain_display_name
n_top_converging_pairs
n_top_diverging_pairs
n_top_pairs_total
mean_delta_in_top_pairs
median_abs_delta_in_top_pairs
```

Make one plot for field level and one for subfield level:

- top 20 entities by total appearances in extreme pairs;
- separate color/stack for convergence vs divergence if possible;
- readable labels.

Interpretation note to add to summary:

```markdown
If a small number of entities dominate many extreme pairs, the pairwise results should be interpreted as evidence of those entities' unusual morphological evolution, not as many independent pairwise phenomena.
```

---

## 8. Add driver diagnostics for top pairs

For each top converging/diverging pair, include how much each entity changed individually.

Use outputs from script 19 when available:

```text
data/processed/temporal/subfield_overall_temporal_change_ranking.parquet
data/processed/temporal/subfield_metric_temporal_changes.parquet
```

For field/domain, aggregate these subfield-level change scores by mean/median.

Add columns to top-pair tables:

```text
entity_a_overall_metric_change_l2
entity_b_overall_metric_change_l2
entity_a_top_changed_metrics
entity_b_top_changed_metrics
```

For `entity_a_top_changed_metrics`, store a compact semicolon-separated string of the top 3 metrics by absolute standardized delta for that entity, e.g.:

```text
embedding_pca_dim_80:-2.3; embedding_knn_median_distance:-1.8; embedding_pca_spectral_entropy:-1.5
```

If driver diagnostics cannot be computed because script 19 outputs are missing, do not fail the whole script. Instead:

- write a warning to `summary.md`;
- leave driver columns as missing.

### New output

```text
outputs/analysis/morphological_similarity_evolution/top_pair_driver_diagnostics.csv
```

---

## 9. Add static morphology nearest/farthest summaries

The static full-period morphology is also useful.

Add static closest/farthest pairs for each level and distance type.

### New outputs

```text
outputs/analysis/morphological_similarity_evolution/static_pairs/<level>_<distance_type>_closest_static_pairs.csv
outputs/analysis/morphological_similarity_evolution/static_pairs/<level>_<distance_type>_farthest_static_pairs.csv
```

Optional figures:

```text
outputs/analysis/morphological_similarity_evolution/field_euclidean_static_closest_farthest_pairs.png
outputs/analysis/morphological_similarity_evolution/field_correlation_static_closest_farthest_pairs.png
```

This gives a simple answer to:

> Which fields have the most similar/different overall morphological profiles across the full period?

---

## 10. Update summary.md and summary.json

The summary should become interpretation-ready.

Add sections:

```markdown
## Purpose
## Methods
## Interpretation of Distance Types
## Entity Retention QA
## Key Static Results
## Key Temporal Results
## Dominant Entities in Extreme Pairs
## Caveats
```

Include:

1. Number of entities retained/dropped by level.
2. Distance types computed.
3. Metrics used.
4. Scaling method.
5. Top 5 static closest/farthest field pairs.
6. Top 5 temporal field-level Euclidean convergences/divergences.
7. Top 5 temporal field-level correlation convergences/divergences.
8. Most frequent entities in top pairs.
9. Warning that subfield extreme pairs may be outlier-driven.
10. Clear distinction from script 22.

---

# CLI expectations

The script should still run like:

```powershell
.\.venv\Scripts\python.exe scripts\23_compute_morphological_similarity_evolution.py `
  --year-min 2000 `
  --year-max 2024 `
  --overwrite
```

If additional flags are useful, add them, but keep defaults thesis-friendly:

```text
--top-n-pairs 10
--heatmap-levels domain field
--distance-types euclidean correlation
--scaler robust
```

---

# Testing

Update/add tests in:

```text
tests/test_temporal_morphology_scripts.py
```

At minimum test:

1. Converging tables include only negative deltas.
2. Diverging tables include only positive deltas.
3. Euclidean and correlation distance types are both produced.
4. Entity retention diagnostics are produced.
5. Top-pair entity frequency counts are correct on a tiny synthetic example.
6. Driver diagnostics do not crash if script 19 outputs are missing.

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall src scripts
.\.venv\Scripts\python.exe -m pytest tests/test_temporal_morphology_scripts.py
```

If full test suite is fast enough, run all tests.

---

# Final chat response required

When done, reply in chat with:

1. Files changed.
2. Exact command used.
3. Number of subfields/fields/domains retained and dropped.
4. Whether converging/diverging tables now filter signs correctly.
5. New figures created.
6. Key preliminary findings if the full run was executed.
7. Any caveats or warnings.

Do not claim full-run findings if only a smoke test was run.
