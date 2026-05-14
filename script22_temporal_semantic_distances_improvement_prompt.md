# Prompt for Agent — Improve Script 22 Temporal Semantic Distances

You are working in the local repository:

```text
C:\Users\Z0058EYW\Workspace\TFM
```

The temporal semantic distance layer has already been implemented in:

```text
scripts/22_compute_temporal_semantic_distances.py
```

Reusable code is likely in:

```text
src/temporal_semantic_distances.py
src/temporal_common.py
```

The current script works and produces useful outputs, but several improvements are needed before the results are thesis-ready.

This task is **not** to redesign the whole analysis. It is to improve interpretability, robustness, and visual outputs of script 22.

---

## Current purpose of script 22

Script 22 studies whether disciplines/subfields become semantically closer or further apart over time.

It compares **centroid positions** in the original SPECTER2 embedding space.

Interpretation:

```text
delta_distance = final_distance - initial_distance
```

- Negative delta → semantic convergence.
- Positive delta → semantic divergence.
- Near zero → stable semantic relationship.

This is different from morphological similarity:

- Semantic distance = topic/content proximity between centroids.
- Morphological distance = similarity of internal structure/metric profiles.

Keep this distinction explicit in all summaries and figure captions.

---

# Main issues to fix

## 1. Strictly separate convergence and divergence tables

Currently, some top-pair tables can be confusing, especially at domain level, because with few pairs the "top converging" and "top diverging" outputs may include pairs with the opposite sign.

Fix this.

Create/overwrite:

```text
outputs/analysis/temporal_semantic_distances/top_semantic_converging_pairs.csv
outputs/analysis/temporal_semantic_distances/top_semantic_diverging_pairs.csv
```

Rules:

```python
converging_pairs = pairs[pairs["delta_distance"] < 0].sort_values("delta_distance", ascending=True)
diverging_pairs = pairs[pairs["delta_distance"] > 0].sort_values("delta_distance", ascending=False)
```

If there are fewer than `top_n` valid converging/diverging pairs for a given level, output fewer rows. Do not include positive deltas in the converging table, and do not include negative deltas in the diverging table.

Also create neutral versions that may include both signs if useful:

```text
outputs/analysis/temporal_semantic_distances/most_negative_delta_pairs.csv
outputs/analysis/temporal_semantic_distances/most_positive_delta_pairs.csv
```

but the primary files should be semantically correct.

Add a QA check to `summary.md` and `summary.json`:

- number of converging pairs by level;
- number of diverging pairs by level;
- number of stable/zero-delta pairs by level;
- min/max delta by level.

---

## 2. Prioritize delta heatmaps over final-distance heatmaps

The current final-distance heatmaps are useful, but for the temporal story the most important object is the **change** in distance.

Create clear delta heatmaps for all levels:

```text
outputs/analysis/temporal_semantic_distances/domain_distance_delta_heatmap.png
outputs/analysis/temporal_semantic_distances/field_distance_delta_heatmap.png
outputs/analysis/temporal_semantic_distances/subfield_distance_delta_heatmap.png
```

For subfields, full 241×241 heatmaps are too dense, so:

- still save the full matrix as CSV;
- create a readable figure using either:
  - top N most dynamic subfields based on absolute semantic distance change, or
  - top N subfields by centroid path movement from script 20, if available.

Add CLI argument:

```bash
--heatmap-top-n 40
```

Default: `40` for subfield delta heatmaps.

Delta heatmap requirements:

- diverging colormap centered at 0;
- symmetric color limits around the maximum absolute delta;
- clear caption/title:
  - "Semantic Distance Change, 2020–2024 minus 2000–2004"
- blue/teal or cool color for convergence if possible;
- orange/red or warm color for divergence if possible;
- diagonal masked or set to 0;
- readable labels for domain and field;
- abbreviated labels for subfields.

Important: do not use separate color scales for initial/final/delta in ways that make comparison misleading.

---

## 3. Add initial/final/delta triptych figures

For domain and field levels, create a single figure with three panels:

```text
initial distance | final distance | delta distance
```

New outputs:

```text
outputs/analysis/temporal_semantic_distances/domain_distance_initial_final_delta_triptych.png
outputs/analysis/temporal_semantic_distances/field_distance_initial_final_delta_triptych.png
```

Optional for selected subfields:

```text
outputs/analysis/temporal_semantic_distances/subfield_distance_initial_final_delta_triptych_topN.png
```

Requirements:

- Initial = `2000-2004`
- Final = `2020-2024`
- Delta = final - initial
- Same ordering across all three panels.
- For initial/final panels, use a sequential colormap.
- For delta panel, use a diverging colormap centered at 0.
- Domain triptych should be compact.
- Field triptych should be readable; if there are too many labels, use smaller font, rotation, or a clustered order. But the order must be the same across panels.

---

## 4. Improve top convergence/divergence barplots

The current field barplot is useful but visually dense.

Create cleaner plots:

```text
outputs/analysis/temporal_semantic_distances/domain_top_semantic_converging_diverging_pairs_clean.png
outputs/analysis/temporal_semantic_distances/field_top_semantic_converging_diverging_pairs_clean.png
outputs/analysis/temporal_semantic_distances/subfield_top_semantic_converging_diverging_pairs_clean.png
```

Requirements:

- Show top 10 converging and top 10 diverging pairs by default.
- Use two separated panels or one diverging horizontal bar chart with a clear zero line.
- Do not include pairs of the wrong sign.
- Use short labels:
  - `"Life Sciences ↔ Social Sciences"`
  - `"Economics ↔ Medicine"`
  - for subfields use `"Subfield A ↔ Subfield B"` but abbreviate if too long.
- Annotate numeric delta values or make axis labels clear.
- Title should mention:
  - "Delta cosine distance, final window minus initial window"
- Caption/note should state:
  - Negative = semantic convergence.
  - Positive = semantic divergence.

Add CLI arguments if not present:

```bash
--top-n-pairs 10
--max-label-length 55
```

---

## 5. Add "movement driver" diagnostics for top pairs

For any top converging/diverging pair, we need to understand whether the change is driven by entity A, entity B, or both.

Use the centroid path metrics produced by script 20 if available:

```text
data/processed/temporal/subfield_centroid_path_metrics.parquet
```

For field/domain levels, aggregate subfield centroid movement using weighted or median summaries.

Add these columns to top-pair tables:

```text
entity_a_path_length
entity_b_path_length
entity_a_early_late_drift
entity_b_early_late_drift
entity_a_n_windows_available
entity_b_n_windows_available
driver_note
```

For `driver_note`, implement a simple rule:

```python
if both have high movement:
    "both_move"
elif entity_a movement much larger:
    "mostly_entity_a"
elif entity_b movement much larger:
    "mostly_entity_b"
else:
    "small_or_balanced_movement"
```

Define "high movement" relative to the median movement at that level. Keep the rule simple and document it.

If script 20 outputs are unavailable, do not fail. Instead:

- write the top-pair tables without driver columns;
- add a warning to `summary.md`.

---

## 6. Add direct paper-level aggregation sensitivity option

The current summary says:

> Field and domain centroids are weighted averages of subfield-window centroids using `n_used`; direct paper-level aggregation can be used as a future sensitivity check.

Add an optional mode to compute field/domain centroids directly from paper-level embeddings if feasible.

CLI argument:

```bash
--aggregation-mode weighted_subfield_centroids
```

and optional:

```bash
--aggregation-mode direct_paper_centroids
```

Default should remain `weighted_subfield_centroids` for speed/backward compatibility.

If `direct_paper_centroids` is used:

- compute field/domain centroids directly from all paper embeddings in each group-window;
- write separate outputs or include `aggregation_mode` in filenames/columns;
- do not overwrite weighted outputs unless explicitly asked;
- document runtime implications.

At minimum, implement the CLI structure and, if direct mode is too heavy, add a clear TODO/warning rather than silently ignoring it.

---

## 7. Add readable subfield-level outputs

Subfield-level matrices are too large for direct interpretation, but they are analytically important.

Create:

```text
outputs/analysis/temporal_semantic_distances/subfield_top_semantic_converging_pairs.csv
outputs/analysis/temporal_semantic_distances/subfield_top_semantic_diverging_pairs.csv
outputs/analysis/temporal_semantic_distances/subfield_top_semantic_converging_diverging_pairs_clean.png
```

Also create an optional simple network plot for extreme subfield pairs:

```text
outputs/analysis/temporal_semantic_distances/subfield_semantic_convergence_divergence_network.png
```

Network plot requirements:

- include top 20 converging and top 20 diverging pairs;
- nodes = subfields;
- edge color = convergence/divergence;
- edge width proportional to absolute delta;
- node labels shortened;
- if `networkx` is unavailable, skip gracefully and write a warning.

This network is optional but useful.

---

## 8. Improve summary.md and summary.json

Update:

```text
outputs/analysis/temporal_semantic_distances/summary.md
outputs/analysis/temporal_semantic_distances/summary.json
```

Add sections:

```markdown
## Interpretation

Semantic distance compares centroid positions in the original embedding space. It measures topic/content proximity, not internal morphological similarity.

## QA Checks

- Number of pair-distance rows.
- Number of pair-change rows.
- Number of converging/diverging/stable pairs by level.
- Min/max delta by level.
- Aggregation mode used for field/domain centroids.
- Whether driver diagnostics were merged.
- Whether direct paper-level aggregation sensitivity was run.

## Main results

- Top 10 semantic convergences by level.
- Top 10 semantic divergences by level.
- Domain-level change summary.
- Field-level notable patterns.
- Warning that domain changes are small and should not be overinterpreted if delta magnitudes are tiny.

## Caveats

- Field/domain centroids are aggregate summaries and may hide subfield-level heterogeneity.
- Weighted subfield-centroid aggregation is not identical to direct paper-level aggregation.
- Full subfield matrices are saved but too dense for direct visual interpretation.
```

Make the summary honest and concise.

---

# Output checklist

After implementation and a full run, these outputs should exist:

```text
data/processed/temporal/semantic_pair_distances_by_window.parquet
data/processed/temporal/semantic_pair_distance_changes.parquet

outputs/analysis/temporal_semantic_distances/top_semantic_converging_pairs.csv
outputs/analysis/temporal_semantic_distances/top_semantic_diverging_pairs.csv
outputs/analysis/temporal_semantic_distances/most_negative_delta_pairs.csv
outputs/analysis/temporal_semantic_distances/most_positive_delta_pairs.csv

outputs/analysis/temporal_semantic_distances/domain_distance_delta_heatmap.png
outputs/analysis/temporal_semantic_distances/field_distance_delta_heatmap.png
outputs/analysis/temporal_semantic_distances/subfield_distance_delta_heatmap.png

outputs/analysis/temporal_semantic_distances/domain_distance_initial_final_delta_triptych.png
outputs/analysis/temporal_semantic_distances/field_distance_initial_final_delta_triptych.png

outputs/analysis/temporal_semantic_distances/domain_top_semantic_converging_diverging_pairs_clean.png
outputs/analysis/temporal_semantic_distances/field_top_semantic_converging_diverging_pairs_clean.png
outputs/analysis/temporal_semantic_distances/subfield_top_semantic_converging_diverging_pairs_clean.png

outputs/analysis/temporal_semantic_distances/subfield_top_semantic_converging_pairs.csv
outputs/analysis/temporal_semantic_distances/subfield_top_semantic_diverging_pairs.csv

outputs/analysis/temporal_semantic_distances/summary.md
outputs/analysis/temporal_semantic_distances/summary.json
```

Existing outputs should not be broken:

```text
outputs/analysis/temporal_semantic_distances/matrices/<level>_<window>_distance_matrix.csv
outputs/analysis/temporal_semantic_distances/matrices/<level>_distance_delta_matrix.csv
outputs/analysis/temporal_semantic_distances/matrices/<level>_distance_slope_matrix.csv
```

---

# Recommended run command

Use the current 2000–2024 SPECTER2 embeddings:

```powershell
cd C:\Users\Z0058EYW\Workspace\TFM
$env:LOCAL_EMBEDDINGS_DIR = "embeddings/specter2_v1_2000_2024_400py"

.\.venv\Scripts\python.exe scripts\22_compute_temporal_semantic_distances.py `
  --embedding-dir embeddings/specter2_v1_2000_2024_400py `
  --year-min 2000 `
  --year-max 2024 `
  --aggregation-mode weighted_subfield_centroids `
  --top-n-pairs 10 `
  --heatmap-top-n 40 `
  --overwrite
```

If `--embedding-dir`, `--year-min`, or `--year-max` are not currently supported by the script, either add them or document why they are unnecessary.

---

# Testing

Update relevant tests in:

```text
tests/test_temporal_morphology_scripts.py
```

At minimum test:

1. Converging table contains only `delta_distance < 0`.
2. Diverging table contains only `delta_distance > 0`.
3. Delta heatmap matrix is symmetric and has zero diagonal.
4. Pair labels are stable and deterministic.
5. Driver diagnostics do not fail if script 20 outputs are missing.
6. `aggregation_mode` is stored in output summary.

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall src scripts
.\.venv\Scripts\python.exe -m pytest tests/test_temporal_morphology_scripts.py
```

If possible, run the full test suite.

---

# Final chat response required

After implementation, report:

1. Files changed.
2. Exact command run.
3. Whether strict convergence/divergence filtering is fixed.
4. Number of converging/diverging pairs by level.
5. New figures created.
6. Whether driver diagnostics were merged.
7. Whether direct paper-level aggregation was implemented or left as TODO.
8. Which outputs are now recommended for the thesis.
9. Any caveats or failures.

Do not overclaim. If only smoke-tested, say so.
