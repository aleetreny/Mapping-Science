# Agent Prompt — Explore Temporal-Trajectory and Hybrid Clustering Sensitivity for Chapter 9

The current Chapter 9 uses morphological clustering over the reduced eleven-metric profile and correctly interprets the result as weak/descriptive. A centroid-embedding sensitivity check was also added in Appendix C. Now explore two additional routes:

1. **Temporal trajectory clustering**
2. **Hybrid centroid + morphology clustering**

The goal is not to force stronger clusters. The goal is to test whether other reasonable representations produce more stable, meaningful, and interpretable groupings.

Do not rewrite Chapter 9 unless the results materially improve or clarify the interpretation. Prefer adding a short appendix subsection and a short bridge paragraph in Chapter 9 if useful.

---

## Current baseline to preserve

Keep the current selected Chapter 9 result as baseline unless the new evidence is clearly superior:

- Main morphology typology: Ward hierarchical clustering, Euclidean, robust scaling, `k=5`.
- Morphology silhouette: `0.133`.
- Interpretation: weak/descriptive typology; useful anchors, fuzzy boundaries.
- Centroid sensitivity: best balanced centroid solution `centroid_average_cosine_k3`.
- Centroid silhouette: `0.236`.
- Centroid clusters align more with OpenAlex domains than with morphology typologies.
- Do not change these conclusions unless the new diagnostics strongly justify it.

---

# Part A — Temporal Trajectory Clustering

## Objective

Cluster subfields by **how their morphology changes over time**, rather than by their full-period static profile.

This should answer:

1. Are there interpretable temporal types such as contracting, densifying, drifting, stabilizing, expanding, or increasing local unevenness?
2. Are temporal trajectory clusters stronger or more interpretable than the full-period morphology typology?
3. Do dynamic groups cut across OpenAlex domains?
4. Do they explain Chapter 7 findings better than the current typology?

## Input

Use the active five-year window metrics:

- 2000--2004
- 2005--2009
- 2010--2014
- 2015--2019
- 2020--2024

Use the eight structural metrics recomputed by window, not the three full-period temporal indicators as repeated window features.

Likely source folders:

```text
outputs/06_temporal_evolution/
outputs/04_reduced_metric_core/
data/processed/
research/final_outputs/
```

Inspect the actual repo and use the authoritative files.

## Feature representations to try

Explore multiple temporal representations:

### A1. First--last change vector

For each subfield:

```text
final_window_metric_z - initial_window_metric_z
```

across the eight windowed structural metrics.

This captures net structural displacement.

### A2. Linear slope vector

For each subfield and each windowed structural metric, fit a simple linear trend across the five windows.

This captures direction and speed of change.

### A3. Full trajectory vector

Concatenate standardized metric values across windows:

```text
metric_1_t1, ..., metric_1_t5, ..., metric_8_t1, ..., metric_8_t5
```

This captures trajectory shape, not only first--last movement. Use carefully because dimensionality is higher.

### A4. Dynamic summary vector

Combine:

- first--last changes;
- slopes;
- volatility/standard deviation across windows;
- accumulated profile movement if already computed.

Use only if it improves interpretability and does not duplicate other representations.

## Methods to test

For each representation, test:

- hierarchical Ward, Euclidean;
- average linkage, Euclidean;
- average linkage, correlation distance if appropriate;
- k-means;
- k-medoids if available;
- spectral clustering if easy and reproducible;
- HDBSCAN only if available and sensible.

Explore `k = 3,...,10` where applicable.

## Diagnostics

For each candidate:

- silhouette;
- cluster sizes;
- Calinski-Harabasz / Davies-Bouldin if available;
- ARI/AMI against current morphology typology;
- ARI/AMI against OpenAlex domains;
- stability under bootstrap/subsampling if feasible;
- interpretability of cluster profiles.

Do not select a solution just because silhouette is higher. Avoid singleton-heavy or degenerate solutions.

## Possible outputs

Create:

```text
outputs/09_morphological_typologies/temporal_trajectory_clustering/
```

Include:

```text
trajectory_cluster_model_comparison.csv
trajectory_cluster_solution_selected.csv
trajectory_cluster_profile_summary.csv
trajectory_cluster_domain_composition.csv
trajectory_cluster_stability_summary.csv
trajectory_clustering_summary.md
```

Possible figures:

```text
fig_c_temporal_cluster_quality.pdf/png
fig_c_temporal_typology_profile_heatmap.pdf/png
fig_c_temporal_trajectory_examples.pdf/png
fig_c_temporal_cluster_map.pdf/png
```

Use only the strongest figures in the appendix.

---

# Part B — Hybrid Centroid + Morphology Clustering

## Objective

Test whether combining **semantic location** and **morphological shape** yields clearer or more meaningful clusters.

This should answer:

1. Do subfields cluster better when we combine where they are in SPECTER2 space with how their paper distributions are shaped?
2. Are hybrid clusters mainly topic/domain clusters, morphology clusters, or something genuinely mixed?
3. Does the hybrid representation improve interpretability without collapsing into OpenAlex domains?
4. Does it clarify why morphology-only clustering is weak?

## Input

Use:

1. Subfield centroid embeddings from the centroid sensitivity pipeline.
2. Robust-scaled eleven-metric morphology profiles from the main typology pipeline.

Do not use UMAP/PCA coordinates as clustering features.

## Feature representations to try

### B1. Concatenated reduced representation

Avoid directly concatenating 768 centroid dimensions with 11 morphology metrics without balancing.

First reduce centroid embeddings using PCA:

- 10 components;
- 25 components;
- enough components for 80 percent variance if sensible.

Then concatenate:

```text
centroid_PCs + morphology_profile
```

Scale blocks so centroid information does not swamp morphology.

### B2. Weighted hybrid distances

Construct a combined distance:

```text
D_hybrid = alpha * D_centroid + (1 - alpha) * D_morphology
```

Try:

```text
alpha = 0.25, 0.50, 0.75
```

where:

- `D_centroid` uses cosine distance between subfield centroid embeddings;
- `D_morphology` uses Euclidean distance between robust-scaled eleven-metric profiles.

Normalize both distance matrices before combining.

This may be more transparent than feature concatenation.

### B3. Two-view consensus clustering

If feasible, build a co-association or consensus view from:

- morphology clustering;
- centroid clustering;
- temporal trajectory clustering if selected.

Use this only if implementation remains clean and interpretable.

## Methods to test

For hybrid representations, test:

- hierarchical clustering on hybrid distance matrices;
- k-medoids if available;
- spectral clustering on affinity matrix if appropriate;
- k-means only for feature-concatenation variants.

Explore `k = 3,...,10`.

## Diagnostics

For each candidate:

- silhouette in the hybrid distance/feature space;
- cluster sizes;
- ARI/AMI against morphology typology;
- ARI/AMI against centroid clusters;
- ARI/AMI against OpenAlex domains;
- stability under alpha changes;
- interpretability of cluster profiles;
- whether clusters are dominated by domains or by morphology.

Important: If hybrid clustering just reconstructs OpenAlex domains, state that. That may still be useful, but it is not a new morphological result.

## Possible outputs

Create:

```text
outputs/09_morphological_typologies/hybrid_centroid_morphology_clustering/
```

Include:

```text
hybrid_cluster_model_comparison.csv
hybrid_cluster_solution_selected.csv
hybrid_cluster_profile_summary.csv
hybrid_cluster_domain_composition.csv
hybrid_cluster_stability_summary.csv
hybrid_clustering_summary.md
```

Possible figures:

```text
fig_c_hybrid_quality_by_alpha.pdf/png
fig_c_hybrid_typology_profile_heatmap.pdf/png
fig_c_hybrid_domain_composition.pdf/png
fig_c_hybrid_embedding_vs_morphology_map.pdf/png
```

Use only if they add value.

---

# Selection and Interpretation Rules

You are not required to replace the current Chapter 9 method.

After the exploration, decide one of the following:

## Outcome 1 — No stronger meaningful clustering

If temporal and hybrid clustering are also weak or unstable, keep Chapter 9 unchanged except for a short paragraph saying that additional trajectory and hybrid checks support the interpretation of a mostly continuous profile space.

## Outcome 2 — Temporal clustering is meaningful

If temporal trajectory clustering produces interpretable and reasonably stable groups, add it as an appendix sensitivity and mention in Chapter 9 that dynamic morphology may be better summarized separately from full-period morphology.

Do not replace the main typology unless the result is clearly superior.

## Outcome 3 — Hybrid clustering is meaningful

If hybrid clustering gives stronger and interpretable groups, explain whether the strength comes from semantic location, morphology, or both.

Be careful: if hybrid clusters mostly reproduce domains, say that semantic location dominates the hybrid representation.

## Outcome 4 — A genuinely better final typology emerges

Only if a new solution is clearly more interpretable, stable, balanced, and methodologically defensible than the current Ward k=5 morphology typology, propose replacing the main Chapter 9 solution.

Do not replace it silently. Report the evidence and explain the tradeoff.

---

# Required Appendix Integration

If any new result adds value, update Appendix C with a compact subsection:

```latex
\section{Temporal and Hybrid Clustering Sensitivity}
```

or split into:

```latex
\section{Temporal-Trajectory Clustering Sensitivity}
\section{Hybrid Centroid--Morphology Clustering Sensitivity}
```

Keep it concise.

Include at most 1--2 appendix figures unless the results are genuinely important.

Possible appendix figure:

- Panel A: morphology profile PCA colored by current typology.
- Panel B: temporal trajectory map colored by selected temporal clusters.
- Panel C: centroid embedding PCA colored by centroid clusters.
- Panel D: hybrid map or domain composition if useful.

Make clear that projections are visualization only.

---

# Possible Chapter 9 Update

If useful, add one short paragraph near the end of Section 9.5:

- Mention that Appendix C compares the morphology typology with centroid, temporal-trajectory, and hybrid clustering.
- State the core result honestly:
  - semantic location separates more than morphology;
  - temporal dynamics may or may not form separate dynamic types;
  - hybrid clustering may be more domain-like if centroid information dominates;
  - none of these checks should be interpreted as proving natural disciplinary clusters unless the evidence is strong.

Do not bloat Chapter 9.

---

# Code Requirements

Create clean code, preferably:

```text
src/temporal_trajectory_clustering.py
src/hybrid_morphology_centroid_clustering.py
scripts/20_cluster_temporal_trajectories.py
scripts/21_cluster_hybrid_centroid_morphology.py
```

or equivalent names consistent with the repo.

Requirements:

- no hard-coded absolute paths;
- reproducible random seeds;
- clear CLI arguments such as `--overwrite`;
- outputs saved under the new subfolders;
- readable diagnostic summaries;
- schema checks;
- no UMAP/PCA coordinates used as clustering features;
- no modification of large raw data files.

Add lightweight tests if feasible:

```text
tests/test_temporal_trajectory_clustering.py
tests/test_hybrid_morphology_centroid_clustering.py
```

At minimum, test:

- feature matrix shapes;
- no missing values in selected feature matrix;
- selected output schema;
- reproducibility under fixed seed;
- no projection coordinates used as clustering input.

---

# Final Verification

Run:

- the new scripts;
- relevant tests;
- thesis compilation with Biber.

Verify:

- no broken references;
- no unresolved citations;
- new appendix figures appear in the List of Figures if included;
- Chapter 9 remains concise;
- no raw repository paths in prose;
- no unsupported claims;
- current main typology is preserved unless replacement is explicitly justified.

---

# Final Report

Report:

1. temporal representations tested;
2. hybrid representations tested;
3. best temporal solution and diagnostics;
4. best hybrid solution and diagnostics;
5. comparison with current morphology typology and centroid-only clusters;
6. whether any result is strong enough to affect Chapter 9;
7. figures/tables added or avoided;
8. files/scripts/tests created;
9. whether compilation succeeded;
10. final recommendation: keep, revise, or replace current Chapter 9 typology.

Be ambitious in exploration, but conservative in interpretation.
