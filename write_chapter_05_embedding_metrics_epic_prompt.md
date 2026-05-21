# Agent Prompt — Write Chapter 5: Embedding-Space Morphological Metrics

Write Chapter 5 of the thesis: **Embedding-Space Morphological Metrics**.

Target file:

```text
memory/chapters/05_embedding_space_metrics.tex
```

The thesis is written in English. This chapter is the methodological core of the thesis. It must have intellectual personality: clear, confident, and conceptually sharp. The style should be academic, but not dull. Avoid generic textbook prose. Every paragraph must either define, justify, or clarify something necessary for the thesis.

Do **not** write a long catalogue of metrics. The chapter must make the reader understand the logic of the metric system: why these indicators exist, what aspect of scientific morphology they capture, and how they turn a cloud of embedded papers into interpretable field-level structure.

---

## Central idea of the chapter

The thesis treats each scientific subfield as a distribution of documents in the original SPECTER2 embedding space. Chapter 5 defines the morphological indicators used to describe those distributions.

The metrics should be presented as a deliberately reduced and interpretable core, not as an arbitrary feature set. The narrative should be:

> A discipline is not only a set of papers. In embedding space, it has shape: it may be compact or dispersed, locally dense or fragmented, hub-dominated or evenly connected, low- or high-dimensional, stable or drifting. The eleven metrics operationalize these geometric properties.

Use this idea as the spine of the chapter, but phrase it in polished academic English.

---

## Required methodology

The active methodology is based only on the original SPECTER2 embedding space.

Do **not** use UMAP metrics.  
Do **not** introduce UMAP as quantitative evidence.  
Do **not** revive archived metric-family comparisons.  
Do **not** introduce Leiden/Louvain clustering, Mutual Proximity, Local Scaling, LID, Dynamic Topic Models, or Stirling-Rafols indicators as active methods.  
Do **not** add new metrics beyond the reduced eleven-metric core.

UMAP can be mentioned only briefly as a visualization tool, not as a metric source.

---

## Reduced eleven-metric core

The chapter must define and justify the eleven active metrics.

### A. Semantic dispersion

1. `centroid_median_distance`
2. `centroid_distance_iqr`
3. `centroid_distance_p90`

Interpretation:
- how far papers lie from the subfield centroid;
- how concentrated or spread out the semantic territory is;
- whether the subfield has a broad tail of distant papers.

### B. Local density and hubness

4. `knn_median_distance`
5. `knn_distance_cv`
6. `knn_indegree_gini`

Interpretation:
- local packing of papers;
- heterogeneity of local neighborhoods;
- whether some papers become disproportionally common nearest neighbors.
- Cite `radovanovic2010hubness` when discussing hubness in high-dimensional spaces.

### C. Intrinsic dimensionality / spectral structure

7. `pca_dim80`
8. `pca_spectral_entropy`

Interpretation:
- how many principal components are needed to explain a large share of variance;
- whether the subfield is organized along a few dominant semantic axes or spread over many directions.
- Do not claim this is the true intrinsic dimension of the field. It is a PCA-based operational proxy.

### D. Temporal movement

9. `centroid_drift`
10. `radial_expansion_slope`
11. `recent_novelty_score`

Interpretation:
- movement of the subfield center across time;
- expansion or contraction of the field’s semantic radius;
- distance of recent papers from previous semantic structure.
- Explain these as temporal morphology indicators, not causal mechanisms.

---

## Expected chapter structure

Use around **4–5 sections maximum**, not 8–9. The table of contents must remain compact.

Recommended structure:

```latex
\section{From Embedded Papers to Morphological Indicators}
\section{Metric Families and Geometric Interpretation}
\section{Operational Definitions}
\section{Reduction to an Interpretable Metric Core}
\section{Interpretation and Limits of the Metrics}
```

You may adjust section titles, but keep the chapter compact.

---

## Visual and tabular content

Tables are encouraged if they carry dense information. Use them to reduce repetitive prose.

Create at least one thesis-ready table:

```text
memory/tables/tab_05_metric_core.tex
```

Suggested columns:

| Family | Metric | What it captures | Interpretation |
|---|---|---|---|

The table should include all eleven metrics.

Optionally create a second compact table if useful:

```text
memory/tables/tab_05_metric_interpretation_rules.tex
```

Suggested purpose:
- low vs high values;
- cautious interpretation;
- known limitations.

Do **not** create oversized tables that are too wide for the page. Use concise wording and sensible column widths.

If useful, create a small conceptual figure or diagram only if it genuinely clarifies the chapter. Do not create decorative visuals.

All tables must:
- use proper `\caption{...}` and `\label{...}`;
- appear in the List of Tables;
- have notes below the table body if needed;
- follow the current thesis table style.

---

## Mathematical content

Use equations where they add precision. Do not overload the chapter with notation.

Define notation clearly:

- Let \(Z_s = \{z_i : i \in s\}\) be the set of embeddings for subfield \(s\).
- Let \(c_s\) be the centroid of subfield \(s\).
- Let \(d(z_i, c_s)\) denote the distance between paper \(i\) and the subfield centroid.
- Let \(N_k(i)\) be the \(k\)-nearest-neighbor set of paper \(i\), if needed.
- Let \(T\) denote temporal windows when defining temporal indicators.

Keep formulas compact. Avoid long derivations unless necessary.

Potential equations to include:

1. Subfield centroid:
```latex
\[
c_s = \frac{1}{n_s}\sum_{i \in s} z_i.
\]
```

2. Centroid distance:
```latex
\[
r_i = d(z_i, c_s).
\]
```

3. PCA variance share / dimensionality proxy:
```latex
\[
D_{80}(s) = \min \left\{m : \frac{\sum_{j=1}^{m} \lambda_j}{\sum_{j=1}^{p} \lambda_j} \geq 0.80 \right\}.
\]
```

4. Spectral entropy:
```latex
\[
H(s) = -\sum_j p_j \log p_j,
\qquad
p_j = \frac{\lambda_j}{\sum_\ell \lambda_\ell}.
\]
```

Only include formulas that match the implemented methodology. If the code uses a slightly different convention, follow the code and document the actual implementation.

---

## Sources to inspect before writing

Inspect the active repo documentation and code before drafting, especially:

```text
docs/embedding_metrics.md
docs/reduced_metric_core.md
docs/temporal_evolution.md
docs/morphological_similarity.md
research/final_outputs/
src/
scripts/11_compute_embedding_space_metrics.py
scripts/12_build_reduced_metric_core.py
scripts/14_compute_temporal_metric_evolution.py
```

Use these to ensure metric names, definitions, and terminology match the actual implementation.

Do not copy README-style file paths into the chapter body. The chapter should explain the method, not the repository structure.

---

## Citations

Use citations sparingly and only where they support the methodological argument.

Likely useful citations from `memory/referencias.bib`:

- `radovanovic2010hubness` for hubness in high-dimensional spaces.
- `cohan2020specter` and `singh2023scirepeval` only if briefly reminding the reader that metrics operate on SPECTER2 scientific document embeddings.
- `chari2023specious`, `mcinnes2018umap`, or `wattenberg2016tsne` only if warning that UMAP projections are not the metric space. Do not overuse them here.
- Avoid citing sources for methods not actively implemented unless explicitly framed as background or limitation.

Do not invent citations.

---

## Tone and personality

The chapter should sound like it has an argument, not like it is merely documenting code.

Use controlled but memorable phrasing. For example, ideas like these are acceptable if written academically:

- “The purpose of the metric core is to turn geometry into interpretable morphology.”
- “A compact field and a fragmented field can contain the same number of papers; their difference lies in how those papers occupy semantic space.”
- “The metrics do not name disciplines; they describe their shape under a specific representation model.”
- “The analysis is deliberately modest: it measures reproducible structure in an embedding space, not the essence of science.”

Do not become poetic or exaggerated. Personality should come from conceptual clarity, not decorative language.

---

## Prose requirements

Every paragraph must add value.

Avoid:
- filler introductions;
- repeated caveats;
- long generic explanations of embeddings already covered in Chapter 4;
- repeated statements that the space is model-dependent;
- saying the same limitation in several sections;
- excessive sectioning;
- raw file paths;
- code-like wording;
- overclaiming.

The reader should finish this chapter knowing:
1. what each metric measures;
2. why these metrics are grouped into families;
3. why the final set has eleven metrics;
4. how to interpret low/high values;
5. what the metrics cannot prove.

---

## Required output

After writing the chapter:

1. Update `memory/chapters/05_embedding_space_metrics.tex`.
2. Create or update `memory/tables/tab_05_metric_core.tex`.
3. Optionally create `memory/tables/tab_05_metric_interpretation_rules.tex` if it adds real value.
4. Compile with Biber.
5. Verify:
   - citations render in APA author-year style;
   - bibliography appears correctly;
   - all new tables appear in the List of Tables;
   - no unresolved citation keys;
   - no internal repository paths in Chapter 5;
   - no UMAP metrics are introduced;
   - no legacy or future-work methods are presented as active methodology;
   - section count is compact, ideally 4–5 sections.
6. Report:
   - final section structure;
   - tables created;
   - citations used;
   - any files changed;
   - whether compilation succeeded;
   - any remaining warnings.
