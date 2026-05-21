# Agent Prompt — Write Chapter 6: Static Comparison of Scientific Disciplines

Write Chapter 6 of the thesis: **Static Comparison of Scientific Disciplines**.

Target file:

```text
memory/chapters/06_static_comparison.tex
```

The thesis is written in English. This is the first empirical results chapter, so it must be visually strong and analytically convincing. The chapter should not merely describe outputs; it must extract patterns from the eleven embedding-space morphological metrics and explain what they reveal about differences between scientific disciplines.

The style should be rigorous, concise, and confident. Use visual evidence heavily, but every figure must serve an argument.

---

## Central objective

Chapter 6 compares scientific disciplines statically, using the reduced eleven-metric core defined in Chapter 5.

The chapter should answer:

1. How do domains, fields, and subfields differ in their embedding-space morphology?
2. Which scientific areas are more compact, dispersed, locally dense, hub-dominated, spectrally complex, or morphologically atypical?
3. Are there broad domain-level patterns, or is most variation found at field/subfield level?
4. Which disciplines occupy extreme positions in the metric space?
5. What can be interpreted from static morphology, and what cannot?

The chapter should make clear that this is a **cross-sectional morphological comparison**, not a causal explanation.

---

## Methodological constraints

Use only the active methodology:

- original SPECTER2 embedding space;
- reduced eleven-metric core;
- static discipline profiles;
- OpenAlex hierarchy: subfields, fields, domains.

Do **not** use UMAP metrics.  
Do **not** treat 2D projections as quantitative evidence.  
Do **not** introduce new metrics.  
Do **not** introduce clustering as the main result here; clustering belongs to Chapter 9.  
Do **not** overclaim that metric differences reveal the true nature of disciplines.

UMAP or PCA-style maps of **metric profiles** may be used only as exploratory visual summaries, not as proof of objective categories.

---

## Data and outputs to inspect

Before writing, inspect the repo and use the actual generated outputs.

Likely relevant sources:

```text
docs/reduced_metric_core.md
docs/embedding_metrics.md
docs/output_organization.md
research/final_outputs/
outputs/04_reduced_metric_core/
outputs/05_static_comparison/
data/processed/subfield_embedding_space_metrics.parquet
scripts/13_analyze_static_discipline_profiles.py
src/static_discipline_profiles.py
```

Use the actual available files and metric names. If some expected output is missing, generate it from the active processed metric tables rather than inventing numbers.

Do not put raw file paths in the main prose. File paths and execution details belong in Appendix A if needed.

---

## Required analytical logic

The chapter should be organized around morphological interpretation, not around code outputs.

Recommended structure: around **4–5 sections maximum**.

Suggested structure:

```latex
\section{Static Morphology as a Cross-Sectional Profile}
\section{Domain-Level Morphological Patterns}
\section{Field and Subfield Variation}
\section{Morphological Extremes and Outlying Profiles}
\section{Interpretive Limits of Static Comparison}
```

You may adjust the section titles, but keep the chapter compact.

---

## Visual expectations

I want this chapter to look visually attractive and information-rich. The graphics should be polished, thesis-ready, and genuinely useful.

Create figures in:

```text
memory/figures/
```

or the existing thesis figure folder used by the template.

Use high resolution and also save source copies where appropriate:

- `.pdf` for LaTeX/vector quality when possible;
- `.png` for quick viewing;
- avoid low-resolution screenshots.

Use a consistent visual style across all figures:
- clean background;
- readable labels;
- no clutter;
- thoughtful ordering;
- restrained color palette;
- domain-consistent colors if useful;
- no unnecessary decoration;
- captions that explain the analytical message, not just the chart type.

Do **not** use overly generic plots. Every chart must answer a real question.

---

## Suggested figures

Create a focused set of **3–5 strong figures**, not many mediocre ones.

### Figure 6.1 — Domain metric profile heatmap

A compact heatmap showing standardized values of the eleven metrics by OpenAlex domain.

Purpose:
- show broad cross-domain morphology;
- identify whether domains differ systematically in dispersion, density, hubness, spectral complexity, or temporal indicators.

Design:
- rows = domains;
- columns = eleven metrics, grouped by metric family;
- values = standardized metric means or medians;
- clear legend;
- metric-family separators if possible;
- use readable metric labels, not raw code names.

### Figure 6.2 — Field-level morphological heatmap

A richer heatmap showing field-level standardized profiles.

Purpose:
- show that variation is not only domain-level;
- identify fields with distinctive profiles.

Design:
- rows = fields, optionally ordered by domain and/or hierarchical clustering of metric profiles;
- columns = eleven metrics;
- domain annotation strip if possible;
- readable y-axis labels;
- avoid overcrowding.

### Figure 6.3 — Distribution of subfield metrics by domain

Use a visually compact plot such as boxplots, violin/box hybrids, or small multiples.

Purpose:
- show within-domain heterogeneity;
- demonstrate whether domain averages hide substantial subfield variation.

Design:
- focus on representative metrics from each family, not necessarily all eleven if too crowded;
- possible metrics:
  - median centroid distance;
  - median kNN distance;
  - kNN in-degree Gini;
  - PCA dim80;
  - centroid drift or recent novelty.
- use facets/small multiples if readable.

### Figure 6.4 — Metric-profile map of subfields

A 2D visualization of subfields based on the eleven standardized metric profiles, not on paper-level embeddings.

Purpose:
- summarize morphological similarity between subfields;
- identify outlying subfields;
- prepare the reader for later similarity and clustering chapters without doing clustering here.

Design:
- use PCA, UMAP, or another projection of the 11-dimensional metric-profile table;
- clearly state this is an exploratory visualization of metric profiles;
- color by domain;
- label only selected outliers or representative subfields;
- do not over-label.

### Figure 6.5 — Morphological extremes / ranked outliers

A ranked chart or compact table/figure showing the most extreme subfields for selected metrics.

Purpose:
- make the results concrete;
- show what “high dispersion”, “high hubness”, or “high spectral complexity” looks like in actual disciplines.

Design:
- top/bottom 5 or 10 subfields for a small set of metrics;
- avoid giant tables;
- consider a multi-panel lollipop chart or compact ranked table.

---

## Tables

Create one or two tables only if they add value. Do not duplicate what the figures show.

Possible table:

```text
memory/tables/tab_06_static_summary_by_domain.tex
```

Suggested content:
- domain;
- number of analysis subfields;
- selected standardized metric summaries;
- one short morphological interpretation.

Another possible table:

```text
memory/tables/tab_06_extreme_subfields.tex
```

Suggested content:
- metric;
- high-end subfields;
- low-end subfields;
- interpretation.

Tables must:
- use proper `\caption{...}` and `\label{...}`;
- appear in the List of Tables;
- be readable on the page;
- not include raw code-like metric names unless unavoidable.

---

## Prose requirements

Every paragraph must add value. The text should interpret patterns, not narrate that “Figure X shows...”.

Good paragraph structure:
1. State the finding.
2. Point to the figure/table.
3. Explain the morphological interpretation.
4. Add caution if necessary.

Avoid:
- describing every cell of a heatmap;
- listing metrics mechanically;
- saying “as can be seen” repeatedly;
- generic claims like “there are differences between fields” without specifying what kind;
- overclaiming causal or epistemological conclusions;
- repeating the same limitation from Chapters 3–5.

Use strong but careful language:
- “suggests”
- “is consistent with”
- “indicates under this representation”
- “should be interpreted as”
- “does not imply”

Avoid:
- “proves”
- “reveals the true structure”
- “disciplines are naturally divided”
- “this field is better/worse”

---

## Static analysis expectations

The chapter should include:

1. A brief reminder that all metrics are computed on the original 768-dimensional SPECTER2 space.
2. A description of how profiles are standardized for cross-metric visual comparison.
3. Domain-level comparison.
4. Field-level comparison.
5. Subfield-level heterogeneity.
6. Outlier/extreme profile interpretation.
7. A clear statement of what static comparison can and cannot support.

---

## Important interpretive distinction

Distinguish carefully between:

- **corpus size**: how many works are in a unit;
- **semantic dispersion**: how spread out works are in embedding space;
- **local density**: how close papers are to their nearest neighbors;
- **hubness**: whether some papers dominate neighborhood structure;
- **spectral complexity**: how many directions are needed to represent variance;
- **temporal indicators**: included in the reduced metric core, but interpreted here only as static summaries of each subfield's full-period trajectory.

Do not confuse these concepts.

---

## Captions

Figure captions should be analytical. Avoid captions like:

> Heatmap of metrics.

Use captions like:

> Standardized domain-level morphological profiles. Values are z-scored across domains for each metric; positive values indicate domains above the cross-domain average on that metric. Metrics are grouped by dispersion, local density, hubness, spectral structure, and temporal movement.

Tables and figures should be understandable without reading the entire section.

---

## Output requirements

After completing the chapter:

1. Update:

```text
memory/chapters/06_static_comparison.tex
```

2. Create/update thesis-ready figures, preferably with both `.pdf` and `.png` versions where appropriate.

3. Create/update any required tables in:

```text
memory/tables/
```

4. Ensure the chapter includes figures using proper LaTeX `figure` environments with:
   - `\caption{...}`;
   - `\label{...}`;
   - correct sizing;
   - captions above or consistent with the template style;
   - no broken paths.

5. Compile with Biber.

6. Verify:
   - citations render correctly;
   - bibliography appears correctly;
   - all figures appear in the List of Figures;
   - all tables appear in the List of Tables;
   - no unresolved references;
   - no overfull visual elements;
   - no raw repository paths in the main chapter;
   - no UMAP metrics or legacy methods introduced;
   - figures are readable in the compiled PDF.

---

## Final output

Report:

1. Final section structure.
2. Figures created, with short description of each.
3. Tables created, if any.
4. Data sources used.
5. Main empirical patterns found.
6. Any warnings about missing outputs, low readability, overfull boxes, or unresolved references.
7. Whether compilation with Biber succeeded.

This chapter should feel like the thesis has moved from methodology into evidence. It should be visually memorable, analytically dense, and cautious in interpretation.
