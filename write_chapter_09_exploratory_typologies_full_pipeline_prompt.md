# Agent Prompt — Explore, Implement, and Write Chapter 9: Exploratory Morphological Typologies

You are working in the TFM repository. Your task is broader than writing a chapter: Chapter 9 does **not** yet have an implemented clustering/typology pipeline. You must first explore the problem seriously, create the necessary scripts and outputs, evaluate multiple reasonable approaches, select the most defensible and interesting one, and then write the chapter.

Target chapter:

```text
memory/chapters/09_clustering.tex
```

Expected new pipeline stage:

```text
outputs/09_morphological_typologies/
```

Suggested new active files, unless the repository structure suggests better names:

```text
src/morphological_typologies.py
scripts/18_explore_morphological_typologies.py
docs/morphological_typologies.md
memory/tables/tab_09_typology_summary.tex
memory/tables/tab_09_typology_examples.tex
```

Use the current repository organization and naming conventions. Do not disturb existing completed chapters or active outputs unless necessary.

---

## Core Objective

Chapter 9 should build an **exploratory typology** of scientific disciplines based on the reduced embedding-space morphological metric core.

The chapter should answer:

1. Can scientific subfields be grouped into interpretable morphological types?
2. Which grouping method gives the most stable and meaningful typology?
3. What characterizes each type in terms of dispersion, density, hubness, spectral structure, and temporal movement?
4. Are typologies aligned with OpenAlex domains, or do they cut across them?
5. Which subfields are prototypical members, border cases, or unusual cases?
6. How robust are the typologies across reasonable methodological choices?
7. What can these typologies mean, and what must they not be interpreted as?

This is **not** a chapter about discovering natural categories of science. It is a chapter about constructing useful, transparent, exploratory descriptions of morphological profile space.

---

## Non-Negotiable Interpretation Rule

Do **not** overclaim.

Clusters are not:

- natural kinds;
- objective disciplinary boundaries;
- proof that disciplines are internally coherent;
- evidence of intellectual quality;
- evidence of causal similarity;
- replacements for OpenAlex taxonomy;
- proof of interdisciplinarity.

Clusters are:

- exploratory typologies;
- summaries of similarity in the eleven-metric morphological profile space;
- heuristic tools for interpreting recurrent combinations of dispersion, density, hubness, spectral structure, and temporal movement.

If the clustering evidence is weak, say so clearly and build the chapter around gradients, instability, or weak typological structure. Do **not** force a neat story.

---

## Active Methodology Constraints

Use only active thesis evidence:

- original SPECTER2 embedding space for metric computation;
- reduced eleven-metric core;
- static full-period subfield profiles;
- optional temporal-profile information only if it genuinely adds value;
- OpenAlex hierarchy only as metadata for interpretation.

Do **not** use:

- UMAP coordinates as clustering input;
- paper-level clustering;
- Leiden/Louvain on paper-level graphs;
- archived UMAP metric outputs;
- legacy mixed UMAP-plus-embedding clustering;
- growth prediction/classification outputs;
- any archived exploratory output as active thesis evidence.

UMAP, PCA, t-SNE, or other 2D projections may be used only to visualize **metric-profile clusters**, not as the clustering geometry.

---

## Required Starting Point

The main clustering input should be the 241 analysis subfields represented by the reduced eleven-metric core.

Use the actual current metric table from the active pipeline. Inspect the repository to find the correct files, likely under:

```text
outputs/04_reduced_metric_core/
data/processed/
outputs/05_static_comparison/
outputs/07_morphological_similarity/
research/final_outputs/
```

Do not invent numbers.

Before choosing a method, verify:

- number of subfields;
- available metrics;
- missing values;
- whether metrics are raw, z-scored, robust-scaled, or already transformed;
- whether any metrics require sign interpretation;
- whether temporal metrics are included in the full static profile.

Document the chosen preprocessing.

---

## Exploration Phase

You must explore several reasonable approaches before selecting the final thesis typology.

At minimum, compare:

### 1. Hierarchical clustering

Try:

- Ward linkage on standardized/robust-scaled Euclidean profiles;
- average linkage on Euclidean distance;
- average linkage on correlation distance if appropriate.

Inspect:

- dendrogram structure;
- cluster sizes;
- interpretability;
- stability across number of clusters;
- whether results are dominated by one or two metrics.

### 2. K-means or k-medoids

Try a range of `k`, for example 3 to 10.

Inspect:

- silhouette score;
- Calinski-Harabasz score;
- Davies-Bouldin score;
- cluster size balance;
- profile interpretability;
- sensitivity to initialization.

K-medoids is preferable if available because actual subfields can serve as medoid examples. If not available, k-means is acceptable but must be interpreted carefully.

### 3. Gaussian mixture models

Try GMMs only as an exploratory comparison if numerically stable.

Inspect:

- BIC/AIC;
- cluster degeneracy;
- whether covariance assumptions are reasonable.

Do not use GMM as final method if it gives fragile or hard-to-interpret results.

### 4. HDBSCAN or density-based clustering

Use only if available and sensible.

Purpose:

- detect whether the profile space has dense groups plus noise;
- identify isolated or border subfields.

Do not force HDBSCAN if the 11-dimensional metric profile space does not support it clearly.

### 5. Dimensionality-assisted exploration

Use PCA of the standardized metric profiles for diagnostic visualization.

Optionally use UMAP of metric profiles for visualization only.

Do not cluster on UMAP unless explicitly justified as a secondary sensitivity check and not used as final thesis evidence.

---

## Robustness and Model Selection

Do not choose the method only because it produces a pretty plot.

Select the final typology using a combination of:

- interpretability;
- cluster size balance;
- stability under preprocessing choices;
- stability under metric subsets/families;
- silhouette or related metrics;
- agreement between methods;
- usefulness for explaining the empirical chapters;
- absence of obvious artifacts.

Include robustness checks.

At minimum, compute/report some combination of:

- silhouette score by k;
- cluster size distribution;
- adjusted Rand index or adjusted mutual information between plausible clustering solutions;
- bootstrap/subsampling stability if feasible;
- sensitivity to:
  - z-score scaling vs robust scaling;
  - all eleven metrics vs family-balanced representation;
  - excluding the three temporal metrics;
  - Euclidean vs correlation distance.

Important: The final choice can be something like “a 5-cluster hierarchical solution is the most interpretable and reasonably stable”, but only if supported by the exploration. If the best answer is “there is no strong discrete cluster structure; typologies are weak but useful as descriptive anchors”, write that.

---

## Possible Final Typology Logic

The final typology should ideally identify interpretable morphological types, such as examples below. These are only examples; do not force these labels unless the data support them.

Possible labels:

- Compact and locally dense fields.
- Broad and spectrally complex fields.
- Hub-dominated fields.
- Temporally novel or drifting fields.
- Low-dimensional/stabilized fields.
- Fragmented or heterogeneous fields.
- Cross-domain bridge profiles.
- Atypical/outlier profiles.

Cluster labels should be descriptive and grounded in the cluster centroid/medoid profile.

Avoid vague labels like:

- Cluster 1;
- Group A;
- Modern sciences;
- Traditional disciplines;
- Advanced fields;
- Complex fields;
- Better/worse fields.

Good labels should describe morphology, not prestige.

---

## Required Outputs

Create a clean output folder:

```text
outputs/09_morphological_typologies/
```

Suggested files:

```text
cluster_exploration_summary.md
cluster_model_comparison.csv
cluster_solution_selected.csv
subfield_typology_assignments.csv
typology_profile_summary.csv
typology_domain_composition.csv
typology_stability_summary.csv
```

Figures, preferably both `.pdf` and `.png` when practical:

```text
fig_09_cluster_quality_by_k.pdf/png
fig_09_typology_profile_heatmap.pdf/png
fig_09_typology_pca_map.pdf/png
fig_09_typology_domain_composition.pdf/png
fig_09_typology_dendrogram.pdf/png
fig_09_typology_stability.pdf/png
```

Only keep figures that add value. Do not include every diagnostic plot in the main chapter.

---

## Suggested Figures for Chapter 9

Use **3--5 strong figures maximum** in the chapter.

### Figure 9.1 — Cluster Quality and Stability Diagnostics

Purpose:

- show that the selected typology was not chosen arbitrarily.

Possible design:

- silhouette score by k;
- cluster size distribution;
- stability/agreement across methods or preprocessing choices.

This figure should be compact. If too technical, move detailed diagnostics to Appendix C and include only the strongest summary in the chapter.

### Figure 9.2 — Typology Profile Heatmap

Purpose:

- show the morphological signature of each typology.

Design:

- rows = selected typologies/clusters;
- columns = eleven metrics, grouped by family;
- values = standardized cluster mean/median profile;
- use readable metric labels;
- include family separators if possible;
- label clusters with descriptive names, not just numbers.

This is likely the most important figure.

### Figure 9.3 — Subfield Map in Metric-Profile Space

Purpose:

- show where typologies sit relative to one another.

Design:

- PCA or UMAP projection of standardized metric profiles;
- points = subfields;
- color = selected typology;
- shape/outline or small annotation = domain if useful;
- label only medoids/prototypical subfields and major outliers;
- state clearly that projection is visualization only.

### Figure 9.4 — Domain Composition of Typologies

Purpose:

- show whether typologies cut across OpenAlex domains.

Design:

- stacked bar chart or mosaic-like composition;
- rows or bars = typologies;
- segments = domains;
- include counts or percentages.

This figure is useful if it shows cross-domain mixing.

### Figure 9.5 — Prototypical and Borderline Subfields

Purpose:

- make the typologies concrete.

Options:

- lollipop chart of distance to cluster centroid/medoid;
- table-style figure;
- small panel showing prototypical and borderline examples.

Use only if it adds value beyond the tables.

---

## Tables for Chapter 9

Use **1--2 tables maximum**.

Create:

```text
memory/tables/tab_09_typology_summary.tex
```

Suggested columns:

- typology label;
- number of subfields;
- dominant morphological traits;
- representative subfields;
- domain mix.

Optional second table:

```text
memory/tables/tab_09_typology_examples.tex
```

Suggested columns:

- typology;
- prototypical subfields;
- borderline/outlier subfields;
- interpretation.

Do not create giant membership tables in the main thesis. Full membership lists should go to output files or Appendix C if needed.

Tables must:

- use proper LaTeX captions and labels;
- appear in the List of Tables;
- be readable;
- not duplicate figures.

---

## Chapter Structure

Use around **4--5 sections maximum**.

Recommended structure:

```latex
\section{From Similarity to Exploratory Typologies}
\section{Exploring Candidate Typology Models}
\section{Selected Morphological Typology}
\section{Typologies Across Domains and Border Cases}
\section{Interpretive Limits of Clustering}
```

You may adjust titles, but keep the chapter compact.

Section roles:

- Section 1: define why clustering is used after similarity/convergence.
- Section 2: summarize exploration and model selection.
- Section 3: present selected typology and its profiles.
- Section 4: discuss domain composition, representative cases, bridges, and border cases.
- Section 5: methodological and epistemological limits.

Do not make one subsection per algorithm. That belongs in the exploration report or appendix, not the main chapter.

---

## Prose Style

The chapter must have personality, but not hype.

Good conceptual framing:

> Clustering is used here as a compression device, not as a discovery of natural kinds.

> The typology is useful only if it makes the morphology easier to reason about without pretending that the boundaries are real disciplinary borders.

> A subfield belongs to a typology because its metric profile is close to a profile pattern, not because its intellectual content is reducible to that group.

Every paragraph must add value. Avoid:

- generic clustering textbook explanations;
- long mechanical lists of algorithms;
- repeating Chapters 6--8;
- saying “clusters reveal hidden structure”;
- overusing “interdisciplinary” unless evidence supports it;
- interpreting clusters as quality, impact, maturity, or importance;
- describing every figure cell.

---

## Required Methodological Clarifications

The chapter must state clearly:

- what unit is clustered: primarily 241 subfields;
- what features are used: reduced eleven-metric morphological profiles;
- how features are scaled: inspect actual code and document it;
- what algorithm is selected and why;
- what alternatives were tested;
- what diagnostics support or weaken the selected solution;
- whether temporal metrics are included;
- whether results are robust to excluding temporal metrics;
- that projections are visualization only;
- that domain labels are used for interpretation, not as clustering targets.

---

## Robustness Checks to Run

At minimum, run and save evidence for:

1. Scaling sensitivity:
   - z-score scaling;
   - robust median/IQR scaling.

2. Metric subset sensitivity:
   - all eleven metrics;
   - eight static/structural metrics only;
   - family-balanced version if appropriate.

3. Algorithm sensitivity:
   - hierarchical clustering;
   - k-means/k-medoids;
   - GMM if stable;
   - HDBSCAN if useful.

4. Number of clusters:
   - explore k = 3 to 10.

5. Domain alignment:
   - compare cluster assignments to OpenAlex domains using contingency tables.
   - Do not use domain purity as the selection criterion; use it only to understand whether typologies cross taxonomy.

6. Stability:
   - if feasible, bootstrap or subsample subfields and measure cluster stability;
   - otherwise compare assignment agreement across reasonable solutions.

Report results honestly.

---

## Selection Criteria

Choose the final method only after exploration.

Preferred final solution characteristics:

- interpretable cluster profiles;
- not too many clusters;
- not one giant cluster plus tiny fragments unless that is genuinely the result;
- stable across small methodological changes;
- visually clear;
- helpful for synthesizing Chapters 6--8.

If no method gives strong discrete clusters, write a chapter about weak typological structure:

- use clusters as descriptive anchors;
- emphasize gradients and outliers;
- avoid pretending the typology is robust.

The worst possible outcome is a neat but unsupported cluster story. Avoid that.

---

## Relationship to Earlier Chapters

Chapter 9 should synthesize, not repeat.

Connect to:

- Chapter 6: static morphology showed extremes and profile variation.
- Chapter 7: temporal evolution showed different movement patterns.
- Chapter 8: similarity showed relational structure and bridges.
- Chapter 9: typologies compress these profile relations into a small set of descriptive morphological types.

Do not repeat the results of Chapters 6--8 except where needed to motivate or interpret the typology.

---

## LaTeX and Thesis Integration

Update:

```text
memory/chapters/09_clustering.tex
```

Create figures in the thesis figure folder and include them with proper LaTeX figure environments:

```latex
\caption[short caption]{Long analytical caption...}
\label{fig:...}
```

Create tables under:

```text
memory/tables/
```

Use proper table captions and labels.

Ensure:

- all figures appear in the List of Figures with short captions;
- all tables appear in the List of Tables;
- no broken references;
- no raw repository paths in the chapter body;
- no overfull tables or unreadable figures.

---

## Code Quality Requirements

Create clean, reusable code.

The new clustering pipeline should:

- follow existing repo style;
- use the active configuration conventions where possible;
- avoid hard-coded absolute paths;
- write outputs to `outputs/09_morphological_typologies/`;
- include clear command-line arguments such as `--overwrite`;
- produce a readable summary report;
- fail clearly if required input files are missing;
- avoid modifying large data files unnecessarily;
- avoid committing generated large artifacts if they are ignored by Git.

Suggested command:

```powershell
.\.venv\Scripts\python.exe scripts\18_explore_morphological_typologies.py --overwrite
```

Add tests if practical, especially for:

- loading the reduced metric core;
- scaling;
- cluster assignment output schema;
- reproducibility with random seeds;
- no UMAP coordinates used as clustering features.

If full tests are expensive, add lightweight smoke tests.

---

## Final Verification

After implementing and writing:

1. Run the new typology script.
2. Verify output files exist and have reasonable contents.
3. Compile the thesis with Biber.
4. Verify:
   - Chapter 9 appears correctly;
   - figures are readable;
   - tables are readable;
   - figures/tables appear in their lists;
   - no unresolved references;
   - no raw paths in the prose;
   - no UMAP metrics used as clustering input;
   - no unsupported claims.
5. Run relevant tests or at least script import checks.

---

## Final Report

When finished, report:

1. scripts and source files created;
2. outputs created;
3. methods explored;
4. selected final method and why;
5. diagnostics supporting the selected method;
6. diagnostics that weaken or qualify the typology;
7. final typology labels and number of subfields per type;
8. figures and tables included in Chapter 9;
9. whether compilation succeeded;
10. any warnings or limitations.

This task is exploratory. Be ambitious in exploration, conservative in interpretation, and selective in what enters the thesis.
