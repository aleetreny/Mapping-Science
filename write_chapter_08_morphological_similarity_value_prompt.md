# Agent Prompt — Write Chapter 8: Morphological Similarity, Convergence and Divergence

Write Chapter 8 of the thesis: **Morphological Similarity, Convergence and Divergence**.

Target file:

```text
memory/chapters/08_morphological_similarity.tex
```

The thesis is written in English. This is the empirical chapter that moves from describing disciplines individually to comparing them relationally. Chapter 6 described static profiles. Chapter 7 described temporal evolution. Chapter 8 must ask whether scientific regions become more similar, remain stable, or diverge in their morphological structure.

The chapter must be visually strong, analytically precise, and conceptually clear. It should not become a generic similarity-matrix report. It must explain what morphological similarity means, how it is measured, what patterns emerge, and what can and cannot be inferred.

Every paragraph, figure, and table must add unique value.

---

## Non-Negotiable Value Rule

This chapter must contain the minimum necessary amount of prose and visuals to make the argument complete.

Do **not** include repeated content across sections, figures, tables, or captions.

Every element must have a distinct analytical role:

- each section must answer a different question;
- each figure must reveal a different pattern;
- each table must add information that is hard to read from the figures;
- captions must clarify interpretation, not repeat the whole paragraph;
- prose must interpret, not describe visuals mechanically.

Before finishing, review the chapter and remove any section, paragraph, figure, or table that duplicates another one.

If two visuals make the same point, keep the clearer one and delete the other.

If a table repeats what a figure already shows, remove or redesign the table.

If a paragraph only restates a caption or previous paragraph, delete or compress it.

The final chapter should feel dense, selective, and necessary.

---

## Central Objective

Chapter 8 must answer:

1. Which scientific disciplines are morphologically similar under the eleven-metric profile representation?
2. Are broad domains internally coherent, or do they contain morphologically diverse subfields?
3. Which pairs or groups of fields/subfields are unexpectedly similar across domains?
4. Which disciplines are morphologically isolated or atypical?
5. Do disciplines become more similar or more different over time?
6. Which areas show the strongest convergence and divergence?
7. What does morphological convergence mean, and what does it not mean?

This is a chapter about **relational structure**: similarity between scientific regions, not the internal morphology of one region alone.

---

## Active Methodology Only

Use only the active thesis methodology:

- original SPECTER2 embedding space for metric computation;
- reduced eleven-metric core;
- static metric profiles;
- temporal metric profiles across five-year windows;
- OpenAlex hierarchy:
  - domains;
  - fields;
  - subfields.

Do **not** introduce:

- paper-level semantic similarity as the main result;
- UMAP metrics;
- document-to-document clustering;
- Leiden/Louvain communities;
- causal explanations of convergence;
- policy claims not supported by the data;
- legacy outputs or archived metric versions.

Clustering and typologies belong to Chapter 9. Chapter 8 may show similarity structure, nearest-neighbor relations, or distance matrices, but it should not present final clusters as if they were the main product.

---

## Data and Outputs to Inspect

Before writing, inspect the active repo outputs, docs, and scripts. Use actual generated results.

Likely relevant sources:

```text
docs/morphological_similarity.md
docs/reduced_metric_core.md
docs/temporal_evolution.md
docs/output_organization.md
research/final_outputs/
outputs/07_morphological_similarity/
outputs/06_temporal_evolution/
outputs/05_static_comparison/
outputs/04_reduced_metric_core/
scripts/15_compute_morphological_similarity_evolution.py
src/
```

Use the actual available outputs and metric names. If expected summaries or figures are missing, generate them from the synchronized processed metric tables rather than inventing numbers.

Do not place raw repository paths in the main chapter body. Implementation-level file names belong in Appendix A.

---

## Conceptual Definition

The chapter must define morphological similarity clearly.

Suggested framing:

> Two disciplines are morphologically similar when their profiles across the eleven embedding-space metrics are close after appropriate standardization. This does not mean that they study the same topics, cite the same papers, or share institutional structures. It means that, under the SPECTER2 representation and the reduced metric core, their distributions of embedded papers have similar shape.

Make this distinction explicit early.

Important distinction:

- **semantic similarity of papers**: closeness between individual documents in SPECTER2 space;
- **morphological similarity of disciplines**: closeness between aggregate metric profiles.

Chapter 8 is about the second, not the first.

---

## Expected Analytical Structure

Use around **4--5 sections maximum**.

Recommended structure:

```latex
\section{From Morphological Profiles to Similarity}
\section{Static Similarity Across Disciplines}
\section{Convergence and Divergence Over Time}
\section{Nearest Neighbors, Isolates, and Cross-Domain Bridges}
\section{Interpretive Limits of Morphological Similarity}
```

You may adjust titles, but keep the chapter compact. Avoid excessive subsections.

Each section must have a unique purpose:

- Section 1 defines the relational object and standardization.
- Section 2 analyzes static pairwise similarity.
- Section 3 analyzes temporal convergence/divergence.
- Section 4 identifies concrete nearest-neighbor, bridge, and isolate cases.
- Section 5 limits interpretation.

Do not let sections 2, 3, and 4 become three versions of the same heatmap discussion.

---

## Required Content

The chapter should cover the following themes.

### A. Static Morphological Similarity

Use the full-period eleven-metric profiles.

Questions:

- Which fields/subfields are closest in metric-profile space?
- Do domains form coherent blocks in similarity matrices?
- Are there cross-domain similarities that are morphologically surprising?
- Which disciplines are morphologically isolated?

This should connect naturally to Chapter 6 but shift from "how does each discipline look?" to "which disciplines look structurally similar?"

### B. Domain and Field Coherence

Analyze whether broad domains are internally cohesive.

Questions:

- Are subfields more similar within the same domain than across domains?
- Which domains are most internally heterogeneous?
- Which domains have stronger cross-domain morphological links?

Be careful: do not claim domains are "wrong" or "natural". The point is whether OpenAlex taxonomy aligns with morphology under this metric system.

### C. Convergence and Divergence

Use temporal profiles to study whether distances between disciplines increase or decrease over time.

Questions:

- Are average pairwise distances shrinking or growing?
- Which pairs of fields/subfields converge most strongly?
- Which pairs diverge most strongly?
- Are convergence/divergence patterns concentrated within domains or across domains?
- Are changes driven by dispersion, local density, hubness, spectral structure, or temporal movement?

Define convergence and divergence operationally:

- convergence = decreasing distance between morphological profiles across time;
- divergence = increasing distance between morphological profiles across time.

Do not imply intellectual agreement, collaboration, interdisciplinarity, or causal influence unless supported by additional evidence.

### D. Nearest Neighbors and Bridges

Use nearest-neighbor relations in profile space to make the similarity structure concrete.

Questions:

- For each field/subfield, what are its closest morphological neighbors?
- Which disciplines act as cross-domain bridges?
- Which are isolated from most others?
- Are some pairs similar despite belonging to different domains?

This can be a strong visual section if done carefully.

### E. Limits

The chapter must clearly state what similarity can and cannot mean.

Morphological similarity does **not** imply:

- same topic;
- same vocabulary;
- same method;
- direct collaboration;
- citation exchange;
- causal convergence;
- disciplinary equivalence.

It means similar metric-profile shape under the chosen corpus, representation, and metric design.

---

## Visual Expectations

This chapter should be visually rich and relational, but not visually repetitive. Create **3--5 strong figures maximum**.

Use publication-quality visuals:

- vector `.pdf` where possible;
- `.png` companion if useful;
- readable labels;
- consistent style with Chapters 6 and 7;
- restrained colors;
- strong captions;
- short captions in the List of Figures via `\caption[short]{long}`.

Every visual must answer a clear relational question.

Do not create six visuals if four are enough. Quality and distinctiveness matter more than quantity.

---

## Suggested Figures

### Figure 8.1 — Static Morphological Similarity Matrix

A heatmap of pairwise distances or similarities between fields or selected subfields based on the standardized eleven-metric profiles.

Purpose:

- show the overall relational structure;
- reveal whether domains form blocks or mix with one another.

Design:

- rows/columns = fields, or a selected readable set of subfields;
- order by domain and/or hierarchical ordering, but do not present this as clustering results;
- domain annotation strip if useful;
- color scale clearly labeled;
- use distance or similarity consistently.

If using distance:
- lower values = more similar.
If using similarity:
- higher values = more similar.

Make the direction unambiguous.

This figure should not repeat the same message as the nearest-neighbor network. If both are used, the heatmap should show global structure and the network should show selected bridge/isolate relations.

---

### Figure 8.2 — Within-Domain vs Cross-Domain Distances

A distribution plot comparing pairwise distances:

- within same domain;
- across different domains.

Purpose:

- test whether taxonomy aligns with morphology at a broad level;
- show whether cross-domain morphological similarity is common.

Design:

- violin/box/ridgeline/density plot;
- show medians clearly;
- avoid overinterpretation as inferential testing unless tests are actually implemented.

Use this figure only if it adds something beyond Figure 8.1. If the heatmap already communicates domain coherence clearly, make this figure more quantitative and concise.

---

### Figure 8.3 — Convergence and Divergence Over Time

A line plot or small multiple showing average pairwise morphological distance across five-year windows.

Possible panels:

- all pairs;
- within-domain pairs;
- cross-domain pairs;
- by domain pair.

Purpose:

- show whether the scientific morphology space becomes more compressed, more separated, or mixed over time.

This figure must be the main temporal contribution of the chapter.

---

### Figure 8.4 — Strongest Converging and Diverging Pairs

A ranked lollipop/bar chart or compact paired plot showing the largest decreases and increases in morphological distance.

Purpose:

- make convergence/divergence concrete;
- identify actual field/subfield pairs that move closer or farther apart.

Design:

- include pair labels;
- indicate domain relationship;
- separate convergence and divergence panels;
- do not overcrowd.

If a table communicates these pairs better, use a table instead of a figure. Do not include both unless each adds distinct value.

---

### Figure 8.5 — Morphological Nearest-Neighbor Network

A network or graph of nearest-neighbor relations in metric-profile space.

Purpose:

- show bridges, isolates, and cross-domain links.

Design:

- nodes = fields or selected subfields;
- edges = nearest-neighbor or mutual-nearest-neighbor links;
- color = domain;
- edge weight = similarity or inverse distance;
- label only important nodes;
- keep the graph readable.

Do **not** turn this into a community detection chapter. This is a relational summary, not the typology/clustering result.

Use this figure only if it reveals relations not already obvious in the heatmap or nearest-neighbor table.

---

### Optional Figure 8.6 — Temporal Similarity Map

A 2D projection of temporal change profiles, not paper-level embeddings.

Purpose:

- summarize which disciplines have similar temporal evolution patterns.

If included:
- state clearly it is exploratory;
- color by domain;
- label selected outliers or bridges only.

This is optional. Use only if it adds a genuinely new angle.

---

## Tables

Use **1--2 tables maximum**, only if they add value.

Possible table:

```text
memory/tables/tab_08_converging_diverging_pairs.tex
```

Suggested columns:

- pair;
- level;
- domains;
- initial distance;
- final distance;
- change;
- interpretation.

Another possible table:

```text
memory/tables/tab_08_nearest_neighbors.tex
```

Suggested columns:

- discipline;
- nearest morphological neighbor;
- same/cross domain;
- distance;
- interpretation.

Do not duplicate the figures. Tables should identify concrete cases that would be hard to read from the figures alone.

Before keeping a table, ask:
- does this table reveal named cases more clearly than the figure?
- does it avoid repeating the figure?
- does it help interpretation?

If not, remove it.

---

## Prose Requirements

Every paragraph must add value.

Good paragraph structure:

1. state the relational pattern;
2. refer to the relevant figure/table only if necessary;
3. explain what kind of similarity or divergence is being observed;
4. add caution if the interpretation could be misunderstood.

Avoid:

- mechanically describing heatmap colors;
- saying "Figure X shows" repeatedly;
- generic phrases like "there are similarities and differences";
- long lists of pairs without interpretation;
- claiming convergence means collaboration or interdisciplinarity;
- repeating the full limitations from Chapters 3--7;
- restating the caption in the prose;
- using a table and then describing every row in the paragraph.

Use strong but careful language:

- "morphologically close";
- "profile-space neighbor";
- "cross-domain bridge";
- "relative convergence";
- "distance compression";
- "structural divergence";
- "under the metric representation".

Avoid:

- "same discipline";
- "true similarity";
- "proof of convergence";
- "natural cluster";
- "causal relation".

---

## Required Methodological Note

Explain standardization.

The chapter should state how profiles are standardized before similarity computation, if that is what the code does. The reader must know whether distance is computed on:

- raw metric values;
- z-scored metric profiles;
- family-balanced metrics;
- static full-period profiles;
- windowed profiles.

Do not invent. Inspect the code and report the actual implementation.

If multiple similarity definitions exist in the outputs, select the active thesis definition and make it explicit.

Keep this methodological note short. Do not repeat standardization explanations already given in Chapters 6 and 7 unless needed for the similarity definition.

---

## Relationship to Chapter 9

Chapter 8 should prepare the ground for Chapter 9 without replacing it.

Good distinction:

- Chapter 8: pairwise similarity, convergence, divergence, nearest neighbors, relational structure.
- Chapter 9: exploratory typologies/clustering based on profile similarity.

Do not introduce final clusters here unless needed as a preview. If any clustering is used only for ordering rows in a heatmap, say so and do not interpret clusters substantively.

---

## Captions

Captions must be analytical and self-contained, but not bloated.

Bad:

> Similarity heatmap.

Good:

> Pairwise morphological distance between fields. Distances are computed from standardized eleven-metric profiles; lower values indicate more similar embedding-space morphology, not topical or citation similarity. Rows and columns are ordered by domain to show whether OpenAlex domains align with profile-space structure.

Use short captions in the List of Figures:

```latex
\caption[Pairwise field morphological distances]{Long analytical caption here...}
```

Captions should not repeat full paragraphs from the prose.

---

## Final Redundancy Audit

Before finishing, perform a redundancy audit.

Check:

1. Does each section answer a different question?
2. Does each figure show a distinct pattern?
3. Does each table add named-case detail that the figures cannot easily show?
4. Are any captions simply repeating the paragraph before or after?
5. Are any paragraphs just rephrasing a figure caption?
6. Are convergence/divergence findings repeated in both a figure, table, and prose without added interpretation?
7. Is there any plot that looks nice but does not change the reader's understanding?

Remove or compress anything that fails this audit.

---

## Output Requirements

After completing the chapter:

1. Update:

```text
memory/chapters/08_morphological_similarity.tex
```

2. Create or update thesis-ready figures in the thesis figure folder.

3. Create or update necessary tables in:

```text
memory/tables/
```

4. Ensure proper LaTeX integration:

- `figure` environments;
- `\caption[short]{long}`;
- `\label{...}`;
- readable sizing;
- proper placement;
- no broken references.

5. Compile with Biber.

6. Verify:

- citations render correctly;
- bibliography appears correctly;
- all figures appear in the List of Figures with short captions;
- all tables appear in the List of Tables;
- no unresolved references;
- no overfull visual elements;
- no raw repository paths in the chapter;
- no legacy methods introduced;
- no clustering or typology is presented as the main result;
- figures remain readable in the compiled PDF.

---

## Final Output

Report:

1. final section structure;
2. figures created, with the unique analytical role of each;
3. tables created, if any, and why they are not redundant;
4. data sources used;
5. main similarity/convergence/divergence findings discovered;
6. what content was removed or avoided to reduce repetition;
7. any warnings about readability, missing outputs, overfull boxes, unresolved references, or ambiguous similarity definitions;
8. whether compilation with Biber succeeded.

This chapter should make the thesis relational. After Chapter 6 shows what disciplines look like and Chapter 7 shows how they move, Chapter 8 must show which disciplines resemble, approach, or separate from one another in morphological space.
