# Agent Prompt — Write Chapter 2: Background and Related Work

You are working in the TFM repository. Chapter 2 is currently a placeholder and must be replaced with a serious, concise, well-cited literature review that supports the actual thesis.

Target file:

```text
memory/chapters/02_background.tex
```

The thesis is written in English.

Do not rewrite empirical chapters. Do not add new analyses. Chapter 2 should frame the thesis intellectually and methodologically, not repeat results.

---

## Current State

Chapter 2 currently contains only a short placeholder about mapping scientific disciplines, classical bibliometric approaches, dense semantic representation, and embedding-space morphology. Replace it completely.

Before writing, inspect:

```text
memory/chapters/03_data_and_corpus.tex
memory/chapters/04_semantic_representation.tex
memory/chapters/05_embedding_space_metrics.tex
memory/chapters/06_static_comparison.tex
memory/chapters/07_temporal_evolution.tex
memory/chapters/08_morphological_similarity.tex
memory/chapters/09_clustering.tex
memory/chapters/10_discussion.tex
memory/chapters/11_conclusion.tex
research/literature_review/
referencias.bib
```

Use the actual thesis vocabulary:
- OpenAlex;
- SPECTER2;
- embedding-space morphology;
- static morphology;
- temporal trajectories;
- semantic location;
- profile space;
- typological anchors;
- visualization only;
- not a new taxonomy of science.

---

## Central Objective

Chapter 2 must explain where this thesis sits in the literature.

The main positioning is:

> Classical science mapping usually represents science through citations, co-citations, bibliographic coupling, co-word relations, or journal/category overlays. Modern scientific document embeddings allow another view: fields can be treated as distributions of papers in a dense semantic space. This thesis builds on that representational shift, but it does not simply cluster papers or draw a map. It measures the morphology of subfield distributions: dispersion, local density, hubness, spectral structure, and temporal movement.

The chapter should make clear why the thesis is not:
- a generic topic model;
- a citation-network clustering project;
- a paper-level community detection study;
- a simple UMAP visualization;
- a productivity/citation-impact study;
- a replacement for OpenAlex taxonomy.

It is a **measurement framework for the shape and movement of scientific fields in embedding space**.

---

## Desired Length and Structure

Write a compact Chapter 2, around **4--5 sections maximum**.

Recommended structure:

```latex
\section{Mapping Science as a Spatial Problem}
\section{From Bibliometric Relations to Semantic Representations}
\section{Scientific Document Embeddings}
\section{From Semantic Location to Morphology}
\section{Positioning of This Thesis}
```

You may adjust titles, but keep the chapter focused. Avoid many subsections; the table of contents is already long.

---

## Section 2.1 — Mapping Science as a Spatial Problem

Purpose:
- introduce science mapping as a tradition;
- explain that maps of science turn relations among scientific objects into spatial structure;
- distinguish entities: papers, journals, categories, fields, disciplines;
- explain why spatial maps are useful but depend on the chosen relation.

Relevant literature:
- Börner, Chen, & Boyack (2003): visualizing knowledge domains;
- Van Eck & Waltman (2010): VOSviewer and distance-based bibliometric mapping;
- Leydesdorff, Carley, & Rafols (2013): global maps of science and overlay maps.

Key points:
- Science mapping has long used spatial metaphors to summarize cognitive structure.
- The map is never neutral: distances depend on the input relation.
- Citation relations, co-citation, bibliographic coupling, co-word relations, and subject categories answer different questions.
- This thesis inherits the spatial logic, but changes the object of measurement from relational bibliographic links to distributions in embedding space.

Do not over-explain all bibliometric methods mechanically. Give enough context to justify the transition.

---

## Section 2.2 — From Bibliometric Relations to Semantic Representations

Purpose:
- explain why content-based semantic representations matter;
- contrast citation/category-based maps with text-based/dense semantic approaches;
- clarify that semantic proximity is not the same as citation proximity.

Relevant literature:
- co-word/content-based science mapping if available in the repo;
- Mikolov et al. (2013) for dense vector representation as conceptual background;
- SciBERT / scientific language models if already in bibliography;
- SPECTER/SPECTER2 references later.

Key points:
- Citation maps capture intellectual and social traces, but not all semantic similarity.
- Category overlays depend on predefined classification systems.
- Text embeddings represent documents directly from language, allowing every paper to occupy a common vector space.
- Dense representation makes distance, neighborhood, centroid, dispersion, and movement measurable.
- But embeddings are model-defined geometries, not direct observations of meaning.

Do not claim embeddings are more objective than bibliometric networks. Say they are complementary.

---

## Section 2.3 — Scientific Document Embeddings

Purpose:
- explain why scientific text requires specialized representation;
- introduce SPECTER and SPECTER2;
- justify using document-level embeddings rather than word counts or generic sentence embeddings.

Relevant literature:
- Beltagy et al. (2019): SciBERT / scientific-language representation;
- Cohan et al. (2020): SPECTER;
- Singh et al. (2023): SciRepEval / SPECTER2.

Key points:
- Scientific documents contain specialized vocabulary, conventional abstract structures, and domain-specific language.
- SPECTER learns document representations using citation-informed supervision.
- SPECTER2 extends scientific document representation in a multi-format benchmark context.
- The thesis uses SPECTER2 because the unit is a scientific work represented as a document-level vector.
- Do not over-describe architecture; focus on why this representation is appropriate for macro-level field morphology.

Important:
- This section should support Chapter 4, not duplicate it. Chapter 4 explains the implemented representation; Chapter 2 provides literature background.

---

## Section 2.4 — From Semantic Location to Morphology

Purpose:
- introduce the conceptual leap from “where is a paper/field located?” to “what shape does a field’s distribution have?”
- connect morphology metrics to existing conceptual traditions.

Relevant literature:
- Stirling (2007): diversity as variety, balance, disparity;
- Uzzi et al. (2013): novelty through atypical combinations;
- Nooteboom et al. (2007): cognitive distance if useful;
- Radovanović et al. (2010): hubness in high-dimensional spaces;
- Levina & Bickel (2004) only as future/intrinsic-dimension background if already in `.bib`;
- Chari et al. (2023), Wattenberg et al. (2016), McInnes et al. (2018) for projection caution if relevant.

Key points:
- A centroid gives semantic location, but it loses information about dispersion and internal organization.
- A field should be understood as a distribution of documents in embedding space.
- Morphology captures spread, local packing, hub concentration, spectral structure, and movement.
- This is related to diversity/disparity thinking, but operationalized in dense semantic geometry rather than categorical distance.
- High-dimensional geometry introduces risks: hubness, distance concentration, projection distortion.
- Dimensionality reduction is useful for communication, but should not become the quantitative measurement space.

This section is crucial. It should clearly prepare the reader for Chapter 5.

---

## Section 2.5 — Positioning of This Thesis

Purpose:
- state the gap and the contribution;
- distinguish this thesis from previous science maps and embedding studies.

Core argument:
- Existing science maps often focus on topology, categories, topics, or two-dimensional visualization.
- Scientific document embeddings offer a shared semantic geometry.
- This thesis uses that geometry to measure field-level morphology, not merely to cluster or visualize papers.
- It separates:
  - semantic location;
  - static morphology;
  - temporal trajectories;
  - morphological similarity;
  - clustering as typological compression.
- It treats OpenAlex subfields as pragmatic analytical units, not natural categories.
- It treats clusters as descriptive typological anchors, not discovered scientific kinds.

End this chapter with a transition to Chapter 3:
- Chapter 3 builds the corpus and analytical unit.
- Chapter 4 defines the SPECTER2 representation.
- Chapter 5 defines the morphology metrics.

---

## Citation Strategy

Use citations properly and sparingly. Chapter 2 should be literature-based, but not citation soup.

Recommended citation targets:

### Core background
- Börner et al. (2003)
- Van Eck & Waltman (2010)
- Leydesdorff et al. (2013)

### Corpus/data context
- Priem et al. (2022)
- Martín-Martín et al. (2021) if relevant to broad database coverage
- Culbert et al. (2025) if discussing OpenAlex limitations

### Scientific embeddings
- Beltagy et al. (2019)
- Cohan et al. (2020)
- Singh et al. (2023)
- Mikolov et al. (2013), only as general dense-vector background

### Morphology/conceptual metrics
- Stirling (2007)
- Uzzi et al. (2013)
- Nooteboom et al. (2007), if useful
- Radovanović et al. (2010)

### Projection and clustering caution
- McInnes et al. (2018)
- Wattenberg et al. (2016)
- Chari et al. (2023)
- Kleinberg (2002), if discussing why clustering is not unique or objective.

Do not cite methods that are not implemented as if they were implemented. If mentioning future-work methods, label them as conceptual background or possible extensions.

Check `referencias.bib`. Add missing bibliography entries only if they are actually cited. Preserve the existing bib style.

---

## What Not to Do

Do not:
- write a generic review of “AI in science”;
- write a long history of bibliometrics;
- explain every clustering algorithm;
- introduce results from Chapters 6--9;
- overuse “interdisciplinarity” unless directly tied to cited literature;
- claim that embeddings recover meaning objectively;
- claim that OpenAlex categories are ground truth;
- claim that UMAP/PCA/PCoA reveal true scientific structure;
- add figures or tables unless absolutely necessary;
- create many subsections.

This chapter should be conceptual scaffolding, not another empirical chapter.

---

## Tone and Style

Use rigorous academic English with personality.

Good tone:
- direct;
- conceptually sharp;
- cautious where needed;
- not defensive;
- not verbose.

Avoid:
- filler;
- vague phrases like “in recent years, technology has evolved rapidly”;
- generic claims about “big data”;
- laundry lists of papers;
- unsupported grand statements.

Strong framing phrases you may adapt:

> A map of science is always a map of a relation.

> The same field can be close in citation space, far in semantic location, and similar in morphology.

> This thesis shifts attention from where a field is located to how its papers occupy space around that location.

> The contribution is not a new universal map, but a disciplined separation of location, shape, trajectory, and taxonomy.

---

## Integration Requirements

Ensure Chapter 2 matches later chapters:

- Chapter 3 uses OpenAlex as the corpus and subfield as analytical unit.
- Chapter 4 uses SPECTER2 document embeddings in the original 768-dimensional space.
- Chapter 5 defines the eleven metric core.
- Chapter 6 studies static profiles.
- Chapter 7 studies windowed temporal evolution.
- Chapter 8 studies similarity/convergence/divergence.
- Chapter 9 studies exploratory typologies.
- Chapter 10 discusses static shape, temporal change, semantic location, and taxonomy as distinct readings.

Do not introduce terminology that later chapters do not use.

---

## Final Verification

After writing:

1. Compile with Biber:
   - `pdflatex -> biber -> pdflatex -> pdflatex`
2. Verify:
   - no unresolved citations;
   - no unresolved references;
   - no TODOs/placeholders remain in Chapter 2;
   - no raw paths;
   - Chapter 2 appears correctly in ToC;
   - bibliography includes every cited source;
   - Chapter 2 does not add unnecessary sections to the index;
   - writing is concise and non-repetitive.

---

## Final Report

When finished, report:

1. final Chapter 2 section structure;
2. main conceptual argument;
3. citations added or reused;
4. whether `referencias.bib` was modified;
5. whether any other chapter was touched;
6. compilation status and remaining warnings.

Chapter 2 should make the reader understand why the thesis exists before they see the corpus, embeddings, and metrics: science has been mapped before, but this thesis measures the morphology and movement of fields as distributions in semantic space.
