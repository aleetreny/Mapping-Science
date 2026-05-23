# Agent Prompt — Write Chapters 10 and 11: Discussion and Conclusion

You are working in the TFM repository. Chapters 10 and 11 are currently placeholders. Your task is to write both chapters together so that the thesis ends with a coherent, mature, and intellectually strong argument.

Target files:

```text
memory/chapters/10_discussion.tex
memory/chapters/11_conclusion.tex
```

The thesis is written in English.

Do not rewrite Chapters 3--9 unless you find a direct inconsistency that must be corrected. The goal is to synthesize the empirical work already completed, not to generate new analyses.

---

## Central Thesis Argument

The final discussion and conclusion should crystallize the main contribution:

> This thesis does not claim to discover the true map of science. It proposes a reproducible way to measure the morphology of scientific fields in a dense scientific embedding space, using OpenAlex subfields as analytical units and SPECTER2 as the representational geometry.

The empirical result is not one simple map. It is a layered interpretation:

1. Scientific fields differ in static morphology: dispersion, local density, hubness, spectral structure, and temporal movement.
2. Broad OpenAlex domains show tendencies, but they do not determine morphology.
3. Temporal evolution is not uniform convergence. Many fields become more separated in morphological profile space, while selected pairs converge.
4. Static morphology is weakly clustered; typologies are useful descriptive anchors, not natural categories.
5. Temporal trajectories form clearer and more stable dynamic typologies.
6. Semantic-location clustering aligns more with domains than morphology, showing that topic/location and shape are distinct axes.
7. The strongest final claim is that **static shape, temporal change, taxonomy, and semantic location provide different readings of scientific structure**.

This is the conceptual spine of Chapters 10 and 11.

---

## Non-Negotiable Style Rule

Do not write a generic “discussion chapter”.

Every paragraph must either:

- interpret a major empirical result;
- connect multiple chapters into a higher-level claim;
- clarify what can and cannot be inferred;
- explain the methodological contribution;
- identify a limitation that genuinely affects interpretation;
- propose a future extension that follows logically from the thesis.

Avoid filler phrases such as:

- “This chapter has discussed...”
- “It is important to note that...”
- “There are many implications...”
- “Further research is needed...” unless followed by specific detail.
- “The results are interesting...”

Use clear, rigorous, university-level English. The writing should have personality and intellectual confidence, but no hype.

---

# Chapter 10 — Discussion

## Desired Length and Structure

Write a compact but substantial discussion. Around **4--5 sections maximum**.

Recommended structure:

```latex
\section{What It Means to Measure the Shape of Science}
\section{Static Shape, Temporal Change, and Semantic Location}
\section{Methodological Contributions}
\section{Limitations and Threats to Interpretation}
\section{Implications and Future Research}
```

You may adjust section titles, but keep the chapter focused and not over-fragmented.

Do not create many subsections. The index is already long enough.

---

## Section 10.1 — What It Means to Measure the Shape of Science

Purpose:
- explain the conceptual contribution of “shape”;
- clarify that morphology is not topic, citation, impact, or taxonomy;
- situate the thesis as a measurement framework rather than a universal map.

Key points to include:
- A field is treated as a distribution of embedded papers.
- Morphology summarizes how that distribution occupies semantic space.
- The metrics capture dispersion, local density, hubness, spectral structure, and movement.
- This gives a complementary view to topic modeling, citation networks, and bibliometric productivity indicators.

Possible wording to adapt:

> The thesis shifts the object of measurement from the content of a field to the geometry of its embedded paper distribution.

Be precise: morphology is conditional on OpenAlex, title--abstract records, SPECTER2, and the sampling design.

---

## Section 10.2 — Static Shape, Temporal Change, and Semantic Location

Purpose:
- synthesize the empirical chapters;
- explain the layered result without repeating every figure.

This section should connect Chapters 6--9.

### Static morphology

From Chapter 6:
- domains differ, but domain averages are blunt;
- Physical Sciences are generally more dispersed and spectrally extended;
- Health Sciences are more locally dense and hub-concentrated;
- field/subfield heterogeneity is substantial;
- outliers matter because different subfields are extreme for different reasons.

Interpretation:
- taxonomy gives orientation but not morphology;
- morphology is multidimensional, not reducible to one axis.

### Temporal evolution

From Chapter 7:
- temporal evolution is broad but not uniform;
- many metrics show contraction/densification patterns;
- kNN variability and hubness can move differently;
- the most dynamic subfields have different drivers;
- temporal movement is a bundle of coordinated and uncoordinated changes.

Interpretation:
- scientific morphology changes through multiple mechanisms;
- dynamic change cannot be inferred from static profile alone.

### Similarity and convergence

From Chapter 8:
- field-level morphology becomes more separated on average over time;
- convergence is selective, not global;
- nearest neighbors and bridges often cross domain boundaries;
- similarity does not imply shared topic, collaboration, or causality.

Interpretation:
- the morphology space has relational structure, but not a simple taxonomic block pattern.

### Typologies

From Chapter 9:
- static typology is weakly clustered: silhouette 0.133;
- temporal trajectory typology is stronger: silhouette 0.396 and subsample ARI 0.725;
- centroid and hybrid clustering show semantic-location/domain structure;
- morphology, dynamic change, and semantic location are distinct axes.

Interpretation:
- the most important final result is not “there are five clusters”;
- it is that static morphology is continuous, dynamic morphology is more typological, and semantic location is more domain-aligned.

Do not over-describe each figure. Write the synthesis.

---

## Section 10.3 — Methodological Contributions

Purpose:
- state what the thesis contributes methodologically.

Include 4--6 concrete contributions, written as prose rather than a bullet list unless a short list improves clarity.

Possible contributions:

1. A reproducible OpenAlex title--abstract corpus design with annual balancing across subfield-year cells.
2. A SPECTER2-based representation strategy where quantitative metrics are computed in the original embedding space.
3. A reduced eleven-metric morphology core covering dispersion, density/hubness, spectral structure, and temporal movement.
4. A distinction between static morphology, temporal morphology, morphological similarity, and semantic-location clustering.
5. A conservative empirical workflow that uses dimensionality reduction only for visualization.
6. A typological analysis that treats clusters as descriptive anchors rather than natural categories.

Emphasize that the contribution is not only technical but epistemological: it separates several notions often confused in science maps.

---

## Section 10.4 — Limitations and Threats to Interpretation

Purpose:
- be honest, precise, and serious;
- avoid generic limitations.

Organize limitations around the actual thesis design.

Include:

### Data limitations
- OpenAlex coverage and classification uncertainty.
- English-only filtering.
- Article/preprint restriction.
- Title--abstract availability.
- Sampling cap and sparse subfield-year cells.
- Primary-topic assignment forces one work into one subfield.

### Representation limitations
- SPECTER2 embeddings are learned representations, not ground truth.
- Title--abstract embeddings compress scientific work.
- Scientific meaning, method, evidence, and social/institutional structure are only partially represented.
- Citation-informed training may carry biases.

### Geometric/statistical limitations
- High-dimensional distance concentration, anisotropy, local density variation, and hubness affect interpretation.
- Metrics are summaries, not full distribution comparisons.
- PCA D80 is an operational spectral proxy, not true intrinsic dimension.
- UMAP/PCA/PCoA projections are visual only.

### Clustering limitations
- Static clusters are weak.
- Temporal clusters are more stable, but still exploratory.
- Centroid/hybrid clusters reveal semantic-location structure, not morphology alone.
- No clustering result should be interpreted as a natural taxonomy.

Write this section with mature restraint. Do not undermine the thesis; clarify its domain of validity.

---

## Section 10.5 — Implications and Future Research

Purpose:
- explain what the work enables;
- propose specific extensions.

Possible implications:
- science mapping can move beyond topic labels and citation communities;
- morphology could be used to monitor how fields reorganize structurally;
- dynamic typologies may help identify fields undergoing contraction, broadening, smoothing, or dimensionalizing changes;
- policy or research management use should be cautious and descriptive, not evaluative.

Future research should be concrete:

1. Compare SPECTER2 with other scientific embedding models.
2. Include multilingual corpora or analyze language effects.
3. Use full-text or section-aware embeddings where available.
4. Compare morphology with citation networks, funding, institutions, or journal structures.
5. Use distributional distances such as Wasserstein, MMD, or Gaussian/Fréchet approximations to compare full subfield distributions.
6. Extend temporal analysis with annual models, change-point detection, or event-based analysis.
7. Validate dynamic typologies against external historical/domain knowledge.
8. Explore paper-level communities separately, while preserving the distinction between paper-topic clustering and field-morphology analysis.

Do not oversell practical applications. Emphasize that the framework is descriptive and diagnostic.

---

# Chapter 11 — Conclusion

## Desired Length and Structure

Chapter 11 should be short, clear, and final. Around **3--4 sections maximum**, or a compact unsectioned conclusion if the template allows it.

Recommended structure:

```latex
\section{Summary of Contributions}
\section{Main Empirical Takeaways}
\section{Final Remarks}
```

If three sections feel excessive, use two sections:

```latex
\section{Main Contributions}
\section{Final Remarks}
```

Keep it concise. The conclusion should not introduce new results, new literature, or new methods.

---

## Chapter 11 Content

The conclusion should answer:

1. What did the thesis set out to do?
2. What did it build?
3. What did it find?
4. What should the reader remember?

Core points:
- The thesis constructed a balanced OpenAlex/SPECTER2 corpus and measured fields as distributions in embedding space.
- It defined and reduced a metric core for morphology.
- It showed that domains differ but do not determine morphology.
- It showed that temporal evolution is structured and heterogeneous.
- It showed that similarity and convergence cut across taxonomy.
- It showed that static morphological clustering is weak, while temporal trajectories provide clearer dynamic typologies.
- It showed that semantic-location clusters are different from morphology clusters.
- The final contribution is a framework for measuring and interpreting the shape and movement of scientific fields.

Potential final paragraph idea:

> The central result is not a new taxonomy of science, but a way to measure why no single taxonomy is enough. Scientific fields have positions, shapes, trajectories, and relationships. Treating those dimensions separately makes the structure of science more measurable, but also more resistant to simplistic maps.

Make the ending strong and elegant.

---

# Required Consistency With Earlier Chapters

Before writing, inspect Chapters 3--9, especially:

```text
memory/chapters/03_data_and_corpus.tex
memory/chapters/04_semantic_representation.tex
memory/chapters/05_embedding_space_metrics.tex
memory/chapters/06_static_comparison.tex
memory/chapters/07_temporal_evolution.tex
memory/chapters/08_morphological_similarity.tex
memory/chapters/09_clustering.tex
```

Use the actual terminology already established:
- “morphology”;
- “embedding-space morphology”;
- “static morphology”;
- “temporal trajectories”;
- “semantic location”;
- “profile space”;
- “typological anchors”;
- “not natural kinds”;
- “SPECTER2 original embedding space”;
- “OpenAlex subfields”.

Do not introduce contradictory names.

---

# Citation Strategy

Use citations sparingly in Chapters 10 and 11.

Chapter 10 may cite the most relevant literature when discussing:
- OpenAlex limitations;
- scientific document embeddings;
- UMAP/projection limitations;
- high-dimensional hubness;
- bibliometric/science mapping context.

But do not overload the discussion with citations. It is primarily a synthesis of the thesis results.

Chapter 11 likely needs few or no citations unless required by the template. It should mostly conclude the thesis.

Ensure all citations use the template’s citation style and compile with Biber.

---

# Figures and Tables

Do not add new figures or tables unless absolutely necessary.

Discussion and conclusion should be prose-led.

If any cross-reference to existing figures/tables is used:
- keep it minimal;
- do not repeat figure captions;
- ensure references compile.

Avoid turning Chapter 10 into another results chapter.

---

# Formatting Requirements

Use clean LaTeX.

Do not include:
- raw repository paths in the chapter body;
- markdown syntax;
- bullet-heavy prose;
- TODO comments;
- placeholders;
- uncompiled references;
- unsupported claims.

Use proper cross-references if needed:
- `Chapter~\ref{...}` only if labels exist;
- otherwise write “Chapter 6” etc. plainly.

If labels for chapters do not exist, do not invent references that break compilation.

---

# Tone Requirements

The tone should be rigorous, concise, and intellectually serious.

Avoid:
- hype;
- apologetic over-qualification;
- vague policy language;
- generic “AI will change everything” language;
- empty claims about “important implications”;
- excessive repetition of “not causal” in every paragraph.

Prefer:
- direct claims with scope conditions;
- clear distinctions;
- careful conceptual synthesis;
- strong final sentences.

---

# Final Verification

After writing:

1. Compile with Biber:
   - `pdflatex -> biber -> pdflatex -> pdflatex`
2. Verify:
   - no unresolved citations;
   - no unresolved references;
   - no overfull boxes introduced by Chapters 10/11;
   - no raw paths in prose;
   - no TODO placeholders remain;
   - Chapters 10 and 11 appear correctly in the table of contents;
   - bibliography still compiles;
   - Chapter 10 does not repeat Chapter 9 figure-by-figure;
   - Chapter 11 does not introduce new methods or results.

---

# Final Report

When finished, report:

1. final Chapter 10 section structure;
2. final Chapter 11 section structure;
3. core discussion argument;
4. main final takeaway;
5. citations added, if any;
6. whether any earlier chapter was touched;
7. compilation status and remaining warnings.

The final two chapters should make the thesis feel complete: Chapter 10 interprets what has been learned, and Chapter 11 closes with a clear, memorable statement of the contribution.
