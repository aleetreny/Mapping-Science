# Agent Prompt — Write Chapter 4: Semantic Representation

Write Chapter 4 of the thesis: **Semantic Representation**.

Target file:

```text
memory/chapters/04_semantic_representation.tex
```

The thesis is written in English. Use rigorous university-level academic language: precise, concise, and methodological. Avoid filler, overclaiming, or README-like wording.

## Chapter Objective

Chapter 4 must explain how the validated OpenAlex title--abstract corpus is transformed into a quantitative semantic representation suitable for downstream morphology analysis.

The chapter should answer:

1. Why title--abstract text is used as the semantic input.
2. Why SPECTER2 is appropriate for scientific document representation.
3. What the embedding space represents conceptually.
4. How the embedding matrix is aligned with corpus metadata.
5. Why all quantitative metrics are computed in the original embedding space.
6. Why UMAP is only used later as auxiliary visualization.
7. What limitations follow from using model-dependent scientific embeddings.

## Required Content

Use the current synchronized thesis snapshot:

- Full corpus: `2,378,036` text-eligible works.
- Full corpus coverage: `252` OpenAlex subfields, `26` fields, `4` domains.
- Analysis embedding subset: `2,344,927` embedded works.
- Analysis subfields: `241`.
- Active embedding model: SPECTER2.
- Embedding dimensionality: `768`.
- Main geometry for all quantitative work: original SPECTER2 embedding space.
- UMAP is not quantitative evidence.

Do **not** include internal repository paths in the main chapter prose. If implementation-level file names are necessary, put them in Appendix A, not in Chapter 4.

## Suggested Structure

Use these sections unless there is a strong reason to adjust them:

```latex
\section{From Text Corpus to Semantic Representation}
\section{Scientific Document Embeddings}
\section{SPECTER2 Representation}
\section{Embedding Matrix and Row Alignment}
\section{Original Embedding Space as Analytical Geometry}
\section{Role of Dimensionality Reduction}
\section{Limitations of Embedding-Based Representation}
\section{Implications for Downstream Morphological Analysis}
```

## Literature and Citations

Use citations from:

```text
memory/referencias.bib
```

At minimum, cite:

- `cohan2020specter` for SPECTER and citation-informed scientific document representations.
- `singh2023scirepeval` for SciRepEval / SPECTER2.
- `beltagy2019scibert` if discussing scientific-domain language models.
- `devlin2018bert` only if needed for transformer background, not as a central source.
- `mcinnes2018umap` only when explaining UMAP as visualization.
- `chari2023specious` and/or `wattenberg2016tsne` when explaining why nonlinear 2D projections are not used as quantitative geometry.

Do not over-cite. The chapter should be methodological, not a generic literature review.

## Key Conceptual Claims to Include

The chapter should clearly state that:

- The thesis represents each scientific work through its title and abstract.
- SPECTER2 embeddings are used because they are designed for scientific document representation, not generic sentence similarity.
- The embedding vector is treated as a model-dependent semantic representation of a paper, not as an objective measurement of scientific meaning.
- Distances, densities, neighborhoods, and centroids are meaningful only within the assumptions of the embedding model.
- The row-aligned matrix is the foundation for all subsequent morphology metrics.
- Quantitative morphology is computed in the original 768-dimensional embedding space.
- UMAP and other dimensionality reductions may be used to visualize the scientific space, but not to calculate the core metrics.
- The analysis subset is slightly smaller than the full corpus because downstream metric computation requires valid embeddings, metadata alignment, and sufficient observations per analytical unit.

## Possible Table

Create a small thesis-ready table if useful:

```text
memory/tables/tab_04_embedding_snapshot.tex
```

Suggested content:

| Component | Value |
|---|---:|
| Full text corpus | 2,378,036 works |
| Full corpus subfields | 252 |
| Analysis embedding subset | 2,344,927 works |
| Analysis subfields | 241 |
| Embedding model | SPECTER2 |
| Embedding dimensionality | 768 |
| Quantitative geometry | Original embedding space |
| Dimensionality reduction | Visualization only |

If you create this table, include it in Chapter 4 using:

```latex
\input{tables/tab_04_embedding_snapshot}
```

Ensure it appears in the List of Tables.

## Style Requirements

- Do not write like repository documentation.
- Do not include raw file paths in the chapter body.
- Do not claim that embeddings reveal the “true” structure of science.
- Do not claim that UMAP maps are evidence of quantitative distances.
- Do not introduce methods not used in the thesis.
- Do not discuss the 11 metrics in detail; that belongs to Chapter 5.
- Keep the prose elegant and controlled.
- Prefer methodological clarity over technical clutter.

## Compilation and Verification

After writing the chapter:

1. Compile with Biber.
2. Verify citations render in APA author-year style.
3. Verify the bibliography still appears correctly.
4. Verify any new table appears in the List of Tables.
5. Verify Chapter 4 contains no unresolved citation keys.
6. Verify Chapter 4 contains no internal repository paths.
7. Report files changed and any warnings.

## Final Output

Summarize:

- chapter structure;
- citations used;
- tables created;
- files changed;
- whether compilation succeeded;
- any remaining issues.
