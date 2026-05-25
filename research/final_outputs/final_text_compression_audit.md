# Final Text Compression Audit

## 1. Current PDF page count

- Current compiled PDF: `memory/memoria.pdf`
- Physical PDF pages: 74
- Front matter: 8 physical pages (cover, summary, three contents pages, two list-of-figures pages, one list-of-tables page)
- Arabic-numbered thesis pages: 66

## 2. Current main-text page count

- Main text, excluding front matter, bibliography, and appendices: 55 pages
- Range: Arabic pages 1--55, from Chapter 1 through Chapter 11

## 3. Current appendix page count

- Bibliography: 2 pages, Arabic pages 56--57
- Appendices: 9 pages, Arabic pages 58--66

## 4. Chapter-by-chapter cut estimate

| Unit | Current Arabic pages | Estimated cut | Rationale |
|---|---:|---:|---|
| Summary/front matter | 8 physical pages | 0--1 page | Summary can be slightly compressed, but lists of figures/tables dominate front matter. |
| Chapter 1 | 4 pages | 1 page | Contributions and thesis structure repeat later methodological framing. |
| Chapter 2 | 4 pages | 0.5--1 page | Several paragraphs re-explain the representation/projection caution later placed in Chapter 4. |
| Chapter 3 | 5 pages | 0.5--1 page | Tables carry counts; prose repeats filters, coverage, and corpus limits. |
| Chapter 4 | 3 pages | 0.5 page | Measurement-space caveat can be stated once, then bridged forward. |
| Chapter 5 | 4 pages | 0.5 page | Definitions must remain, but explanatory warnings after the metric table can be shorter. |
| Chapter 6 | 7 pages | 0.5--1 page | Empirical content should remain; reduce chapter opening, captions, and closing caveat. |
| Chapter 7 | 9 pages | 0.5--1 page | Figures/tables drive length; text can tighten around repeated "not causal" and "visualization only" reminders. |
| Chapter 8 | 6 pages | 0.5 page | Relational caveats are repeated in the opening, captions, and ending. |
| Chapter 9 | 7 pages | 1 page | Strong candidate for structural compression; repeated clustering/projection cautions occur in opening, captions, comparison, and ending. |
| Chapter 10 | 4 pages | 1 page | Discussion repeats chapter-by-chapter results and full limitation taxonomy. |
| Chapter 11 | 2 pages | 0.5--1 page | Conclusion re-summarizes contribution, empirical takeaways, and final claim. |
| Appendices | 9 pages | 0.5--1 page | Appendix D prose and captions repeat the UMAP caution and surrounding interpretation. |

Working target: reduce main text from 55 pages toward roughly 48--51 pages and total PDF from 74 pages toward roughly 64--67 pages. Reaching 60--63 pages may require typography or figure-layout changes beyond a text-first pass, so the primary honest target is a sharper 64--67 page PDF.

## 5. Repeated caveats and locations

- OpenAlex is pragmatic, not natural taxonomy:
  - Chapter 1 contributions and final contribution paragraph.
  - Chapter 2 OpenAlex paragraph and positioning section.
  - Chapter 3 analytical-frame and corpus-support sections.
  - Chapter 10 data limitations.
  - Chapter 11 opening and final remarks.
- SPECTER2 is learned representation, not ground truth:
  - Chapter 2 scientific document embeddings.
  - Chapter 4 representation and limits.
  - Chapter 10 representation limitations.
  - Chapter 11 summary of contributions.
- Measurement is in original SPECTER2 space, projections are displays:
  - Chapter 1 objective and contributions.
  - Chapter 4 measurement-space and visual-display sections.
  - Chapter 5 opening, Table 5.1 note, and limits.
  - Chapters 6, 7, and 9 figure captions.
  - Chapter 10 methodological contributions and limitations.
  - Chapter 11 summary.
- Morphology is not quality, impact, productivity, causality, maturity, or mechanism:
  - Chapter 3 corpus-support section.
  - Chapter 5 interpretation and limits.
  - Chapter 6 closing section.
  - Chapter 7 most-dynamic caption and closing section.
  - Chapter 8 closing section.
  - Chapter 9 closing section.
  - Chapter 10 implications.
- Clusters are descriptive anchors, not natural categories:
  - Chapter 1 contribution list.
  - Chapter 2 positioning.
  - Chapter 9 opening, static typology, projection comparison, and ending.
  - Appendix C centroid sensitivity.
  - Chapter 10 empirical synthesis and clustering limitations.
  - Chapter 11 typology takeaway and final remarks.
- Semantic location and morphology are distinct:
  - Chapter 1 contributions and thesis structure.
  - Chapter 7 centroid paths.
  - Chapter 9 semantic-location comparison.
  - Appendix C.
  - Chapter 10 synthesis and methods.
  - Chapter 11 summary and typology takeaway.

Compression rule: keep full explanations in Chapters 3, 4, 5, 9, 10, and Appendix D as specified by the prompt; replace other repetitions with short reminders.

## 6. Repeated chapter-opening formulas

- Chapter 5 opens by naming the previous chapter and "turning" a matrix into indicators.
- Chapter 6 opens with "This chapter begins..." and a methodological recap.
- Chapter 7 opens with "This chapter treats..." and a procedural setup.
- Chapter 8 opens with "This chapter turns..." and a definition.
- Chapter 9 opens with "The previous chapters..." and another "This chapter uses..." roadmap.
- Chapter 10 opens with broad restatement of the whole thesis.
- Chapter 11 opens with another broad restatement of the whole thesis.

Compression rule: keep direct conceptual openings; remove administrative "this chapter" roadmaps when the next sentence already performs the transition.

## 7. Repeated chapter-ending formulas

- Chapter 3 ends with "what the corpus can support" caveat.
- Chapter 4 ends by previewing the next chapters.
- Chapter 5 ends with metric interpretation limits.
- Chapter 6 ends with what static profiles establish.
- Chapter 7 ends with what temporal profiles can say.
- Chapter 8 ends with what morphological relations mean.
- Chapter 9 ends with keeping typologies in bounds.
- Chapter 10 ends by restating the separate-dimensions thesis.
- Chapter 11 ends by restating the separate-dimensions thesis again.

Compression rule: preserve endings only where they add new interpretation; turn others into brief bridge paragraphs.

## 8. Captions that repeat surrounding prose

- Figure 2.1 repeats the shift from graph/category maps to distributions in semantic space.
- Figure 3.1 repeats local checks and pipeline details also carried by Table 3.2 and Appendix A.
- Figure 6.2 and Figure 6.3 repeat the within-domain heterogeneity point stated in the paragraphs around them.
- Figure 6.4 repeats that the PCA map is visual, not evidence of categories; this is also in the paragraph above.
- Figure 6.5 repeats that outlyingness is not typology; the paragraph below already says the same.
- Figure 7.3 repeats that PCA is only visualization; the paragraph above says ranking uses original SPECTER2 space.
- Figure 7.6 repeats that the dynamic ranking is not improvement, impact, or causality; the text and Chapter 5 already cover this.
- Figure 7.7 repeats the examples as "not one common trajectory"; the next paragraph says it again.
- Figure 8.1 and Figure 8.2 repeat "not topical/citation similarity."
- Figure 9.1 and Figure 9.4 repeat the same diagnostic/projection cautions as the surrounding Chapter 9 text.
- Figure C.1 repeats centroid/morphology/PCA caution already in Appendix C prose.
- Appendix D captions interpret each UMAP panel and repeat the atlas rationale.

## 9. Tables whose notes or prose repeat the same information

- Table 3.1 note repeats that the analysis rows feed downstream morphology metrics.
- Table 4.1 plus surrounding prose both state original embedding space and visualization-only reduction.
- Table 5.1 note repeats original 768-dimensional SPECTER2-space computation from the surrounding section.
- Table 6.1 surrounding prose repeats the domain summary carried in the table.
- Table 7.1 surrounding prose repeats the driver column interpretation.
- Table 8.1 surrounding prose repeats the convergence/divergence driver column.
- Tables 9.1 and 9.2 notes repeat clustering method details already stated in Chapter 9.

## 10. Sections to merge, shorten, or turn into bridges

- Chapter 1: merge contributions and thesis structure into a shorter contribution-and-route section if numbering flexibility allows; otherwise sharply compress both.
- Chapter 2: shorten "Positioning of This Thesis" and remove the final chapter roadmap.
- Chapter 3: merge "From Validated Corpus to Analysis Subset" with "What the Corpus Can Support" into one shorter bridge.
- Chapter 4: merge "Visual Display and Representational Limits" and "Implications for Morphological Analysis" into a shorter closing bridge, or compress the latter.
- Chapter 5: compress "Reduction to an Interpretable Metric Core" and "Interpretation and Limits" into tighter paragraphs.
- Chapter 6: shorten "Reading Shape in One Cross-Section" and convert "What Static Profiles Establish" into a short bridge.
- Chapter 7: shorten opening setup and "What Temporal Profiles Can Say."
- Chapter 8: shorten "Turning Profiles into Relations" and "What Morphological Relations Mean."
- Chapter 9: shorten "Clustering as Compression," remove repeated projection warnings, and compress "Keeping Typologies in Bounds."
- Chapter 10: replace chapter-by-chapter recap with synthesis by analytical dimension.
- Chapter 11: collapse summary and takeaways into a lean final claim.
- Appendix D: compress opening caution and make captions functional.

## 11. Proposed cut target by chapter

| Unit | Proposed text cut |
|---|---:|
| Summary | 10--15 percent |
| Chapter 1 | 25--35 percent |
| Chapter 2 | 20--25 percent |
| Chapter 3 | 15--20 percent |
| Chapter 4 | 15--20 percent |
| Chapter 5 | 10--15 percent |
| Chapter 6 | 8--12 percent |
| Chapter 7 | 8--12 percent |
| Chapter 8 | 8--12 percent |
| Chapter 9 | 12--18 percent |
| Chapter 10 | 25--35 percent |
| Chapter 11 | 25--40 percent |
| Appendices | 10--20 percent |

The most valuable cuts are not evenly distributed. The best compression comes from Chapter 1, Chapter 2, Chapter 9, Chapter 10, Chapter 11, and Appendix D, while Chapters 6--8 should remain empirically conservative.
