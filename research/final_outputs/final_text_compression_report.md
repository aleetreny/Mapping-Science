# Final Text Compression Report

## 1. Page Counts

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Physical PDF pages | 74 | 65 | -9 |
| Front matter pages | 8 | 7 | -1 |
| Main-text pages | 55 | 48 | -7 |
| Bibliography pages | 2 | 2 | 0 |
| Appendix pages | 9 | 8 | -1 |

Estimated total PDF reduction: 12.2 percent.

Estimated main-text reduction: 12.7 percent.

Estimated appendix reduction: 11.1 percent.

The final PDF did not reach 60--63 pages, but the text-first pass removed nine pages without changing empirical results, figure data, table values, methods, or bibliography. Further reduction to 60--63 pages would likely require more aggressive figure/table layout decisions rather than only prose compression.

## 2. Chapter-by-Chapter Cuts

| Unit | Before | After | Page change | Main changes |
|---|---:|---:|---:|---|
| Chapter 1 | 4 | 3 | -1 | Merged five-section rhythm into three sections; compressed contributions and structure. |
| Chapter 2 | 4 | 3 | -1 | Removed repeated roadmap and shortened representation/projection caveats. |
| Chapter 3 | 5 | 5 | 0 | Compressed prose while retaining all corpus counts, validation checks, and sampling logic. |
| Chapter 4 | 3 | 3 | 0 | Folded implications into representation limits and shortened measurement/display explanation. |
| Chapter 5 | 4 | 4 | 0 | Preserved definitions and formulas; merged metric reduction and interpretation. |
| Chapter 6 | 7 | 6 | -1 | Shortened empirical framing, captions, and closing caveat. |
| Chapter 7 | 9 | 7 | -2 | Merged subfield and field movement discussion; shortened captions and final limits. |
| Chapter 8 | 6 | 6 | 0 | Tightened relational caveats and figure captions. |
| Chapter 9 | 7 | 6 | -1 | Compressed clustering setup and ending; kept diagnostics and sensitivity values. |
| Chapter 10 | 4 | 3 | -1 | Replaced chapter-by-chapter recap with synthesis, contributions, limitations, implications. |
| Chapter 11 | 2 | 2 | 0 | Removed section headings and turned conclusion into one lean final argument. |
| Appendices | 9 | 8 | -1 | Compressed Appendix D and shortened appendix captions/notes. |

## 3. Structure Changes

- Chapter 1: 5 sections reduced to 3: Motivation; Research Gap and Questions; Contribution and Structure.
- Chapter 2: 5 sections reduced to 4 and renamed around relation, document geometry, morphology, and positioning.
- Chapter 3: 5 sections reduced to 4; "From Validated Corpus to Analysis Subset" and "What the Corpus Can Support" became "From Corpus to Analysis."
- Chapter 4: 5 sections reduced to 4; visual-display and implication material was consolidated.
- Chapter 5: 5 sections reduced to 4; "Reduction to an Interpretable Metric Core" and "Interpretation and Limits" became "Interpreting the Core."
- Chapter 6: 5 sections reduced to 4; the defensive final section became a short closing paragraph.
- Chapter 7: 6 sections reduced to 4; "Subfields as Moving Semantic Paths" and "Field-Level Departures and Full-Period Signals" were merged.
- Chapter 8: 5 sections reduced to 4; "What Morphological Relations Mean" became a closing paragraph.
- Chapter 9: 5 sections reduced to 4; "Keeping Typologies in Bounds" was folded into the final comparison.
- Chapter 10: 5 sections reduced to 4: Synthesis; Methodological Contributions; Limitations; Implications and Future Research.
- Chapter 11: section headings removed entirely to avoid another mini-thesis structure.

The five-section rhythm was deliberately broken in Chapters 1--11 except where a four-part empirical flow was clearer. It was preserved nowhere as a template; it remained only where chapter content naturally required multiple analytical stages.

## 4. Caveats Consolidated

- OpenAlex taxonomy caveats are now concentrated in Chapters 2, 3, 10, and brief conclusion language.
- SPECTER2 and title--abstract representation limits are concentrated in Chapters 4 and 10.
- Measurement-space versus projection cautions are concentrated in Chapters 4, 5, 9, Appendix D, and brief figure-caption reminders.
- "Not quality, impact, productivity, causality, or maturity" language was shortened in Chapters 6--8 and retained more fully in Chapter 10.
- Clustering as descriptive, not natural categories, is concentrated in Chapter 9 and Chapter 10.
- Semantic location versus morphology is preserved in Chapter 9, Appendix C, Chapter 10, and the conclusion without repeating full warning paragraphs.

## 5. Captions and Tables

Captions shortened:

- Figures 1.1, 2.1, 3.1, 5.1.
- Figures 6.1--6.5.
- Figures 7.1--7.7.
- Figures 8.1--8.4.
- Figures 9.1--9.4.
- Figure C.1.
- Figures D.1--D.6.

Tables preserved:

- All table values and table structures were preserved.
- Table notes shortened in Tables 3.1, 4.1, 5.1, 9.1, and 9.2.
- No tables were moved or deleted.

Figures changed:

- No figure files or figure data were changed.
- Only captions and surrounding prose were shortened.

## 6. Empirical Integrity

Confirmed unchanged in substance:

- empirical results and numerical values;
- corpus counts;
- metric definitions and formulas;
- clustering methods and diagnostics;
- interpretation of weak static clustering;
- interpretation of stronger temporal clustering;
- semantic location versus morphology distinction;
- measurement-space versus visualization distinction;
- bibliography.

## 7. Compilation and QA

Build commands run from `memory/`:

```text
pdflatex -interaction=nonstopmode -halt-on-error memoria.tex
biber memoria
pdflatex -interaction=nonstopmode -halt-on-error memoria.tex
pdflatex -interaction=nonstopmode -halt-on-error memoria.tex
pdflatex -interaction=nonstopmode -halt-on-error memoria.tex
```

Final output:

- Compiled PDF: `memory/memoria.pdf`
- Final page count: 65
- No unresolved citations or references found in the final log.
- No overfull or underfull boxes found in the final log after cleanup.
- No raw local paths, placeholders, TODO/FIXME markers, or introduced em dashes found in edited thesis sources.

Remaining build notices:

- `pdfx` reports that `memoria.xmpdata` is missing, so metadata may be incomplete. This warning existed independently of the text pass because the repository contains `output.xmpdata`, not `memoria.xmpdata`.
- `pdfx` reports its standard PDF/A color-command warning.
- MiKTeX prints its update-check notice.

## 8. Manual Review Points

- Read Chapter 1 to confirm that the compressed contribution list still matches the author's preferred framing.
- Check Chapter 10 to ensure the synthesis feels sufficiently discussion-like after removing the chapter-by-chapter recap.
- Inspect Appendix D visually; captions are now intentionally minimal.
- Decide whether reaching 60--63 pages is worth figure/layout intervention. The current 65-page version is a sharper text-first result without changing the empirical apparatus.
