# Editorial and Visual Polish Audit

Scope: `memory/memoria.tex`, chapters, appendices, tables, figures, outputs, and supporting docs were inspected before editing.

## 1. Repeated concepts

- OpenAlex as pragmatic taxonomy appears in Chapters 1, 2, 3, 9, 10, and 11. Keep the full explanation in Chapters 2 and 3; compress later reminders.
- SPECTER2 as a learned representation, not ground truth, appears in Chapters 2, 4, 10, and 11. Keep the full caution in Chapter 4 and the consolidated limitation in Chapter 10.
- UMAP/PCA/PCoA as visualization only appears in Chapters 2, 4, 5, 7, 9, Appendix C, and Appendix D. Keep the methodological statement in Chapter 4; later captions can use shorter reminders.
- Metrics computed in original embedding space appears in Chapters 1, 2, 4, 5, 6, 7, 9, 10, 11, and appendices. It remains central, but repeated long forms should be shortened after Chapter 5.
- Clusters as descriptive rather than natural categories appears in Chapters 1, 9, 10, and 11. Keep it in Chapter 9 and in the final synthesis, but reduce defensive phrasing.
- Morphology as not quality, impact, productivity, causality, or topic appears in Chapters 5, 6, 7, 8, 10, and 11. Keep the concept, but stop ending every empirical chapter with the same warning.

## 2. Heaviest prose

- Chapter 10 is the longest chapter and repeats several limitations already established in Chapters 3 to 5 and 9. It should be tightened without losing the consolidated limitations role.
- Chapter 9 has dense paragraphs in the semantic-location comparison. The content is valuable, but the rhythm can be improved with sharper topic sentences.
- Chapter 7 is figure-rich and mostly effective, but its closing section repeats warnings from Chapters 4 and 5.
- Chapter 2 closes with a long negative definition of what the thesis is not. It is useful, but it can become more compact and more confident.

## 3. Mechanical five-section rhythm

- Chapters 1 to 6, 8, 9, and 10 mostly follow a predictable five-section form.
- The repetition is most visible in Chapters 3 to 6, where each chapter ends with an "Implications" or "Interpretive Limits" section.
- The empirical chapters should preserve numbering and claims, but their section titles and closings can become more varied.

## 4. Repeated caveat paragraphs

- Chapter 6 final paragraph restates morphology is not superiority, simplicity, or mechanism.
- Chapter 7 final paragraphs restate no causal explanation, no simplicity/maturity interpretation, and z-scores as relative.
- Chapter 8 final section restates no topic, method, citation, causality, or quality interpretation.
- Chapter 9 final section is necessary, but can be written as a boundary of the claim rather than another defensive caveat.
- Chapter 10 should consolidate these cautions, allowing Chapters 6 to 8 to close more actively.

## 5. Tables that duplicate nearby prose

- Tables 3.1 to 3.3 anchor corpus counts, filters, and domain distribution. They should stay.
- Table 4.1 anchors the row-aligned embedding snapshot. It should stay.
- Table 5.1 is central to metric definitions. It should stay.
- Table A.1 is operational and belongs in the appendix. It should stay, but the note can remain short.
- Tables 6.2, 7.1, 8.1, 9.1, and 9.2 add interpretation beyond nearby prose. They should stay.

## 6. Figures that are weak or repetitive

- Early conceptual chapters have no figures. This makes the first third of the thesis text-heavy and increases reliance on repeated prose.
- Chapter 6 has five figures. They are useful, but the outlier ranking and PCA map partly overlap. Both can stay because one gives geometry and the other gives rank.
- Chapter 7 is visually varied after the centroid-path addition. The mix is strong enough to preserve.
- Chapter 8 Figure 8.3 is modest but necessary because it summarizes temporal pair-distance expansion.
- Chapter 9 figures are dense but purposeful. Appendix C correctly carries supporting diagnostics.
- Appendix D is useful as visual companion material, but its captions repeat "illustrative only" often. Shortening some captions would help if time allows.

## 7. Better visual formats

- A conceptual workflow diagram in Chapter 1 can replace part of the long explanatory load and make the thesis logic visible early.
- A metric-family schematic can make the eight-metric structural framework easier to remember and reduce reliance on prose plus table alone.
- A full pipeline diagram in Chapter 3 would be useful, but Table A.1 already provides reproducibility detail. Adding both might overburden the front half.
- A reader compass for Chapters 6 to 9 would be pleasant but risks feeling decorative. Do not add unless text reduction creates a clear opening.

## 8. Abrupt concept introductions

- "Morphology" is introduced clearly in Chapter 1, but a compact visual workflow would make the shift from paper vectors to field profiles easier.
- "Hubness" is defined well in Chapter 2 and Chapter 5.
- "PCA D80" and "spectral entropy" are technically clear in Chapter 5, but a schematic could help non-specialist readers retain the metric families.
- "Typological anchor" is understandable in Chapter 9, but Chapter 10 can make it more direct as a named compression device.

## 9. Caption length

- Most main-body captions now have short optional captions for the lists. Long captions in Chapters 8 and 9 still carry analytic explanation. They are acceptable because the figures are dense.
- Appendix D captions repeat the visualization-only caution. The appendix introduction already carries that caution, so individual captions can be shorter.

## 10. Repeated grammatical structures

- Repeated openings include "This chapter...", "The analysis...", "The result...", and "This does not mean...".
- Chapters 6 to 9 often use the same rhythm: define, show figure, warn. Later edits should vary openings and make caveats shorter.

## 11. Stronger transitions

- Chapter 1 can transition from motivation to research gap through the idea that a centroid is not a shape.
- Chapter 5 can frame metrics as a "grammar of shape" rather than another technical inventory.
- Chapter 9 can frame clustering as compression, not discovery, then keep the weak result as a feature of the interpretation.
- Chapter 11 can close with a stronger final paragraph that states what the thesis enables.

## 12. Safe opportunities to cut

- Compress repeated caveats in Chapters 6 to 8 by one paragraph each.
- Shorten Chapter 2's "what this thesis is not" paragraph.
- Tighten Chapter 10 limitations by folding representation and geometry cautions into fewer paragraphs.
- Replace some generic roadmap prose in Chapter 1 with the conceptual workflow figure.
- Avoid cutting corpus counts, metric definitions, diagnostic values, and selected clustering results.

## 13. Risks

- Cutting too much from Chapter 4 could weaken the measurement-space versus visualization-space distinction.
- Cutting too much from Chapter 9 could make weak clustering appear stronger than it is.
- Removing tables in Chapters 3 or 5 would damage auditability.
- Adding too many visuals would increase page count and create a new kind of monotony.
- Redesigning empirical figures without rerunning the analysis could create mismatch risk. Prefer conceptual figures and light caption/text edits.

## Proposed implementation

- Add two generated conceptual figures: Chapter 1 workflow and Chapter 5 metric-family schematic.
- Rewrite the Chapter 1 opening and structure section to be more direct and less boilerplate.
- Tighten Chapters 2 to 5 where repetition is high, preserving methods and counts.
- Lightly retitle or sharpen empirical chapter sections without changing results.
- Compress repeated closing caveats in Chapters 6 to 9.
- Tighten Chapter 10 into a stronger synthesis and consolidated limitation section.
- Strengthen Chapter 11's final voice.
- Preserve all empirical results, tables, metrics, clustering choices, and diagnostics.
