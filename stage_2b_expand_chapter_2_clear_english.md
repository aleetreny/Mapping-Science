# Stage 2B — Strengthen Chapter 2 with targeted literature and clear, natural English

Use the approved style guide:

`stage_1_style_diagnostic_report.md`

Revise **only Chapter 2: Background and Related Work** and the bibliography entries required by that chapter. Do not modify Chapter 1, Chapter 3 or any later chapter, figures, tables, analytical outputs, formulas, or numerical results.

## Objective

The current Chapter 2 is clear and well written, but at approximately one and a half pages it moves too quickly from the existing literature to the thesis position. Expand it modestly—approximately 500–700 words, aiming for roughly 2.25–2.75 well-composed pages after compilation—without turning it into a long or generic literature review.

The additional material must deepen the argument, not merely increase the number of citations.

Maintain the style already achieved in the revised chapters:

- direct and problem-led;
- compact;
- technically explicit;
- varied in sentence and paragraph structure;
- free of repeated “not X, but Y” constructions;
- no generic literature inventory;
- no repeated announcement of the thesis contribution;
- no copied or imitated phrasing from the supervisor’s papers.

---

## Accessible and realistic English

The thesis is written by a Spanish student and will be read mainly by Spanish professors. The English must therefore be **correct, academic, natural, and easy to understand**, but it should not sound like a C2 language exercise or like highly polished literary prose.

Write in clear international academic English, approximately at a strong B2–C1 level:

- prefer common, precise words over rare or ornate vocabulary;
- prefer short or medium-length sentences when the idea is complex;
- avoid excessive subordinate clauses and heavily nested sentence structures;
- avoid idioms, rhetorical flourishes, literary expressions, and unusual phrasal verbs;
- avoid unnecessary nominalisations when a direct verb is clearer;
- avoid long strings of abstract nouns;
- do not use advanced vocabulary merely to sound academic;
- use technical terms when they are necessary, but explain them plainly;
- make each sentence understandable on a first reading;
- keep paragraph logic explicit without excessive signposting;
- allow a natural non-native academic voice, but do not introduce grammatical mistakes or artificial awkwardness.

The goal is not to make the English simplistic. The goal is to make the argument easy to follow for professors who are experts in statistics but may not be native English speakers.

When two formulations are equally accurate, choose the simpler one.

Examples of the preferred direction:

- prefer **“This method reduces memory use”** to **“This approach yields a substantial alleviation of computational storage requirements.”**
- prefer **“The two measures capture different relations between papers”** to **“The two measures instantiate non-equivalent relational ontologies.”**
- prefer **“The representation depends on how the model was trained”** to **“The resulting geometry remains contingent upon the underlying training objective.”**

Do not make every sentence short. Use natural variation, but clarity must come before sophistication.

---

## Literature to inspect

Read and verify the following sources before writing. Use only those that genuinely strengthen a specific claim. Do not cite all of them mechanically.

### 1. Different bibliometric relations produce different maps

**Boyack, K. W., & Klavans, R. (2010).**  
“Co-citation analysis, bibliographic coupling, and direct citation: Which citation approach represents the research front most accurately?”  
*Journal of the American Society for Information Science and Technology*, 61(12), 2389–2404.  
DOI: `10.1002/asi.21419`

Intended use: support the argument that citation-based maps are conditional on the relation used and that co-citation, bibliographic coupling, and direct citation should not be treated as interchangeable views of scientific structure.

### 2. Semantic and relational representations encode different dimensions

**Kozlowski, D., Dusdal, J., Pang, J., & Zilian, A. (2021).**  
“Semantic and relational spaces in science of science: Deep learning models for article vectorisation.”  
*Scientometrics*, 126, 5881–5910.  
DOI: `10.1007/s11192-021-03984-1`

Intended use: explain that article relatedness is multidimensional and that text-based and network-based embeddings encode different kinds of proximity. This is useful for positioning SPECTER2 geometry as a chosen scientific-document representation rather than a neutral or unique map of science.

### 3. Cognitive extent is different from publication volume

**Milojević, S. (2015).**  
“Quantifying the cognitive extent of science.”  
*Journal of Informetrics*, 9(4), 962–973.  
DOI: `10.1016/j.joi.2015.10.005`

Intended use: support the broader motivation for measuring the internal extent or structure of scientific fields independently of publication volume. Do not imply that its lexical-diversity measure is equivalent to the morphology metrics used in this thesis.

### 4. Scientific subfields change their relations over time

**Pan, R. K., Sinha, S., Kaski, K., & Saramäki, J. (2012).**  
“The evolution of interdisciplinarity in physics research.”  
*Scientific Reports*, 2, 551.  
DOI: `10.1038/srep00551`

Intended use: provide an empirical example of longitudinal change in the organization and interaction of scientific subfields. Make clear that interdisciplinarity or network integration is not the same object as convergence of eight-metric morphology profiles.

### 5. Broader science-of-science context

**Fortunato, S., Bergstrom, C. T., Börner, K., Evans, J. A., Helbing, D., Milojević, S., Petersen, A. M., Radicchi, F., Sinatra, R., Uzzi, B., Vespignani, A., Waltman, L., Wang, D., & Barabási, A.-L. (2018).**  
“Science of science.”  
*Science*, 359(6379), eaao0185.  
DOI: `10.1126/science.aao0185`

Intended use: situate the thesis within the broader quantitative study of the structure and evolution of science. Use sparingly; it should frame the problem, not replace more specific references.

### Optional source for temporal emergence

Use only if the temporal paragraph genuinely needs an additional concrete example:

**Small, H., Boyack, K. W., & Klavans, R. (2014).**  
“Identifying emerging topics in science and technology.”  
*Research Policy*, 43(8), 1450–1467.  
DOI: `10.1016/j.respol.2014.02.005`

Intended use: show that changing research fronts and emerging topics have been studied through dynamic citation structures. Do not conflate topic emergence with morphological convergence or divergence.

---

## Required substantive expansion

Strengthen Chapter 2 in four places.

### A. Explain what different science-mapping traditions measure

Go beyond listing citation, co-citation, bibliographic coupling, co-word, and taxonomy-based maps. Explain briefly that they construct proximity from different evidence:

- direct citation records explicit scholarly links;
- bibliographic coupling reflects shared reference bases;
- co-citation reflects later joint recognition;
- co-word approaches measure lexical association;
- taxonomies assign papers to predefined or algorithmic categories;
- document embeddings derive proximity from a learned representation of text and, depending on training, relational signals.

The purpose is not to declare one method superior. It is to establish that every map has a measurement object and that these objects are not interchangeable.

### B. Deepen the discussion of scientific-document embeddings

Explain in accessible terms that a document embedding compresses title–abstract information into coordinates optimized under a learned objective. Clarify:

- why this is richer than raw word counts;
- why scientific-domain training matters;
- why proximity depends on the model and training signal;
- why a fixed representation can nevertheless support consistent relative comparisons.

Do not repeat the mathematical mapping or implementation details reserved for Chapter 3.

### C. Connect prior measurement work to field morphology

Develop the transition from semantic location to distributional shape. Use the literature to explain why publication volume, a centroid, a topic label, or a projected map does not exhaust the structure of a field.

Show how the thesis’s four families address different properties of the same document cloud:

- global dispersion;
- local density;
- hubness;
- spectral structure.

Do not describe all eight metrics in detail or reproduce Chapter 3. The aim is to establish the conceptual need for a multidimensional structural profile and to make clear that the four families combine ideas that are usually studied separately.

### D. Give the temporal question a stronger theoretical basis

Add a concise paragraph explaining that scientific organization changes through specialization, emerging topics, cross-field interaction, and changing cognitive boundaries.

Then distinguish carefully:

- interdisciplinarity and integration concern relations or combinations across domains;
- topic emergence concerns the appearance and growth of research fronts;
- this thesis defines convergence and divergence more narrowly as decreasing or increasing distance between fields’ structural morphology profiles.

This distinction must prepare Research Question 3 without repeating Chapter 6.

---

## Structure

Prefer to retain the chapter’s current compact structure unless a third subsection materially improves clarity.

A suitable structure is:

- `2.1 Science Mapping and Scientific Document Representations`
- `2.2 From Semantic Location to Field Morphology`

A third subsection may be introduced only if it gives the temporal literature a clear and non-repetitive place. Do not create a standalone “Positioning of the Thesis” section that repeats Chapter 1.

---

## Citation and bibliography rules

- Verify every bibliographic field against the DOI or publisher record.
- Add BibTeX entries only for sources actually cited in the chapter.
- Read the relevant source before attributing a claim to it.
- Do not cite a paper for a stronger claim than it supports.
- Preserve all useful existing citations.
- Remove an existing citation only when the associated claim has genuinely disappeared.
- Avoid citation clusters that function only as decoration.
- Do not use secondary descriptions when the original paper is available.

---

## Boundaries

Do not move the following detailed justifications into Chapter 2:

- the annual sample cap;
- the seeded OpenAlex sampling procedure;
- the exact choice of five-year windows;
- `k = 15`;
- robust median/IQR scaling;
- the exact use of cosine distance;
- Euclidean field-profile distance;
- clustering algorithms or selected `k`;
- formulas for the eight metrics.

Those belong to Chapter 3 or the empirical chapters.

Do not change:

- the research questions;
- the active eight-metric framework;
- the four metric families;
- the separation of centroid drift;
- the interpretation of SPECTER2 as the active measurement space;
- any empirical claim or numerical result.

---

## Compile and inspect

After editing:

1. compile the full thesis;
2. render or screenshot every page containing Chapter 2;
3. check that the chapter is neither visibly underfilled nor padded;
4. avoid a final page with only a few lines where a concise edit or sensible page break can solve it;
5. do not alter later figures or add filler for layout purposes.

---

## Final report

Report:

1. sources inspected;
2. sources ultimately cited and the claim supported by each;
3. bibliography entries added or corrected;
4. approximate word-count change;
5. structural changes to Chapter 2;
6. repetitions avoided between Chapters 1 and 2;
7. examples of sentences simplified to improve accessibility;
8. page-layout result after compilation;
9. confirmation that no other chapter or analytical output was modified.
