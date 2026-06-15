# Stage 1 — Internalize the target writing style and define the revision standard

You are working on the current version of the Master’s Thesis:

**“Measuring the Shape of Science: Morphological Indicators and Evolution of Research Fields.”**

You will receive:

1. the current thesis PDF;
2. three academic papers co-authored by the thesis supervisor, María Luz Durbán;
3. the written-assessment rubric.

At this stage, **do not rewrite the thesis and do not modify any LaTeX, figures, tables, captions, bibliography, or source files**. Your only task is to study the material, understand the revision objective, and produce an actionable style and quality guide that will be used in later chapter-by-chapter prompts.

---

## 1. Main objective

The supervisor’s feedback is that the current thesis **sounds AI-generated**.

The problem is not the technical content. The problem is the written voice and the repeated rhetorical patterns. The final version must sound like a serious academic text written by a person who understands the analysis, makes explicit methodological decisions, and explains them directly.

The future rewrite must:

- preserve the thesis’s empirical results, formulas, definitions, citations, and analytical meaning;
- retain approximately the current overall length, preferably becoming slightly shorter rather than longer;
- eliminate repeated information, even when it is expressed with different wording;
- go directly to the point;
- explain every relevant methodological choice and its rationale;
- address realistic alternatives when they help justify a decision;
- use a recognisable academic voice with personality;
- avoid predictable, repetitive paragraph and sentence structures;
- avoid unnecessary meta-commentary about what the thesis “does” or “does not do”;
- avoid repeatedly restating the central contribution at the end of sections;
- keep necessary limitations and cautions, but state each one once in the most appropriate place.

The final text should be clear, technically precise, compact, and natural.

---

## 2. Use the reference papers as style evidence, not as text to copy

Study the three supplied papers closely. Extract the recurring principles of their academic writing, including:

- how they open a problem;
- how they state the purpose of a section or method;
- how they introduce mathematical notation;
- how they justify modelling and computational choices;
- how they compare an adopted method with alternatives;
- how they move from theory to implementation and examples;
- how they discuss limitations or uncertainty;
- how they close sections;
- typical paragraph length, sentence rhythm, level of signposting, and use of first person.

Do **not** imitate distinctive phrases, reproduce sentences, or construct a pastiche of a particular author. The papers are co-authored and should be treated as evidence of a target academic register, not as a source of reusable wording.

The desired adaptation is:

> direct, problem-led, technically explicit, restrained, and methodologically honest.

The thesis must retain its own subject matter and the student’s own voice.

---

## 3. Diagnose why the current thesis sounds artificial

Read the entire current thesis and identify concrete recurring patterns that may create an AI-written impression. Look especially for:

- repeated “not X, but Y” constructions;
- repeated “This matters because…” or equivalent transitions;
- sections that restate the same claim in the opening and closing paragraphs;
- formulaic conclusions after every figure;
- excessive signposting;
- repeated warnings and interpretive disclaimers;
- paragraphs with identical internal structure;
- overly balanced lists and symmetrical prose;
- uniform sentence length;
- generic academic filler;
- claims about the contribution that are repeated instead of demonstrated;
- unnecessary recaps of information already visible in tables or figures;
- arguments that are delayed, fragmented, or repeated across chapters;
- methodological decisions that are stated but not properly justified.

Distinguish between:

1. information that should be deleted as repetition;
2. information that should be moved to a more appropriate location;
3. information that needs a stronger explanation;
4. text that is already effective and should remain close to its current form.

Do not perform the rewrite yet.

---

## 4. Written-assessment rubric as hard acceptance criteria

The future chapter-by-chapter rewrite must be explicitly optimized for the **Excellent** level of the supplied written-assessment rubric, except for the “Tiempo de desarrollo” criterion, which should be ignored.

Treat the following as hard acceptance criteria:

### Presentation

- no spelling or grammatical errors;
- well-constructed sentences;
- easy reading;
- high-quality figures;
- figures should be sufficiently self-explanatory to support understanding.

### Description of the problem

- describe the current state of the problem;
- position the thesis clearly;
- use updated and relevant references;
- use appropriate technical language.

### Tools used

- describe and justify the tools used;
- explain why OpenAlex, SPECTER2, the embedding-space metrics, scaling, temporal windows, distance choices, clustering methods, and visual tools are appropriate;
- avoid software inventories without analytical purpose.

### Description of the work developed

- make clear what work was actually carried out by the student;
- describe both the adopted solution and the steps used to reach it;
- distinguish data construction, validation, embedding alignment, metric design, static analysis, temporal analysis, and relational analysis;
- preserve reproducibility without turning the thesis into a code manual.

### Exposition of the work

- maintain a logical and ordered structure;
- allow the reader to identify the key aspects of the project quickly;
- make objectives, methods, findings, and conclusions easy to locate;
- remove fragmentation and unnecessary repetition.

The guide you produce must show how the target style supports each of these criteria.

---

## 5. Page composition and figure placement

The final PDF should not contain visibly underfilled or awkward pages where avoidable.

During the later chapter-by-chapter revisions, after each chapter is edited and compiled, you will be required to:

1. render or screenshot every page of that chapter;
2. inspect the page composition visually;
3. identify:
   - pages with large unexplained blank areas;
   - pages containing almost only one figure;
   - isolated captions;
   - figures separated awkwardly from the text that discusses them;
   - short paragraph fragments pushed onto a new page;
   - headings stranded at the bottom of a page;
4. correct these issues through concise textual editing and sensible LaTeX float/layout adjustments.

All substantive figures currently selected for the thesis must be retained unless a later prompt explicitly says otherwise.

When balancing pages:

- prefer removing repetition and tightening prose;
- then improve float placement, figure sizing, or page breaks;
- only add text when a genuine explanation or justification is missing;
- never add filler merely to fill a page;
- do not force every page to contain the same amount of text;
- aim for professional, balanced pages rather than mechanically full pages.

The final layout must support the argument, not determine it.

---

## 6. Deliverable for this stage

Produce a concise but substantive diagnostic report with the following sections:

### A. Style profile of the three reference papers

Identify the transferable principles of their writing, supported by representative observations. Do not quote long passages.

### B. Diagnosis of the current thesis

List the main patterns that make it sound artificial, with examples identified by chapter or section.

### C. Proposed thesis writing standard

Create a practical guide for future rewrites covering:

- paragraph construction;
- sentence rhythm;
- transitions;
- use of first person;
- presentation of methods;
- justification of choices;
- interpretation of results;
- treatment of limitations;
- section openings and endings;
- figure and table discussion.

### D. Rubric-alignment checklist

Translate every relevant “Excellent” criterion into concrete checks to apply to each chapter. Exclude “Tiempo de desarrollo”.

### E. Chapter-by-chapter risk map

For each chapter, identify:

- the main stylistic problems;
- repeated information that may be removable;
- arguments or methodological reasons that need strengthening;
- likely page-layout risks;
- text that should not be substantially changed.

### F. Non-negotiable preservation rules

State clearly what must remain unchanged during stylistic rewriting:

- verified numerical results;
- formulas and definitions unless a correction is required;
- citations and attribution;
- analytical distinctions;
- active eight-metric framework;
- separate centroid-drift analysis;
- figure and table meanings;
- research questions;
- reproducible methodological facts.

End with a short recommended workflow for the later chapter-by-chapter revision.

Do not edit the thesis in this stage.
