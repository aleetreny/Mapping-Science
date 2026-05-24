# Agent Prompt — Final Text Compression and Voice Pass (Text-First, Figures Stable)

You are working in the TFM repository. The thesis is scientifically complete and the figures/diagrams are broadly acceptable. The remaining problem is editorial: the text is too long, too repetitive, and too structurally predictable. This task is a **final text-first slimming and rhythm pass**.

Do not treat this as another visual redesign task. Treat it as a rigorous editorial compression task.

The current PDF is around 72 pages including front matter, bibliography, and appendices. The target is to move the final document toward **60 to 63 PDF pages if feasible**, or as close as possible without damaging the argument. If the exact target cannot be reached honestly, prioritize a sharper thesis over an arbitrary page count.

---

## Core Objective

Produce the same thesis, but leaner, less repetitive, and more readable.

The thesis must keep its scientific claims, empirical results, counts, methods, tables needed for verification, and core limitations. The writing, however, should no longer feel like it is repeatedly defending the same idea.

The current problem is not that the thesis lacks rigor. It is that rigor is being paid for several times.

---

## What Must Not Change

Do not change:

- empirical results;
- numerical values;
- corpus counts;
- table values;
- figure data;
- metric definitions;
- selected clustering methods;
- selected clustering diagnostics;
- interpretation of weak static clustering;
- interpretation of stronger temporal clustering;
- distinction between semantic location and morphology;
- distinction between measurement space and visualization;
- bibliography unless a citation is broken;
- figure files unless a caption or reference requires a minor fix.

Do not add new empirical claims.

Do not add new figures unless absolutely necessary to replace a large block of text. Assume existing visual scaffolding is enough.

Do not use UMAP, PCA, or PCoA as quantitative evidence.

Do not introduce em dashes. Preserve legitimate LaTeX ranges such as `2000--2024` and `title--abstract`.

---

## Main Editorial Diagnosis to Act On

The thesis repeats several protective distinctions too often:

- OpenAlex is a pragmatic taxonomy, not a natural taxonomy.
- SPECTER2 is a learned representation, not ground truth.
- UMAP/PCA/PCoA are visualization tools, not measurement spaces.
- Metrics are computed in original SPECTER2 space.
- Morphology is not impact, quality, productivity, causality, maturity, or truth.
- Clusters are descriptive anchors, not natural kinds.
- Semantic location and morphology are distinct.

All of these points are important. Keep them. But apply a **caveat budget**:

1. Explain each distinction fully only once, at the place where it belongs.
2. Later mentions should be one clause or one short sentence.
3. If the same warning appears three times, keep the strongest version and cut or compress the others.
4. Never end several consecutive chapters with the same defensive tone.

The reader should remember the distinctions because they are well placed, not because they are repeated.

---

## Required First Step: Build a Repetition Map

Before editing, create:

```text
research/final_outputs/final_text_compression_audit.md
```

The audit must include:

1. Current PDF page count.
2. Current main-text page count, excluding front matter, bibliography, and appendices.
3. Current appendix page count.
4. A chapter-by-chapter estimate of which pages can be cut.
5. A list of repeated caveats and where they appear.
6. A list of repeated chapter-opening formulas.
7. A list of repeated chapter-ending formulas.
8. A list of captions that repeat the surrounding prose.
9. Tables whose notes or surrounding paragraphs repeat the same information.
10. Sections that can be merged, shortened, or turned into bridge paragraphs.
11. A proposed cut target by chapter.

Do not start broad rewriting before this audit exists.

---

## Page Reduction Targets

Use these as guidance, not as mechanical quotas.

### Front matter and Summary
Target reduction: moderate.

- Compress the Summary if it exceeds what is needed for a thesis abstract.
- Avoid repeating the entire thesis logic in miniature if the introduction already does it.
- Keep corpus counts and central findings.

### Chapter 1: Introduction
Target reduction: 25 to 35 percent.

Priorities:

- Keep the strong opening line.
- Keep research questions.
- Shorten the thesis structure section aggressively.
- Merge or compress contributions if they read like later methodological repetition.
- Avoid re-explaining representation, measurement space, visualization, and typology in full.

### Chapter 2: Background and Related Work
Target reduction: 20 to 30 percent.

Priorities:

- Keep the literature positioning.
- Cut generic science-mapping exposition once the contrast is clear.
- Avoid repeating Chapter 1’s motivation.
- Make the chapter sharper: what the literature gives, what it does not measure, and where this thesis enters.

### Chapter 3: Data and Corpus Construction
Target reduction: 15 to 25 percent.

Priorities:

- Keep all corpus counts and eligibility logic.
- Reduce repeated caveats about OpenAlex and coverage.
- Let tables carry exact values; do not repeat every table value in prose.
- Compress validation paragraphs if they list facts already visible in tables.

### Chapter 4: Semantic Representation
Target reduction: 15 to 25 percent.

Priorities:

- Keep SPECTER2 justification and row-alignment logic.
- Merge or shorten repeated statements about title--abstract limits, learned representations, and visualization limits.
- Explain measurement space once, cleanly.
- Remove extra defensive phrasing.

### Chapter 5: Morphological Metrics
Target reduction: 10 to 20 percent.

Priorities:

- Preserve definitions.
- Preserve the eleven-metric core.
- Cut repeated interpretive warnings after Table 5.1 and after the formulas.
- Keep the metric-family schematic if it helps reduce prose.
- Avoid turning metric definitions into a tutorial.

### Chapters 6 to 9: Empirical Core
Target reduction: 8 to 15 percent each.

Be more conservative here.

Priorities:

- Preserve all empirical findings.
- Preserve the core figures and tables unless a table note or caption can be shortened.
- Remove boilerplate openings such as “This chapter begins...” when the same function can be stated directly.
- Compress closing caveat sections into shorter bridge paragraphs where possible.
- Do not cut the actual interpretation of results.
- Do cut repeated statements that results are descriptive, not causal, when the point has already been established.

### Chapter 10: Discussion
Target reduction: 25 to 35 percent.

Priorities:

- This chapter currently risks repeating the whole thesis.
- Keep synthesis, not summary.
- Do not restate every chapter in sequence unless it produces a new interpretive layer.
- Consolidate limitations into a tighter taxonomy: data, representation, metrics, projections, clustering.
- Keep future research, but make it lean.

### Chapter 11: Conclusion
Target reduction: 25 to 40 percent.

Priorities:

- Do not repeat the Summary and Discussion.
- Keep only the final contribution, four main takeaways if needed, and final claim.
- Make the ending strong and concise.
- Avoid another full recap of all chapters.

### Appendices
Target reduction: 10 to 25 percent if possible.

Priorities:

- Keep reproducibility and formulas.
- Shorten explanatory prose around appendix figures.
- Keep UMAP atlas caution, but do not repeat it in full for every paragraph.
- If a paragraph only says “this is only illustrative” again, compress it.

---

## Structural Freedom

You may break the five-section rhythm where it improves readability.

Allowed changes:

- merge short sections;
- remove a section heading and turn its content into a bridge paragraph;
- rename generic section titles;
- shorten predictable “what this chapter can say” endings;
- compress chapter roadmaps;
- move minor methodological reminders to appendices;
- reduce caption length;
- shorten table notes;
- delete paragraphs that only re-state already-established caveats.

Do not change chapter numbering unless necessary. Section numbering may change.

## Anti-Monotony Restructuring Pass

Use the reduction pass as an opportunity to break the thesis' predictable architecture, not merely to make the same architecture shorter. The current document often feels as if every chapter must contain five sections, open with a roadmap, proceed through the same explanatory rhythm, and close with the same warning paragraph. That pattern should be actively challenged.

For each chapter, ask whether its current section structure is earned. If a section only exists because the surrounding chapters have a similar section, merge it, rename it, or turn it into a short bridge paragraph. The final thesis should not look like eleven chapters generated from the same template.

Allowed and encouraged:

- break the five-section pattern when it creates monotony;
- allow some chapters to have three or four stronger sections rather than five predictable ones;
- merge generic endings such as “What X can say” into sharper concluding paragraphs;
- rename flat section titles so they carry analytical movement rather than administrative labels;
- replace repeated chapter roadmaps with direct conceptual openings;
- vary openings: some chapters may begin with a claim, others with a methodological contrast, others with a transition from the previous empirical result;
- vary endings: not every chapter needs to end defensively; some can close by stating what the result enables;
- use shorter bridge paragraphs to connect chapters instead of long “this chapter does...” explanations;
- avoid repeating the same paragraph shape: claim, caveat, claim, caveat;
- alternate dense technical explanation with concise interpretive sentences.

Do not innovate by making the thesis informal, theatrical, or unstable. Innovate through rhythm, hierarchy, and compression: fewer template sections, better transitions, sharper topic sentences, more purposeful paragraph lengths, and less predictable chapter endings.

Concrete structural targets:

- Chapters 1--5 may be re-shaped more freely because they currently carry much of the setup repetition.
- Chapters 6--9 should keep their empirical logic, but their openings, captions, and closing caveat sections can be made less formulaic.
- Chapters 10--11 should not repeat the same chapter-by-chapter recap structure. They should synthesize and conclude, not re-walk the thesis.
- Appendices should be functional, not essay-like.

The final report must explicitly state where the five-section rhythm was broken, where it was deliberately preserved, and why.

---

## Style Rules

Make the prose more varied and less mechanical.

Avoid overusing:

```text
This thesis...
This chapter...
The analysis...
The result is therefore...
This does not mean...
It is important to note...
In this sense...
The following section...
The central point is...
```

Replace them with:

- direct topic sentences;
- tighter transitions;
- varied paragraph lengths;
- occasional short sentences after technical paragraphs;
- stronger verbs;
- fewer nominalizations;
- less bureaucratic phrasing.

The target voice is: rigorous academic essay, not lab report; confident, not promotional; careful, not apologetic.

Preserve strong existing lines such as:

```text
Scientific fields are usually named before they are measured.
A map of science is always a map of a relation.
The central result is not a new taxonomy of science, but a way to measure why no single taxonomy is enough.
```

Build around these lines rather than drowning them in explanation.

---

## Caption and Table Economy

For every figure caption:

1. Keep what the reader needs to understand the figure.
2. Remove interpretation that is repeated in the paragraph below.
3. Remove methodological caveats already stated elsewhere unless the figure could otherwise be misread.
4. Prefer one compact caption over a long explanatory block.

For every table note:

1. Keep exact definitions or data-source clarification.
2. Remove repeated “not causal”, “not taxonomy”, “visualization only”, or “computed in original space” language if already clear nearby.
3. Do not repeat table values in prose unless interpreting them.

---

## Caveat Placement Rules

Use this distribution:

- Chapter 3: OpenAlex, sampling, coverage, and corpus limits.
- Chapter 4: SPECTER2, title--abstract, embedding, and projection limits.
- Chapter 5: metric interpretation limits.
- Chapter 9: clustering limits.
- Chapter 10: consolidated limitations.
- Appendix D: UMAP atlas caution.

Outside those locations, use only short reminders when necessary.

Examples of compressed reminders:

```text
Under the representation used here, ...
As a morphology measure, not a quality measure, ...
The projection is only a display of the profile space.
The cluster label is descriptive, not taxonomic.
```

Do not write a full warning paragraph every time.

---

## Editing Method

Work chapter by chapter.

For each chapter:

1. Identify duplicated ideas.
2. Mark paragraphs as KEEP, COMPRESS, MERGE, MOVE, or DELETE.
3. Edit conservatively around numbers, formulas, citations, and figure references.
4. After editing, check that every figure and table is still introduced and interpreted.
5. Check that no empirical result has changed.

Do not rewrite everything in a uniform new style. The goal is not to make the thesis sound like a different document; it is to remove the padding and strengthen the existing voice.

---

## Compilation and QA

After editing, run the full LaTeX build:

```text
pdflatex memoria.tex
biber memoria
pdflatex memoria.tex
pdflatex memoria.tex
```

or the repository’s equivalent build command if documented.

Then inspect the PDF.

Check:

- page count;
- contents;
- summary;
- Chapter 1 opening;
- all chapter endings;
- pages with tables;
- pages with figures;
- appendix start;
- Appendix D;
- bibliography;
- no unresolved references;
- no missing figures;
- no overfull boxes that visibly damage the PDF;
- no raw paths;
- no placeholder comments;
- no em dashes introduced.

---

## Final Deliverables

Create:

```text
research/final_outputs/final_text_compression_report.md
```

The report must include:

1. Old PDF page count.
2. New PDF page count.
3. Old and new main-text page count.
4. Old and new appendix page count.
5. Estimated percentage reduction.
6. Chapter-by-chapter cuts made.
7. Sections merged, renamed, deleted, or converted into bridge paragraphs.
8. Repeated caveats removed or consolidated.
9. Captions shortened.
10. Tables shortened, preserved, or moved.
11. Any figures changed, if any.
12. Confirmation that empirical results and numerical values were not changed.
13. Compilation status and remaining warnings.
14. Manual review points for the thesis author.

Also produce the final compiled PDF.

---

## Success Criterion

The final thesis should feel like a sharper version of the same work:

- fewer defensive loops;
- less repetition;
- more varied rhythm;
- less mechanical section structure;
- shorter captions and table notes;
- clearer chapter openings;
- stronger chapter endings;
- main contribution easier to see;
- closer to 60 pages without sacrificing scientific substance.

If forced to choose between hitting exactly 60 pages and preserving the quality of the argument, preserve the argument and explain why.
