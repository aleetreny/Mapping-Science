# Stage 1 Style Diagnostic Report

Source basis: `memory/memoria.tex`, the compiled `memory/memoria.pdf`, the active chapter and appendix sources included from `memoria.tex`, the three supervisor writing samples, and the written-memory rubric in `research/writing_sample/normativa_tfm_2014.pdf`.

No thesis source, figure, table, bibliography, or compiled output was edited for this stage.

## A. Style Profile Of The Reference Papers

The three papers share a direct, problem-led academic register. They do not open by announcing a broad contribution in abstract terms. They start from a concrete difficulty: large multidimensional grids make straightforward smoothing computationally impractical; array data lose useful structure when flattened into ordinary regression form; mortality forecasting matters because long-horizon forecasts are practically necessary but intrinsically uncertain. The method then appears as a response to that difficulty, not as a preselected tool.

Their strongest transferable habit is that methodological choices are justified by consequences. A method is not only named; the reader is told what it makes possible, what it avoids, and what assumption it imposes. Tensor-product bases are useful but memory hungry; array-preserving computation avoids constructing the full model matrix; treating future mortality as missing values makes forecasting part of smoothing, but makes the penalty order substantively important. This is the model the thesis should follow for OpenAlex, SPECTER2, robust scaling, five-year windows, cosine distance, Euclidean profile distance, PCA, UMAP, and clustering.

Notation is introduced in working order. The papers begin with the data structure, then define matrices, coefficients, penalties, and algorithms only as needed. Equations are followed by a short explanation of why the form matters computationally or statistically. They do not surround every formula with broad claims. The thesis can use the same discipline: introduce a symbol, state what it measures, explain the decision it supports, then move on.

The papers compare alternatives without overbalancing the prose. APC, Lee-Carter, Loess, thin-plate splines, direct Kronecker products, mixed-model approaches, and low-level implementations are discussed when they clarify the adopted route. Alternatives are not listed as literature inventory. They are used to locate a decision: why this representation, why this penalty, why this computation, why this limitation.

Section movement is practical: problem, model or algorithm, example, consequences, limits. The papers often move from one-dimensional cases to higher-dimensional generalizations, or from data description to method to application. This gives readers a ladder. In the thesis, the equivalent ladder is: corpus construction, representation, structural metrics, static profiles, temporal profiles, field relations.

Limitations are handled plainly and locally. The papers do not apologize repeatedly. They state the boundary once, usually where it affects interpretation: forecasts are uncertain; light smoothing of very large matrices remains hard; penalty order shapes extrapolation. The thesis should do the same with OpenAlex coverage, title-abstract evidence, SPECTER2 model dependence, projected-map distortion, descriptive rather than causal claims, and clustering instability.

The style uses signposting, but not as a substitute for argument. Roadmaps are brief. Paragraphs are usually medium length and carry one technical purpose. First-person plural is normal when describing author decisions or computations, but it does not become self-advertising. For the thesis, first-person singular is acceptable when naming work actually carried out by the student, especially in methods, but it should not be used to repeatedly announce the contribution.

## B. Diagnosis Of The Current Thesis

The current thesis has a solid empirical spine. The eight-metric structural profile, the separation of centroid drift, the high-dimensional SPECTER2 measurement space, the robust scaling decision, and the static-temporal-relational sequence are clear. The main problem is not technical content. It is cadence: the text often explains the same boundary in the same kind of sentence, so the prose begins to sound generated even when the argument is valid.

The most visible artificial pattern is repeated analytical separation. The introduction, background, methods, temporal chapter, relations chapter, discussion, and conclusion all return to variants of: morphology is not taxonomy, not semantic location, not visualization, not causality, not a natural classification. This distinction is essential, but it is currently stated too many times. Later revisions should place the full distinction in the introduction and methods, then use shorter reminders only where a specific result could otherwise be misread.

The thesis also overuses symmetrical caution. Examples occur in the introduction's descriptive-not-causal paragraph, the static chapter's warning that compactness is not importance or coherence, the temporal chapter's clustering caveat, the relations chapter's distinction between morphological similarity and topical/citation similarity, and the discussion's limits. Each caution is defensible. Together they create a repeated "not X, but Y" rhythm.

Several chapter endings restate the central contribution instead of adding a clean handoff. `memory/chapters/06_static_comparison.tex` closes by reasserting that domains orient but do not determine field shape. `memory/chapters/07_temporal_evolution.tex` closes by restating trajectory vocabulary and centroid drift. `memory/chapters/08_morphological_similarity.tex` ends with a full answer to Research Question 3 that overlaps with the conclusion. `memory/chapters/10_discussion.tex` and `memory/chapters/11_conclusion.tex` again restate the separation between labels, locations, shapes, trajectories, and relations.

Some figure discussion is formulaic. The pattern is often: figure introduced, visible result listed, caveat added, thesis claim restated. This occurs especially in the static, temporal, and relations chapters. Future revisions should make the prose do work the figure cannot do by itself: compare panels, explain why a contrast matters, name the metric driving the result, or connect the observation to the next analytical step.

The methods chapter is strong but dense. It contains corpus design, validation, representation, metrics, temporal windows, scaling, aggregation, workflow, and evidential limits. The main risk is not length; it is that some methodological choices are stated more than argued. The annual cap, no redistribution from sparse cells, title-abstract choice, SPECTER2 choice, cosine distance, `k=15`, robust median/IQR scaling, five-year windows, Euclidean profile distances, and clustering settings should each have one precise rationale and, where useful, one rejected alternative.

The background chapter is efficient, but it sometimes moves too quickly from literature to thesis position. The gap should be sharpened in the reference-paper style: existing maps answer relation, topic, and location questions; the missing object is the distributional shape of field-level document clouds under a fixed scientific-document representation.

The conclusion is compact and clear, but it repeats language already used in the introduction and discussion. The final rewrite should make the conclusion feel like earned closure: what was measured, what was found, what the reader can now distinguish, and what remains unvalidated.

## C. Proposed Thesis Writing Standard

Paragraphs should carry one job. A strong paragraph should usually have this structure: claim or decision; evidence, definition, or numerical result; consequence for interpretation. Avoid opening a paragraph with a broad claim and closing it by repeating the same claim in more abstract terms.

Use varied sentence rhythm. Mix short interpretive sentences with longer technical ones. Avoid long runs of sentences beginning with "The", "This", or "These". Do not make every paragraph a balanced list of three or four items.

Transitions should be technical rather than theatrical. Prefer "Because", "To avoid", "Under this design", "For comparison", "The consequence is", and "This affects interpretation by..." over generic bridges such as "This matters because" or repeated "therefore" conclusions.

Use first person only for work actually done or decisions actually made: "I use OpenAlex because...", "I compute metrics in the original SPECTER2 space...", "I treat centroid drift separately...". Do not use first person to keep re-announcing the thesis contribution.

Present methods as decisions under constraints. For each major tool or choice, give the problem it solves, the alternative it avoids, and the interpretive cost it introduces. This is especially important for OpenAlex, primary-topic subfields, title-abstract text, balanced sampling, SPECTER2, original-space computation, cosine distance, `k=15`, robust scaling, five-year windows, Euclidean profile distance, PCA, UMAP, and clustering.

Interpret results at the metric level. Do not say only that fields differ or converge. Say which structural family drives the difference: global dispersion, local density, hubness, or spectral structure. When possible, separate "same direction" from "same mechanism".

Treat limitations once, in the most relevant place. Coverage limits belong in corpus construction; model dependence belongs in representation; projection distortion belongs with PCA/UMAP; descriptive-not-causal interpretation belongs in the introduction and discussion; clustering instability belongs with clustering. Later chapters can refer back briefly without restating the whole warning.

Open sections from a problem or analytical need. Avoid openings that only describe what the section will do. Close sections by handing the reader to the next step, not by repeating the chapter's contribution.

For figures and tables, captions should be self-contained, while body text should be selective. Do not narrate every visible element. State the comparison, name the main pattern, and explain why that pattern changes the interpretation.

## D. Rubric-Alignment Checklist

Presentation:
- Sentences are grammatical, direct, and varied in length.
- No paragraph reads like generic academic filler.
- Figures and tables have self-contained captions.
- Body text explains why a figure matters, not just what it contains.
- After each chapter compile, all pages are visually checked for large blank areas, stranded headings, isolated captions, and figure-only pages that are not intentional.

Description of the problem:
- The chapter states the current state of the relevant problem before presenting the thesis decision.
- The thesis position is precise: field distributions are compared by structural shape in a fixed scientific-document embedding space.
- References are current and relevant to the chapter's actual task.
- Technical language is used accurately and without unnecessary inflation.

Tools used:
- OpenAlex is justified as open, auditable, hierarchical, and scalable, while its coverage and classification limits are stated once.
- SPECTER2 is justified as a scientific document representation, not a generic embedding model.
- Original 768-dimensional computation is justified as protection against projection artifacts.
- Cosine distance, `k=15`, robust scaling, five-year windows, Euclidean profile distance, PCA, UMAP, and clustering choices each have a clear rationale.
- Software or pipeline details appear only when they support reproducibility or interpretation.

Description of the work developed:
- The student's work is visible: corpus construction, validation, embedding alignment, metric design, static comparison, temporal analysis, centroid-drift calculation, relational analysis, and figure/table generation.
- The adopted solution and the steps leading to it are both described.
- Reproducibility is preserved without turning the thesis into a code manual.
- Data construction, validation, metric computation, visualization, and interpretation remain distinguishable.

Exposition of the work:
- Objectives, methods, findings, and conclusions are easy to locate.
- Chapters advance rather than repeat the same central distinction.
- Each empirical chapter has a clear internal order: setup, main evidence, interpretation, handoff.
- Repeated cautions are consolidated.
- The conclusion synthesizes rather than recycles.

## E. Chapter-By-Chapter Risk Map

Chapter 1, Introduction (`01_introduction.tex`): The research questions and active framework are clear. Main risks are repeated contribution language, a long roadmap, and early overuse of cautions. Strengthen the concrete problem: existing maps and labels do not measure distributional shape. Preserve the research questions, the eight-metric framework, and the separate centroid-drift distinction.

Chapter 2, Background (`02_background.tex`): The literature is relevant and compact. Main risks are compression and repeated "not just location/label" phrasing. The gap should be made more problem-led: what existing science maps answer, and what they leave unmeasured. Preserve the citation structure and the distinction between bibliometric relations, semantic representations, and morphology.

Chapter 3, Data, Representation, and Morphology Metrics (`03_data_and_corpus.tex`): This is the highest methodological-load chapter. Main risks are density, repeated coverage cautions, and choices that are stated before their rationale is fully explicit. Strengthen the rationale for the annual cap, no redistribution, title-abstract records, SPECTER2, original-space metrics, cosine distance, `k=15`, robust scaling, and five-year windows. Preserve all counts, formulas, validation results, metric definitions, and the row-alignment contract.

Chapter 4, Static Morphology (`06_static_comparison.tex`): The empirical sequence is good: diagnostics, domains, fields/subfields, extremes, profile PCA. Main risks are figure-heavy pages, formulaic figure discussion, and repeated taxonomy caveats. Reduce recap endings and make each figure paragraph identify the metric family doing the work. Preserve correlations, domain tendencies, field/subfield examples, extremes, and PCA variance values.

Chapter 5, Temporal Evolution (`07_temporal_evolution.tex`): The chapter has strong results but many moving parts: trajectories, deltas, subfield rankings, centroid drift, and clustering. Main risks are ranking overload, repeated separation of centroid drift, and clustering caveats that sound like disclaimers. Strengthen the reason for five-year windows and for slope-based dynamic clustering. Preserve the 1,204 of 1,205 computation fact, first-last values, dynamic subfields, centroid-drift results, clustering methods, silhouette, and ARI.

Chapter 6, Relations (`08_morphological_similarity.tex`): The chapter answers Research Question 3 clearly. Main risks are repeated warnings that similarity is not topical/citation similarity, plus a final paragraph that duplicates the conclusion. Strengthen why Euclidean distance between robust-scaled field profiles is the right operational relation here, and what taxonomy is allowed to do as context. Preserve all pair counts, distance values, nearest-neighbor claims, bridge/isolate cases, and convergence/divergence results.

Chapter 7, Discussion (`10_discussion.tex`): The discussion is compact and honest. Main risks are overlap with introduction and conclusion, and a somewhat generic contribution paragraph. Make it more synthetic: what the three empirical chapters jointly establish, what they do not establish, and what validation would test. Preserve the comparative-only claim boundary and future-validation directions.

Chapter 8, Conclusion (`11_conclusion.tex`): The conclusion is short and structurally sound. Main risks are repeated contribution language and possible duplication between the paragraphs and the research-question table. The final version should sound earned rather than promotional. Preserve the main numerical findings, the centroid-drift separation, and the RQ summary unless later layout review shows the table creates avoidable page imbalance.

Appendix A (`appendix_metric_definitions.tex`): Strong and should remain close to current form. Main risk is only consistency with any terminology changes in the main text. Preserve formulas unless a technical correction is required.

Appendix B (`appendix_umap_atlas.tex`): The role of UMAP as visual diagnostic is clear. Main layout risk is intentional figure-only appendix pages. Preserve all selected atlas figures unless a later prompt explicitly removes them.

Current PDF layout risks from visual scan: front matter sparsity is normal; empirical pages are figure-heavy but mostly balanced; pages around the end of Chapter 6 and the end of Discussion appear underfilled; appendix atlas pages are figure-only by design; bibliography final page is naturally sparse. Later chapter edits should first tighten repeated prose, then adjust float sizing or placement if needed.

## F. Non-Negotiable Preservation Rules

Preserve verified numerical results, including corpus sizes, subfield counts, metric values, correlation values, drift values, distance values, convergence/divergence counts, clustering diagnostics, and all table/figure meanings.

Preserve formulas and definitions unless a technical correction is explicitly required.

Preserve citations and attribution. Do not remove a citation merely to make prose shorter if it supports a methodological or literature claim.

Preserve the analytical distinction between taxonomy, semantic location, structural morphology, temporal structural change, field relations, and centroid drift.

Preserve the active eight-metric framework as the structural morphology profile: global dispersion, local density, hubness, and spectral structure.

Preserve centroid drift as a separate semantic-displacement indicator. Do not fold it into the eight-metric profile, PCA, clustering, or distance matrices.

Preserve the fact that quantitative metrics are computed in the original 768-dimensional SPECTER2 space and that PCA/UMAP are display or exploratory profile tools, not the measurement basis.

Preserve the research questions and their descriptive, comparative scope.

Preserve reproducible methodological facts: OpenAlex source, primary-topic subfields, title-abstract filtering, annual cap, row alignment, validation checks, temporal windows, robust scaling, aggregation rules, and field-profile distance construction.

Preserve substantive figures currently selected for the thesis unless a later prompt explicitly authorizes removal.

## Recommended Workflow For Later Chapter Revision

For each chapter, first mark the claims and numbers that cannot change. Then rewrite for one chapter-level purpose: remove repeated cautions, strengthen underjustified methods, and replace generic recap with specific consequence. Compile immediately after editing. Visually inspect every page of the edited chapter in the PDF. Fix layout by tightening repeated prose first, then by modest float sizing or placement adjustments. Finally, check that formulas, citations, figure references, table references, and numerical claims still match the original evidence.
