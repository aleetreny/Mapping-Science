# Agent Prompt — Strict Bibliographic Audit of Literature Review Sources

Audit and correct every reference in:

```text
research/literature_review/candidate_references.bib
```

This is a strict bibliographic verification task. Do **not** treat “BibTeX compiles” as sufficient. A BibTeX entry can compile and still be factually wrong. Your goal is to make every source academically valid, correctly cited, and safe to use in a Master’s Thesis.

## Goal

Every reference must be manually verified against at least one reliable external source:

- publisher page;
- ACL Anthology;
- arXiv;
- DOI resolver / CrossRef;
- OpenAlex;
- journal/conference official page;
- official documentation page for data/tool sources.

If a reference cannot be verified, do not silently keep it. Move it to `sources_to_verify.md` or remove it from the candidate bibliography.

## Required Process

For each BibTeX entry, verify and correct:

1. title;
2. authors;
3. year;
4. publication type;
5. journal / booktitle;
6. volume, issue, pages if applicable;
7. DOI;
8. arXiv ID if applicable;
9. URL;
10. whether it is peer-reviewed, preprint, software paper, documentation, blog, or working paper.

Open the DOI, arXiv, ACL, publisher, or OpenAlex page when needed. Do not rely only on existing metadata in the repo.

## Known Issues to Fix First

These entries are already suspicious and must be corrected manually:

### 1. `singh2023scirepeval`

Check ACL Anthology. The current DOI/pages may be wrong. Verify the correct EMNLP paper ID, DOI, and pages.

### 2. `mcinnes2018umap`

Prefer the peer-reviewed JOSS citation if valid:

```text
McInnes, Healy, Saul, and Großberger,
“UMAP: Uniform Manifold Approximation and Projection”,
Journal of Open Source Software,
DOI: 10.21105/joss.00861
```

Use arXiv only as secondary if needed.

### 3. `visser2020google`

The current title/authors/DOI may be mixed with a different coverage-comparison paper. Verify the exact authors and DOI. If the DOI belongs to Martín-Martín et al., correct the entry or replace it with the intended Visser et al. source.

### 4. `scheidsteger2025reference`

Verify the DOI and authors. The current DOI may correspond to a different article. Correct or remove.

### 5. `salton1983vector`

Fix the year typo if present. It should not be `1083`. Verify title, pages, DOI.

### 6. `boyack2014creation`

Verify title, venue, DOI, pages. The current entry may confuse a Scientometrics record with the JASIST paper:

```text
“Creation of a highly detailed, dynamic, global model and map of science”
```

### 7. OpenAlex IDs in `note` fields

Check whether IDs are correct. If uncertain, remove OpenAlex IDs from the BibTeX rather than keeping wrong identifiers.

## Classification Rules

For each source, assign or update its classification in the literature review markdown files:

- `core-active`: directly supports an actually implemented method;
- `core-justification`: supports a key design choice but is not itself implemented;
- `background`: useful for framing;
- `future-work`: useful but not part of active methodology;
- `exclude`: unreliable, wrong, redundant, or not useful.

Do not mark theoretical papers as `implemented` unless the method is actually computed in the active codebase.

Examples:

- SPECTER2 source: `core-active`.
- OpenAlex official/source paper: `core-active`.
- UMAP paper: `core-justification` or `core-active-visualization-only`, never quantitative evidence.
- Chari & Pachter: `core-justification`, not implemented.
- Kleinberg: `core-justification`, not implemented.
- Mikolov/Word2Vec: probably `background`, not implemented.
- Leiden/Louvain: `future-work` unless there is an active cleaned clustering script using them, which there should not be.

## Files to Update

Update all relevant files consistently:

```text
research/literature_review/candidate_references.bib
research/literature_review/active_methodology_sources.md
research/literature_review/future_work_sources.md
research/literature_review/key_references_by_chapter.md
research/literature_review/claims_to_sources.md
research/literature_review/methodological_caveats.md
research/literature_review/literature_review_report.md
research/literature_review/sources_to_verify.md
```

If a source is corrected in BibTeX, update every markdown file that cites or describes it.

## Required New File

Create a new file:

```text
research/literature_review/reference_audit_log.md
```

For every reference, include a table with:

| key | status | action | verification source | notes |
|---|---|---|---|---|

Where:

- `status` is one of `verified`, `corrected`, `moved_to_verify`, `removed`;
- `action` explains what changed;
- `verification source` is the DOI/publisher/arXiv/ACL/OpenAlex/stable URL actually used;
- `notes` explain any caveats.

## Quality Requirements

Before finishing:

1. Re-run a BibTeX parser/validation script.
2. Confirm there are no placeholder DOIs.
3. Confirm there are no impossible years.
4. Confirm no source has mismatched DOI/authors/title.
5. Confirm no blog/preprint is silently presented as peer-reviewed.
6. Confirm every non-peer-reviewed source is explicitly marked in `sources_to_verify.md`.
7. Confirm the final bibliography has no duplicated or wrong OpenAlex IDs.
8. Confirm all “core” sources are genuinely useful for this thesis.
9. Confirm the audit log includes every single entry in `candidate_references.bib`.

## Important Warning

Passing BibTeX compilation is **not enough**. You must verify bibliographic truth, not syntax.

Do not stop until the audit log is complete and every source has been manually checked.
