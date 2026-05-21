# Agent Prompt — Refactor the `memory/` Overleaf Thesis Folder

You are working inside the `TFM` repository. The repository already contains an active thesis pipeline focused on SPECTER2 embedding-space morphology of scientific disciplines.

Your task is to refactor the `memory/` folder so it becomes a clean, modular, Overleaf-compatible thesis workspace. Do **not** write the full thesis content yet. This task is about structure, maintainability, compilation hygiene, and preparing the folder for chapter-by-chapter writing.

## Main Goal

Reorganize `memory/` into a clean LaTeX thesis workspace while preserving the important parts of the official UC3M / Overleaf template.

The final result should make it easy to:

- write the thesis chapter by chapter;
- include figures and tables from the analysis pipeline;
- manage bibliography cleanly;
- compile in Overleaf;
- understand how the written thesis maps to the cleaned SPECTER2 embedding-space pipeline.

## Important Constraint

Preserve the important institutional and formatting elements of the current template:

- document class and page format;
- UC3M cover structure;
- PDF/A support via `pdfx`;
- `output.xmpdata` and PDF/A metadata files;
- bibliography system with `biblatex` and `biber`;
- APA bibliography style unless project requirements later say otherwise;
- margins, line spacing, title styles, captions, headers/footers, list of figures, and list of tables;
- UC3M logo and template image assets.

You may reorganize files, but do not destroy or bypass the template’s institutional requirements.

## Desired Folder Structure

Use judgment, but aim for something close to:

```text
memory/
  memoria.tex
  referencias.bib
  output.xmpdata
  pdfa.xmpi

  chapters/
    01_introduction.tex
    02_background.tex
    03_data_and_corpus.tex
    04_semantic_representation.tex
    05_embedding_space_metrics.tex
    06_static_comparison.tex
    07_temporal_evolution.tex
    08_morphological_similarity.tex
    09_visualization.tex
    10_discussion.tex
    11_conclusion.tex

  appendices/
    appendix_pipeline_details.tex
    appendix_metric_definitions.tex
    appendix_additional_figures.tex

  figures/
    README.md

  tables/
    README.md

  notes/
    writing_plan.md
```

The exact structure is flexible. The important point is that the thesis is modular and easy to edit.

## Required Changes

### 1. Make `memoria.tex` the master file

Refactor `memory/memoria.tex` so that it remains the main Overleaf file but delegates chapter content to separate files using `\input{...}`.

The thesis body should contain chapter calls conceptually like:

```latex
\chapter{Introduction}
\input{chapters/01_introduction}

\chapter{Background and Related Work}
\input{chapters/02_background}

\chapter{Data and Corpus Construction}
\input{chapters/03_data_and_corpus}

\chapter{Semantic Representation}
\input{chapters/04_semantic_representation}

\chapter{Embedding-Space Morphological Metrics}
\input{chapters/05_embedding_space_metrics}

\chapter{Static Comparison of Scientific Disciplines}
\input{chapters/06_static_comparison}

\chapter{Temporal Evolution of Scientific Morphology}
\input{chapters/07_temporal_evolution}

\chapter{Morphological Similarity, Convergence and Divergence}
\input{chapters/08_morphological_similarity}

\chapter{Visualization of the Scientific Space}
\input{chapters/09_visualization}

\chapter{Discussion}
\input{chapters/10_discussion}

\chapter{Conclusion}
\input{chapters/11_conclusion}
```

Use chapter titles consistent with the current thesis direction.

### 2. Add chapter placeholder files

Create the chapter files with concise placeholders, not full content.

Each placeholder should include:

- 3–6 LaTeX comments describing what will go there;
- notes on which repository outputs or docs feed that chapter;
- reminders for citations where relevant;
- TODOs for final numbers, figures, and tables.

Do not hallucinate final results.

Example:

```latex
% This chapter will describe the OpenAlex corpus, filtering rules,
% sampling design, and validation diagnostics.
% Inputs: docs/data_corpus.md, outputs/01_corpus_construction/, data/processed/.
% TODO: add final corpus counts once the pipeline is rerun.
```

### 3. Keep UMAP in the right place

The thesis uses UMAP only for visualization.

Therefore:

- Do not create a chapter where UMAP metrics are treated as main evidence.
- The visualization chapter may mention UMAP maps as interpretive figures.
- Methodology chapters should emphasize the original SPECTER2 embedding space as the quantitative geometry.
- Do not resurrect archived UMAP-metric analysis.

### 4. Clean LaTeX build artifacts

Remove committed LaTeX build artifacts from `memory/` if present, for example:

```text
*.aux
*.log
*.bcf
*.run.xml
*.toc
*.lof
*.lot
*.out
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.synctex.gz
```

Update `.gitignore` so these files do not return.

Do **not** remove source files, bibliography files, metadata files, logos, or template assets.

### 5. Bibliography hygiene

Keep `memory/referencias.bib`.

If it currently contains only template examples, either keep them commented or replace them with a clear comment block explaining that real references will be added after the literature review.

Do not add fake references.

Make sure `memoria.tex` still points to the correct bibliography path, ideally:

```latex
\addbibresource{referencias.bib}
```

unless the bibliography file is moved.

### 6. Add a writing plan

Create `memory/notes/writing_plan.md`.

It should explain:

- active thesis direction;
- chapter structure;
- which repository outputs feed each chapter;
- which chapters can be drafted now;
- which chapters require final analysis outputs first;
- pending literature review needs;
- where UMAP belongs in the thesis.

Keep it concise and practical.

### 7. Add figure and table instructions

Create README files in:

```text
memory/figures/README.md
memory/tables/README.md
```

They should explain how to name exported figures and tables.

Suggested convention:

```text
fig_05_static_metric_rankings.png
fig_06_temporal_domain_trajectories.png
fig_07_morphological_similarity_heatmap.png
tab_05_top_compact_subfields.tex
tab_07_top_converging_pairs.tex
```

The convention should connect figures/tables to thesis chapters.

### 8. Preserve compilation

After the refactor, the project should still compile in Overleaf.

The template uses `biblatex` with `biber`, so document the expected compilation flow:

```text
pdflatex
biber
pdflatex
pdflatex
```

or equivalent Overleaf automatic compilation.

If local LaTeX tooling is available, run a compile check. If not available, at least inspect the file paths and LaTeX syntax carefully.

### 9. Do not overfill the thesis yet

Do not write long prose sections at this stage.

This refactor should produce a clean skeleton ready for progressive writing, not a fake first draft.

## Expected Final Deliverable

Produce a concise summary with:

1. Files created.
2. Files modified.
3. Files deleted or moved.
4. LaTeX artifacts added to `.gitignore`.
5. Whether `memoria.tex` still follows the UC3M / Overleaf template.
6. Whether the thesis folder is ready for chapter-by-chapter writing.
7. Any open questions or risky changes.

## Quality Bar

A new reader should be able to open `memory/memoria.tex` and immediately understand:

- where each chapter lives;
- where references live;
- where figures and tables should go;
- how the thesis relates to the cleaned SPECTER2 embedding-space pipeline;
- that UMAP is only auxiliary visualization.
