# Agent Prompt — Add an Illustrative UMAP Atlas Appendix

You are working in the TFM repository. The thesis is mostly complete. Your task is to add a **short illustrative UMAP atlas** as an appendix, using existing PNG figures already generated in the repository.

This task is visual and interpretive. It must not change the empirical claims of the thesis.

---

## Objective

Add a new appendix containing selected **per-subfield UMAP scatter + density maps**.

The atlas should help the reader see concrete examples of paper-cloud morphology at subfield level. It should be beautiful and useful, but methodologically cautious.

The atlas must support the existing thesis story:

> The thesis measures morphology quantitatively in the original SPECTER2 embedding space. UMAP is used only as a visual diagnostic. The selected maps illustrate how different subfields can look as embedded paper clouds, but they do not define distances, clusters, typologies, or quantitative results.

Do **not** present these UMAP maps as scientific proof of clusters or objective subfield structure.

---

## Existing Figure Location

The relevant PNG files are already in:

```text
outputs/08_visualization/per_subfield_umap_smooth_density/figures/
```

Use the existing PNGs. Do not regenerate them unless a selected file is missing or unreadable.

Each selected PNG already contains two panels:

- Panel A: paper-level UMAP scatter.
- Panel B: smoothed density map.

The user likes these PNGs as they are. Preserve their design unless there is a technical problem.

---

## Selected Subfields

Locate and include exactly these six subfield maps:

1. **Human Factors and Ergonomics**
   - OpenAlex subfield ID visible in the figure/title: `3307`
   - Domain/field: Social Sciences / Human Factors and Ergonomics
   - Reason: one of the strongest visually separated paper-cloud structures and a highly atypical profile in the thesis.

2. **Nuclear and High Energy Physics**
   - ID: `3106`
   - Domain/field: Physical Sciences / Physics and Astronomy / Nuclear and High Energy Physics
   - Reason: strong example of a fragmented/dispersed physical-science paper cloud.

3. **Applied Microbiology and Biotechnology**
   - ID: `2402`
   - Domain/field: Life Sciences or Immunology and Microbiology context as displayed in the PNG
   - Reason: useful contrast between a main dense region and a separated component; connects to compactness/local structure.

4. **Computer Vision and Pattern Recognition**
   - ID: `1707`
   - Domain/field: Computer Science / Computer Vision and Pattern Recognition
   - Reason: visually rich Computer Science case and important for novelty/movement patterns.

5. **Signal Processing**
   - ID: `1711`
   - Domain/field: Computer Science / Signal Processing
   - Reason: visually strong, separated components; replaces General Dentistry as the preferred dynamic/Computer Science example.

6. **General Materials Science**
   - ID: `2500`
   - Domain/field: Materials Science / General Materials Science
   - Reason: preferred over Electrochemistry; visually clean example of a broad, continuous material-science paper cloud.

Do not include Artificial Intelligence, Electrical and Electronic Engineering, General Energy, General Dentistry, or Electrochemistry unless needed as fallback because one of the six selected files is missing.

Avoid duplicates.

---

## Where to Add It

Prefer creating a new appendix:

```latex
\chapter{Illustrative UMAP Atlas of Selected Subfields}
\input{appendices/appendix_umap_atlas}
```

Add this after Appendix C in `memory/memoria.tex`.

Create:

```text
memory/appendices/appendix_umap_atlas.tex
```

Copy the six selected PNGs into a stable thesis figure folder, preferably:

```text
memory/figures/umap_atlas/
```

Use clean stable filenames such as:

```text
fig_d_umap_human_factors_ergonomics.png
fig_d_umap_nuclear_high_energy_physics.png
fig_d_umap_applied_microbiology_biotechnology.png
fig_d_umap_computer_vision_pattern_recognition.png
fig_d_umap_signal_processing.png
fig_d_umap_general_materials_science.png
```

Do not reference raw `outputs/...` paths directly from the LaTeX body.

---

## Appendix Text

The appendix should begin with a concise methodological warning.

Use or adapt the following wording:

```latex
This appendix provides an illustrative atlas of selected subfield-level UMAP projections. The panels are not used to compute morphology metrics, assign clusters, measure distances, or support quantitative claims. They are included as visual diagnostics to make selected paper-cloud morphologies interpretable at the document-cloud level. All quantitative claims in the thesis remain based on metrics computed in the original SPECTER2 embedding space. Because each panel is a separate subfield-specific UMAP projection, axes, distances, shapes, and density scales should not be compared across subfields.
```

Then add a short paragraph explaining the selection logic:

```latex
The selected cases are not the visually prettiest maps in isolation. They are chosen because they connect to recurring empirical themes in the thesis: atypical profiles, dispersed physical-science structure, compact or separated local organization, computer-science novelty, and broad material-science morphology.
```

Keep the prose short. This appendix is visual, not another results chapter.

---

## Figure Layout

Use one existing PNG per figure.

Recommended layout:

```latex
\begin{figure}[p]
    \centering
    \includegraphics[width=\textwidth]{figures/umap_atlas/fig_d_umap_human_factors_ergonomics.png}
    \caption[Illustrative UMAP atlas: Human Factors and Ergonomics]{Illustrative UMAP atlas: Human Factors and Ergonomics. The left panel shows the paper-level UMAP scatter and the right panel shows the smoothed density representation. The projection is illustrative only and is not used for quantitative morphology measurement.}
    \label{fig:umap_atlas_human_factors}
\end{figure}
```

Use `[p]` or `[htbp]` depending on what compiles and looks best. Since the PNGs are wide, prefer one figure per page or one per float. Do not force two figures per page if readability suffers.

The figures should be readable in the compiled PDF:
- titles legible;
- axes legible;
- density colorbar legible;
- captions not excessive.

---

## Suggested Captions

Use short LoF captions and informative full captions.

### Human Factors and Ergonomics

Short:
```text
Illustrative UMAP atlas: Human Factors and Ergonomics
```

Full:
```text
Illustrative UMAP atlas: Human Factors and Ergonomics. The separated paper-cloud components make this case visually useful for interpreting the atypical profile behavior discussed in the thesis. The projection is illustrative only; quantitative metrics are computed in the original SPECTER2 space.
```

### Nuclear and High Energy Physics

Short:
```text
Illustrative UMAP atlas: Nuclear and High Energy Physics
```

Full:
```text
Illustrative UMAP atlas: Nuclear and High Energy Physics. The fragmented visual structure provides an example of a dispersed physical-science paper cloud. UMAP is used only as a subfield-level visualization and not as a measurement space.
```

### Applied Microbiology and Biotechnology

Short:
```text
Illustrative UMAP atlas: Applied Microbiology and Biotechnology
```

Full:
```text
Illustrative UMAP atlas: Applied Microbiology and Biotechnology. The map illustrates a dense main region together with a separated component, making the case useful for visualizing local organization beyond a single centroid. The projection is illustrative only.
```

### Computer Vision and Pattern Recognition

Short:
```text
Illustrative UMAP atlas: Computer Vision and Pattern Recognition
```

Full:
```text
Illustrative UMAP atlas: Computer Vision and Pattern Recognition. The panel provides a visually rich Computer Science example connected to the thesis discussion of novelty and movement. The projection is not used to define clusters or distances.
```

### Signal Processing

Short:
```text
Illustrative UMAP atlas: Signal Processing
```

Full:
```text
Illustrative UMAP atlas: Signal Processing. The separated components make this a useful visual companion to the temporal and novelty-oriented Computer Science cases. The projection is illustrative only and should not be compared geometrically with other panels.
```

### General Materials Science

Short:
```text
Illustrative UMAP atlas: General Materials Science
```

Full:
```text
Illustrative UMAP atlas: General Materials Science. The broad continuous cloud provides a contrast to the more fragmented examples in the atlas and illustrates why morphology is not reducible to a single centroid. Quantitative interpretation remains based on original-space metrics.
```

You may refine captions, but keep the same methodological caution.

---

## Storytelling Requirements

The appendix should not feel like a random image dump.

It should tell a simple story:

1. UMAP is only a visual diagnostic.
2. The selected maps show different visual manifestations of paper-cloud morphology.
3. Some subfields look separated into components; others look broad and continuous.
4. These visual impressions motivate why the thesis measures morphology with metrics rather than relying on projections.
5. The maps are illustrative companions to the metric-based chapters, not evidence replacing them.

Avoid:
- saying that UMAP components are real topics;
- naming visual islands as research areas unless there is metadata to support it;
- comparing distances across different subfield-specific UMAPs;
- claiming that density peaks equal communities;
- overinterpreting colorbar values across panels;
- adding lots of prose.

---

## Integration With Existing Thesis

Do not modify main chapters unless a single sentence in Chapter 4, 6, or 10 is needed to point readers to the atlas. Prefer not to touch main chapters.

If you add a cross-reference, keep it minimal, for example in Appendix C or at the end of Appendix D introduction:

```latex
These figures are intended as visual companions to the metric-based interpretation in Chapters 6--9.
```

Do not disturb Chapter 9’s current argument. The main empirical claim remains static/dynamic morphology and clustering diagnostics, not UMAP visuals.

---

## Technical Requirements

1. Locate the six PNGs in the output folder.
2. Copy them into `memory/figures/umap_atlas/`.
3. Add `memory/appendices/appendix_umap_atlas.tex`.
4. Add the new appendix to `memory/memoria.tex` after Appendix C.
5. Do not touch the Dedication section.
6. Do not change the Summary text unless there is a direct compilation issue.
7. Do not alter existing figures or tables.
8. Do not use raw repository paths in prose.
9. Keep figure labels unique.
10. Ensure the List of Figures updates correctly.

---

## Render and Review the PDF

After adding the appendix, compile:

```text
pdflatex -> biber -> pdflatex -> pdflatex
```

Then render the new appendix pages to images and inspect them visually.

Use whatever the repository already uses for PDF rendering, or a reliable local command such as:

```text
pdftoppm -png -f <first_appendix_d_page> -l <last_appendix_d_page> memoria.pdf appendix_d_page
```

or the existing project rendering workflow.

Review:
- all six figures appear;
- captions are readable;
- images are not too small;
- no figures are cut off;
- LoF entries are short and clean;
- Appendix D appears in the ToC;
- no unresolved references/citations;
- no raw `outputs/...` paths appear in the PDF;
- Dedication remains untouched.

If a figure is too small, adjust `width=\textwidth` or use a landscape page only if necessary. Do not leave unreadable figures.

---

## Final Report

Report:

1. whether Appendix D was created or another appendix structure was used;
2. which six PNGs were included;
3. where the PNGs were copied;
4. whether `memoria.tex` was updated;
5. whether Dedication was left untouched;
6. whether the PDF compiled successfully;
7. which pages of the PDF contain the atlas;
8. whether the rendered appendix pages were visually inspected;
9. any remaining warnings.

The final appendix should be visually attractive, methodologically safe, and consistent with the thesis style: it should add interpretive value without pretending that UMAP is the measurement space.
