from __future__ import annotations

import shutil
from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "16_selected_insert_visuals"
SOURCE_UMAP_DIR = ROOT / "outputs" / "08_visualization" / "per_subfield_umap_smooth_density" / "figures"

DOMAIN_COLORS = {
    "Life Sciences": "#2f7f5f",
    "Social Sciences": "#8b6bb5",
    "Physical Sciences": "#ba6a36",
    "Health Sciences": "#4c78a8",
}

QUADRANT_STYLES = {
    "Broad + concentrated": ("#f3e5d4", "#8a4f22"),
    "Broad + smooth": ("#e2eef7", "#2f5f8f"),
    "Compact + concentrated": ("#eee3f2", "#6d4a80"),
    "Compact + smooth": ("#e6f0dc", "#4f722e"),
}

SOURCE_IMAGES = {
    "Philosophy": "1211__Philosophy.png",
    "Artificial Intelligence": "1702__Artificial_Intelligence.png",
    "Applied Microbiology and Biotechnology": "2402__Applied_Microbiology_and_Biotechnology.png",
    "Research and Theory": "2922__Research_and_Theory.png",
    "Human Factors and Ergonomics": "3307__Human_Factors_and_Ergonomics.png",
}

SHORT_NAMES = {
    "Applied Microbiology and Biotechnology": "Applied Microbiology",
    "Artificial Intelligence": "Artificial Intelligence",
    "Human Factors and Ergonomics": "Human Factors",
    "Research and Theory": "Research & Theory",
}


def setup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 130,
            "savefig.dpi": 260,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def crop_density_panel(name: str) -> np.ndarray:
    image_path = SOURCE_UMAP_DIR / SOURCE_IMAGES[name]
    image = Image.open(image_path).convert("RGB")
    # Crop the rendered density panel without title, axis labels, or per-plot colorbar.
    # The colorbar would wrongly invite cross-panel density-scale comparisons.
    cropped = image.crop((1195, 100, 1995, 790))
    return np.asarray(cropped)


def load_scores() -> pd.DataFrame:
    score_path = ROOT / "outputs" / "15_glanceable_visual_candidates" / "ch06_compass_scores.csv"
    if score_path.exists():
        return pd.read_csv(score_path)
    raise FileNotFoundError("Run scripts/31_build_glanceable_visual_candidates.py first.")


def short_name(name: str, width: int = 31) -> str:
    return SHORT_NAMES.get(name, shorten(name, width=width, placeholder="..."))


def metric_line(row: pd.Series) -> str:
    return (
        f"breadth {row['semantic_breadth']:+.1f}  |  "
        f"hubness {row['local_concentration']:+.1f}  |  "
        f"novelty {row['temporal_novelty']:+.1f}"
    )


def draw_shape_card(
    ax: plt.Axes,
    *,
    title: str,
    subtitle: str,
    name: str,
    quadrant: str,
    scores: pd.DataFrame,
    note: str,
    accent_override: str | None = None,
    fill_override: str | None = None,
) -> None:
    row = scores.set_index("subfield_display_name").loc[name]
    fill, accent = QUADRANT_STYLES[quadrant]
    if accent_override is not None:
        accent = accent_override
    if fill_override is not None:
        fill = fill_override
    domain = row["domain_display_name"]
    domain_color = DOMAIN_COLORS[domain]
    density = crop_density_panel(name)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.02),
            0.97,
            0.95,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=fill,
            edgecolor=accent,
            linewidth=1.25,
            alpha=0.96,
        )
    )
    ax.add_patch(Rectangle((0.055, 0.84), 0.025, 0.07, facecolor=domain_color, edgecolor="none"))
    ax.text(0.095, 0.895, title, fontsize=12.0, weight="bold", color=accent, va="top")
    ax.text(0.095, 0.835, subtitle, fontsize=8.5, color="#333333", va="top")
    ax.imshow(density, extent=(0.06, 0.94, 0.32, 0.76), aspect="auto", zorder=2)
    ax.add_patch(Rectangle((0.06, 0.32), 0.88, 0.44, facecolor="none", edgecolor="#ffffff", linewidth=1.0, zorder=3))
    ax.text(0.06, 0.25, short_name(name), fontsize=10.8, weight="bold", color="#111111", ha="left")
    ax.text(0.06, 0.19, domain, fontsize=8.0, color=domain_color, ha="left")
    ax.text(0.94, 0.19, row["typology_id"], fontsize=8.0, color=accent, ha="right", weight="bold")
    ax.text(0.06, 0.12, metric_line(row), fontsize=7.7, color="#333333", ha="left")
    ax.text(0.06, 0.065, note, fontsize=8.1, color=accent, ha="left", weight="bold")


def figure_quadrant_umap_shapes(scores: pd.DataFrame) -> None:
    cards = [
        {
            "title": "Broad + concentrated",
            "subtitle": "large territory, strong high-density foci",
            "name": "Philosophy",
            "quadrant": "Broad + concentrated",
            "note": "wide footprint with dominant regions",
        },
        {
            "title": "Broad + smooth",
            "subtitle": "large territory, weak hub concentration",
            "name": "Artificial Intelligence",
            "quadrant": "Broad + smooth",
            "note": "many islands without one simple core",
        },
        {
            "title": "Compact + concentrated",
            "subtitle": "tight territory, strong local concentration",
            "name": "Applied Microbiology and Biotechnology",
            "quadrant": "Compact + concentrated",
            "note": "compact body with separated dense island",
        },
        {
            "title": "Compact + smooth",
            "subtitle": "tight territory, weak hub concentration",
            "name": "Research and Theory",
            "quadrant": "Compact + smooth",
            "note": "small, even semantic footprint",
        },
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    for ax, card in zip(axes.reshape(-1), cards):
        draw_shape_card(ax, scores=scores, **card)
    fig.suptitle("Four morphology zones, four visible shapes", fontsize=17.5, weight="bold", y=0.992)
    fig.text(
        0.5,
        0.952,
        "The quadrant summary becomes concrete by showing one UMAP density shape for each structural zone.",
        ha="center",
        fontsize=10.0,
        color="#444444",
    )
    fig.text(
        0.5,
        0.018,
        "Note. UMAP panels are illustrative per-subfield projections fitted separately; shapes are visual diagnostics, not common-coordinate comparisons.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.925])
    save(fig, "fig_06y_quadrant_umap_shapes_candidate")


def figure_rebel_umap_shapes(scores: pd.DataFrame) -> None:
    cards = [
        {
            "title": "AI is broad, not hubbed",
            "subtitle": "wide but locally diffuse",
            "name": "Artificial Intelligence",
            "quadrant": "Broad + smooth",
            "note": "broad and recent, but not hub-dominated",
            "accent_override": "#2f5f8f",
            "fill_override": "#eef6fb",
        },
        {
            "title": "Applied Microbiology is compact and hubbed",
            "subtitle": "tight but strongly concentrated",
            "name": "Applied Microbiology and Biotechnology",
            "quadrant": "Compact + concentrated",
            "note": "compact does not mean uniform",
            "accent_override": "#735285",
            "fill_override": "#f3edf6",
        },
        {
            "title": "Philosophy is broad and hubbed",
            "subtitle": "unexpected humanities morphology",
            "name": "Philosophy",
            "quadrant": "Broad + concentrated",
            "note": "a social-science label with a broad shape",
            "accent_override": "#965323",
            "fill_override": "#f8eedf",
        },
        {
            "title": "Human Factors is the novelty outlier",
            "subtitle": "the strongest temporal-novelty case",
            "name": "Human Factors and Ergonomics",
            "quadrant": "Compact + concentrated",
            "note": "temporally unusual, visibly segmented",
            "accent_override": "#b7483c",
            "fill_override": "#fbefec",
        },
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    for ax, card in zip(axes.reshape(-1), cards):
        draw_shape_card(ax, scores=scores, **card)
    fig.suptitle("Four diagnostic cases with visible semantic shapes", fontsize=17.5, weight="bold", y=0.992)
    fig.text(
        0.5,
        0.952,
        "These cases make abstract morphology scores readable as concrete projected density shapes.",
        ha="center",
        fontsize=10.0,
        color="#444444",
    )
    fig.text(
        0.5,
        0.018,
        "Note. UMAP panels are illustrative per-subfield projections fitted separately; shapes are visual diagnostics, not common-coordinate comparisons.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.925])
    save(fig, "fig_06y_rebel_umap_shapes_candidate")


def copy_selected_dynamic() -> None:
    src = ROOT / "outputs" / "15_glanceable_visual_candidates" / "fig_09y_dynamic_movement_cards_candidate.png"
    dst = OUTPUT_DIR / "fig_09y_dynamic_movement_cards_candidate.png"
    if src.exists():
        shutil.copy2(src, dst)
    src_pdf = src.with_suffix(".pdf")
    if src_pdf.exists():
        shutil.copy2(src_pdf, dst.with_suffix(".pdf"))


def build_gallery() -> None:
    stems = [
        "fig_06y_quadrant_umap_shapes_candidate",
        "fig_06y_rebel_umap_shapes_candidate",
        "fig_09y_dynamic_movement_cards_candidate",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2))
    axes = axes.reshape(-1)
    for ax, stem in zip(axes, stems):
        image = mpimg.imread(OUTPUT_DIR / f"{stem}.png")
        ax.imshow(image)
        ax.set_title(stem.replace("_candidate", "").replace("_", " "), fontsize=9.5)
        ax.axis("off")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "selected_insert_visuals_gallery.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_report() -> None:
    text = """# Selected Insert Visual Prototypes

Generated from existing TFM outputs only. No thesis file was modified.

## Chapter 6

1. `fig_06y_quadrant_umap_shapes_candidate`
   - My preferred version.
   - It keeps the quadrant idea you liked, but adds one UMAP density shape per quadrant.
   - Best insertion: after the field-level heatmap, replacing the need to keep these cases only in the appendix.

2. `fig_06y_rebel_umap_shapes_candidate`
   - More narrative and memorable.
   - It uses the rebel cases directly: AI, Applied Microbiology, Philosophy, and Human Factors.
   - Risk: two examples occupy the compact/concentrated zone, so it is less structurally balanced.

## Chapter 9

1. `fig_09y_dynamic_movement_cards_candidate`
   - This is the dynamic typology card figure you liked.
   - Best insertion: immediately after the dynamic typology heatmap.

## Appendix implication

If the Chapter 6 UMAP-shape figure enters the main text, Appendix D should be reduced rather than deleted entirely:

- remove the duplicated main-text examples from the appendix,
- keep only a short methodological note or extra examples not used in Chapter 6,
- preserve the caution that per-subfield UMAPs are illustrative and not a shared coordinate system.
"""
    (OUTPUT_DIR / "selected_insert_visuals_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    scores = load_scores()
    figure_quadrant_umap_shapes(scores)
    figure_rebel_umap_shapes(scores)
    copy_selected_dynamic()
    build_gallery()
    write_report()


if __name__ == "__main__":
    main()
