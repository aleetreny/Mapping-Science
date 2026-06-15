from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import shorten, wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.temporal_common import CENTROID_DRIFT_METRIC, STRUCTURAL_MORPHOLOGY_METRICS


OUTPUT_DIR = ROOT / "outputs" / "11_active_8metric_rebuild"
FIGURE_DIR = ROOT / "memory" / "figures"
TABLE_DIR = ROOT / "memory" / "tables"

METRICS = list(STRUCTURAL_MORPHOLOGY_METRICS)
METRIC_LABELS = {
    "embedding_distance_to_centroid_median": "Centroid\nmedian",
    "embedding_distance_to_centroid_iqr": "Centroid\nIQR",
    "embedding_distance_to_centroid_p90": "Centroid\nP90",
    "embedding_knn_median_distance": "kNN\nmedian",
    "embedding_knn_distance_cv": "kNN\nCV",
    "embedding_knn_indegree_gini": "Hub\nGini",
    "embedding_pca_dim_80": "PCA\nD80",
    "embedding_pca_spectral_entropy": "PCA\nentropy",
}
METRIC_TEXT = {
    "embedding_distance_to_centroid_median": "Centroid median",
    "embedding_distance_to_centroid_iqr": "Centroid IQR",
    "embedding_distance_to_centroid_p90": "Centroid P90",
    "embedding_knn_median_distance": "kNN median",
    "embedding_knn_distance_cv": "kNN CV",
    "embedding_knn_indegree_gini": "Hub Gini",
    "embedding_pca_dim_80": "PCA D80",
    "embedding_pca_spectral_entropy": "PCA entropy",
}
METRIC_TABLE_ROWS = [
    (
        "Centroid median",
        "Global dispersion",
        r"Median document distance to the subfield centroid.",
        r"Typical spread around the semantic center.",
    ),
    (
        "Centroid IQR",
        "Global dispersion",
        r"Interquartile range of document distances to the centroid.",
        r"Inequality of radial spread inside the subfield.",
    ),
    (
        "Centroid P90",
        "Global dispersion",
        r"90th percentile of document distances to the centroid.",
        r"Outer-tail dispersion of the subfield cloud.",
    ),
    (
        "kNN median",
        "Local density",
        r"Median distance from each document to its local nearest-neighbor set.",
        r"Typical local spacing among neighboring papers.",
    ),
    (
        "kNN CV",
        "Local density",
        r"Coefficient of variation of local kNN distances.",
        r"Unevenness of local density.",
    ),
    (
        "Hub Gini",
        "Hubness",
        r"Gini coefficient of kNN indegree counts.",
        r"Concentration of local-neighborhood centrality.",
    ),
    (
        "PCA D80",
        "Spectral structure",
        r"Number of principal components needed to explain 80\% variance.",
        r"Effective dimensionality of the subfield cloud.",
    ),
    (
        "PCA entropy",
        "Spectral structure",
        r"Entropy of the normalized PCA spectrum.",
        r"Evenness of variance across latent directions.",
    ),
]

DOMAIN_COLORS = {
    "Life Sciences": "#2f7f5f",
    "Social Sciences": "#8b6bb5",
    "Physical Sciences": "#ba6a36",
    "Health Sciences": "#4c78a8",
}
DOMAIN_ORDER = ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"]
DYNAMIC_LABELS = {
    1: "Compacting locally uneven trajectories",
    2: "Broadening hub-diffusing trajectories",
    3: "Smoothing sparse-neighborhood trajectories",
    4: "Dimensionalizing hub-concentrating trajectories",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, stem: str, *, pdf: bool = False) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    if pdf:
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def robust_scaled_frame(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    scaled = frame.copy()
    scaler = RobustScaler()
    scaled[metrics] = scaler.fit_transform(frame[metrics].to_numpy(dtype=float))
    return scaled


def ordered_fields(core: pd.DataFrame) -> list[str]:
    fields = (
        core[["field_display_name", "domain_display_name"]]
        .drop_duplicates()
        .assign(domain_order=lambda x: x["domain_display_name"].map({d: i for i, d in enumerate(DOMAIN_ORDER)}))
        .sort_values(["domain_order", "field_display_name"], kind="mergesort")
    )
    return fields["field_display_name"].tolist()


def check_active_core(core: pd.DataFrame) -> None:
    missing = [metric for metric in METRICS if metric not in core.columns]
    if missing:
        raise ValueError("Missing active structural metrics: " + ", ".join(missing))
    embedding_columns = [col for col in core.columns if col.startswith("embedding_")]
    if embedding_columns != METRICS:
        raise ValueError(
            "The active profile matrix must contain exactly the eight structural metrics "
            f"in canonical order; got {embedding_columns}"
        )
    if CENTROID_DRIFT_METRIC in core.columns:
        raise ValueError("Centroid drift must not appear in the active structural profile matrix.")


def load_inputs() -> dict[str, pd.DataFrame]:
    core = pd.read_parquet(OUTPUT_DIR.parent / "04_reduced_metric_core" / "reduced_interpretable_core_metrics.parquet")
    check_active_core(core)
    drift = pd.read_parquet(OUTPUT_DIR.parent / "04_reduced_metric_core" / "centroid_drift_early_late.parquet")
    temporal = pd.read_parquet(ROOT / "data" / "processed" / "temporal" / "subfield_window_embedding_metrics.parquet")
    if "metric_status" in temporal.columns:
        temporal = temporal.loc[temporal["metric_status"].isin({"completed", "completed_with_warnings"})].copy()
    return {"core": core, "drift": drift, "temporal": temporal}


def write_metric_core_table() -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Structural embedding-space morphology metrics]{Structural embedding-space morphology metrics}",
        r"\label{tab:metric_core}",
        r"\scriptsize",
        r"\begin{tabular}{@{}p{0.18\textwidth}p{0.18\textwidth}p{0.31\textwidth}p{0.25\textwidth}@{}}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Family} & \textbf{Definition} & \textbf{Interpretation} \\",
        r"\hline",
    ]
    for metric, family, definition, interpretation in METRIC_TABLE_ROWS:
        lines.append(
            " & ".join(
                [
                    latex_escape(metric),
                    latex_escape(family),
                    definition,
                    interpretation,
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\par\vspace{0.12cm}",
            (
                r"\footnotesize\textit{Note.} These eight metrics define the active "
                r"structural morphology profile. Net semantic displacement is measured "
                r"separately with early--late centroid drift and is not included in the "
                r"profile matrix."
            ),
            r"\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "tab_05_metric_core.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_pipeline_figure() -> None:
    from matplotlib.patches import Rectangle, Polygon
    
    # Ensure correct font and configuration
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    
    # Define data for the 8 steps
    steps = [
        {
            "header": "OpenAlex\nworks",
            "subheader": "current global\nWorks index",
            "value": "314.9M",
            "pct": "100%",
            "height": 1.15,
            "group": "live",
        },
        {
            "header": "2000-2024",
            "subheader": "publication-date\nwindow",
            "value": "205.1M",
            "pct": "65%",
            "height": 0.98,
            "group": "live",
        },
        {
            "header": "article or\npreprint",
            "subheader": "document-type\nfilter",
            "value": "150.8M",
            "pct": "74%",
            "height": 0.88,
            "group": "live",
        },
        {
            "header": "English\nrecords",
            "subheader": "language\nfilter",
            "value": "106.6M",
            "pct": "71%",
            "height": 0.78,
            "group": "live",
        },
        {
            "header": "abstract,\nnot retracted",
            "subheader": "broad API\ntext pool",
            "value": "71.8M",
            "pct": "67%",
            "height": 0.68,
            "group": "live",
        },
        {
            "header": "planned\nsample",
            "subheader": "252 subfields;\n<=400/year",
            "value": "2.43M",
            "pct": "3.39%",
            "height": 0.54,
            "group": "frozen",
        },
        {
            "header": "validated\ncorpus",
            "subheader": "local text\nand metadata",
            "value": "2.38M",
            "pct": "3.31%",
            "height": 0.50,
            "group": "frozen",
        },
        {
            "header": "analysis\nsubset",
            "subheader": "row-aligned\nSPECTER2",
            "value": "2.34M",
            "pct": "3.27%",
            "height": 0.46,
            "group": "frozen",
        },
    ]

    # Set positions
    spacing = 1.65
    x_positions = np.arange(len(steps)) * spacing
    W = 0.8  # width of each rectangle
    half_W = W / 2.0

    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.axis("off")
    ax.set_facecolor("white")

    # Set limits
    ax.set_xlim(x_positions[0] - 0.9, x_positions[-1] + 0.9)
    ax.set_ylim(-1.20, 2.05)

    # Draw connecting funnels first
    for i in range(len(steps) - 1):
        x_curr = x_positions[i]
        x_next = x_positions[i + 1]
        h_curr = steps[i]["height"]
        h_next = steps[i + 1]["height"]
        
        # Coordinates of the polygon corners
        poly_pts = [
            [x_curr + half_W, h_curr / 2.0],
            [x_next - half_W, h_next / 2.0],
            [x_next - half_W, -h_next / 2.0],
            [x_curr + half_W, -h_curr / 2.0]
        ]
        
        # Determine funnel line / shade style
        is_bridge = (steps[i]["group"] != steps[i+1]["group"])
        
        # Draw polygon fill
        poly_color = "#E5E7EB" if not is_bridge else "#F3F4F6"
        poly = Polygon(poly_pts, facecolor=poly_color, edgecolor="none", alpha=0.5, zorder=1)
        ax.add_patch(poly)
        
        # Draw top/bottom boundary lines for funnel
        line_style = ":" if is_bridge else "--"
        line_color = "#9CA3AF" if not is_bridge else "#D1D5DB"
        ax.plot([x_curr + half_W, x_next - half_W], [h_curr / 2.0, h_next / 2.0], 
                linestyle=line_style, color=line_color, linewidth=0.8, zorder=2)
        ax.plot([x_curr + half_W, x_next - half_W], [-h_curr / 2.0, -h_next / 2.0], 
                linestyle=line_style, color=line_color, linewidth=0.8, zorder=2)

    # Draw steps (rectangles and texts)
    for i, step in enumerate(steps):
        x = x_positions[i]
        h = step["height"]
        group = step["group"]
        
        # Styling based on group
        if group == "live":
            rect = Rectangle((x - half_W, -h / 2.0), W, h, 
                             facecolor="#FFFFFF", edgecolor="#374151", linewidth=1.2, zorder=3)
            num_color = "#1F2937"
        else:
            rect = Rectangle((x - half_W, -h / 2.0), W, h, 
                             facecolor="#F3F4F6", edgecolor="#111827", linewidth=1.2, zorder=3)
            num_color = "#111827"
            
        ax.add_patch(rect)
        
        # Text inside the box (Value) - Reduced to 10.0
        ax.text(x, 0, step["value"], ha="center", va="center", 
                fontsize=10.0, fontweight="bold", color=num_color, zorder=4)
        
        # Header text above the box - Increased to 11.5
        ax.text(x, 0.88, step["header"], ha="center", va="bottom", 
                fontsize=11.5, fontweight="bold", color="#1F2937", linespacing=1.1, zorder=4)
        
        # Subheader text between header and box - Increased to 9.5
        ax.text(x, 0.82, step["subheader"], ha="center", va="top", 
                fontsize=9.5, color="#4B5563", style="italic", linespacing=1.1, zorder=4)
        
        # Percentage text below the box - Increased to 11.5
        ax.text(x, -0.78, step["pct"], ha="center", va="top", 
                fontsize=11.5, fontweight="bold", color="#374151", zorder=4)

    # Draw group brackets at the top
    y_bracket = 1.52
    tick_len = 0.05

    # Group 1: Live (Columns 0 to 4)
    x_start_g1 = x_positions[0] - half_W
    x_end_g1 = x_positions[4] + half_W
    x_center_g1 = (x_start_g1 + x_end_g1) / 2.0

    ax.plot([x_start_g1, x_end_g1], [y_bracket, y_bracket], color="#4B5563", linewidth=1.2, zorder=5)
    ax.plot([x_start_g1, x_start_g1], [y_bracket, y_bracket - tick_len], color="#4B5563", linewidth=1.2, zorder=5)
    ax.plot([x_end_g1, x_end_g1], [y_bracket, y_bracket - tick_len], color="#4B5563", linewidth=1.2, zorder=5)
    ax.text(x_center_g1, y_bracket + 0.08, "OpenAlex API live counts, queried 25 May 2026", 
            ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#111827", zorder=5)

    # Group 2: Frozen (Columns 5 to 7)
    x_start_g2 = x_positions[5] - half_W
    x_end_g2 = x_positions[7] + half_W
    x_center_g2 = (x_start_g2 + x_end_g2) / 2.0

    ax.plot([x_start_g2, x_end_g2], [y_bracket, y_bracket], color="#4B5563", linewidth=1.2, zorder=5)
    ax.plot([x_start_g2, x_start_g2], [y_bracket, y_bracket - tick_len], color="#4B5563", linewidth=1.2, zorder=5)
    ax.plot([x_end_g2, x_end_g2], [y_bracket, y_bracket - tick_len], color="#4B5563", linewidth=1.2, zorder=5)
    ax.text(x_center_g2, y_bracket + 0.08, "Frozen TFM pipeline snapshot", 
            ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#111827", zorder=5)

    savefig(fig, "fig_03_openalex_corpus_pipeline", pdf=True)


def _make_eda_plot(core: pd.DataFrame, style: str) -> plt.Figure:
    metric_titles = {
        "embedding_distance_to_centroid_median": "Centroid-Distance Median",
        "embedding_distance_to_centroid_iqr": "Centroid-Distance IQR",
        "embedding_distance_to_centroid_p90": "Centroid-Distance P90",
        "embedding_knn_median_distance": "Median kNN Distance",
        "embedding_knn_distance_cv": "kNN Distance CV",
        "embedding_knn_indegree_gini": "kNN In-Degree Gini",
        "embedding_pca_dim_80": "PCA D80",
        "embedding_pca_spectral_entropy": "PCA Spectral Entropy",
    }
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 7.0), constrained_layout=True)
    for i, (ax, metric) in enumerate(zip(axes.ravel(), METRICS)):
        values = pd.to_numeric(core[metric], errors="coerce").dropna()
        counts, bins, patches = ax.hist(values, bins=min(28, max(8, values.nunique())), color="#4c78a8", edgecolor="white", linewidth=0.4)
        
        # Median vertical line removed as requested ("quitame la barra de la media")
        
        # Increase Y-limit to leave space above the highest bar
        max_y = float(np.max(counts)) if len(counts) > 0 else 10.0
        ax.set_ylim(0, max_y * 1.35)
        
        # Render titles inside the subplot area, regular font weight (fontweight="normal")
        if style == "centered_box":
            ax.text(0.5, 0.88, metric_titles[metric], transform=ax.transAxes,
                    ha="center", va="center", fontsize=11.0, fontweight="normal",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="#ffffff", edgecolor="#e5e7eb", alpha=0.9, linewidth=0.6))
        elif style == "left_box":
            ax.text(0.05, 0.88, metric_titles[metric], transform=ax.transAxes,
                    ha="left", va="center", fontsize=11.0, fontweight="normal",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="#ffffff", edgecolor="#e5e7eb", alpha=0.9, linewidth=0.6))
        elif style == "centered":
            ax.text(0.5, 0.88, metric_titles[metric], transform=ax.transAxes,
                    ha="center", va="center", fontsize=11.0, fontweight="normal")
        elif style == "left":
            ax.text(0.05, 0.88, metric_titles[metric], transform=ax.transAxes,
                    ha="left", va="center", fontsize=11.0, fontweight="normal")
        
        # Only label Y axis on the leftmost subplots
        if i % 4 == 0:
            ax.set_ylabel("Subfields", fontsize=12.0, labelpad=4)
            
        # Only label X axis on the bottom row subplots
        if i >= 4:
            ax.set_xlabel("Raw value", fontsize=12.0, labelpad=4)
            
        ax.tick_params(labelsize=11.0)
    return fig


def plot_eda(core: pd.DataFrame) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    for metric in METRICS:
        values = pd.to_numeric(core[metric], errors="coerce").dropna()
        diagnostics[f"{metric}_median"] = float(values.median())
        diagnostics[f"{metric}_skew"] = float(values.skew())
        
    for style in ["centered", "left", "centered_box", "left_box"]:
        fig = _make_eda_plot(core, style)
        savefig(fig, f"fig_06_structural_metric_raw_distributions_{style}")
        plt.close(fig)
        
    # Set default main figure as centered
    fig = _make_eda_plot(core, "centered")
    savefig(fig, "fig_06_structural_metric_raw_distributions")
    plt.close(fig)

    corr = core[METRICS].corr(method="spearman")
    corr.to_csv(OUTPUT_DIR / "structural_metric_raw_spearman_correlation.csv")
    fig, ax = plt.subplots(figsize=(8.8, 6.4), constrained_layout=True)
    image = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_yticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11.0)
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11.0)
    for i in range(len(METRICS)):
        for j in range(len(METRICS)):
            value = corr.iloc[i, j]
            color = "white" if abs(value) > 0.62 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9.5, color=color)
    for boundary in [2.5, 5.5]:
        ax.axhline(boundary, color="#222222", linewidth=0.8)
        ax.axvline(boundary, color="#222222", linewidth=0.8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Spearman rho", fontsize=11.0, labelpad=12.0)
    cbar.ax.tick_params(labelsize=10.0)
    savefig(fig, "fig_06_structural_metric_spearman_correlation")
    return diagnostics


def static_profiles(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaled = robust_scaled_frame(core, METRICS)
    subfield = scaled.copy()
    field = (
        scaled.groupby(["field_id", "field_display_name", "domain_display_name"], sort=True)[METRICS]
        .mean()
        .reset_index()
    )
    domain = scaled.groupby(["domain_id", "domain_display_name"], sort=True)[METRICS].mean().reset_index()
    for frame, path in [
        (subfield, OUTPUT_DIR / "subfield_structural_profiles_scaled.csv"),
        (field, OUTPUT_DIR / "field_structural_profiles_scaled.csv"),
        (domain, OUTPUT_DIR / "domain_structural_profiles_scaled.csv"),
    ]:
        frame.to_csv(path, index=False)
    return subfield, field, domain


def heatmap(
    values: pd.DataFrame,
    path_stem: str,
    *,
    row_labels: list[str],
    column_labels: list[str],
    figsize: tuple[float, float],
    colorbar_label: str,
    pdf: bool = False,
    vlim: float | None = None,
    annotate_threshold: float | None = None,
) -> None:
    matrix = values.to_numpy(dtype=float)
    if vlim is None:
        vlim = float(np.nanmax(np.abs(matrix)))
        vlim = max(vlim, 0.5)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    
    # Tick sizes and padding matching correlation matrix visual polish
    names_fontsize = 10.5
    cbar_label_fontsize = 11.0
    cbar_ticks_fontsize = 10.0
    cbar_labelpad = 12.0
    
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=names_fontsize)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, fontsize=names_fontsize, rotation=0)
    ax.tick_params(length=0)
    
    # Cell values overlaid for significant cells
    if annotate_threshold is not None:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if abs(val) >= annotate_threshold:
                    # White text for highly saturated cells, dark gray for lighter ones
                    color = "white" if (abs(val) / vlim) > 0.55 else "#222222"
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=9.0, color=color, fontweight="semibold")
                    
    cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.015)
    cbar.set_label(colorbar_label, fontsize=cbar_label_fontsize, labelpad=cbar_labelpad)
    cbar.ax.tick_params(labelsize=cbar_ticks_fontsize)
    savefig(fig, path_stem, pdf=pdf)


def plot_static_profiles(core: pd.DataFrame, subfield: pd.DataFrame, field: pd.DataFrame, domain: pd.DataFrame) -> dict[str, float]:
    domain_ordered = domain.assign(
        domain_order=domain["domain_display_name"].map({d: i for i, d in enumerate(DOMAIN_ORDER)})
    ).sort_values("domain_order", kind="mergesort")
    heatmap(
        domain_ordered[METRICS],
        "fig_06_domain_metric_profile_heatmap",
        row_labels=domain_ordered["domain_display_name"].tolist(),
        column_labels=[METRIC_LABELS[m] for m in METRICS],
        figsize=(11.5, 3.8),
        colorbar_label="Robust-scaled profile",
        pdf=True,
        annotate_threshold=0.25,
    )

    field_ordered = field.assign(
        domain_order=field["domain_display_name"].map({d: i for i, d in enumerate(DOMAIN_ORDER)})
    ).sort_values(["domain_order", "field_display_name"], kind="mergesort")
    heatmap(
        field_ordered[METRICS],
        "fig_06_field_metric_profile_heatmap",
        row_labels=[shorten(name, width=42, placeholder="...") for name in field_ordered["field_display_name"]],
        column_labels=[METRIC_LABELS[m] for m in METRICS],
        figsize=(11.5, 9.8),
        colorbar_label="Robust-scaled profile",
        pdf=True,
        annotate_threshold=0.50,
    )

    metric_titles = {
        "embedding_distance_to_centroid_median": "Centroid-Distance Median",
        "embedding_distance_to_centroid_iqr": "Centroid-Distance IQR",
        "embedding_distance_to_centroid_p90": "Centroid-Distance P90",
        "embedding_knn_median_distance": "Median kNN Distance",
        "embedding_knn_distance_cv": "kNN Distance CV",
        "embedding_knn_indegree_gini": "kNN In-Degree Gini",
        "embedding_pca_dim_80": "PCA D80",
        "embedding_pca_spectral_entropy": "PCA Spectral Entropy",
    }
    plot_frame = subfield.melt(
        id_vars=["subfield_id", "subfield_display_name", "domain_display_name"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 8.5), sharey=True, constrained_layout=True)
    for i, (ax, metric) in enumerate(zip(axes.ravel(), METRICS)):
        data = [plot_frame.loc[(plot_frame["metric"] == metric) & (plot_frame["domain_display_name"] == domain), "value"].dropna() for domain in DOMAIN_ORDER]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            tick_labels=[d.replace(" ", "\n") for d in DOMAIN_ORDER],
            showfliers=False,
        )
        for patch, domain in zip(bp["boxes"], DOMAIN_ORDER):
            patch.set_facecolor(DOMAIN_COLORS[domain])
            patch.set_alpha(0.78)
        ax.axhline(0, color="#555555", linewidth=0.6)
        ax.set_title(metric_titles[metric], fontsize=11.5, fontweight="normal")
        ax.tick_params(axis="x", labelsize=10.5)
        ax.tick_params(axis="y", labelsize=11.0)
        ax.grid(True, axis="y", alpha=0.18)
        if i % 4 == 0:
            ax.set_ylabel("Robust-scaled subfield value", fontsize=12.0)
    savefig(fig, "fig_06_subfield_metric_distributions_by_domain", pdf=True)

    matrix = subfield[METRICS].to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)
    pca_frame = subfield[
        ["subfield_id", "subfield_display_name", "field_display_name", "domain_display_name"]
    ].copy()
    pca_frame["pca_1"] = coords[:, 0]
    pca_frame["pca_2"] = coords[:, 1]
    pca_frame["pca_radius"] = np.linalg.norm(coords, axis=1)
    pca_frame["pca_1_explained_variance"] = float(pca.explained_variance_ratio_[0])
    pca_frame["pca_2_explained_variance"] = float(pca.explained_variance_ratio_[1])
    pca_frame.to_csv(OUTPUT_DIR / "subfield_profile_pca_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.3, 6.2), constrained_layout=True)
    for domain in DOMAIN_ORDER:
        subset = pca_frame.loc[pca_frame["domain_display_name"] == domain]
        ax.scatter(
            subset["pca_1"],
            subset["pca_2"],
            s=28,
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.83,
            label=domain,
        )
    for row in pca_frame.sort_values("pca_radius", ascending=False).head(8).itertuples():
        ax.text(row.pca_1, row.pca_2, shorten(str(row.subfield_display_name), width=24, placeholder="..."), fontsize=6.5)
    ax.axhline(0, color="#777777", linewidth=0.6)
    ax.axvline(0, color="#777777", linewidth=0.6)
    ax.set_xlabel(f"Profile PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"Profile PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.grid(True, alpha=0.16)
    ax.legend(fontsize=7.8, frameon=False, loc="best")
    savefig(fig, "fig_06_metric_profile_pca_map", pdf=True)
    return {
        "pca_1_share": float(pca.explained_variance_ratio_[0]),
        "pca_2_share": float(pca.explained_variance_ratio_[1]),
    }


def write_extremes_table(subfield: pd.DataFrame) -> None:
    rows_for_csv = []
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Selected static metric extremes]{Selected static structural-metric extremes at subfield level}",
        r"\label{tab:extreme_subfields}",
        r"\scriptsize",
        r"\begin{tabular}{@{}p{0.22\textwidth}p{0.34\textwidth}p{0.34\textwidth}@{}}",
        r"\hline",
        r"\textbf{Metric} & \textbf{High end} & \textbf{Low end} \\",
        r"\hline",
    ]
    for metric in METRICS:
        high = subfield.sort_values([metric, "subfield_display_name"], ascending=[False, True]).head(2)
        low = subfield.sort_values([metric, "subfield_display_name"], ascending=[True, True]).head(2)
        high_text = "; ".join(
            f"{latex_escape(row.subfield_display_name)} ({getattr(row, metric):+.2f})"
            for row in high.itertuples()
        )
        low_text = "; ".join(
            f"{latex_escape(row.subfield_display_name)} ({getattr(row, metric):+.2f})"
            for row in low.itertuples()
        )
        lines.append(f"{latex_escape(METRIC_TEXT[metric])} & {high_text} & {low_text} \\\\")
        for direction, subset in [("high", high), ("low", low)]:
            for rank, row in enumerate(subset.itertuples(), start=1):
                rows_for_csv.append(
                    {
                        "metric": metric,
                        "direction": direction,
                        "rank": rank,
                        "subfield_id": row.subfield_id,
                        "subfield_display_name": row.subfield_display_name,
                        "field_display_name": row.field_display_name,
                        "domain_display_name": row.domain_display_name,
                        "robust_scaled_value": float(getattr(row, metric)),
                    }
                )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\par\vspace{0.12cm}",
            (
                r"\footnotesize\textit{Note.} Values in parentheses are robust-scaled "
                r"subfield profile values under the active eight-metric structural "
                r"specification."
            ),
            r"\end{table}",
            "",
        ]
    )
    pd.DataFrame(rows_for_csv).to_csv(OUTPUT_DIR / "static_metric_extremes.csv", index=False)
    (TABLE_DIR / "tab_06_extreme_subfields.tex").write_text("\n".join(lines), encoding="utf-8")


def write_centroid_drift_table(drift: pd.DataFrame) -> dict[str, float]:
    frame = drift.dropna(subset=[CENTROID_DRIFT_METRIC]).copy()
    top = frame.sort_values([CENTROID_DRIFT_METRIC, "subfield_display_name"], ascending=[False, True]).head(10)
    bottom = frame.sort_values([CENTROID_DRIFT_METRIC, "subfield_display_name"], ascending=[True, True]).head(10)
    pd.concat(
        [top.assign(drift_group="highest"), bottom.assign(drift_group="lowest")],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "centroid_drift_extreme_subfields.csv", index=False)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Net semantic displacement extremes]{Highest and lowest early--late centroid drift by subfield}",
        r"\label{tab:centroid_drift_extremes}",
        r"\scriptsize",
        r"\begin{tabular}{@{}p{0.12\textwidth}p{0.34\textwidth}p{0.22\textwidth}p{0.17\textwidth}r@{}}",
        r"\hline",
        r"\textbf{Group} & \textbf{Subfield} & \textbf{Field} & \textbf{Domain} & \textbf{Drift} \\",
        r"\hline",
    ]
    for group_label, subset in [("Highest", top), ("Lowest", bottom)]:
        for row in subset.itertuples():
            lines.append(
                " & ".join(
                    [
                        group_label,
                        latex_escape(row.subfield_display_name),
                        latex_escape(row.field_display_name),
                        latex_escape(row.domain_display_name),
                        f"{getattr(row, CENTROID_DRIFT_METRIC):.3f}",
                    ]
                )
                + r" \\"
            )
        if group_label == "Highest":
            lines.append(r"\hline")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\par\vspace{0.12cm}",
            (
                r"\footnotesize\textit{Note.} Centroid drift measures net semantic "
                r"displacement between the early and late parts of a subfield's "
                r"publication history. Low values indicate the most stable centroids "
                r"in this corpus. Drift is reported separately and is not part of the "
                r"structural morphology profile."
            ),
            r"\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "tab_07_centroid_drift_extremes.tex").write_text("\n".join(lines), encoding="utf-8")
    return {
        "centroid_drift_max": float(frame[CENTROID_DRIFT_METRIC].max()),
        "centroid_drift_median": float(frame[CENTROID_DRIFT_METRIC].median()),
    }


def temporal_scaled(temporal: pd.DataFrame) -> pd.DataFrame:
    required = {"window_label", "window_start", "window_end", *METRICS}
    missing = required - set(temporal.columns)
    if missing:
        raise ValueError("Missing temporal columns: " + ", ".join(sorted(missing)))
    scaled = robust_scaled_frame(temporal, METRICS)
    scaled.to_csv(OUTPUT_DIR / "subfield_window_structural_profiles_scaled.csv", index=False)
    return scaled


def plot_temporal_figures(temporal: pd.DataFrame) -> dict[str, float]:
    scaled = temporal_scaled(temporal)
    domain_time = (
        scaled.groupby(["domain_display_name", "window_label", "window_start"], sort=True)[METRICS]
        .mean()
        .reset_index()
        .sort_values(["window_start", "domain_display_name"], kind="mergesort")
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.6), sharex=True, constrained_layout=True)
    for ax, metric in zip(axes.ravel(), METRICS):
        for domain in DOMAIN_ORDER:
            subset = domain_time.loc[domain_time["domain_display_name"] == domain]
            ax.plot(
                subset["window_label"],
                subset[metric],
                marker="o",
                linewidth=1.5,
                markersize=3,
                color=DOMAIN_COLORS[domain],
                label=domain,
            )
        ax.axhline(0, color="#777777", linewidth=0.6)
        ax.text(0.02, 0.93, METRIC_TEXT[metric], transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
        ax.tick_params(axis="x", labelrotation=35, labelsize=7.2)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(True, axis="y", alpha=0.17)
    axes[0, 0].set_ylabel("Mean standardized value")
    axes[1, 0].set_ylabel("Mean standardized value")
    axes[0, 3].legend(fontsize=7.5, frameon=False, loc="best")
    savefig(fig, "fig_07_domain_temporal_trajectories", pdf=True)

    first_label = scaled.sort_values("window_start")["window_label"].iloc[0]
    last_label = scaled.sort_values("window_start")["window_label"].iloc[-1]
    field_time = (
        scaled.groupby(["field_display_name", "domain_display_name", "window_label", "window_start"], sort=True)[METRICS]
        .mean()
        .reset_index()
    )
    first = field_time.loc[field_time["window_label"] == first_label].set_index("field_display_name")
    last = field_time.loc[field_time["window_label"] == last_label].set_index("field_display_name")
    delta = last[METRICS] - first[METRICS]
    metadata = field_time[["field_display_name", "domain_display_name"]].drop_duplicates().set_index("field_display_name")
    field_order = ordered_fields(scaled)
    delta = delta.reindex(field_order)
    heatmap(
        delta,
        "fig_07_field_temporal_delta_heatmap",
        row_labels=[shorten(name, width=42, placeholder="...") for name in delta.index],
        column_labels=[METRIC_LABELS[m] for m in METRICS],
        figsize=(10.8, 8.6),
        colorbar_label=f"{last_label} minus {first_label}",
        pdf=True,
    )

    sub_first = scaled.loc[scaled["window_label"] == first_label].set_index("subfield_id")
    sub_last = scaled.loc[scaled["window_label"] == last_label].set_index("subfield_id")
    common_ids = sub_first.index.intersection(sub_last.index)
    changes = sub_last.loc[common_ids, METRICS] - sub_first.loc[common_ids, METRICS]
    change_norm = np.linalg.norm(changes.to_numpy(dtype=float), axis=1)
    dynamic = sub_last.loc[common_ids, ["subfield_display_name", "field_display_name", "domain_display_name"]].copy()
    dynamic["overall_structural_change_l2"] = change_norm
    dynamic = dynamic.sort_values(["overall_structural_change_l2", "subfield_display_name"], ascending=[False, True]).head(15)
    dynamic.to_csv(OUTPUT_DIR / "most_dynamic_subfields.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.0, 5.7), constrained_layout=True)
    plot = dynamic.iloc[::-1]
    colors = [DOMAIN_COLORS.get(domain, "#888888") for domain in plot["domain_display_name"]]
    ax.barh([shorten(name, width=42, placeholder="...") for name in plot["subfield_display_name"]], plot["overall_structural_change_l2"], color=colors)
    ax.set_xlabel("First--last structural profile displacement")
    ax.tick_params(axis="y", labelsize=7.7)
    ax.grid(True, axis="x", alpha=0.18)
    savefig(fig, "fig_07_most_dynamic_subfields", pdf=True)

    profile = pd.read_csv(
        OUTPUT_DIR.parent / "09_morphological_typologies" / "temporal_trajectory_clustering" / "trajectory_cluster_profile_summary.csv"
    ).sort_values("trajectory_cluster_id", kind="mergesort")
    value_columns = [f"slope_{metric}" for metric in METRICS]
    values = profile[value_columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.3, 4.6), constrained_layout=True)
    limit = max(1.2, float(np.nanmax(np.abs(values))))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            color = "white" if abs(value) > 0.72 * limit else "#222222"
            ax.text(col_idx, row_idx, f"{value:+.2f}", ha="center", va="center", fontsize=8, color=color)
    row_labels = [
        f"D{int(row.trajectory_cluster_id)} {DYNAMIC_LABELS[int(row.trajectory_cluster_id)]}\n(n={int(row.n_subfields)})"
        for row in profile.itertuples()
    ]
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.8)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=8.8)
    for boundary in [2.5, 5.5]:
        ax.axvline(boundary, color="#2d2d2d", linewidth=0.8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.015)
    cbar.set_label("Mean standardized slope", fontsize=9)
    savefig(fig, "fig_09_dynamic_typology_profile_heatmap", pdf=True)

    return {
        "temporal_first_window": first_label,
        "temporal_last_window": last_label,
        "most_dynamic_subfield_change": float(dynamic["overall_structural_change_l2"].max()),
    }


def field_domain_map(core: pd.DataFrame) -> dict[str, str]:
    return dict(
        core[["field_display_name", "domain_display_name"]]
        .drop_duplicates()
        .set_index("field_display_name")["domain_display_name"]
    )


def plot_similarity_figures(core: pd.DataFrame) -> dict[str, float]:
    matrix_path = OUTPUT_DIR.parent / "07_morphological_similarity" / "matrices" / "field_euclidean_static_morphological_distance_matrix.csv"
    matrix = pd.read_csv(matrix_path, index_col=0)
    name_to_domain = field_domain_map(core)
    field_order = ordered_fields(core)
    matrix = matrix.reindex(index=field_order, columns=field_order)

    fig, ax = plt.subplots(figsize=(9.5, 8.5), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis_r", aspect="equal")
    ax.set_xticks(np.arange(len(field_order)))
    ax.set_yticks(np.arange(len(field_order)))
    ax.set_xticklabels([shorten(name, width=19, placeholder="...") for name in field_order], rotation=90, fontsize=6.3)
    ax.set_yticklabels([shorten(name, width=31, placeholder="...") for name in field_order], fontsize=6.5)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Morphological distance", fontsize=8.5)
    savefig(fig, "fig_08_field_static_distance_heatmap", pdf=True)

    nearest_rows = []
    for field in matrix.index:
        distances = matrix.loc[field].drop(index=field).sort_values()
        neighbor = distances.index[0]
        nearest_rows.append(
            {
                "field": field,
                "domain": name_to_domain[field],
                "nearest_field": neighbor,
                "nearest_domain": name_to_domain[neighbor],
                "nearest_distance": float(distances.iloc[0]),
                "outside_domain": bool(name_to_domain[field] != name_to_domain[neighbor]),
            }
        )
    nearest = pd.DataFrame(nearest_rows).sort_values("field", kind="mergesort")
    nearest.to_csv(OUTPUT_DIR / "field_nearest_morphological_neighbors.csv", index=False)

    pair_rows = []
    for idx, left in enumerate(matrix.index):
        for right in matrix.index[idx + 1 :]:
            pair_rows.append(
                {
                    "field_a": left,
                    "field_b": right,
                    "domain_a": name_to_domain[left],
                    "domain_b": name_to_domain[right],
                    "same_domain": name_to_domain[left] == name_to_domain[right],
                    "distance": float(matrix.loc[left, right]),
                    "domain_pair": " -- ".join(sorted([name_to_domain[left], name_to_domain[right]])),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    domain_pair = pairs.groupby("domain_pair", sort=True)["distance"].median().reset_index(name="median_distance")
    domain_pair.to_csv(OUTPUT_DIR / "field_domain_pair_median_distances.csv", index=False)

    domain_matrix = pd.DataFrame(np.nan, index=DOMAIN_ORDER, columns=DOMAIN_ORDER, dtype=float)
    for a in DOMAIN_ORDER:
        for b in DOMAIN_ORDER:
            mask = pairs["domain_pair"] == " -- ".join(sorted([a, b]))
            if mask.any():
                domain_matrix.loc[a, b] = float(pairs.loc[mask, "distance"].median())

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), gridspec_kw={"width_ratios": [1.25, 0.85]}, constrained_layout=True)
    ax = axes[0]
    image = ax.imshow(domain_matrix.to_numpy(dtype=float), cmap="viridis_r", aspect="equal")
    ax.set_xticks(np.arange(len(DOMAIN_ORDER)))
    ax.set_yticks(np.arange(len(DOMAIN_ORDER)))
    ax.set_xticklabels([d.replace(" ", "\n") for d in DOMAIN_ORDER], fontsize=8)
    ax.set_yticklabels(DOMAIN_ORDER, fontsize=8)
    for i in range(len(DOMAIN_ORDER)):
        for j in range(len(DOMAIN_ORDER)):
            value = domain_matrix.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 1.85 else "#222222")
    ax.text(-0.08, 1.04, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.015)
    cbar.set_label("Median distance", fontsize=8.5)
    ax = axes[1]
    counts = pd.Series(
        {
            "Within domain": int((~nearest["outside_domain"]).sum()),
            "Cross domain": int(nearest["outside_domain"].sum()),
        }
    )
    ax.bar(counts.index, counts.values, color=["#777777", "#4c78a8"], width=0.62)
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 0.4, str(value), ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(counts.values) + 4)
    ax.set_ylabel("Fields")
    ax.text(-0.08, 1.04, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.18)
    savefig(fig, "fig_08_domain_pair_distance_structure", pdf=True)

    means = []
    matrix_dir = OUTPUT_DIR.parent / "07_morphological_similarity" / "matrices"
    for file in sorted(matrix_dir.glob("field_euclidean_20*-20*_morphological_distance_matrix.csv")):
        window = file.name.split("_")[2]
        mat = pd.read_csv(file, index_col=0)
        rows = []
        for idx, left in enumerate(mat.index):
            for right in mat.index[idx + 1 :]:
                rows.append(
                    {
                        "same_domain": name_to_domain[left] == name_to_domain[right],
                        "distance": float(mat.loc[left, right]),
                    }
                )
        tmp = pd.DataFrame(rows)
        means.append(
            {
                "window": window,
                "mean_all": float(tmp["distance"].mean()),
                "mean_within": float(tmp.loc[tmp["same_domain"], "distance"].mean()),
                "mean_cross": float(tmp.loc[~tmp["same_domain"], "distance"].mean()),
            }
        )
    temporal_summary = pd.DataFrame(means)
    temporal_summary.to_csv(OUTPUT_DIR / "field_temporal_distance_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.3, 4.3), constrained_layout=True)
    for column, label, color in [
        ("mean_all", "All field pairs", "#222222"),
        ("mean_within", "Within domain", "#777777"),
        ("mean_cross", "Cross domain", "#4c78a8"),
    ]:
        ax.plot(temporal_summary["window"], temporal_summary[column], marker="o", linewidth=1.8, color=color, label=label)
    ax.set_ylabel("Mean morphological distance")
    ax.set_xlabel("Window")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(fontsize=8, frameon=False, loc="best")
    savefig(fig, "fig_08_temporal_distance_trajectories")

    conv = pd.read_csv(OUTPUT_DIR.parent / "07_morphological_similarity" / "top_pairs" / "field_euclidean_top_converging_pairs.csv").head(5)
    div = pd.read_csv(OUTPUT_DIR.parent / "07_morphological_similarity" / "top_pairs" / "field_euclidean_top_diverging_pairs.csv").head(5)
    combo = pd.concat([conv.assign(direction="Converging"), div.assign(direction="Diverging")], ignore_index=True)
    combo.to_csv(OUTPUT_DIR / "field_top_convergence_divergence_pairs.csv", index=False)
    labels = [
        shorten(f"{row.entity_a_name} -- {row.entity_b_name}", width=55, placeholder="...")
        for row in combo.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(9.7, 5.8), constrained_layout=True)
    y = np.arange(len(combo))[::-1]
    colors = combo["direction"].map({"Converging": "#2f7f5f", "Diverging": "#b65b3a"}).to_numpy()
    ax.hlines(y, combo["initial_distance"], combo["final_distance"], color=colors, linewidth=2.0, alpha=0.85)
    ax.scatter(combo["initial_distance"], y, color="#f1f1f1", edgecolor="#333333", s=34, label="2000--2004", zorder=3)
    ax.scatter(combo["final_distance"], y, color=colors, edgecolor="#333333", s=42, label="2020--2024", zorder=3)
    for y_pos, delta in zip(y, combo["delta_distance"]):
        ax.text(max(combo["initial_distance"].max(), combo["final_distance"].max()) + 0.06, y_pos, f"{delta:+.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlabel("Morphological distance")
    ax.grid(True, axis="x", alpha=0.16)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    savefig(fig, "fig_08_field_convergence_divergence_pairs")

    changes = pd.read_parquet(ROOT / "data" / "processed" / "temporal" / "morphological_pair_distance_changes.parquet")
    field_changes = changes.loc[(changes["level"] == "field") & (changes["distance_metric"] == "euclidean")]
    return {
        "outside_domain_nearest_count": int(nearest["outside_domain"].sum()),
        "n_fields": int(len(nearest)),
        "within_domain_median_distance": float(pairs.loc[pairs["same_domain"], "distance"].median()),
        "cross_domain_median_distance": float(pairs.loc[~pairs["same_domain"], "distance"].median()),
        "closest_field_pair_distance": float(pairs["distance"].min()),
        "farthest_field_pair_distance": float(pairs["distance"].max()),
        "mean_distance_initial": float(temporal_summary["mean_all"].iloc[0]),
        "mean_distance_final": float(temporal_summary["mean_all"].iloc[-1]),
        "n_converging_field_pairs": int((field_changes["delta_morphological_distance"] < 0).sum()),
        "n_diverging_field_pairs": int((field_changes["delta_morphological_distance"] > 0).sum()),
    }


def clustering_summary(subfield: pd.DataFrame) -> dict[str, float]:
    selected = pd.read_csv(OUTPUT_DIR.parent / "09_morphological_typologies" / "cluster_solution_selected.csv")
    assignments = pd.read_csv(OUTPUT_DIR.parent / "09_morphological_typologies" / "subfield_typology_assignments.csv")
    labels = assignments.sort_values("subfield_id", kind="mergesort")["typology_id"].to_numpy()
    matrix = subfield.sort_values("subfield_id", kind="mergesort")[METRICS].to_numpy(dtype=float)
    silhouette = silhouette_score(matrix, labels, metric="euclidean")
    return {
        "static_ward_k": float(selected["k"].iloc[0]),
        "static_ward_silhouette": float(silhouette),
    }


def main() -> None:
    ensure_dirs()
    data = load_inputs()
    core = data["core"]
    drift = data["drift"]
    temporal = data["temporal"]

    write_metric_core_table()
    plot_pipeline_figure()
    eda = plot_eda(core)
    subfield, field, domain = static_profiles(core)
    pca = plot_static_profiles(core, subfield, field, domain)
    write_extremes_table(subfield)
    drift_summary = write_centroid_drift_table(drift)
    temporal_summary = plot_temporal_figures(temporal)
    similarity = plot_similarity_figures(core)
    clustering = clustering_summary(subfield)

    summary = {
        "active_metric_count": len(METRICS),
        "active_metrics": METRICS,
        **pca,
        **drift_summary,
        **temporal_summary,
        **similarity,
        **clustering,
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "active_8metric_results_summary.csv", index=False)
    (OUTPUT_DIR / "active_8metric_results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "eda_distribution_diagnostics.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
