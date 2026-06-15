import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from pathlib import Path

# Setup directories
out_dir = Path(r"C:\Users\Z0058EYW\.gemini\antigravity\brain\158259c6-593f-495f-84b0-63312e6908bd")
out_dir.mkdir(parents=True, exist_ok=True)
png_path = out_dir / "fig_03_clean_pipeline.png"

project_fig_dir = Path(r"c:\Users\Z0058EYW\Workspace\TFM\memory\figures")
project_fig_dir.mkdir(parents=True, exist_ok=True)
project_png_path = project_fig_dir / "fig_03_clean_pipeline.png"
project_pdf_path = project_fig_dir / "fig_03_clean_pipeline.pdf"

# Setup matplotlib styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "figure.dpi": 200,
    "savefig.dpi": 300,
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

# Draw connecting funnels first (so they go behind the rectangles)
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
        # Live counts: solid clean border, white background
        rect = Rectangle((x - half_W, -h / 2.0), W, h, 
                         facecolor="#FFFFFF", edgecolor="#374151", linewidth=1.2, zorder=3)
        num_color = "#1F2937"
    else:
        # Frozen snapshot: light gray background, solid border
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

# Save figure in all locations
plt.savefig(png_path, bbox_inches="tight", dpi=300)
plt.savefig(project_png_path, bbox_inches="tight", dpi=300)
plt.savefig(project_pdf_path, bbox_inches="tight")
plt.close()
print(f"Figure saved to artifact folder: {png_path}")
print(f"Figure saved to project PNG: {project_png_path}")
print(f"Figure saved to project PDF: {project_pdf_path}")
