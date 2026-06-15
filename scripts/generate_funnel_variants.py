import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from pathlib import Path

# Setup directories
out_dir = Path(r"C:\Users\Z0058EYW\.gemini\antigravity\brain\158259c6-593f-495f-84b0-63312e6908bd")
project_fig_dir = Path(r"c:\Users\Z0058EYW\Workspace\TFM\memory\figures")

for d in [out_dir, project_fig_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Common Data
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

def save_variant(fig, variant_num):
    png_name = f"fig_03_variant_{variant_num}.png"
    pdf_name = f"fig_03_variant_{variant_num}.pdf"
    
    fig.savefig(out_dir / png_name, bbox_inches="tight", dpi=300)
    fig.savefig(project_fig_dir / png_name, bbox_inches="tight", dpi=300)
    fig.savefig(project_fig_dir / pdf_name, bbox_inches="tight")
    plt.close(fig)
    print(f"Variant {variant_num} saved successfully.")

# ==========================================
# VARIANT 1: Ultra-Minimalist Serif (IEEE/ACM style)
# ==========================================
def make_variant_1():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9.0,
    })
    
    spacing = 1.65
    x_positions = np.arange(len(steps)) * spacing
    W = 0.8
    half_W = W / 2.0
    
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_xlim(x_positions[0] - 0.9, x_positions[-1] + 0.9)
    ax.set_ylim(-1.05, 1.8)
    
    # Draw non-filled funnel borders
    for i in range(len(steps) - 1):
        x_curr = x_positions[i]
        x_next = x_positions[i + 1]
        h_curr = steps[i]["height"]
        h_next = steps[i + 1]["height"]
        is_bridge = (steps[i]["group"] != steps[i+1]["group"])
        
        line_style = ":" if is_bridge else "--"
        ax.plot([x_curr + half_W, x_next - half_W], [h_curr / 2.0, h_next / 2.0], 
                linestyle=line_style, color="#111111", linewidth=0.8)
        ax.plot([x_curr + half_W, x_next - half_W], [-h_curr / 2.0, -h_next / 2.0], 
                linestyle=line_style, color="#111111", linewidth=0.8)

    # Draw boxes
    for i, step in enumerate(steps):
        x = x_positions[i]
        h = step["height"]
        # Hollow black box
        rect = Rectangle((x - half_W, -h / 2.0), W, h, 
                         facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.0, zorder=3)
        ax.add_patch(rect)
        
        ax.text(x, 0, step["value"], ha="center", va="center", 
                fontsize=11.0, fontweight="bold", color="#000000", zorder=4)
        ax.text(x, 0.80, step["header"], ha="center", va="bottom", 
                fontsize=9.5, fontweight="bold", color="#000000", linespacing=1.1, zorder=4)
        ax.text(x, 0.75, step["subheader"], ha="center", va="top", 
                fontsize=8.0, color="#333333", style="italic", linespacing=1.1, zorder=4)
        ax.text(x, -0.72, step["pct"], ha="center", va="top", 
                fontsize=9.5, color="#000000", zorder=4)

    # Brackets
    y_bracket = 1.35
    tick_len = 0.04
    for start_idx, end_idx, title in [(0, 4, "OpenAlex API live counts, queried 25 May 2026"), 
                                      (5, 7, "Frozen TFM pipeline snapshot")]:
        xs = x_positions[start_idx] - half_W
        xe = x_positions[end_idx] + half_W
        xc = (xs + xe) / 2.0
        ax.plot([xs, xe], [y_bracket, y_bracket], color="#000000", linewidth=1.0)
        ax.plot([xs, xs], [y_bracket, y_bracket - tick_len], color="#000000", linewidth=1.0)
        ax.plot([xe, xe], [y_bracket, y_bracket - tick_len], color="#000000", linewidth=1.0)
        ax.text(xc, y_bracket + 0.08, title, ha="center", va="bottom", 
                fontsize=10.0, fontweight="bold", color="#000000")

    save_variant(fig, 1)

# ==========================================
# VARIANT 2: Flowchart Blocks with Arrow Connectors
# ==========================================
def make_variant_2():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
    })
    
    spacing = 1.55
    x_positions = np.arange(len(steps)) * spacing
    W = 0.95
    half_W = W / 2.0
    h_box = 0.58  # Uniform height for flowchart
    
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_xlim(x_positions[0] - 0.9, x_positions[-1] + 0.9)
    ax.set_ylim(-1.05, 1.8)
    
    # Draw arrows
    for i in range(len(steps) - 1):
        x_curr = x_positions[i]
        x_next = x_positions[i + 1]
        is_bridge = (steps[i]["group"] != steps[i+1]["group"])
        
        # Draw horizontal connecting arrow
        arrow_style = dict(arrowstyle="->", lw=1.2, color="#4B5563" if not is_bridge else "#9CA3AF")
        if is_bridge:
            # Dotted arrow for transition
            ax.annotate("", xy=(x_next - half_W - 0.05, 0), xytext=(x_curr + half_W + 0.05, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color="#9CA3AF", linestyle=":"))
        else:
            ax.annotate("", xy=(x_next - half_W - 0.05, 0), xytext=(x_curr + half_W + 0.05, 0),
                        arrowprops=arrow_style)
            
    # Draw boxes
    for i, step in enumerate(steps):
        x = x_positions[i]
        group = step["group"]
        
        face_color = "#FFFFFF" if group == "live" else "#F3F4F6"
        rect = Rectangle((x - half_W, -h_box / 2.0), W, h_box, 
                         facecolor=face_color, edgecolor="#2D3748", linewidth=1.1, zorder=3)
        ax.add_patch(rect)
        
        ax.text(x, 0, step["value"], ha="center", va="center", 
                fontsize=11.5, fontweight="bold", color="#111827", zorder=4)
        ax.text(x, 0.80, step["header"], ha="center", va="bottom", 
                fontsize=9.5, fontweight="bold", color="#1F2937", linespacing=1.1, zorder=4)
        ax.text(x, 0.75, step["subheader"], ha="center", va="top", 
                fontsize=8.0, color="#4B5563", style="italic", linespacing=1.1, zorder=4)
        
        # Put percentage below the box
        ax.text(x, -0.72, step["pct"], ha="center", va="top", 
                fontsize=9.5, fontweight="bold", color="#374151", zorder=4)

    # Brackets
    y_bracket = 1.35
    tick_len = 0.04
    for start_idx, end_idx, title in [(0, 4, "OpenAlex API live counts, queried 25 May 2026"), 
                                      (5, 7, "Frozen TFM pipeline snapshot")]:
        xs = x_positions[start_idx] - half_W
        xe = x_positions[end_idx] + half_W
        xc = (xs + xe) / 2.0
        ax.plot([xs, xe], [y_bracket, y_bracket], color="#4B5563", linewidth=1.2)
        ax.plot([xs, xs], [y_bracket, y_bracket - tick_len], color="#4B5563", linewidth=1.2)
        ax.plot([xe, xe], [y_bracket, y_bracket - tick_len], color="#4B5563", linewidth=1.2)
        ax.text(xc, y_bracket + 0.08, title, ha="center", va="bottom", 
                fontsize=10.0, fontweight="bold", color="#111827")

    save_variant(fig, 2)

# ==========================================
# VARIANT 3: Modern Flat (Nature/Science style)
# ==========================================
def make_variant_3():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
    })
    
    spacing = 1.65
    x_positions = np.arange(len(steps)) * spacing
    W = 0.8
    half_W = W / 2.0
    
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_xlim(x_positions[0] - 0.9, x_positions[-1] + 0.9)
    ax.set_ylim(-1.05, 1.8)
    
    # Draw filled flat connectors
    for i in range(len(steps) - 1):
        x_curr = x_positions[i]
        x_next = x_positions[i + 1]
        h_curr = steps[i]["height"]
        h_next = steps[i + 1]["height"]
        
        poly_pts = [
            [x_curr + half_W, h_curr / 2.0],
            [x_next - half_W, h_next / 2.0],
            [x_next - half_W, -h_next / 2.0],
            [x_curr + half_W, -h_curr / 2.0]
        ]
        
        # Clean light grey flat fills, no borders at all
        poly_color = "#F3F4F6"
        poly = Polygon(poly_pts, facecolor=poly_color, edgecolor="none", zorder=1)
        ax.add_patch(poly)

    # Draw steps as borderless flat blocks
    for i, step in enumerate(steps):
        x = x_positions[i]
        h = step["height"]
        group = step["group"]
        
        # Borderless shaded blocks
        fill_color = "#E5E7EB" if group == "live" else "#D1D5DB"
        rect = Rectangle((x - half_W, -h / 2.0), W, h, 
                         facecolor=fill_color, edgecolor="none", zorder=3)
        ax.add_patch(rect)
        
        ax.text(x, 0, step["value"], ha="center", va="center", 
                fontsize=11.5, fontweight="bold", color="#111827", zorder=4)
        ax.text(x, 0.80, step["header"], ha="center", va="bottom", 
                fontsize=9.5, fontweight="bold", color="#1F2937", linespacing=1.1, zorder=4)
        ax.text(x, 0.75, step["subheader"], ha="center", va="top", 
                fontsize=8.0, color="#4B5563", style="italic", linespacing=1.1, zorder=4)
        ax.text(x, -0.72, step["pct"], ha="center", va="top", 
                fontsize=9.5, fontweight="bold", color="#374151", zorder=4)

    # Thin sleek bracket lines
    y_bracket = 1.35
    for start_idx, end_idx, title in [(0, 4, "OpenAlex API live counts, queried 25 May 2026"), 
                                      (5, 7, "Frozen TFM pipeline snapshot")]:
        xs = x_positions[start_idx] - half_W + 0.05
        xe = x_positions[end_idx] + half_W - 0.05
        xc = (xs + xe) / 2.0
        # Simple line bracket with no drop ticks, or very minimal ones
        ax.plot([xs, xe], [y_bracket, y_bracket], color="#9CA3AF", linewidth=1.0)
        ax.text(xc, y_bracket + 0.06, title, ha="center", va="bottom", 
                fontsize=10.0, color="#374151")

    save_variant(fig, 3)

# ==========================================
# VARIANT 4: Stepped Funnel (Continuous / Compact)
# ==========================================
def make_variant_4():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
    })
    
    W = 1.15  # wider boxes since they are adjacent
    x_positions = np.arange(len(steps)) * W
    half_W = W / 2.0
    
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_xlim(x_positions[0] - 0.7, x_positions[-1] + 0.7)
    ax.set_ylim(-1.05, 1.8)
    
    # Draw continuous funnel background first
    all_x = []
    all_y_top = []
    all_y_bot = []
    
    for i, step in enumerate(steps):
        x = x_positions[i]
        h = step["height"]
        # Top points (left and right of each block)
        all_x.extend([x - half_W, x + half_W])
        all_y_top.extend([h / 2.0, h / 2.0])
        all_y_bot.extend([-h / 2.0, -h / 2.0])
        
    # Build a single big polygon for the stepped funnel
    poly_pts = []
    # top line left-to-right
    for x, y in zip(all_x, all_y_top):
        poly_pts.append([x, y])
    # bottom line right-to-left
    for x, y in zip(reversed(all_x), reversed(all_y_bot)):
        poly_pts.append([x, y])
        
    poly = Polygon(poly_pts, facecolor="#F3F4F6", edgecolor="none", zorder=1)
    ax.add_patch(poly)
    
    # Draw steps and outer borders
    # Top and bottom outlines
    for i in range(len(all_x) - 1):
        # Top stepped line segments
        ax.plot([all_x[i], all_x[i+1]], [all_y_top[i], all_y_top[i+1]], color="#4B5563", linewidth=1.0, zorder=2)
        # Bottom stepped line segments
        ax.plot([all_x[i], all_x[i+1]], [all_y_bot[i], all_y_bot[i+1]], color="#4B5563", linewidth=1.0, zorder=2)
        
    # Draw vertical dividers and overlays
    for i, step in enumerate(steps):
        x = x_positions[i]
        h = step["height"]
        group = step["group"]
        
        # Left boundary divider (except for the first one, which is start edge)
        if i > 0:
            # Check if it's the transition boundary
            is_bridge = (steps[i-1]["group"] != steps[i]["group"])
            line_style = ":" if is_bridge else "-"
            ax.plot([x - half_W, x - half_W], [-h / 2.0, h / 2.0], 
                    color="#9CA3AF" if is_bridge else "#D1D5DB", linestyle=line_style, linewidth=0.8, zorder=2)
            
        # Add a light gray fill for the frozen group to distinguish it
        if group == "frozen":
            rect = Rectangle((x - half_W, -h / 2.0), W, h, 
                             facecolor="#E5E7EB", edgecolor="none", alpha=0.6, zorder=1.5)
            ax.add_patch(rect)
            
        # Start and end vertical edges
        if i == 0:
            ax.plot([x - half_W, x - half_W], [-h / 2.0, h / 2.0], color="#4B5563", linewidth=1.0, zorder=2)
        if i == len(steps) - 1:
            ax.plot([x + half_W, x + half_W], [-h / 2.0, h / 2.0], color="#4B5563", linewidth=1.0, zorder=2)

        ax.text(x, 0, step["value"], ha="center", va="center", 
                fontsize=11.5, fontweight="bold", color="#111827", zorder=4)
        ax.text(x, 0.80, step["header"], ha="center", va="bottom", 
                fontsize=9.5, fontweight="bold", color="#1F2937", linespacing=1.1, zorder=4)
        ax.text(x, 0.75, step["subheader"], ha="center", va="top", 
                fontsize=8.0, color="#4B5563", style="italic", linespacing=1.1, zorder=4)
        ax.text(x, -0.72, step["pct"], ha="center", va="top", 
                fontsize=9.5, fontweight="bold", color="#374151", zorder=4)

    # Brackets
    y_bracket = 1.35
    tick_len = 0.04
    for start_idx, end_idx, title in [(0, 4, "OpenAlex API live counts, queried 25 May 2026"), 
                                      (5, 7, "Frozen TFM pipeline snapshot")]:
        xs = x_positions[start_idx] - half_W
        xe = x_positions[end_idx] + half_W
        xc = (xs + xe) / 2.0
        ax.plot([xs, xe], [y_bracket, y_bracket], color="#4B5563", linewidth=1.2)
        ax.plot([xs, xs], [y_bracket, y_bracket - tick_len], color="#4B5563", linewidth=1.2)
        ax.plot([xe, xe], [y_bracket, y_bracket - tick_len], color="#4B5563", linewidth=1.2)
        ax.text(xc, y_bracket + 0.08, title, ha="center", va="bottom", 
                fontsize=10.0, fontweight="bold", color="#111827")

    save_variant(fig, 4)

if __name__ == "__main__":
    make_variant_1()
    make_variant_2()
    make_variant_3()
    make_variant_4()
