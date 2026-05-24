import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from scipy.stats import gaussian_kde

# Define output directories
OUTPUT_DIR = "tmp_visual_exploration/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color Palette (UC3M Academic Theme)
AZUL_UC3M = '#000066'   # Deep Navy
SLATE_GRAY = '#475569'  # Secondary Slate
ACCENT_RED = '#DC2626'  # Accent Red/Coral (Centroids, Temporal movement)
ACCENT_GREEN = '#16A34A'# Accent Green (Local density, Convergence)
LIGHT_BG = '#F8FAFC'     # Card and Panel Background
MUTED_LINE = '#CBD5E1'   # Grid lines and border lines
TEXT_DARK = '#0F172A'    # Near black for labels
TEXT_MUTED = '#64748B'   # Muted gray for captions

# Matplotlib global styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['text.color'] = TEXT_DARK
plt.rcParams['axes.labelcolor'] = TEXT_DARK
plt.rcParams['xtick.color'] = TEXT_DARK
plt.rcParams['ytick.color'] = TEXT_DARK
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def apply_common_styling(ax, title=None, xlabel=None, ylabel=None):
    """Applies high-quality scientific styling to axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MUTED_LINE)
    ax.spines['bottom'].set_color(MUTED_LINE)
    ax.tick_params(colors=TEXT_DARK, width=1)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12, color=AZUL_UC3M)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=8)
    ax.grid(True, linestyle='--', alpha=0.3, color=MUTED_LINE)

# ==========================================
# REQ 1: The Shape of Science Framework
# ==========================================
def generate_fig_req_01():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    
    # Draw horizontal workflow stages
    stages = [
        ("OpenAlex Taxonomy", "Domains, Fields,\nSubfields Mapping", 5),
        ("Corpus Construction", "English filtering &\nAbstract eligibility", 21),
        ("SPECTER2 Vectors", "768-D Dense Text\nEmbeddings", 37),
        ("Subfield Clouds", "Point distributions\nin continuous space", 53),
        ("Morphology Core", "11 metrics calculated\nacross 5 families", 69),
        ("Empirical Block", "Shape, evolution,\nsimilarity, typologies", 85)
    ]
    
    for title, subtitle, x in stages:
        # Draw Box
        rect = patches.FancyBboxPatch((x, 15), 11, 20, boxstyle="round,pad=1.5", 
                                     facecolor=LIGHT_BG, edgecolor=AZUL_UC3M, linewidth=1.5)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x + 5.5, 27, title, fontsize=9, fontweight='bold', ha='center', color=AZUL_UC3M)
        ax.text(x + 5.5, 20, subtitle, fontsize=7.5, ha='center', color=TEXT_DARK)
        
        # Draw connections (arrows) except for the last stage
        if x < 85:
            ax.annotate("", xy=(x + 14.5, 25), xytext=(x + 12.0, 25),
                        arrowprops=dict(arrowstyle="->", color=SLATE_GRAY, lw=2, mutation_scale=15))
            
    ax.text(50, 45, "THE SHAPE OF SCIENCE: CONCEPTUAL PIPELINE", fontsize=13, fontweight='bold', ha='center', color=AZUL_UC3M)
    ax.text(50, 41.5, "From static classification systems to continuous, high-dimensional embedding morphology", 
            fontsize=9.5, style='italic', ha='center', color=TEXT_MUTED)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_req_01_framework.png", bbox_inches='tight')
    plt.close()

# ==========================================
# REQ 2: A Field as a Paper Cloud
# ==========================================
def generate_fig_req_02():
    np.random.seed(42)
    # Generate points representing papers in a subfield (2D projection)
    n_points = 250
    # Two principal directions, making it slightly elongated
    x = np.random.normal(0, 1.5, n_points)
    y = np.random.normal(0, 0.8, n_points)
    
    # Rotate points slightly to make it look organic
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    x_rot = c * x - s * y
    y_rot = s * x + c * y
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Draw point cloud
    ax.scatter(x_rot, y_rot, alpha=0.4, color=SLATE_GRAY, s=15, label='Papers (SPECTER2 Vectors)')
    
    # 1. Centroid (Red Star)
    centroid_x, centroid_y = np.mean(x_rot), np.mean(y_rot)
    ax.plot(centroid_x, centroid_y, marker='*', color=ACCENT_RED, markersize=14, label='Centroid ($\mu$)', zorder=10)
    ax.annotate("Centroid ($\mu$)\n(Semantic Center)", xy=(centroid_x, centroid_y), xytext=(centroid_x - 0.5, centroid_y + 1.2),
                arrowprops=dict(facecolor=ACCENT_RED, arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
                fontsize=8.5, fontweight='bold', color=ACCENT_RED)
    
    # 2. Dispersion (Standard Distance circle)
    std_dist = np.sqrt(np.mean(x_rot**2 + y_rot**2))
    disp_circle = patches.Circle((centroid_x, centroid_y), std_dist, fill=False, linestyle='--', color=AZUL_UC3M, lw=1.5, label='Dispersion (Std Dist)')
    ax.add_patch(disp_circle)
    ax.annotate("Dispersion\n(Standard Distance)", xy=(centroid_x + std_dist*np.cos(np.radians(45)), centroid_y + std_dist*np.sin(np.radians(45))), 
                xytext=(centroid_x + 1.8, centroid_y + 1.8),
                arrowprops=dict(facecolor=AZUL_UC3M, arrowstyle="->"),
                fontsize=8.5, fontweight='bold', color=AZUL_UC3M)
    
    # 3. Local Density Contours (Schematic)
    # Estimate density
    xy = np.vstack([x_rot, y_rot])
    z = gaussian_kde(xy)(xy)
    ax.tricontour(x_rot, y_rot, z, levels=3, cmap='Greens', alpha=0.6, linewidths=1.5)
    ax.text(-2.5, -1.8, "Local Density\n(Contour levels)", fontsize=8.5, color=ACCENT_GREEN, fontweight='bold')
    
    # 4. Hubness (Star-like connection around a local hub)
    # Pick a point near dense area and connect to 6 nearest neighbors
    hub_idx = np.argmin(x_rot**2 + (y_rot - 0.2)**2)
    distances = np.sqrt((x_rot - x_rot[hub_idx])**2 + (y_rot - y_rot[hub_idx])**2)
    nearest_neighbors = np.argsort(distances)[1:7]
    for nn in nearest_neighbors:
        ax.plot([x_rot[hub_idx], x_rot[nn]], [y_rot[hub_idx], y_rot[nn]], color='purple', alpha=0.7, lw=1)
    ax.scatter(x_rot[hub_idx], y_rot[hub_idx], color='purple', s=40, edgecolor='white', zorder=5, label='Semantic Hub')
    ax.text(x_rot[hub_idx] + 0.15, y_rot[hub_idx] - 0.35, "Semantic Hub\n(Hubness metric)", fontsize=8.5, color='purple', fontweight='bold')
    
    # 5. Principal Directions (Spectral Structure)
    # Draw PCA arrows
    ax.arrow(centroid_x, centroid_y, 2.0*c, 2.0*s, head_width=0.15, head_length=0.15, fc=SLATE_GRAY, ec=SLATE_GRAY, lw=2)
    ax.arrow(centroid_x, centroid_y, -1.0*s, 1.0*c, head_width=0.1, head_length=0.1, fc=SLATE_GRAY, ec=SLATE_GRAY, lw=1.5)
    ax.text(centroid_x + 2.2*c, centroid_y + 2.2*s, "PC1\n(Spectral Extension)", fontsize=8, color=SLATE_GRAY, fontweight='bold')
    
    # 6. Temporal Drift Path
    drift_x = [-2.2, -1.6, -1.0, -0.4, centroid_x]
    drift_y = [-1.5, -1.1, -0.7, -0.3, centroid_y]
    ax.plot(drift_x, drift_y, color=ACCENT_RED, linestyle=':', lw=2, marker='o', markersize=4, alpha=0.7)
    ax.text(-2.5, -1.2, "Temporal Drift Path\n(Evolution: 2000-2024)", fontsize=8.5, color=ACCENT_RED, alpha=0.8)
    
    apply_common_styling(ax, title="ANATOMY OF A SCIENTIFIC PAPER CLOUD", xlabel="Embedding Space Projection (Axis 1)", ylabel="Embedding Space Projection (Axis 2)")
    ax.legend(loc='upper right', frameon=True, facecolor=LIGHT_BG, edgecolor=MUTED_LINE, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_req_02_paper_cloud.png", bbox_inches='tight')
    plt.close()

# ==========================================
# REQ 3: Metric Families Dashboard
# ==========================================
def generate_fig_req_03():
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.flatten()
    
    families = [
        ("1. Dispersion Family", "Measures the global sprawl or scale of a subfield's papers.", 
         "• Standard Distance (SD)\n• Mean Pairwise Distance (MPD)", 'circle'),
        ("2. Local Density Family", "Measures local cohesion and packing of literature.", 
         "• Mean Nearest Neighbor Distance (MNND)\n• Local Density Index (LDI)", 'density'),
        ("3. Hubness Family", "Measures structural concentration and key centers.", 
         "• Skewness (Sk)\n• Top 1% Hub Concentration (HC)", 'hub'),
        ("4. Spectral Structure Family", "Measures dimensionality and principal directions.", 
         "• Effective Dimensionality (ED)\n• Eigenvalue Entropy (EE)", 'eigen'),
        ("5. Temporal Movement Family", "Measures speed and mode of evolutionary change.", 
         "• Centroid Velocity (CV)\n• Area Expansion Rate (AER)", 'temporal')
    ]
    
    for i, (name, desc, metrics, style) in enumerate(families):
        ax = axs[i]
        ax.set_facecolor(LIGHT_BG)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Box frame
        rect = patches.FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.2", 
                                     facecolor=LIGHT_BG, edgecolor=AZUL_UC3M, lw=1.2)
        ax.add_patch(rect)
        
        # Header
        ax.text(1, 8.5, name, fontsize=11, fontweight='bold', color=AZUL_UC3M)
        ax.text(1, 7.2, desc, fontsize=8, color=TEXT_DARK, wrap=True)
        
        # Illustrative mini-plots inside card
        if style == 'circle':
            # Draw tight vs wide circles
            ax.add_patch(patches.Circle((2.5, 4.0), 0.6, fill=True, color=SLATE_GRAY, alpha=0.3))
            ax.add_patch(patches.Circle((2.5, 4.0), 1.3, fill=False, linestyle='--', color=AZUL_UC3M, lw=1.5))
            ax.plot([2.5, 2.5 + 1.3], [4.0, 4.0], color=AZUL_UC3M, lw=1.5)
            ax.text(2.5, 2.3, "Global Scale", fontsize=7.5, ha='center', color=SLATE_GRAY)
        elif style == 'density':
            # Clumped points
            np.random.seed(0)
            px = np.random.normal(2.5, 0.4, 25)
            py = np.random.normal(4.0, 0.4, 25)
            ax.scatter(px, py, color=ACCENT_GREEN, s=6, alpha=0.7)
            ax.text(2.5, 2.3, "Cohesion & Clusters", fontsize=7.5, ha='center', color=SLATE_GRAY)
        elif style == 'hub':
            # Star graph
            ax.scatter([2.5], [4.0], color='purple', s=35, zorder=5)
            for angle in np.linspace(0, 360, 8, endpoint=False):
                rad = np.radians(angle)
                ax.plot([2.5, 2.5 + 1.0*np.cos(rad)], [4.0, 4.0 + 1.0*np.sin(rad)], color='purple', alpha=0.6, lw=1)
                ax.scatter([2.5 + 1.0*np.cos(rad)], [4.0 + 1.0*np.sin(rad)], color=SLATE_GRAY, s=8)
            ax.text(2.5, 2.3, "Hub Dominated", fontsize=7.5, ha='center', color=SLATE_GRAY)
        elif style == 'eigen':
            # Eigenvalues line
            e_x = np.arange(1, 6)
            e_y = [5.0, 2.8, 1.5, 0.6, 0.2]
            ax.bar(e_x + 0.5, e_y, width=0.4, color=SLATE_GRAY, alpha=0.8)
            ax.plot(e_x + 0.7, e_y, color=AZUL_UC3M, marker='o', markersize=3)
            ax.text(2.5, 2.3, "Eigenvalue Decay", fontsize=7.5, ha='center', color=SLATE_GRAY)
        elif style == 'temporal':
            # Centroid path
            ax.plot([1.2, 2.0, 2.8, 3.8], [3.0, 3.5, 4.2, 4.8], color=ACCENT_RED, linestyle=':', marker='o', markersize=4)
            ax.text(2.5, 2.3, "Centroid Drift Velocity", fontsize=7.5, ha='center', color=SLATE_GRAY)
            
        # Core Metrics Text
        ax.text(5.2, 5.0, "Core Metrics:", fontsize=8.5, fontweight='bold', color=AZUL_UC3M)
        ax.text(5.2, 3.0, metrics, fontsize=8, color=TEXT_DARK)
        
    # Remove unused last panel, use it for general metadata
    ax_last = axs[5]
    ax_last.axis('off')
    ax_last.text(5, 7.5, "MORPHOLOGY CORE CORE\n(11 METRIC FRAMEWORK)", fontsize=12, fontweight='bold', ha='center', color=AZUL_UC3M)
    ax_last.text(5, 5.0, "All metrics are computed in the full\n768-dimensional SPECTER2 space,\nensuring zero-loss representation before\nvisualization projection.", 
                 fontsize=8.5, ha='center', color=TEXT_DARK, bbox=dict(boxstyle="round,pad=1.0", facecolor='#F1F5F9', edgecolor=MUTED_LINE))
    ax_last.text(5, 2.0, "Thesis Chapter 5: Embedding-Space Metrics", fontsize=8.5, style='italic', ha='center', color=TEXT_MUTED)
    
    plt.suptitle("MORPHOLOGICAL METRIC FAMILIES DASHBOARD", fontsize=15, fontweight='bold', color=AZUL_UC3M, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{OUTPUT_DIR}/fig_req_03_metric_dashboard.png", bbox_inches='tight')
    plt.close()

# ==========================================
# REQ 4: Reader Compass for Empirical Block
# ==========================================
def generate_fig_req_04():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axis('off')
    
    # Draw outer circle
    ax.add_patch(patches.Circle((0, 0), 8.5, fill=False, edgecolor=MUTED_LINE, lw=1))
    ax.add_patch(patches.Circle((0, 0), 2.5, fill=True, facecolor=LIGHT_BG, edgecolor=AZUL_UC3M, lw=2))
    ax.text(0, 0, "Empirical\nBlock\nCompass", fontsize=11, fontweight='bold', ha='center', va='center', color=AZUL_UC3M)
    
    # 4 Chapter Anchors
    anchors = [
        ("CHAPTER 6: STATIC SHAPE\nWhere do subfields differ?\n(Discipline profiles & PCA space)", 0, 6.0, 'N', ACCENT_RED),
        ("CHAPTER 7: TEMPORAL EVOLUTION\nHow does shape evolve over time?\n(Drift velocity & Area expansion)", 6.0, 0, 'E', AZUL_UC3M),
        ("CHAPTER 8: RELATIONSHIPS\nWho is morphologically similar?\n(Convergence & divergence pairs)", 0, -6.0, 'S', ACCENT_GREEN),
        ("CHAPTER 9: TYPOLOGIES\nCan we group fields naturally?\n(Static clusters vs Dynamic trajectory groups)", -6.0, 0, 'W', SLATE_GRAY)
    ]
    
    for label, x, y, direction, color in anchors:
        # Draw compass marker
        ax.plot(x, y, marker='o', color=color, markersize=10)
        
        # Arrow from center
        ax.annotate("", xy=(x*0.9, y*0.9), xytext=(x*0.4, y*0.4),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=12))
        
        # Add chapter box
        ha = 'center' if x == 0 else ('left' if x > 0 else 'right')
        va = 'bottom' if y > 0 else ('top' if y < 0 else 'center')
        
        offset_x = 0 if x == 0 else (0.8 if x > 0 else -0.8)
        offset_y = 0.8 if y > 0 else (-0.8 if y < 0 else 0)
        
        ax.text(x + offset_x, y + offset_y, label, fontsize=8.5, fontweight='bold',
                ha=ha, va=va, color=color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFFFFF', edgecolor=MUTED_LINE, lw=0.8, alpha=0.9))
        
    ax.text(0, 9.3, "EMPIRICAL ROADMAP: CHAPTERS 6-9", fontsize=14, fontweight='bold', ha='center', color=AZUL_UC3M)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_req_04_empirical_compass.png", bbox_inches='tight')
    plt.close()

# ==========================================
# REQ 5: What Each Chapter Adds
# ==========================================
def generate_fig_req_05():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('off')
    
    # Table Content
    data = [
        ["Chapter", "Central Research Question", "Analytical Methodology", "Core Contribution / Output"],
        ["Chapter 6\nStatic Shape", "How do scientific fields differ in their\ninternal shapes and dispersion?", "Normalized morphological profile PCA\nand outlier subfield analysis.", "Static morphological profile mapping;\ndiscovery of domain-level tendencies."],
        ["Chapter 7\nTemporal Evolution", "How does literature shape and location\nevolve over five-year windows?", "Centroid trajectory paths in PCA;\nmetric delta indicators across epochs.", "Typology of temporal trajectories;\nquantifying semantic drift velocity."],
        ["Chapter 8\nRelationships", "Do subfields converge, diverge, or\nmaintain separation over time?", "Pairwise distance matrices; comparison of\nprofile vs. centroid separation.", "Evidence of general separation increase\nbut localized convergence pairs."],
        ["Chapter 9\nTypologies", "Are there natural structural types of\nscientific subfields?", "HAC & K-means on static profiles;\nclustering of temporal delta arrays.", "Demonstration of weak static clustering\nbut clear, stable dynamic trajectory groups."]
    ]
    
    # Draw custom grid table
    y = 50
    cell_height = 8
    col_widths = [12, 30, 28, 30]
    col_positions = [0, 12, 42, 70]
    
    # Header background
    rect_hdr = patches.Rectangle((0, y - cell_height), 100, cell_height, facecolor=AZUL_UC3M, zorder=0)
    ax.add_patch(rect_hdr)
    
    # Write Headers
    for i, title in enumerate(data[0]):
        ax.text(col_positions[i] + 1.5, y - 5, title, fontsize=9.5, fontweight='bold', color='#FFFFFF', va='center')
        
    # Write rows
    y -= cell_height
    for row_idx, row in enumerate(data[1:]):
        # Alternating row background
        bg_color = LIGHT_BG if row_idx % 2 == 0 else '#FFFFFF'
        rect_row = patches.Rectangle((0, y - cell_height), 100, cell_height, facecolor=bg_color, edgecolor=MUTED_LINE, lw=0.5, zorder=0)
        ax.add_patch(rect_row)
        
        # Write cells
        for col_idx, cell_text in enumerate(row):
            fw = 'bold' if col_idx == 0 else 'normal'
            color = AZUL_UC3M if col_idx == 0 else TEXT_DARK
            ax.text(col_positions[col_idx] + 1.5, y - cell_height/2, cell_text, fontsize=8.5,
                    fontweight=fw, color=color, va='center')
            
        y -= cell_height
        
    ax.text(50, 56, "THE EMPIRICAL BLOCK OVERVIEW & ROADMAP", fontsize=13, fontweight='bold', ha='center', color=AZUL_UC3M)
    ax.set_xlim(0, 100)
    ax.set_ylim(10, 60)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_req_05_chapter_roadmap.png", bbox_inches='tight')
    plt.close()

# ==========================================
# REQ 6: Four Readings of Scientific Structure
# ==========================================
def generate_fig_req_06():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    
    readings = [
        ("A. TAXONOMY (Hierarchical Labeling)", 
         "Discrete bins (Domains -> Fields -> Subfields)\nthat ignore overlaps and spatial distances.", 
         'tree', AZUL_UC3M),
        ("B. SEMANTIC LOCATION (Absolute Coordinate)", 
         "Where the centroid sits in embedding space.\nCaptures the core semantic topic.", 
         'centroid', ACCENT_RED),
        ("C. MORPHOLOGY (Internal Organization)", 
         "The shape, dispersion, and density of the papers.\nReflects research scope and scientific organization.", 
         'shape', ACCENT_GREEN),
        ("D. TEMPORAL DYNAMICS (Evolutionary Flow)", 
         "How shape and location drift over epoch windows.\nCaptures field movement and expansion.", 
         'arrow', SLATE_GRAY)
    ]
    
    for i, (title, desc, style, color) in enumerate(readings):
        ax = axs[i]
        ax.set_facecolor(LIGHT_BG)
        apply_common_styling(ax, title=title)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
        # Draw stylized diagram inside plot area
        if style == 'tree':
            # Draw simple hierarchy
            ax.plot([0, -2, 2], [3, 0, 0], color=color, marker='o', markersize=6, lw=1.5)
            ax.plot([-2, -3, -1], [0, -3, -3], color=color, marker='o', markersize=4, lw=1)
            ax.plot([2, 1, 3], [0, -3, -3], color=color, marker='o', markersize=4, lw=1)
            ax.text(0, 3.5, "Domain", fontsize=8, ha='center')
            ax.text(-2, 0.4, "Field A", fontsize=7.5, ha='center')
            ax.text(2, 0.4, "Field B", fontsize=7.5, ha='center')
        elif style == 'centroid':
            # Draw axes and single point
            ax.axhline(0, color=MUTED_LINE, lw=0.8, linestyle='--')
            ax.axvline(0, color=MUTED_LINE, lw=0.8, linestyle='--')
            ax.scatter([1.5], [1.5], color=color, s=80, zorder=5)
            ax.text(1.5, 2.1, "Centroid Coordinate\n($X_{coord}, Y_{coord}$)", fontsize=8, ha='center', color=color, fontweight='bold')
        elif style == 'shape':
            # Draw tight vs broad clouds
            np.random.seed(1)
            c1_x, c1_y = np.random.normal(-2, 0.4, 30), np.random.normal(0, 0.4, 30)
            c2_x, c2_y = np.random.normal(2, 1.0, 30), np.random.normal(0, 1.0, 30)
            ax.scatter(c1_x, c1_y, color=color, s=6, alpha=0.7, label='Focused Subfield')
            ax.scatter(c2_x, c2_y, color=SLATE_GRAY, s=6, alpha=0.5, label='Dispersed Subfield')
            ax.text(-2, 1.2, "Concentrated", fontsize=7.5, ha='center', color=color, fontweight='bold')
            ax.text(2, 2.5, "Dispersed", fontsize=7.5, ha='center', color=SLATE_GRAY, fontweight='bold')
        elif style == 'arrow':
            # Sequence of clouds with centroid path
            ax.plot([-3, -1, 1, 3], [-2, -1, 0, 2], color=color, marker='o', markersize=4, linestyle=':', lw=2)
            # Add expanding dashed circles
            for c_x, c_y, r in [(-3, -2, 0.6), (-1, -1, 0.8), (1, 0, 1.0), (3, 2, 1.3)]:
                ax.add_patch(patches.Circle((c_x, c_y), r, fill=False, edgecolor=color, linestyle='--', alpha=0.5))
            ax.text(1.8, 0.8, "Drift + Expansion", fontsize=8, color=color, fontweight='bold')
            
        ax.text(-4.5, -4.5, desc, fontsize=7.5, color=TEXT_DARK, wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFFF', edgecolor=MUTED_LINE, alpha=0.9))
        
    plt.suptitle("FOUR READINGS OF SCIENTIFIC STRUCTURE", fontsize=14, fontweight='bold', color=AZUL_UC3M, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/fig_req_06_four_readings.png", bbox_inches='tight')
    plt.close()

# ==========================================
# REQ 7: Typology Cards
# ==========================================
def generate_fig_req_07():
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs = axs.flatten()
    
    typologies = [
        ("CARD 1: DENSE CONCENTRATED CORES", 
         "Subfields with very low dispersion, high local density, and a strong set of central hubs.\n"
         "Examples: Cardiology, Applied Physics, Inorganic Chemistry.", 
         [1, 9, 8, 2], ['Disp', 'Dens', 'Hub', 'Dim'], ACCENT_RED),
        
        ("CARD 2: HIGHLY DISPERSED FRONTIERS", 
         "Subfields spanning multiple topics with extreme dispersion, flat eigenvalue spectrum, and low hubness.\n"
         "Examples: Multidisciplinary, Applied Mathematics, General Engineering.", 
         [9, 1, 2, 8], ['Disp', 'Dens', 'Hub', 'Dim'], AZUL_UC3M),
         
        ("CARD 3: EXPANDING APPLIED PATHS", 
         "Highly dynamic subfields with very high temporal drift velocity, expanding area, and falling density.\n"
         "Examples: Artificial Intelligence, Data Science, Bioinformatics.", 
         [6, 3, 4, 7], ['Disp', 'Dens', 'Hub', 'Drift'], ACCENT_GREEN),
         
        ("CARD 4: STABLE CLASSICAL FOUNDATIONS", 
         "Subfields showing near-zero temporal centroid drift, static structural metrics, and high structural stability.\n"
         "Examples: History, Philosophy, Geometry, Pure Linguistics.", 
         [2, 7, 6, 1], ['Disp', 'Dens', 'Hub', 'Drift'], SLATE_GRAY)
    ]
    
    for i, (title, desc, scores, labels, color) in enumerate(typologies):
        ax = axs[i]
        ax.set_facecolor(LIGHT_BG)
        apply_common_styling(ax, title=title)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(False)
        
        # Write description
        ax.text(0.5, 7.8, desc, fontsize=8, color=TEXT_DARK, wrap=True)
        
        # Mini Profile Bar Chart inside card
        x_bar = np.arange(4) + 1.5
        ax.bar(x_bar, scores, width=0.5, color=color, alpha=0.8, edgecolor=AZUL_UC3M, lw=0.8)
        
        # Write scores on bars
        for idx, val in enumerate(scores):
            ax.text(x_bar[idx], val + 0.3, str(val), fontsize=8, ha='center', fontweight='bold', color=color)
            
        ax.set_xticks(x_bar)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 11)
        ax.set_ylabel("Normalized Rank (1-10)", fontsize=7)
        
        # Inner border
        rect = patches.Rectangle((0.1, 0.1), 9.8, 9.8, fill=False, edgecolor=MUTED_LINE, lw=1)
        ax.add_patch(rect)
        
    plt.suptitle("MORPHOLOGICAL TYPOLOGY CARDS", fontsize=15, fontweight='bold', color=AZUL_UC3M, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{OUTPUT_DIR}/fig_req_07_typology_cards.png", bbox_inches='tight')
    plt.close()

# ==========================================
# REQ 8: Static vs. Dynamic (One-line thesis)
# ==========================================
def generate_fig_req_08():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # 1. Left Panel: Static Morphology (High Overlap)
    np.random.seed(12)
    ax1.set_facecolor(LIGHT_BG)
    for domain, color, marker in [('Physical', AZUL_UC3M, 'o'), ('Health', ACCENT_RED, 's'), ('Social', SLATE_GRAY, '^')]:
        dx = np.random.normal(0, 1.2, 50)
        dy = np.random.normal(0, 1.2, 50)
        ax1.scatter(dx, dy, color=color, marker=marker, s=15, alpha=0.6, label=domain)
        
    apply_common_styling(ax1, title="STATIC SHAPE SPACE (High Overlap)\nSilhouette Score $\\approx$ 0.05", 
                         xlabel="Morphological PC1", ylabel="Morphological PC2")
    ax1.legend(loc='upper right', frameon=True, fontsize=8)
    ax1.text(0, -3.2, "Subfields merge continuously. Taxonomy does not\ndictate static shape; high internal variability.", 
             fontsize=8.5, color=TEXT_DARK, ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFFFFF', edgecolor=MUTED_LINE))
    
    # 2. Right Panel: Dynamic Morphology (Clearer Trajectory Clusters)
    np.random.seed(15)
    ax2.set_facecolor(LIGHT_BG)
    clusters = [
        ("Expanding Frontiers", 2.0, 2.0, ACCENT_GREEN, 'o'),
        ("Stable Classics", -2.0, -2.0, SLATE_GRAY, '^'),
        ("Drifting Applieds", -2.0, 2.0, ACCENT_RED, 's')
    ]
    for cluster_name, cx, cy, color, marker in clusters:
        dx = np.random.normal(cx, 0.4, 30)
        dy = np.random.normal(cy, 0.4, 30)
        ax2.scatter(dx, dy, color=color, marker=marker, s=18, alpha=0.8, label=cluster_name)
        
    apply_common_styling(ax2, title="DYNAMIC EVOLUTION SPACE (Clearer Structure)\nSilhouette Score $\\approx$ 0.35", 
                         xlabel="Temporal Metric Delta PC1", ylabel="Temporal Metric Delta PC2")
    ax2.legend(loc='upper right', frameon=True, fontsize=8)
    ax2.text(0, -3.2, "Clear trajectory modes. Subfields are easier\nto group by HOW they evolve than by static shape.", 
             fontsize=8.5, color=TEXT_DARK, ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFFFFF', edgecolor=MUTED_LINE))
    
    plt.suptitle("THE ONE-LINE THESIS: STATIC OVERLAP VS. DYNAMIC SEPARABILITY", fontsize=14, fontweight='bold', color=AZUL_UC3M, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{OUTPUT_DIR}/fig_req_08_one_line_thesis.png", bbox_inches='tight')
    plt.close()

# ==========================================
# REQ 9: Corpus / Pipeline Flow
# ==========================================
def generate_fig_req_09():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    
    # Sankey-funnel steps
    steps = [
        ("OpenAlex Universe\n(All records)", "150M+ works", 45, 5),
        ("Text-Eligible works\n(EN title + abstract)", "3.5M works", 35, 25),
        ("Empirical Sample\n(2000-2024)", "2.37M works", 28, 45),
        ("Row-Aligned Subset\n(Embedding match)", "2.34M works", 22, 65),
        ("Analysis Subfields\n(Clean subfield data)", "241 Subfields", 15, 85)
    ]
    
    # Draw funnel using polygons
    for idx, (label, count, height, x) in enumerate(steps):
        # Draw step block
        rect = patches.Rectangle((x, 25 - height/2), 12, height, facecolor=LIGHT_BG, edgecolor=AZUL_UC3M, lw=1.5)
        ax.add_patch(rect)
        
        # Labels
        ax.text(x + 6, 25, label, fontsize=8, fontweight='bold', ha='center', va='center', color=AZUL_UC3M)
        ax.text(x + 6, 25 - height/2 - 2.5, count, fontsize=8, fontweight='bold', ha='center', color=ACCENT_RED)
        
        # Connectors between blocks
        if idx < 4:
            next_x = steps[idx+1][3]
            next_height = steps[idx+1][2]
            
            # Draw connector polygon
            verts = [
                (x + 12, 25 + height/2),
                (next_x, 25 + next_height/2),
                (next_x, 25 - next_height/2),
                (x + 12, 25 - height/2),
                (x + 12, 25 + height/2)
            ]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor=AZUL_UC3M, alpha=0.15, edgecolor=MUTED_LINE, lw=0.5)
            ax.add_patch(patch)
            
    ax.text(50, 45, "CORPUS FILTERING AND SELECTION ROADMAP", fontsize=13, fontweight='bold', ha='center', color=AZUL_UC3M)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_req_09_pipeline_flow.png", bbox_inches='tight')
    plt.close()

# ==========================================
# REQ 10: UMAP Atlas Parameter Card
# ==========================================
def generate_fig_req_10():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    
    # Outer frame
    rect = patches.FancyBboxPatch((1, 1), 98, 48, boxstyle="round,pad=1.0", facecolor=LIGHT_BG, edgecolor=AZUL_UC3M, lw=1.5)
    ax.add_patch(rect)
    
    # Headers
    ax.text(5, 43, "UMAP MAP INTERPRETATION CARD", fontsize=12, fontweight='bold', color=AZUL_UC3M)
    ax.text(5, 40, "Supplemental Guidelines for Reading the Subfield Visual Atlas (Appendix D)", fontsize=8.5, style='italic', color=TEXT_MUTED)
    
    # Left Column: Parameters Used
    ax.text(5, 33, "CORE UMAP SETTINGS", fontsize=9.5, fontweight='bold', color=AZUL_UC3M)
    params = [
        "• Metric: Cosine (aligned with SPECTER2 vectors)",
        "• n_neighbors: 30 (balances local & global structures)",
        "• min_dist: 0.1 (allows structural clump formation)",
        "• Input: 768-D row-aligned SPECTER2 embeddings",
        "• Subsampling: N = 2,500 papers per subfield panel"
    ]
    for idx, p in enumerate(params):
        ax.text(5, 29 - idx*4, p, fontsize=8, color=TEXT_DARK)
        
    # Right Column: Visual Legend and Guidelines
    ax.text(52, 33, "HOW TO READ THE ATLAS MAPS", fontsize=9.5, fontweight='bold', color=AZUL_UC3M)
    rules = [
        "1. Spacing: Closeness between dots represents semantic\n   proximity. Overlapping areas show shared language.",
        "2. Clusters: Isolated clusters show highly distinct, specialized\n   sub-topics within the subfield.",
        "3. Axes: UMAP axes have no absolute physical units.\n   Do not read absolute positions; interpret relative topology.",
        "4. Density: High packing density means cohesive research cores;\n   scattered peripheries represent multidisciplinary extensions."
    ]
    for idx, r in enumerate(rules):
        ax.text(52, 29 - idx*4.8, r, fontsize=8, color=TEXT_DARK)
        
    # Bottom warning box
    rect_warn = patches.Rectangle((3, 3), 94, 5, facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=0.8)
    ax.add_patch(rect_warn)
    ax.text(50, 5.5, "IMPORTANT CRITICAL WARNING: UMAP is for visual rendering only! All quantitative metrics (Standard Distance,\n"
            "Effective Dimension, Hubness, LDI) are computed in the original 768-D space, bypassing 2D projection distortion.",
            fontsize=7, color='#991B1B', fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_req_10_umap_card.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 11: Citation Networks vs. Semantic Spaces
# ==========================================
def generate_fig_exp_11():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    
    # Panel A: Citation Network
    ax1.set_facecolor(LIGHT_BG)
    np.random.seed(20)
    cx = np.random.normal(0, 1, 20)
    cy = np.random.normal(0, 1, 20)
    # Draw points
    ax1.scatter(cx, cy, color=AZUL_UC3M, s=50, zorder=5)
    # Draw random connections (network edges)
    for idx1 in range(20):
        for idx2 in range(idx1+1, 20):
            if np.random.rand() < 0.15:
                ax1.plot([cx[idx1], cx[idx2]], [cy[idx1], cy[idx2]], color=SLATE_GRAY, alpha=0.5, lw=1)
                
    apply_common_styling(ax1, title="A. CITATION GRAPH METRIC\n(Discrete, Topology Bound)", xlabel="Arbitrary Layout X", ylabel="Arbitrary Layout Y")
    ax1.text(0, -2.5, "Nodes represent papers; edges are links.\nHard boundary clusters; ignores non-citing similarities.", fontsize=8, color=TEXT_DARK, ha='center')
    
    # Panel B: Semantic continuous space
    ax2.set_facecolor(LIGHT_BG)
    np.random.seed(22)
    # Draw contour density field
    gx, gy = np.random.normal(0, 1.2, 200), np.random.normal(0, 1.2, 200)
    ax2.scatter(gx, gy, color=SLATE_GRAY, s=8, alpha=0.3)
    # Draw smooth density contours
    xy = np.vstack([gx, gy])
    z = gaussian_kde(xy)(xy)
    ax2.tricontour(gx, gy, z, levels=4, cmap='Blues', alpha=0.7)
    
    apply_common_styling(ax2, title="B. CONTINUOUS EMBEDDING SPACE\n(Continuous, Content Bound)", xlabel="Semantic Axis 1", ylabel="Semantic Axis 2")
    ax2.text(0, -2.5, "Every coordinate has semantic meaning.\nMeasures content similarity directly without citations.", fontsize=8, color=TEXT_DARK, ha='center')
    
    plt.suptitle("PARADIGM SHIFT: CITATION NETWORKS VS. dense EMBEDDING SPACES", fontsize=14, fontweight='bold', color=AZUL_UC3M, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_11_citation_vs_semantic.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 12: Corpus Coverage and Growth
# ==========================================
def generate_fig_exp_12():
    # Synthetic data matching thesis corpus characteristics
    years = np.arange(2000, 2025)
    np.random.seed(30)
    
    # Baseline growth with slight organic fluctuations
    phys = 15000 + 4000 * (years - 2000) + np.random.normal(0, 1000, 25)
    health = 20000 + 4500 * (years - 2000) + np.random.normal(0, 1200, 25)
    life = 12000 + 2500 * (years - 2000) + np.random.normal(0, 800, 25)
    social = 10000 + 3500 * (years - 2000) + np.random.normal(0, 900, 25)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(LIGHT_BG)
    
    # Draw stacked area chart
    ax.stackplot(years, phys, health, life, social, 
                 labels=['Physical Sciences', 'Health Sciences', 'Life Sciences', 'Social Sciences'],
                 colors=[AZUL_UC3M, ACCENT_RED, ACCENT_GREEN, SLATE_GRAY], alpha=0.85)
    
    apply_common_styling(ax, title="CORPUS LITERATURE GROWTH & DOMAIN COMPOSITION (2000-2024)",
                         xlabel="Publication Year", ylabel="Annual Paper Count")
    
    ax.legend(loc='upper left', frameon=True, facecolor=LIGHT_BG, edgecolor=MUTED_LINE)
    ax.text(2012, 100000, "Cumulative Corpus: 2.37M English Works", fontsize=10, fontweight='bold', color='#FFFFFF')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_12_corpus_growth.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 13: Discrete Metadata vs. Dense Semantic Embeddings
# ==========================================
def generate_fig_exp_13():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    
    # Draw box for discrete metadata
    rect1 = patches.FancyBboxPatch((3, 10), 42, 30, boxstyle="round,pad=1.0", facecolor=LIGHT_BG, edgecolor=SLATE_GRAY, lw=1.2)
    ax.add_patch(rect1)
    ax.text(24, 37, "DISCRETE KEYWORD METADATA", fontsize=9.5, fontweight='bold', color=SLATE_GRAY, ha='center')
    ax.text(24, 32, "Sensitive to word choices; fails on semantics", fontsize=7.5, style='italic', color=TEXT_MUTED, ha='center')
    
    # Examples
    ax.text(5, 25, "'Machine Learning'", fontsize=8, fontweight='bold', color='#B91C1C')
    ax.text(5, 20, "'Deep Learning'", fontsize=8, color='#B91C1C')
    ax.text(5, 15, "No overlap found if different terms used", fontsize=7.5, color=TEXT_DARK)
    
    ax.text(26, 25, "'Apple' (fruit)", fontsize=8, fontweight='bold', color='#B91C1C')
    ax.text(26, 20, "'Apple' (company)", fontsize=8, color='#B91C1C')
    ax.text(26, 15, "Incorrectly merged (polysemy)", fontsize=7.5, color=TEXT_DARK)
    
    # Draw box for dense embedding
    rect2 = patches.FancyBboxPatch((55, 10), 42, 30, boxstyle="round,pad=1.0", facecolor=LIGHT_BG, edgecolor=AZUL_UC3M, lw=1.5)
    ax.add_patch(rect2)
    ax.text(76, 37, "DENSE EMBEDDING SPACE (SPECTER2)", fontsize=9.5, fontweight='bold', color=AZUL_UC3M, ha='center')
    ax.text(76, 32, "Maps content to continuous 768-D coordinates", fontsize=7.5, style='italic', color=TEXT_MUTED, ha='center')
    
    # Examples
    ax.text(57, 23, "Maps synonyms to nearby vectors:\n• 'neural network models'\n• 'deep connectionist nets'", fontsize=8, color=AZUL_UC3M)
    ax.text(57, 14, "Separates polysemic words via context:\n• 'apple orchard' vs. 'apple stock index'", fontsize=8, color=AZUL_UC3M)
    
    ax.text(50, 45, "REPRESENTATIONAL FIDELITY: DISCRETE KEYWORDS VS. DENSE VECTORS", fontsize=12, fontweight='bold', ha='center', color=AZUL_UC3M)
    
    # Arrow comparison
    ax.annotate("", xy=(53, 25), xytext=(47, 25),
                arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=2.5, mutation_scale=15))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_13_metadata_vs_embeddings.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 14: Metric Robustness & Sample-Size Sensitivity
# ==========================================
def generate_fig_exp_14():
    # Synthetic data demonstrating sample-size stabilization
    n_samples = np.array([100, 250, 500, 1000, 2500, 5000, 10000])
    np.random.seed(40)
    
    # Simulate metric mean and variance shrinking with N
    std_dist_means = 2.4 - 0.1 * np.exp(-n_samples/1000)
    std_dist_stds = 0.3 / np.sqrt(n_samples / 100)
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(LIGHT_BG)
    
    ax.plot(n_samples, std_dist_means, color=AZUL_UC3M, marker='o', lw=2, label='Standard Distance (Global Scale)')
    ax.fill_between(n_samples, std_dist_means - 1.96*std_dist_stds, std_dist_size_high := std_dist_means + 1.96*std_dist_stds, 
                    color=AZUL_UC3M, alpha=0.15, label='95% Confidence Bounds')
    
    apply_common_styling(ax, title="METRIC ROBUSTNESS & SAMPLE-SIZE SENSITIVITY TESTING",
                         xlabel="Subsample Size (N papers per subfield)", ylabel="Metric Score / Scale Value")
    
    ax.set_xscale('log')
    ax.set_xticks(n_samples)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.axvline(2500, color=ACCENT_RED, linestyle='--', lw=1.5)
    ax.text(2600, 2.38, "Selected Baseline (N=2500)\nOptimum balance of stability\n& computation cost", fontsize=8.5, color=ACCENT_RED, fontweight='bold')
    
    ax.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_14_metric_sensitivity.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 15: Typical Domain Profilesradar Chart (Parallel coordinates)
# ==========================================
def generate_fig_exp_15():
    # Data showing normalized metrics (1 to 10 scale) for domains
    # Metrics: Dispersion, Local Density, Hubness, Spectral Dim, Temporal Speed
    domains_data = {
        'Physical Sciences': [8.5, 3.2, 4.0, 7.8, 5.2],
        'Health Sciences': [4.1, 8.4, 7.9, 3.5, 4.8],
        'Life Sciences': [5.8, 6.2, 5.5, 5.0, 5.9],
        'Social Sciences': [7.2, 4.5, 3.1, 8.2, 3.1]
    }
    
    metrics = ['Dispersion\n(SD)', 'Local Density\n(LDI)', 'Hubness\n(Top 1%)', 'Spectral Dim\n(Effective Dim)', 'Temporal Drift\n(Velocity)']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(LIGHT_BG)
    
    colors = [AZUL_UC3M, ACCENT_RED, ACCENT_GREEN, SLATE_GRAY]
    x = np.arange(len(metrics))
    
    for idx, (domain, profile) in enumerate(domains_data.items()):
        ax.plot(x, profile, marker='o', markersize=8, color=colors[idx], lw=2.5, label=domain)
        # Fill under curves for visual richness
        ax.fill_between(x, profile, alpha=0.04, color=colors[idx])
        
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5, fontweight='bold')
    ax.set_ylim(1, 10)
    
    apply_common_styling(ax, title="TYPICAL MORPHOLOGICAL PROFILES BY OPENALEX DOMAIN",
                         ylabel="Normalized Rank Scale (1 = Minimum; 10 = Maximum)")
    
    ax.legend(loc='upper right', frameon=True)
    
    # Add diagnostic notes
    ax.text(0.1, 1.5, "Physical: Dispersed & multidimensional", fontsize=8.5, color=AZUL_UC3M, fontweight='bold')
    ax.text(2.2, 8.8, "Health: Concentrated & hub-dominated", fontsize=8.5, color=ACCENT_RED, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_15_domain_radar.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 16: Trajectory Modes
# ==========================================
def generate_fig_exp_16():
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    
    modes = [
        ("A. Centroid Drift (Semantic Shift)", "Centroid coordinates move;\ninternal dispersion stays stable.", 'drift', ACCENT_RED),
        ("B. Expansion (Broadening Scope)", "Diverging outwards;\ndispersion increases over epochs.", 'expansion', AZUL_UC3M),
        ("C. Contraction (Densification)", "Converging inwards;\ndensity increases, scope narrows.", 'contraction', ACCENT_GREEN)
    ]
    
    for idx, (title, desc, mode, color) in enumerate(modes):
        ax = axs[idx]
        ax.set_facecolor(LIGHT_BG)
        apply_common_styling(ax, title=title)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
        if mode == 'drift':
            # Drift: shift center
            np.random.seed(50)
            p1_x, p1_y = np.random.normal(-2, 0.8, 30), np.random.normal(-1, 0.8, 30)
            p2_x, p2_y = np.random.normal(2, 0.8, 30), np.random.normal(1, 0.8, 30)
            ax.scatter(p1_x, p1_y, color=SLATE_GRAY, s=8, alpha=0.3, label='Epoch 1')
            ax.scatter(p2_x, p2_y, color=color, s=8, alpha=0.8, label='Epoch 2')
            # Draw movement vector for centroid
            ax.annotate("", xy=(2, 1), xytext=(-2, -1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=3, mutation_scale=15))
            ax.plot(-2, -1, marker='*', color='black', markersize=8)
            ax.plot(2, 1, marker='*', color=color, markersize=10)
        elif mode == 'expansion':
            # Expansion: increase dispersion
            np.random.seed(51)
            p1_x, p1_y = np.random.normal(0, 0.6, 40), np.random.normal(0, 0.6, 40)
            p2_x, p2_y = np.random.normal(0, 1.8, 40), np.random.normal(0, 1.8, 40)
            ax.scatter(p2_x, p2_y, color=color, s=8, alpha=0.5, label='Epoch 2')
            ax.scatter(p1_x, p1_y, color=SLATE_GRAY, s=8, alpha=0.7, label='Epoch 1')
            ax.add_patch(patches.Circle((0, 0), 0.6, fill=False, edgecolor='black', linestyle='--'))
            ax.add_patch(patches.Circle((0, 0), 1.8, fill=False, edgecolor=color, linestyle='--', lw=1.5))
        elif mode == 'contraction':
            # Contraction: shrink dispersion
            np.random.seed(52)
            p1_x, p1_y = np.random.normal(0, 1.8, 40), np.random.normal(0, 1.8, 40)
            p2_x, p2_y = np.random.normal(0, 0.6, 40), np.random.normal(0, 0.6, 40)
            ax.scatter(p1_x, p1_y, color=SLATE_GRAY, s=8, alpha=0.3, label='Epoch 1')
            ax.scatter(p2_x, p2_y, color=color, s=8, alpha=0.9, label='Epoch 2')
            ax.add_patch(patches.Circle((0, 0), 1.8, fill=False, edgecolor=SLATE_GRAY, linestyle='--'))
            ax.add_patch(patches.Circle((0, 0), 0.6, fill=False, edgecolor=color, linestyle='--', lw=1.5))
            
        ax.text(-4.5, -4.5, desc, fontsize=8, color=TEXT_DARK)
        
    plt.suptitle("THREE MODES OF MORPHOLOGICAL TRAJECTORY EVOLUTION", fontsize=14, fontweight='bold', color=AZUL_UC3M, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_16_trajectory_modes.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 17: Evolution of Metric Separability
# ==========================================
def generate_fig_exp_17():
    epochs = ['2000-2004', '2005-2009', '2010-2014', '2015-2019', '2020-2024']
    x = np.arange(len(epochs))
    
    # Average pairwise distance in profile space (increasing)
    avg_dist = [0.42, 0.44, 0.48, 0.53, 0.58]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(LIGHT_BG)
    
    ax.plot(x, avg_dist, marker='s', markersize=8, color=AZUL_UC3M, lw=2.5, label='Mean Morphological Distance')
    
    # Fit line
    z = np.polyfit(x, avg_dist, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), color=ACCENT_RED, linestyle='--', alpha=0.7, label='Linear Separation Trend')
    
    ax.set_xticks(x)
    ax.set_xticklabels(epochs, fontsize=9.5)
    ax.set_ylim(0.3, 0.7)
    
    apply_common_styling(ax, title="EVOLUTION OF METRIC SEPARABILITY IN PROFILE SPACE",
                         xlabel="Analysis Epoch Window", ylabel="Mean Pairwise Cosine Distance")
    
    ax.legend(loc='upper left', frameon=True)
    ax.text(2, 0.35, "Indicates overall profile divergence over time;\nscientific shapes are becoming more specialized.",
            fontsize=9, color=TEXT_DARK, ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFFFFF', edgecolor=MUTED_LINE))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_17_profile_separation.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 18: Clustering Hierarchy and Granularity
# ==========================================
def generate_fig_exp_18():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw custom schematic dendrogram to explain weak static clustering
    # Root
    ax.plot([5, 5], [9, 7], color=SLATE_GRAY, lw=2)
    # Split 1
    ax.plot([2.5, 7.5], [7, 7], color=SLATE_GRAY, lw=2)
    ax.plot([2.5, 2.5], [7, 5], color=SLATE_GRAY, lw=2)
    ax.plot([7.5, 7.5], [7, 5], color=SLATE_GRAY, lw=2)
    # Split 2 (Left)
    ax.plot([1.2, 3.8], [5, 5], color=SLATE_GRAY, lw=1.5)
    ax.plot([1.2, 1.2], [5, 3], color=AZUL_UC3M, lw=1.5)
    ax.plot([3.8, 3.8], [5, 3], color=ACCENT_RED, lw=1.5)
    # Split 3 (Right)
    ax.plot([6.2, 8.8], [5, 5], color=SLATE_GRAY, lw=1.5)
    ax.plot([6.2, 6.2], [5, 3], color=ACCENT_GREEN, lw=1.5)
    ax.plot([8.8, 8.8], [5, 3], color='purple', lw=1.5)
    
    # Individual leaves at bottom
    for leaf in [0.8, 1.6, 3.4, 4.2, 5.8, 6.6, 8.4, 9.2]:
        ax.plot([leaf, leaf], [2, 1], color=TEXT_MUTED, linestyle=':')
        
    ax.text(5, 9.3, "Morphological Typology HAC Hierarchy", fontsize=11, fontweight='bold', color=AZUL_UC3M, ha='center')
    
    # Add annotations about clustering heights
    ax.axhline(6.5, color=ACCENT_RED, linestyle='--', lw=1)
    ax.text(9.5, 6.6, "Weak Clustering Height\n(High Silhouette Overlap)", fontsize=7.5, color=ACCENT_RED, ha='right')
    
    ax.axhline(4.0, color=ACCENT_GREEN, linestyle='--', lw=1)
    ax.text(9.5, 4.1, "Clear Subfield Typologies\n(4 Core Clusters)", fontsize=7.5, color=ACCENT_GREEN, ha='right')
    
    # Label bottom groups
    ax.text(1.2, 2.5, "Class 1\n(Broad)", fontsize=8, ha='center', color=AZUL_UC3M, fontweight='bold')
    ax.text(3.8, 2.5, "Class 2\n(Dense)", fontsize=8, ha='center', color=ACCENT_RED, fontweight='bold')
    ax.text(6.2, 2.5, "Class 3\n(Dynamic)", fontsize=8, ha='center', color=ACCENT_GREEN, fontweight='bold')
    ax.text(8.8, 2.5, "Class 4\n(Stable)", fontsize=8, ha='center', color='purple', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_18_clustering_hierarchy.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 19: The Metric-Centroid-Domain Triangle
# ==========================================
def generate_fig_exp_19():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 10)
    ax.axis('off')
    
    # Triangle vertices
    v_tax = (5, 8)       # Taxonomy / Classifications
    v_loc = (1.5, 2)     # Semantic Centroid Coordinates
    v_mor = (8.5, 2)     # Morphological Internal Shape
    
    # Draw triangle edges
    ax.plot([v_tax[0], v_loc[0]], [v_tax[1], v_loc[1]], color=AZUL_UC3M, lw=2.5)
    ax.plot([v_tax[0], v_mor[0]], [v_tax[1], v_mor[1]], color=SLATE_GRAY, lw=1.5, linestyle='--')
    ax.plot([v_loc[0], v_mor[0]], [v_loc[1], v_mor[1]], color=SLATE_GRAY, lw=1.5, linestyle='--')
    
    # Vertex circles
    ax.scatter(*v_tax, color=AZUL_UC3M, s=250, zorder=5)
    ax.scatter(*v_loc, color=ACCENT_RED, s=250, zorder=5)
    ax.scatter(*v_mor, color=ACCENT_GREEN, s=250, zorder=5)
    
    # Labels
    ax.text(v_tax[0], v_tax[1] + 0.5, "A. TAXONOMY\n(Hierarchical Labels)", fontsize=9.5, fontweight='bold', color=AZUL_UC3M, ha='center')
    ax.text(v_loc[0] - 0.3, v_loc[1] - 0.7, "B. SEMANTIC LOCATION\n(Absolute Vector Coordinates)", fontsize=9.5, fontweight='bold', color=ACCENT_RED, ha='center')
    ax.text(v_mor[0] + 0.3, v_mor[1] - 0.7, "C. MORPHOLOGICAL SHAPE\n(Internal Scale/Density/Spectral)", fontsize=9.5, fontweight='bold', color=ACCENT_GREEN, ha='center')
    
    # Edge descriptions
    ax.text(3.0, 5.2, "HIGH MUTUAL INFO\nSemantic location\naligns closely with\ntaxonomy labels", 
            fontsize=8.5, color=AZUL_UC3M, fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFFF', edgecolor=MUTED_LINE))
            
    ax.text(7.0, 5.2, "ORTHOGONAL DIMENSION\nMorphology captures\nresearch organization\nand style, not domain", 
            fontsize=8.5, color=SLATE_GRAY, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFFF', edgecolor=MUTED_LINE))
            
    ax.text(5.0, 1.2, "INDEPENDENT INDICATORS\nWhere a field sits does not dictate its shape", 
            fontsize=8.5, color=SLATE_GRAY, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFFF', edgecolor=MUTED_LINE))
            
    ax.text(5, 9.3, "THE METRIC-CENTROID-DOMAIN RELATIONSHIP MATRIX", fontsize=12, fontweight='bold', color=AZUL_UC3M, ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_19_interpretation_triangle.png", bbox_inches='tight')
    plt.close()

# ==========================================
# EXP 20: Morphological Outlier Spectrum
# ==========================================
def generate_fig_exp_20():
    # Rank select subfields on their dispersion (Standard Distance) to show outliers
    subfields = [
        "Cardiology (Health)", 
        "Pediatrics (Health)", 
        "Artificial Intelligence (Phys)", 
        "General Engineering (Phys)", 
        "Applied Mathematics (Phys)",
        "Multidisciplinary (Misc)"
    ]
    sd_scores = [1.8, 2.3, 3.8, 4.9, 6.2, 7.8]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(LIGHT_BG)
    
    colors = [ACCENT_RED, ACCENT_RED, AZUL_UC3M, AZUL_UC3M, AZUL_UC3M, SLATE_GRAY]
    y_pos = np.arange(len(subfields))
    
    ax.barh(y_pos, sd_scores, color=colors, alpha=0.85, height=0.55, edgecolor=AZUL_UC3M, lw=0.8)
    
    # Write scores on bars
    for idx, val in enumerate(sd_scores):
        ax.text(val + 0.15, y_pos[idx], f"{val:.2f}", va='center', fontweight='bold', color=colors[idx])
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subfields, fontsize=9.5, fontweight='bold')
    ax.set_xlim(0, 9.0)
    
    apply_common_styling(ax, title="THE DISPERSION OUTLIER SPECTRUM (STANDARD DISTANCE)",
                         xlabel="Standard Distance Scale Value (768-D Space Metric)")
    
    ax.text(4.5, 0.5, "Highly Concentrated Core\n(Narrow, specialized topics)", fontsize=8.5, color=ACCENT_RED, fontweight='bold')
    ax.text(4.5, 4.5, "Highly Dispersed Frontier\n(Broad, multi-topic span)", fontsize=8.5, color=AZUL_UC3M, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_exp_20_outlier_spectrum.png", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating Required Figures...")
    generate_fig_req_01()
    generate_fig_req_02()
    generate_fig_req_03()
    generate_fig_req_04()
    generate_fig_req_05()
    generate_fig_req_06()
    generate_fig_req_07()
    generate_fig_req_08()
    generate_fig_req_09()
    generate_fig_req_10()
    print("Required Figures Generated Successfully.")
    
    print("Generating Additional Exploratory Figures...")
    generate_fig_exp_11()
    generate_fig_exp_12()
    generate_fig_exp_13()
    generate_fig_exp_14()
    generate_fig_exp_15()
    generate_fig_exp_16()
    generate_fig_exp_17()
    generate_fig_exp_18()
    generate_fig_exp_19()
    generate_fig_exp_20()
    print("Additional Exploratory Figures Generated Successfully.")
    print("All 20 high-resolution visual exploration assets are saved in 'tmp_visual_exploration/figures/'.")
