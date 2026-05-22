import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# Set up paths
ROOT = Path("c:/Users/Z0058EYW/Workspace/TFM")
matrix_path = ROOT / "outputs/07_morphological_similarity/matrices/field_static_morphological_distance_matrix.csv"

# Actual targets in thesis memory directory
fig_png = ROOT / "memory/figures/fig_08_field_static_distance_heatmap.png"
fig_pdf = ROOT / "memory/figures/fig_08_field_static_distance_heatmap.pdf"

# Scratch validation target
scratch_png = ROOT / "scratch/test_heatmap.png"

# Read static distance matrix
matrix = pd.read_csv(matrix_path, index_col=0)

# The 26 fields and their corresponding domains, in the exact order shown in Figure 8.1
field_domain_mapping = {
    # Life Sciences
    'Agricultural and Biological Sciences': 'Life Sciences',
    'Biochemistry, Genetics and Molecular Biology': 'Life Sciences',
    'Immunology and Microbiology': 'Life Sciences',
    'Neuroscience': 'Life Sciences',
    'Pharmacology, Toxicology and Pharmaceutics': 'Life Sciences',
    
    # Social Sciences
    'Arts and Humanities': 'Social Sciences',
    'Business, Management and Accounting': 'Social Sciences',
    'Decision Sciences': 'Social Sciences',
    'Economics, Econometrics and Finance': 'Social Sciences',
    'Psychology': 'Social Sciences',
    'Social Sciences': 'Social Sciences',
    
    # Physical Sciences
    'Chemical Engineering': 'Physical Sciences',
    'Chemistry': 'Physical Sciences',
    'Computer Science': 'Physical Sciences',
    'Earth and Planetary Sciences': 'Physical Sciences',
    'Energy': 'Physical Sciences',
    'Engineering': 'Physical Sciences',
    'Environmental Science': 'Physical Sciences',
    'Materials Science': 'Physical Sciences',
    'Mathematics': 'Physical Sciences',
    'Physics and Astronomy': 'Physical Sciences',
    
    # Health Sciences
    'Dentistry': 'Health Sciences',
    'Health Professions': 'Health Sciences',
    'Medicine': 'Health Sciences',
    'Nursing': 'Health Sciences',
    'Veterinary': 'Health Sciences'
}

# Ordered fields as list
ordered_fields = list(field_domain_mapping.keys())

# Abbreviated display names for each field (from Figure 8.1)
abbreviations = {
    'Agricultural and Biological Sciences': 'Agric. & Bio.',
    'Biochemistry, Genetics and Molecular Biology': 'Biochem. & Gen.',
    'Immunology and Microbiology': 'Immunol. & Microbio.',
    'Neuroscience': 'Neuroscience',
    'Pharmacology, Toxicology and Pharmaceutics': 'Pharm. & Tox.',
    'Arts and Humanities': 'Arts & Hum.',
    'Business, Management and Accounting': 'Business & Mgmt.',
    'Decision Sciences': 'Decision Sci.',
    'Economics, Econometrics and Finance': 'Econ. & Finance',
    'Psychology': 'Psychology',
    'Social Sciences': 'Social Sci.',
    'Chemical Engineering': 'Chem. Eng.',
    'Chemistry': 'Chemistry',
    'Computer Science': 'Computer Sci.',
    'Earth and Planetary Sciences': 'Earth & Planet.',
    'Energy': 'Energy',
    'Engineering': 'Engineering',
    'Environmental Science': 'Environ. Sci.',
    'Materials Science': 'Materials Sci.',
    'Mathematics': 'Mathematics',
    'Physics and Astronomy': 'Physics & Astron.',
    'Dentistry': 'Dentistry',
    'Health Professions': 'Health Prof.',
    'Medicine': 'Medicine',
    'Nursing': 'Nursing',
    'Veterinary': 'Veterinary'
}

# Domain display names and their corresponding colors (harmonious, publication-ready palette)
domain_colors = {
    'Life Sciences': '#3fa45b',     # Green
    'Social Sciences': '#8c6bb1',   # Purple
    'Physical Sciences': '#c65a4a', # Rust Red
    'Health Sciences': '#4e86a6'    # Slate Blue
}

# Reorder the matrix to match the ordered fields
ordered_matrix = matrix.loc[ordered_fields, ordered_fields]

# Set diagonal values to 0.0 explicitly (representing zero distance to self)
values = ordered_matrix.to_numpy(dtype=float).copy()
np.fill_diagonal(values, 0.0)

# Create domain vectors for category strips
domain_list = [field_domain_mapping[field] for field in ordered_fields]
domain_to_int = {domain: idx for idx, domain in enumerate(domain_colors.keys())}
domain_ints = np.array([domain_to_int[d] for d in domain_list])

# Custom colormap for domain strips
domain_cmap = ListedColormap(list(domain_colors.values()))

# Create figure
fig = plt.subplots(figsize=(8, 8.2), dpi=300)
fig = plt.gcf()
fig.clf()

# Set up pixel-perfect coordinates for absolute layout
# Main Heatmap
heatmap_left = 0.26
heatmap_bottom = 0.22
heatmap_width = 0.58
heatmap_height = 0.58

# Left Category Strip
strip_thickness = 0.015
gap = 0.008
left_strip_left = heatmap_left - strip_thickness - gap

# Top Category Strip (now a thin discrete line matching Y-axis)
top_strip_bottom = heatmap_bottom + heatmap_height + gap

# Colorbar
cbar_gap = 0.025
cbar_width = 0.025
cbar_left = heatmap_left + heatmap_width + cbar_gap

# Legend Area (above top strip)
legend_bottom = top_strip_bottom + strip_thickness + 0.04
legend_height = 0.03

# Define all axes
ax_heatmap = fig.add_axes([heatmap_left, heatmap_bottom, heatmap_width, heatmap_height])
ax_left_strip = fig.add_axes([left_strip_left, heatmap_bottom, strip_thickness, heatmap_height])
ax_top_strip = fig.add_axes([heatmap_left, top_strip_bottom, heatmap_width, strip_thickness])
ax_cbar = fig.add_axes([cbar_left, heatmap_bottom, cbar_width, heatmap_height])
ax_legend = fig.add_axes([heatmap_left, legend_bottom, heatmap_width, legend_height])

# 1. Plot Main Heatmap
# Set vmin and vmax exactly as in prompt (0.0 to 4.5), using reverse viridis (viridis_r)
im_heatmap = ax_heatmap.imshow(values, aspect='auto', cmap='viridis_r', vmin=0.0, vmax=4.5)

# Style main heatmap grid
ax_heatmap.set_xticks(np.arange(len(ordered_fields)))
ax_heatmap.set_yticks(np.arange(len(ordered_fields)))

# Get short display names
display_labels = [abbreviations[field] for field in ordered_fields]

# Set tick labels with perfect rotated alignment to prevent overlap
ax_heatmap.set_xticklabels(display_labels, rotation=65, ha='right', rotation_mode='anchor', fontsize=7.5)
ax_heatmap.tick_params(axis='y', which='both', left=False, labelleft=False)

# Thin white grid lines separating cells
ax_heatmap.set_xticks(np.arange(len(ordered_fields)) - 0.5, minor=True)
ax_heatmap.set_yticks(np.arange(len(ordered_fields)) - 0.5, minor=True)
ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=0.25)
ax_heatmap.tick_params(which='minor', size=0)

# Draw thicker white line boundaries separating major domains
# Domain blocks end at indices:
# - Life Sciences: ends at 4 (between 4 and 5)
# - Social Sciences: ends at 10 (between 10 and 11)
# - Physical Sciences: ends at 20 (between 20 and 21)
domain_boundaries = [4.5, 10.5, 20.5]
for boundary in domain_boundaries:
    ax_heatmap.axhline(boundary, color='white', linewidth=1.2)
    ax_heatmap.axvline(boundary, color='white', linewidth=1.2)

# Adjust tick params
ax_heatmap.tick_params(axis='both', which='major', length=0, pad=4)

# Set outline border spine linewidth for heatmap
for spine in ax_heatmap.spines.values():
    spine.set_color('#111111')
    spine.set_linewidth(0.4)

# 2. Plot Left Category Strip (Vertical)
# Reshape domain_ints as column vector and enforce vmin/vmax to map indices correctly
left_strip_data = domain_ints.reshape(-1, 1)
ax_left_strip.imshow(left_strip_data, aspect='auto', cmap=domain_cmap, vmin=0, vmax=3)
ax_left_strip.set_xticks([])
ax_left_strip.set_yticks(np.arange(len(ordered_fields)))
ax_left_strip.set_yticklabels(display_labels, fontsize=7.5)
ax_left_strip.tick_params(axis='y', which='both', length=0, pad=6)

# Outline border spines for left strip to match main plot style
for spine in ax_left_strip.spines.values():
    spine.set_visible(True)
    spine.set_color('#111111')
    spine.set_linewidth(0.4)

# Draw white lines on left strip to match domain boundaries
for boundary in domain_boundaries:
    ax_left_strip.axhline(boundary, color='white', linewidth=1.2)

# 3. Plot Top Category Strip (Horizontal)
# Reshape domain_ints as row vector and enforce vmin/vmax to map indices correctly
top_strip_data = domain_ints.reshape(1, -1)
ax_top_strip.imshow(top_strip_data, aspect='auto', cmap=domain_cmap, vmin=0, vmax=3)
ax_top_strip.set_xticks([])
ax_top_strip.set_yticks([])

# Outline border spines for top strip to match main plot style
for spine in ax_top_strip.spines.values():
    spine.set_visible(True)
    spine.set_color('#111111')
    spine.set_linewidth(0.4)

# Draw white lines on top strip to match domain boundaries
for boundary in domain_boundaries:
    ax_top_strip.axvline(boundary, color='white', linewidth=1.2)

# 4. Colorbar
cbar = fig.colorbar(im_heatmap, cax=ax_cbar)
cbar.set_label("Euclidean profile distance", fontsize=9, labelpad=8)
cbar.ax.tick_params(labelsize=8)
# Set ticks every 0.5 from 0.0 to 4.5
cbar.set_ticks(np.arange(0.0, 4.6, 0.5))

# Outer border spine for colorbar
for spine in ax_cbar.spines.values():
    spine.set_color('#111111')
    spine.set_linewidth(0.4)

# 5. Legend
ax_legend.axis('off')
legend_patches = []
for domain, color in domain_colors.items():
    patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, label=domain)
    legend_patches.append(patch)

# Arrange the 4 items horizontally
ax_legend.legend(
    handles=legend_patches,
    loc='center',
    ncol=4,
    frameon=False,
    fontsize=8.5,
    handlelength=1.4,
    handleheight=0.9,
    columnspacing=1.8
)

# Create output directories if needed
fig_png.parent.mkdir(parents=True, exist_ok=True)

# Save high-res PNG and vector PDF for the thesis
plt.savefig(fig_png, dpi=300)
plt.savefig(fig_pdf, format='pdf')
plt.savefig(scratch_png, dpi=300)
plt.close()

print(f"Success! Regenerated and saved plots to:")
print(f"  PNG: {fig_png}")
print(f"  PDF: {fig_pdf}")
print(f"  Scratch validation copy: {scratch_png}")
