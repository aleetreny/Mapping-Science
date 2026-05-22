import collections
from PIL import Image

image_path = "memory/figures/fig_07_field_temporal_delta_heatmap.png"
img = Image.open(image_path)
width, height = img.size
print(f"Image dimensions: {width} x {height}")

# Convert to RGB if not already
img_rgb = img.convert("RGB")

# Let's sample colors in the left vertical strip.
# In Matplotlib plots, the left strip is usually located around 10-15% of the width.
# Let's scan from x = 0 to x = width // 3 and list the colors that have a distinct hue.
# A color is distinct if the standard deviation between R, G, B is high (not grey/white/black).
distinct_colors = []
for y in range(0, height, 5):
    row_colors = []
    for x in range(0, width // 3, 2):
        r, g, b = img_rgb.getpixel((x, y))
        # standard deviation to filter out greyscale (R=G=B) and white (all ~255) and black (all ~0)
        mean_val = (r + g + b) / 3
        std_val = ((r - mean_val)**2 + (g - mean_val)**2 + (b - mean_val)**2)**0.5
        if std_val > 15 and mean_val < 240 and mean_val > 20:
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            row_colors.append((hex_color, r, g, b))
    if row_colors:
        # Get the one with the highest std_val in the row, as it will be the most saturated (the strip color)
        best_color = max(row_colors, key=lambda item: ((item[1]-((item[1]+item[2]+item[3])/3))**2 + (item[2]-((item[1]+item[2]+item[3])/3))**2 + (item[3]-((item[1]+item[2]+item[3])/3))**2))
        distinct_colors.append((y, best_color[0]))

# Group consecutive rows with similar colors to identify the 4 major domain color blocks
print("\nSampled colorful pixels down the Y-axis:")
last_color = None
run_start = 0
runs = []
for y, c in distinct_colors:
    if last_color is None:
        last_color = c
        run_start = y
    elif c != last_color:
        # Check if color is close enough or different
        # Parse hex
        r1, g1, b1 = int(last_color[1:3], 16), int(last_color[3:5], 16), int(last_color[5:7], 16)
        r2, g2, b2 = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        dist = ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)**0.5
        if dist > 30: # clearly different color
            runs.append((run_start, y, last_color))
            last_color = c
            run_start = y
if last_color is not None:
    runs.append((run_start, height, last_color))

for start, end, color in runs:
    print(f"Y-range {start:4d} to {end:4d}: {color}")

