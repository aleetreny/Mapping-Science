from PIL import Image
import collections

image_path = "memory/figures/fig_07_field_temporal_delta_heatmap.png"
img = Image.open(image_path)
img_rgb = img.convert("RGB")
width, height = img.size

# The strip is on the left edge. Let's find it by scanning horizontally for each Y-range.
# We will look at x from 0 to 150.
# We want to find the x-range where there is a color with high std_dev.
y_centers = {
    "Life Sciences": 250,
    "Social Sciences": 700,
    "Physical Sciences": 1300,
    "Health Sciences": 2000
}

for domain, y in y_centers.items():
    # Let's count colors for x from 0 to 200 at this y
    row_colors = []
    for x in range(200):
        r, g, b = img_rgb.getpixel((x, y))
        mean_val = (r + g + b) / 3
        std_val = ((r - mean_val)**2 + (g - mean_val)**2 + (b - mean_val)**2)**0.5
        if std_val > 15 and mean_val < 240 and mean_val > 20:
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            row_colors.append((x, hex_color, r, g, b))
    
    # We expect a contiguous block of x coordinates for the strip.
    # Let's print out the midpoints of the strip and their colors.
    if row_colors:
        min_x = min(item[0] for item in row_colors)
        max_x = max(item[0] for item in row_colors)
        # Find the dominant color in the middle 50% of the horizontal strip width to avoid borders
        mid_start = min_x + (max_x - min_x) // 4
        mid_end = min_x + 3 * (max_x - min_x) // 4
        
        strip_pixels = []
        for x in range(mid_start, mid_end + 1):
            r, g, b = img_rgb.getpixel((x, y))
            strip_pixels.append((r, g, b))
            
        counter = collections.Counter(strip_pixels)
        most_common = counter.most_common(1)[0][0]
        hex_color = f"#{most_common[0]:02x}{most_common[1]:02x}{most_common[2]:02x}"
        print(f"Domain: {domain:18s} | X-range: {min_x:3d}-{max_x:3d} | Precise Color: {hex_color} | RGB: {most_common}")
    else:
        print(f"Domain: {domain:18s} | No strip detected at y={y}")

