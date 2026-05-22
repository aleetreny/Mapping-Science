from PIL import Image
import collections

image_path = "memory/figures/fig_07_field_temporal_delta_heatmap.png"
img = Image.open(image_path)
img_rgb = img.convert("RGB")
width, height = img.size

# Let's crop the domain strip.
# In Figure 7, the category strip is on the left.
# Let's find the horizontal region of this strip.
# We will scan across x for a few rows and find where the color is non-white and non-black.
strip_x_coords = []
for x in range(width // 4):
    colors = []
    for y in [height // 10, height // 3, height // 2, 2 * height // 3, 9 * height // 10]:
        r, g, b = img_rgb.getpixel((x, y))
        mean_val = (r + g + b) / 3
        std_val = ((r - mean_val)**2 + (g - mean_val)**2 + (b - mean_val)**2)**0.5
        if std_val > 15 and mean_val < 240 and mean_val > 20:
            colors.append(True)
    if len(colors) == 5: # All 5 check points are colorful!
        strip_x_coords.append(x)

if strip_x_coords:
    min_x, max_x = min(strip_x_coords), max(strip_x_coords)
    mid_x = (min_x + max_x) // 2
    print(f"Detected domain strip at X-range: {min_x} to {max_x}, using midpoint X={mid_x}")
    
    # Now let's divide the height into 4 domain sections and find the dominant color in each
    sections = [
        ("Life Sciences", 0.05, 0.2),
        ("Social Sciences", 0.25, 0.4),
        ("Physical Sciences", 0.45, 0.75),
        ("Health Sciences", 0.8, 0.95)
    ]
    
    for domain, y_start_frac, y_end_frac in sections:
        y_start = int(y_start_frac * height)
        y_end = int(y_end_frac * height)
        pixels = []
        # Sample all pixels in this section of the strip
        for y in range(y_start, y_end):
            for x in range(min_x + 2, max_x - 2):
                r, g, b = img_rgb.getpixel((x, y))
                pixels.append((r, g, b))
        
        # Find the most common pixel color
        counter = collections.Counter(pixels)
        most_common = counter.most_common(5)
        print(f"\nDomain: {domain}")
        for idx, ((r, g, b), count) in enumerate(most_common):
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            print(f"  Top {idx+1}: {hex_color} ({count} pixels) RGB: ({r}, {g}, {b})")
else:
    print("Could not automatically locate the domain strip. Let's do a fallback analysis.")

