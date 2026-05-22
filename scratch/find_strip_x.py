from PIL import Image

image_path = "memory/figures/fig_07_field_temporal_delta_heatmap.png"
img = Image.open(image_path)
img_rgb = img.convert("RGB")
width, height = img.size

y = 250
colorful_xs = []
for x in range(width):
    r, g, b = img_rgb.getpixel((x, y))
    mean_val = (r + g + b) / 3
    std_val = ((r - mean_val)**2 + (g - mean_val)**2 + (b - mean_val)**2)**0.5
    # Look for colorful pixel
    if std_val > 15 and mean_val < 240:
        colorful_xs.append((x, (r, g, b)))

# Let's print out the colorful ranges
if colorful_xs:
    print(f"Total colorful pixels at y=250: {len(colorful_xs)}")
    print(f"First colorful x: {colorful_xs[0][0]}, RGB: {colorful_xs[0][1]}")
    print(f"Last colorful x: {colorful_xs[-1][0]}, RGB: {colorful_xs[-1][1]}")
    
    # Let's print out all continuous ranges of colorful pixels
    ranges = []
    start_x = colorful_xs[0][0]
    prev_x = start_x
    for x, rgb in colorful_xs[1:]:
        if x > prev_x + 2:
            ranges.append((start_x, prev_x))
            start_x = x
        prev_x = x
    ranges.append((start_x, prev_x))
    
    print("\nColorful X-ranges:")
    for idx, (start, end) in enumerate(ranges):
        # Sample mid pixel color in this range
        mid_x = (start + end) // 2
        r_mid, g_mid, b_mid = img_rgb.getpixel((mid_x, y))
        hex_color = f"#{r_mid:02x}{g_mid:02x}{b_mid:02x}"
        print(f"  Range {idx+1}: x={start:4d} to x={end:4d} | Midpoint x={mid_x:4d} color: {hex_color} RGB: ({r_mid},{g_mid},{b_mid})")
else:
    print("No colorful pixels found at y=250")

