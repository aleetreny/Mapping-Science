from PIL import Image

image_path = "memory/figures/fig_07_field_temporal_delta_heatmap.png"
img = Image.open(image_path)
img_rgb = img.convert("RGB")
width, height = img.size
print(f"Image dimensions: {width} x {height}")

# Let's print out the RGB values of the first 120 pixels along X at Y = 250, 700, 1300, 2000
for y in [250, 700, 1300, 2000]:
    print(f"\n--- Y = {y} ---")
    row_pixels = []
    for x in range(0, 150, 5):
        r, g, b = img_rgb.getpixel((x, y))
        row_pixels.append(f"x={x}:({r},{g},{b})")
    print(" ".join(row_pixels[:15]))
    print(" ".join(row_pixels[15:]))

