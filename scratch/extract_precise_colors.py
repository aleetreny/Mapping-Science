from PIL import Image

image_path = "memory/figures/fig_07_field_temporal_delta_heatmap.png"
img = Image.open(image_path)
img_rgb = img.convert("RGB")

x = 748
y_centers = {
    "Life Sciences": 250,
    "Social Sciences": 700,
    "Physical Sciences": 1300,
    "Health Sciences": 2000
}

print("Precise domain colors from the category strip:")
for domain, y in y_centers.items():
    r, g, b = img_rgb.getpixel((x, y))
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    print(f"  {domain:18s} : '{hex_color}' (RGB: {r}, {g}, {b})")

