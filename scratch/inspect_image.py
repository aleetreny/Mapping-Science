from PIL import Image

image_path = "memory/figures/fig_08_field_static_distance_heatmap.png"
try:
    with Image.open(image_path) as img:
        print("Image Format:", img.format)
        print("Image Size:", img.size)
        print("Image Mode:", img.mode)
        if "dpi" in img.info:
            print("Image DPI:", img.info["dpi"])
except Exception as e:
    print(f"Error reading image: {e}")
