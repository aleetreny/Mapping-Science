import os
from pathlib import Path
import re

ROOT = Path("c:/Users/Z0058EYW/Workspace/TFM")
color_pattern = re.compile(r"#[0-9a-fA-F]{6}")

print("Searching for hex colors in python files...")
for py_file in ROOT.glob("**/*.py"):
    if ".venv" in py_file.parts or "__pycache__" in py_file.parts:
        continue
    try:
        with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            matches = color_pattern.findall(content)
            if matches:
                # Find if it mentions domains
                mentions_domain = any(d in content for d in ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"])
                if mentions_domain or "color" in py_file.name or "plot" in py_file.name or "umap" in py_file.name:
                    print(f"File: {py_file.relative_to(ROOT)}")
                    lines = content.splitlines()
                    for idx, line in enumerate(lines):
                        if any(m in line for m in matches):
                            print(f"  Line {idx+1}: {line.strip()}")
    except Exception as e:
        pass
