from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.per_category_umap_runner import main as run_category_main


FIXED_LEVEL = "domain"


def main() -> None:
    run_category_main(fixed_level=FIXED_LEVEL)


if __name__ == "__main__":
    main()
