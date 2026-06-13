from __future__ import annotations

from pathlib import Path
from textwrap import shorten, wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "memory" / "figures" / "alternatives"

DOMAIN_COLORS = {
    "Life Sciences": "#2f7f5f",
    "Social Sciences": "#8b6bb5",
    "Physical Sciences": "#ba6a36",
    "Health Sciences": "#4c78a8",
}

ZONES = [
    {
        "zone": "Broad + concentrated",
        "count": 47,
        "reading": "wide territory, strong hubs",
        "breadth": "Broad",
        "concentration": "Concentrated",
        "accent": "#8a4f22",
        "fill": "#f3e5d4",
        "examples": [
            ("Philosophy", "Social Sciences"),
            ("Pharmaceutical Sci.", "Life Sciences"),
            ("Physical and Theoretical Chemistry", "Physical Sciences"),
            ("General Materials", "Physical Sciences"),
            ("Organic Chemistry", "Physical Sciences"),
        ],
    },
    {
        "zone": "Broad + smooth",
        "count": 64,
        "reading": "wide territory, weak hubs",
        "breadth": "Broad",
        "concentration": "Smooth",
        "accent": "#2f5f8f",
        "fill": "#e2eef7",
        "examples": [
            ("Artificial Intelligence", "Physical Sciences"),
            ("Molecular Biology", "Life Sciences"),
            ("Electrical Engineering", "Physical Sciences"),
            ("Biomedical Eng.", "Physical Sciences"),
            ("Cognitive Neuroscience", "Life Sciences"),
        ],
    },
    {
        "zone": "Compact + concentrated",
        "count": 77,
        "reading": "tight territory, strong hubs",
        "breadth": "Compact",
        "concentration": "Concentrated",
        "accent": "#6d4a80",
        "fill": "#eee3f2",
        "examples": [
            ("Applied Microbiology", "Life Sciences"),
            ("Molecular Medicine", "Life Sciences"),
            ("Toxicology", "Life Sciences"),
            ("Radiological and Ultrasound Technology", "Health Sciences"),
            ("Med. Lab Technology", "Health Sciences"),
        ],
    },
    {
        "zone": "Compact + smooth",
        "count": 53,
        "reading": "tight territory, weak hubs",
        "breadth": "Compact",
        "concentration": "Smooth",
        "accent": "#4f722e",
        "fill": "#e6f0dc",
        "examples": [
            ("Research & Theory", "Health Sciences"),
            ("Leadership and Management", "Health Sciences"),
            ("Business and International Management", "Social Sciences"),
            ("Aging", "Life Sciences"),
            ("Equine", "Health Sciences"),
        ],
    },
]

TABLE_SHORT_NAMES = {
    "Physical and Theoretical Chemistry": "Phys. & Theor. Chem.",
    "Radiological and Ultrasound Technology": "Radiology & Ultrasound",
    "Leadership and Management": "Leadership & Mgmt.",
    "Business and International Management": "Business & Intl. Mgmt.",
}


def setup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 140,
            "savefig.dpi": 280,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> Path:
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return png


def add_title(fig: plt.Figure, title: str, subtitle: str, *, color: str = "#141414") -> None:
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=17.5, weight="bold", color=color)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10.3, color="#4c4c4c")


def text_lines(name: str, max_width: int = 34) -> str:
    return "\n".join(wrap(name, width=max_width, break_long_words=False, break_on_hyphens=False))


def table_name(name: str) -> str:
    return TABLE_SHORT_NAMES.get(name, name)


def domain_legend(fig: plt.Figure, y: float = 0.035) -> None:
    x = 0.22
    for domain, color in DOMAIN_COLORS.items():
        fig.patches.append(Rectangle((x, y - 0.006), 0.012, 0.012, transform=fig.transFigure, facecolor=color, edgecolor="none"))
        fig.text(x + 0.016, y, domain, transform=fig.transFigure, ha="left", va="center", fontsize=7.8, color="#333333")
        x += 0.16


def card_examples(ax: plt.Axes, zone: dict, *, x: float, y0: float, row_gap: float, name_size: float, domain_size: float) -> None:
    for idx, (name, domain) in enumerate(zone["examples"]):
        y = y0 - idx * row_gap
        color = DOMAIN_COLORS[domain]
        ax.add_patch(Rectangle((x, y - 0.020), 0.020, 0.040, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(x + 0.033, y + 0.011, shorten(name, width=43, placeholder="..."), transform=ax.transAxes, ha="left", va="center", fontsize=name_size, weight="bold", color="#151515")
        ax.text(x + 0.033, y - 0.019, domain, transform=ax.transAxes, ha="left", va="center", fontsize=domain_size, color=color)


def variant_01_clean_matrix() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5))
    fig.patch.set_facecolor("#fbfaf7")
    add_title(
        fig,
        "Static morphology zones",
        "Four structural readings from semantic breadth and local concentration.",
    )

    for ax, zone in zip(axes.reshape(-1), ZONES):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.025, 0.035),
                0.95,
                0.88,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=ax.transAxes,
                facecolor="#ffffff",
                edgecolor="#d3d0c9",
                linewidth=1.0,
                path_effects=[pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.12), pe.Normal()],
            )
        )
        ax.add_patch(Rectangle((0.025, 0.895), 0.95, 0.020, transform=ax.transAxes, facecolor=zone["accent"], edgecolor="none"))
        ax.text(0.065, 0.825, zone["zone"], transform=ax.transAxes, ha="left", va="top", fontsize=15.5, weight="bold", color=zone["accent"])
        ax.text(0.065, 0.755, zone["reading"], transform=ax.transAxes, ha="left", va="top", fontsize=9.6, color="#383838")
        ax.text(0.875, 0.820, f"n={zone['count']}", transform=ax.transAxes, ha="right", va="top", fontsize=14.0, weight="bold", color="#222222")

        ax.add_patch(
            FancyBboxPatch(
                (0.065, 0.665),
                0.31,
                0.055,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                transform=ax.transAxes,
                facecolor=zone["fill"],
                edgecolor="none",
            )
        )
        ax.text(0.220, 0.692, zone["breadth"], transform=ax.transAxes, ha="center", va="center", fontsize=8.2, weight="bold", color=zone["accent"])
        ax.add_patch(
            FancyBboxPatch(
                (0.395, 0.665),
                0.38,
                0.055,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                transform=ax.transAxes,
                facecolor=zone["fill"],
                edgecolor="none",
            )
        )
        ax.text(0.585, 0.692, zone["concentration"], transform=ax.transAxes, ha="center", va="center", fontsize=8.2, weight="bold", color=zone["accent"])

        card_examples(ax, zone, x=0.070, y0=0.575, row_gap=0.105, name_size=9.3, domain_size=7.4)

    domain_legend(fig)
    fig.tight_layout(rect=[0.02, 0.055, 0.98, 0.885])
    return save(fig, "fig_06_static_morphology_zones_alt_01_clean_matrix")


def variant_02_axis_map() -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 8.2))
    fig.patch.set_facecolor("#f7f8f5")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_title(
        fig,
        "A two-axis map of static morphology",
        "Position, count and examples are read in one coordinate system.",
    )

    left, right, bottom, top = 0.085, 0.940, 0.115, 0.850
    mid_x, mid_y = (left + right) / 2, (bottom + top) / 2
    ax.add_patch(Rectangle((left, bottom), right - left, top - bottom, facecolor="#ffffff", edgecolor="#c7c7c7", linewidth=1.1))

    quadrants = {
        "Broad + concentrated": (mid_x, mid_y, right - mid_x, top - mid_y),
        "Broad + smooth": (mid_x, bottom, right - mid_x, mid_y - bottom),
        "Compact + concentrated": (left, mid_y, mid_x - left, top - mid_y),
        "Compact + smooth": (left, bottom, mid_x - left, mid_y - bottom),
    }
    for zone in ZONES:
        x, y, w, h = quadrants[zone["zone"]]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=zone["fill"], edgecolor="none", alpha=0.46))

    ax.plot([mid_x, mid_x], [bottom, top], color="#474747", linewidth=1.0)
    ax.plot([left, right], [mid_y, mid_y], color="#474747", linewidth=1.0)
    ax.add_patch(FancyArrowPatch((left, bottom - 0.030), (right, bottom - 0.030), arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#2d2d2d"))
    ax.add_patch(FancyArrowPatch((left - 0.030, bottom), (left - 0.030, top), arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#2d2d2d"))
    ax.text(mid_x, bottom - 0.067, "Semantic breadth: compact -> broad", ha="center", va="center", fontsize=9.6, color="#222222")
    ax.text(left - 0.068, mid_y, "Local concentration: smooth -> hub-dominated", ha="center", va="center", rotation=90, fontsize=9.6, color="#222222")

    positions = {
        "Broad + concentrated": (0.708, 0.675),
        "Broad + smooth": (0.708, 0.302),
        "Compact + concentrated": (0.282, 0.675),
        "Compact + smooth": (0.282, 0.302),
    }
    for zone in ZONES:
        cx, cy = positions[zone["zone"]]
        ax.scatter([cx], [cy], s=1150, color=zone["accent"], alpha=0.95, zorder=4)
        ax.text(cx, cy, str(zone["count"]), ha="center", va="center", fontsize=18, weight="bold", color="#ffffff", zorder=5)
        ax.text(cx, cy + 0.110, zone["zone"], ha="center", va="bottom", fontsize=14.2, weight="bold", color=zone["accent"])
        ax.text(cx, cy + 0.078, zone["reading"], ha="center", va="bottom", fontsize=8.9, color="#333333")
        examples = [shorten(item[0], width=31, placeholder="...") for item in zone["examples"][:4]]
        ax.text(cx, cy - 0.092, "\n".join(examples), ha="center", va="top", fontsize=8.7, color="#1f1f1f", linespacing=1.45)

    domain_legend(fig, y=0.035)
    return save(fig, "fig_06_static_morphology_zones_alt_02_axis_map")


def variant_03_editorial_cards() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.7))
    fig.patch.set_facecolor("#f4f1ea")
    add_title(
        fig,
        "Four named morphology profiles",
        "Each profile keeps the structural claim visible before the examples.",
        color="#171717",
    )

    for idx, (ax, zone) in enumerate(zip(axes.reshape(-1), ZONES), start=1):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.035, 0.030),
                0.93,
                0.89,
                boxstyle="round,pad=0.012,rounding_size=0.015",
                transform=ax.transAxes,
                facecolor="#ffffff",
                edgecolor="#d7d0c3",
                linewidth=0.9,
                path_effects=[pe.SimplePatchShadow(offset=(2.0, -2.0), alpha=0.12), pe.Normal()],
            )
        )
        ax.add_patch(Rectangle((0.035, 0.030), 0.075, 0.89, transform=ax.transAxes, facecolor=zone["accent"], edgecolor="none"))
        ax.text(0.072, 0.855, f"{idx:02d}", transform=ax.transAxes, ha="center", va="top", fontsize=12.5, weight="bold", color="#ffffff")
        ax.text(0.145, 0.840, zone["zone"], transform=ax.transAxes, ha="left", va="top", fontsize=14.7, weight="bold", color="#151515")
        ax.text(0.145, 0.780, zone["reading"], transform=ax.transAxes, ha="left", va="top", fontsize=9.3, color="#555555")

        ax.add_patch(
            FancyBboxPatch(
                (0.735, 0.775),
                0.145,
                0.070,
                boxstyle="round,pad=0.006,rounding_size=0.010",
                transform=ax.transAxes,
                facecolor=zone["fill"],
                edgecolor=zone["accent"],
                linewidth=0.7,
            )
        )
        ax.text(0.807, 0.809, f"n={zone['count']}", transform=ax.transAxes, ha="center", va="center", fontsize=11.7, weight="bold", color=zone["accent"])

        ax.text(0.145, 0.675, "Extreme named subfields", transform=ax.transAxes, ha="left", va="center", fontsize=8.0, weight="bold", color=zone["accent"])
        for item_idx, (name, domain) in enumerate(zone["examples"]):
            y = 0.590 - item_idx * 0.092
            color = DOMAIN_COLORS[domain]
            ax.add_patch(Rectangle((0.145, y - 0.018), 0.018, 0.036, transform=ax.transAxes, facecolor=color, edgecolor="none"))
            ax.text(0.180, y + 0.010, shorten(name, width=42, placeholder="..."), transform=ax.transAxes, ha="left", va="center", fontsize=9.0, weight="bold", color="#171717")
            ax.text(0.180, y - 0.018, domain, transform=ax.transAxes, ha="left", va="center", fontsize=7.0, color=color)

        ax.add_patch(Rectangle((0.145, 0.100), 0.710, 0.010, transform=ax.transAxes, facecolor=zone["fill"], edgecolor="none"))

    domain_legend(fig, y=0.035)
    fig.tight_layout(rect=[0.02, 0.055, 0.98, 0.885])
    return save(fig, "fig_06_static_morphology_zones_alt_03_editorial_cards")


def variant_04_compact_table() -> Path:
    fig, ax = plt.subplots(figsize=(12.8, 7.3))
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_title(
        fig,
        "Static morphology zones as a compact lookup table",
        "The dense version emphasizes comparability and uses color only as a guide.",
    )

    top = 0.825
    row_h = 0.155
    x_zone, x_reading, x_count, x_examples = 0.070, 0.345, 0.555, 0.640

    ax.text(x_zone, top + 0.045, "Zone", ha="left", va="center", fontsize=8.8, weight="bold", color="#333333")
    ax.text(x_reading, top + 0.045, "Structural reading", ha="left", va="center", fontsize=8.8, weight="bold", color="#333333")
    ax.text(x_count, top + 0.045, "n", ha="center", va="center", fontsize=8.8, weight="bold", color="#333333")
    ax.text(x_examples, top + 0.045, "Extreme examples", ha="left", va="center", fontsize=8.8, weight="bold", color="#333333")
    ax.plot([0.055, 0.945], [top + 0.017, top + 0.017], color="#222222", linewidth=1.0)

    for idx, zone in enumerate(ZONES):
        y = top - idx * row_h
        ax.add_patch(Rectangle((0.055, y - row_h + 0.010), 0.890, row_h - 0.016, facecolor=zone["fill"], edgecolor="none", alpha=0.54))
        ax.add_patch(Rectangle((0.055, y - row_h + 0.010), 0.010, row_h - 0.016, facecolor=zone["accent"], edgecolor="none"))

        ax.text(x_zone, y - 0.034, zone["zone"], ha="left", va="center", fontsize=12.2, weight="bold", color=zone["accent"])
        ax.text(x_zone, y - 0.081, f"{zone['breadth']} / {zone['concentration']}", ha="left", va="center", fontsize=8.2, color="#333333")

        ax.text(x_reading, y - 0.054, zone["reading"], ha="left", va="center", fontsize=10.0, color="#1f1f1f")
        ax.text(x_count, y - 0.054, str(zone["count"]), ha="center", va="center", fontsize=15.4, weight="bold", color="#222222")

        for ex_idx, (name, domain) in enumerate(zone["examples"]):
            line_y = y - 0.024 - ex_idx * 0.025
            col_x = x_examples
            color = DOMAIN_COLORS[domain]
            ax.add_patch(Rectangle((col_x, line_y - 0.011), 0.010, 0.022, facecolor=color, edgecolor="none"))
            ax.text(
                col_x + 0.016,
                line_y,
                shorten(table_name(name), width=37, placeholder="..."),
                ha="left",
                va="center",
                fontsize=7.7,
                weight="bold",
                color="#1f1f1f",
            )

    domain_legend(fig, y=0.040)
    return save(fig, "fig_06_static_morphology_zones_alt_04_compact_table")


def variant_05_domain_only_cards() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("The four easiest-to-read static morphology zones", fontsize=17, weight="bold", y=0.99, color="#111111")
    fig.text(
        0.5,
        0.94,
        "Each card shows the most extreme named subfields inside one simple structural quadrant.",
        ha="center",
        fontsize=10.8,
        color="#444444",
    )

    for ax, zone in zip(axes.reshape(-1), ZONES):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.018,rounding_size=0.030",
                transform=ax.transAxes,
                facecolor="#fbfbfa",
                edgecolor="#c9c7c2",
                linewidth=1.35,
                alpha=1.0,
            )
        )
        ax.text(0.06, 0.91, zone["zone"], transform=ax.transAxes, fontsize=17, weight="bold", color="#1d1d1b", va="top")
        ax.text(0.06, 0.815, zone["reading"], transform=ax.transAxes, fontsize=11.2, color="#424242", va="top")
        ax.text(0.88, 0.885, f"n={zone['count']}", transform=ax.transAxes, fontsize=15, weight="bold", color="#2f2f2f", ha="right", va="top")

        y = 0.690
        for name, domain in zone["examples"]:
            color = DOMAIN_COLORS[domain]
            ax.add_patch(Rectangle((0.06, y - 0.031), 0.028, 0.062, transform=ax.transAxes, facecolor=color, edgecolor="none"))
            ax.text(
                0.105,
                y + 0.017,
                shorten(name, 40, placeholder="..."),
                transform=ax.transAxes,
                fontsize=10.8,
                color="#111111",
                va="center",
                weight="bold",
            )
            ax.text(0.105, y - 0.033, domain, transform=ax.transAxes, fontsize=9.0, color=color, va="center")
            y -= 0.123

    domain_legend(fig, y=0.030)
    fig.tight_layout(rect=[0, 0.050, 1, 0.91])
    return save(fig, "fig_06_static_morphology_zones_alt_05_domain_only_cards")


def variant_06_group_only_cards() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("The four easiest-to-read static morphology zones", fontsize=17, weight="bold", y=0.99, color="#111111")
    fig.text(
        0.5,
        0.94,
        "Each card shows the most extreme named subfields inside one simple structural quadrant.",
        ha="center",
        fontsize=10.8,
        color="#444444",
    )

    for ax, zone in zip(axes.reshape(-1), ZONES):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.018,rounding_size=0.030",
                transform=ax.transAxes,
                facecolor=zone["fill"],
                edgecolor=zone["accent"],
                linewidth=1.40,
                alpha=0.96,
            )
        )
        ax.text(0.06, 0.91, zone["zone"], transform=ax.transAxes, fontsize=17, weight="bold", color=zone["accent"], va="top")
        ax.text(0.06, 0.815, zone["reading"], transform=ax.transAxes, fontsize=11.2, color="#333333", va="top")
        ax.text(0.88, 0.885, f"n={zone['count']}", transform=ax.transAxes, fontsize=15, weight="bold", color=zone["accent"], ha="right", va="top")

        y = 0.690
        for name, domain in zone["examples"]:
            ax.add_patch(Rectangle((0.06, y - 0.031), 0.028, 0.062, transform=ax.transAxes, facecolor=zone["accent"], edgecolor="none", alpha=0.90))
            ax.text(
                0.105,
                y + 0.017,
                shorten(name, 40, placeholder="..."),
                transform=ax.transAxes,
                fontsize=10.8,
                color="#111111",
                va="center",
                weight="bold",
            )
            ax.text(0.105, y - 0.033, domain, transform=ax.transAxes, fontsize=9.0, color="#555555", va="center")
            y -= 0.123

    fig.tight_layout(rect=[0, 0.020, 1, 0.91])
    return save(fig, "fig_06_static_morphology_zones_alt_06_group_only_cards")


def build_contact_sheet(paths: list[Path]) -> Path:
    thumbs: list[Image.Image] = []
    labels = [
        "Alt 01 - clean matrix",
        "Alt 02 - axis map",
        "Alt 03 - editorial cards",
        "Alt 04 - compact table",
        "Alt 05 - domain only cards",
        "Alt 06 - group only cards",
    ]
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((980, 680), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())

    pad = 36
    label_h = 34
    cell_w = max(img.width for img in thumbs) + pad * 2
    cell_h = max(img.height for img in thumbs) + pad * 2 + label_h
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), "#f5f5f2")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for i, (image, label) in enumerate(zip(thumbs, labels)):
        col = i % cols
        row = i // cols
        x0 = col * cell_w
        y0 = row * cell_h
        draw.text((x0 + pad, y0 + 16), label, fill="#222222", font=font)
        x = x0 + (cell_w - image.width) // 2
        y = y0 + pad + label_h
        sheet.paste(image, (x, y))

    out = OUTPUT_DIR / "fig_06_static_morphology_zones_alternatives_contact.png"
    sheet.save(out)
    return out


def main() -> None:
    setup()
    paths = [
        variant_01_clean_matrix(),
        variant_02_axis_map(),
        variant_03_editorial_cards(),
        variant_04_compact_table(),
        variant_05_domain_only_cards(),
        variant_06_group_only_cards(),
    ]
    contact = build_contact_sheet(paths)
    print("Generated:")
    for path in paths + [contact]:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
