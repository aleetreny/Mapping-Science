from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "10_global_map_candidates" / "neighbor_variants"
FIGURE_DIR = ROOT / "memory" / "figures"

DOMAIN_ORDER = [
    "Life Sciences",
    "Social Sciences",
    "Physical Sciences",
    "Health Sciences",
]

DOMAIN_COLORS = {
    "Life Sciences": "#2f7f5f",
    "Social Sciences": "#8b6bb5",
    "Physical Sciences": "#ba6a36",
    "Health Sciences": "#4c78a8",
}

FIELD_LABELS = {
    "Agricultural and Biological Sciences": "Agric. & Bio.",
    "Arts and Humanities": "Arts & Hum.",
    "Biochemistry, Genetics and Molecular Biology": "Biochem. & Gen.",
    "Business, Management and Accounting": "Business & Mgmt.",
    "Chemical Engineering": "Chem. Eng.",
    "Chemistry": "Chemistry",
    "Computer Science": "Computer Sci.",
    "Decision Sciences": "Decision Sci.",
    "Dentistry": "Dentistry",
    "Earth and Planetary Sciences": "Earth & Planet.",
    "Economics, Econometrics and Finance": "Econ. & Finance",
    "Energy": "Energy",
    "Engineering": "Engineering",
    "Environmental Science": "Environ. Sci.",
    "Health Professions": "Health Prof.",
    "Immunology and Microbiology": "Immunol. & Microbio.",
    "Materials Science": "Materials Sci.",
    "Mathematics": "Mathematics",
    "Medicine": "Medicine",
    "Neuroscience": "Neuroscience",
    "Nursing": "Nursing",
    "Pharmacology, Toxicology and Pharmaceutics": "Pharm. & Tox.",
    "Physics and Astronomy": "Physics & Astron.",
    "Psychology": "Psychology",
    "Social Sciences": "Social Sci.",
    "Veterinary": "Veterinary",
}


def _configure_matplotlib() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    for folder in (OUTPUT_DIR, FIGURE_DIR):
        fig.savefig(folder / f"{stem}.png", bbox_inches="tight")
        fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_neighbor_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(ROOT / "outputs" / "09_morphological_typologies" / "subfield_typology_assignments.csv")
    fields = (
        assignments.groupby(["field_id", "field_display_name", "domain_display_name"], as_index=False)
        .agg(n_subfields=("subfield_id", "nunique"))
        .sort_values(["domain_display_name", "field_display_name"])
    )
    fields["field_id"] = fields["field_id"].astype(str)

    distances = pd.read_parquet(ROOT / "data" / "processed" / "temporal" / "morphological_pair_distances_static.parquet")
    distances = distances[
        (distances["metric_set"] == "static_full_reduced")
        & (distances["level"] == "field")
        & (distances["distance_metric"] == "euclidean")
    ].copy()

    field_names = sorted(fields["field_display_name"].tolist())
    matrix = pd.DataFrame(0.0, index=field_names, columns=field_names)
    for row in distances.itertuples(index=False):
        a = str(row.entity_display_name_a)
        b = str(row.entity_display_name_b)
        if a not in matrix.index or b not in matrix.columns:
            continue
        matrix.loc[a, b] = float(row.morphological_distance)
        matrix.loc[b, a] = float(row.morphological_distance)

    domain_by_field = dict(zip(fields["field_display_name"], fields["domain_display_name"]))
    size_by_field = dict(zip(fields["field_display_name"], fields["n_subfields"]))
    nearest_rows = []
    for field in field_names:
        row = matrix.loc[field].copy()
        row.loc[field] = np.inf
        neighbor = str(row.idxmin())
        nearest_rows.append(
            {
                "field": field,
                "nearest_field": neighbor,
                "distance": float(row.loc[neighbor]),
                "domain": domain_by_field[field],
                "nearest_domain": domain_by_field[neighbor],
                "cross_domain": domain_by_field[field] != domain_by_field[neighbor],
                "n_subfields": int(size_by_field[field]),
            }
        )
    nearest = pd.DataFrame(nearest_rows)
    nearest.to_csv(OUTPUT_DIR / "field_nearest_morphological_neighbors.csv", index=False)
    return fields, matrix, nearest


def _pcoa(distance: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float]]:
    labels = list(distance.index)
    d = distance.to_numpy(dtype=float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d**2) @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = eigvals > 1e-12
    total = eigvals[positive].sum()
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    shares = (
        float(eigvals[0] / total) if total > 0 else 0.0,
        float(eigvals[1] / total) if total > 0 else 0.0,
    )
    return pd.DataFrame(coords, index=labels, columns=["x", "y"]), shares


def _node_size(n_subfields: int) -> float:
    return 92 + n_subfields * 19


def _legend_handles(cross_count: int, n_fields: int) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DOMAIN_COLORS[d], markeredgecolor="white", markersize=9, label=d)
        for d in DOMAIN_ORDER
    ] + [
        Line2D([0], [0], color="#111111", linewidth=1.8, label=f"Cross-domain nearest neighbor ({cross_count}/{n_fields})"),
        Line2D([0], [0], color="#858585", linewidth=1.3, label="Within-domain nearest neighbor"),
    ]


def variant_01_clean_pcoa(fields: pd.DataFrame, matrix: pd.DataFrame, nearest: pd.DataFrame) -> None:
    coords, shares = _pcoa(matrix)
    frame = fields.set_index("field_display_name").join(coords).reset_index()
    domain_by_field = dict(zip(frame["field_display_name"], frame["domain_display_name"]))
    size_by_field = dict(zip(frame["field_display_name"], frame["n_subfields"]))

    fig, ax = plt.subplots(figsize=(11.2, 8.0))
    ax.grid(color="#e8e8e8", linewidth=0.55, alpha=0.8)

    for row in nearest.itertuples(index=False):
        a = coords.loc[row.field]
        b = coords.loc[row.nearest_field]
        cross = bool(row.cross_domain)
        arrow = FancyArrowPatch(
            (a.x, a.y),
            (b.x, b.y),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.25 if cross else 0.8,
            color="#111111" if cross else "#b8b8b8",
            alpha=0.62 if cross else 0.38,
            connectionstyle="arc3,rad=0.07",
            zorder=1,
        )
        ax.add_patch(arrow)

    for domain in DOMAIN_ORDER:
        subset = frame[frame["domain_display_name"] == domain]
        ax.scatter(
            subset["x"],
            subset["y"],
            s=[_node_size(int(v)) for v in subset["n_subfields"]],
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
            zorder=3,
        )

    for row in frame.itertuples(index=False):
        label = FIELD_LABELS.get(row.field_display_name, row.field_display_name)
        ax.text(row.x + 0.04, row.y + 0.035, label, fontsize=8.1, color="#222222", zorder=4)

    cross_count = int(nearest["cross_domain"].sum())
    ax.set_title("Nearest morphological neighbors between fields", fontsize=15.5, pad=13)
    ax.set_xlabel(f"Morphology PCoA 1 ({shares[0] * 100:.1f}% of positive-axis inertia)")
    ax.set_ylabel(f"Morphology PCoA 2 ({shares[1] * 100:.1f}% of positive-axis inertia)")
    ax.legend(handles=_legend_handles(cross_count, len(nearest)), loc="upper right", frameon=True, framealpha=0.92, fontsize=8.6)
    _save(fig, "fig_08_field_neighbor_variant_01_clean_pcoa")


def variant_02_domain_ring(fields: pd.DataFrame, nearest: pd.DataFrame) -> None:
    ordered = []
    for domain in DOMAIN_ORDER:
        subset = fields[fields["domain_display_name"] == domain].sort_values("field_display_name")
        ordered.extend(subset["field_display_name"].tolist())

    domain_ranges = {
        "Life Sciences": (118, 190),
        "Social Sciences": (200, 278),
        "Physical Sciences": (288, 48),
        "Health Sciences": (58, 108),
    }
    positions: dict[str, tuple[float, float]] = {}
    angles: dict[str, float] = {}
    for domain in DOMAIN_ORDER:
        subset = fields[fields["domain_display_name"] == domain].sort_values("field_display_name")
        start, end = domain_ranges[domain]
        if end < start:
            end += 360
        angle_values = np.linspace(start, end, len(subset), endpoint=True)
        for angle_deg, field in zip(angle_values, subset["field_display_name"]):
            angle = np.deg2rad(angle_deg % 360)
            positions[field] = (cos(angle), sin(angle))
            angles[field] = angle

    size_by_field = dict(zip(fields["field_display_name"], fields["n_subfields"]))
    domain_by_field = dict(zip(fields["field_display_name"], fields["domain_display_name"]))

    fig, ax = plt.subplots(figsize=(14.8, 10.8))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.84, bottom=0.22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.34, 1.34)
    ax.set_ylim(-1.28, 1.28)

    # Pale domain sectors as context.
    for domain in DOMAIN_ORDER:
        xs = [positions[f][0] for f, d in domain_by_field.items() if d == domain]
        ys = [positions[f][1] for f, d in domain_by_field.items() if d == domain]
        ax.scatter(xs, ys, s=1, color=DOMAIN_COLORS[domain], alpha=0.1)

    for row in nearest.itertuples(index=False):
        a = positions[row.field]
        b = positions[row.nearest_field]
        cross = bool(row.cross_domain)
        rad = 0.24 if cross else 0.12
        arrow_kwargs = {
            "arrowstyle": "-|>",
            "mutation_scale": 14 if cross else 11,
            "linewidth": 1.15 if cross else 0.9,
            "color": "#1f1f1f" if cross else "#858585",
            "alpha": 0.60 if cross else 0.55,
            "connectionstyle": f"arc3,rad={rad}",
            "shrinkA": 10,
            "shrinkB": 15,
            "zorder": 2,
        }
        arrow = FancyArrowPatch(a, b, **arrow_kwargs)
        ax.add_patch(arrow)

    for field, (x, y) in positions.items():
        domain = domain_by_field[field]
        ax.scatter(
            [x],
            [y],
            s=_node_size(int(size_by_field[field])),
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        angle = angles[field]
        lx, ly = 1.13 * cos(angle), 1.13 * sin(angle)
        ha = "left" if lx >= 0 else "right"
        ax.text(lx, ly, FIELD_LABELS.get(field, field), fontsize=7.6, ha=ha, va="center", color="#222222")

    for domain in DOMAIN_ORDER:
        xs = np.array([positions[f][0] for f, d in domain_by_field.items() if d == domain])
        ys = np.array([positions[f][1] for f, d in domain_by_field.items() if d == domain])
        ax.text(xs.mean() * 0.58, ys.mean() * 0.58, domain.replace(" ", "\n"), ha="center", va="center", fontsize=11, color=DOMAIN_COLORS[domain], weight="bold")

    cross_count = int(nearest["cross_domain"].sum())
    ax.set_title(
        f"Nearest-neighbor ring map: {cross_count}/26 fields link outside their domain",
        fontsize=15.5,
        pad=34,
    )
    legend_handles = _legend_handles(cross_count, len(nearest))
    domain_legend = ax.legend(
        handles=legend_handles[: len(DOMAIN_ORDER)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=len(DOMAIN_ORDER),
        frameon=False,
        fontsize=9.6,
        borderaxespad=0.0,
        columnspacing=2.1,
        handletextpad=0.55,
    )
    ax.add_artist(domain_legend)
    ax.legend(
        handles=legend_handles[len(DOMAIN_ORDER) :],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=9.6,
        borderaxespad=0.0,
        columnspacing=2.4,
        handlelength=2.8,
        handletextpad=0.6,
    )
    _save(fig, "fig_08_field_neighbor_variant_02_domain_ring")


def variant_03_domain_flow(fields: pd.DataFrame, nearest: pd.DataFrame) -> None:
    domain_by_field = dict(zip(fields["field_display_name"], fields["domain_display_name"]))
    size_by_field = dict(zip(fields["field_display_name"], fields["n_subfields"]))

    y_positions: dict[tuple[str, str], float] = {}
    domain_centers: dict[str, float] = {}
    cursor = 0.0
    gap = 1.8
    for domain in DOMAIN_ORDER:
        subset = fields[fields["domain_display_name"] == domain].sort_values("field_display_name")
        ys = np.arange(cursor, cursor + len(subset), dtype=float)
        center = float(ys.mean())
        domain_centers[domain] = center
        for field, y in zip(subset["field_display_name"], ys):
            y_positions[("left", field)] = float(y)
            y_positions[("right", field)] = float(y)
        cursor += len(subset) + gap

    fig, ax = plt.subplots(figsize=(12.5, 9.2))
    ax.axis("off")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(cursor - gap + 0.5, -1.0)

    for domain in DOMAIN_ORDER:
        center = domain_centers[domain]
        ax.text(-0.10, center, domain, ha="right", va="center", fontsize=10.5, color=DOMAIN_COLORS[domain], weight="bold")
        ax.text(1.10, center, domain, ha="left", va="center", fontsize=10.5, color=DOMAIN_COLORS[domain], weight="bold")

    for row in nearest.itertuples(index=False):
        y0 = y_positions[("left", row.field)]
        y1 = y_positions[("right", row.nearest_field)]
        cross = bool(row.cross_domain)
        arrow = FancyArrowPatch(
            (0.24, y0),
            (0.76, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.1 if cross else 0.7,
            color="#111111" if cross else "#b8b8b8",
            alpha=0.44 if cross else 0.24,
            connectionstyle="arc3,rad=0.18",
            zorder=1,
        )
        ax.add_patch(arrow)

    for side, x in [("left", 0.20), ("right", 0.80)]:
        for row in fields.sort_values(["domain_display_name", "field_display_name"]).itertuples(index=False):
            y = y_positions[(side, row.field_display_name)]
            domain = domain_by_field[row.field_display_name]
            ax.scatter([x], [y], s=45 + int(size_by_field[row.field_display_name]) * 7, color=DOMAIN_COLORS[domain], edgecolor="white", linewidth=0.6, zorder=3)
            label = FIELD_LABELS.get(row.field_display_name, row.field_display_name)
            if side == "left":
                ax.text(x - 0.018, y, label, ha="right", va="center", fontsize=7.2, color="#222222")
            else:
                ax.text(x + 0.018, y, label, ha="left", va="center", fontsize=7.2, color="#222222")

    cross_count = int(nearest["cross_domain"].sum())
    ax.text(0.20, -0.55, "Field", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.text(0.80, -0.55, "Nearest morphological neighbor", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.set_title(f"Field-to-neighbor flow map ({cross_count}/26 cross-domain links)", fontsize=15.5, pad=16)
    ax.legend(handles=_legend_handles(cross_count, len(nearest)), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True, fontsize=8.4)
    _save(fig, "fig_08_field_neighbor_variant_03_domain_flow")


def variant_04_force_network(fields: pd.DataFrame, nearest: pd.DataFrame) -> None:
    domain_by_field = dict(zip(fields["field_display_name"], fields["domain_display_name"]))
    size_by_field = dict(zip(fields["field_display_name"], fields["n_subfields"]))
    graph = nx.DiGraph()
    for field in fields["field_display_name"]:
        graph.add_node(field)
    for row in nearest.itertuples(index=False):
        graph.add_edge(row.field, row.nearest_field, cross_domain=bool(row.cross_domain), weight=1.0 / max(float(row.distance), 0.01))

    layout_graph = graph.to_undirected()
    pos = nx.spring_layout(layout_graph, seed=17, k=0.92, iterations=700, weight="weight")

    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")
    ax.set_title("Nearest-neighbor morphology network", fontsize=15.5, pad=13)

    for u, v, data in graph.edges(data=True):
        cross = bool(data["cross_domain"])
        a = pos[u]
        b = pos[v]
        arrow = FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.35 if cross else 0.8,
            color="#111111" if cross else "#b8b8b8",
            alpha=0.52 if cross else 0.30,
            connectionstyle="arc3,rad=0.10",
            zorder=1,
        )
        ax.add_patch(arrow)

    for field, (x, y) in pos.items():
        domain = domain_by_field[field]
        ax.scatter([x], [y], s=_node_size(int(size_by_field[field])), color=DOMAIN_COLORS[domain], edgecolor="white", linewidth=0.9, zorder=3)

    # Label fields with high degree or visually important outliers, plus small labels for all others.
    degrees = dict(layout_graph.degree())
    for field, (x, y) in pos.items():
        label = FIELD_LABELS.get(field, field)
        fontsize = 8.7 if degrees[field] >= 3 or field in {"Computer Science", "Medicine", "Materials Science"} else 7.4
        ax.text(x + 0.025, y + 0.018, label, fontsize=fontsize, color="#222222", zorder=4)

    cross_count = int(nearest["cross_domain"].sum())
    ax.legend(handles=_legend_handles(cross_count, len(nearest)), loc="lower left", frameon=True, framealpha=0.92, fontsize=8.6)
    _save(fig, "fig_08_field_neighbor_variant_04_force_network")


def write_report(nearest: pd.DataFrame) -> None:
    cross_count = int(nearest["cross_domain"].sum())
    lines = [
        "# Field Neighbor Map Variants",
        "",
        "All four variants use the same field-level full-period morphology distances.",
        "",
        f"- Cross-domain nearest-neighbor fields: {cross_count} of {len(nearest)}.",
        "",
        "## Figures",
        "",
        "- `fig_08_field_neighbor_variant_01_clean_pcoa`: closest to the current candidate, but with directed arrows.",
        "- `fig_08_field_neighbor_variant_02_domain_ring`: circular domain map with cross-domain links as visible chords.",
        "- `fig_08_field_neighbor_variant_03_domain_flow`: left-right flow from each field to its nearest morphological neighbor.",
        "- `fig_08_field_neighbor_variant_04_force_network`: force-directed relational network of nearest-neighbor links.",
    ]
    (OUTPUT_DIR / "field_neighbor_map_variants_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _configure_matplotlib()
    fields, matrix, nearest = _load_neighbor_data()
    variant_01_clean_pcoa(fields, matrix, nearest)
    variant_02_domain_ring(fields, nearest)
    variant_03_domain_flow(fields, nearest)
    variant_04_force_network(fields, nearest)
    write_report(nearest)


if __name__ == "__main__":
    main()
