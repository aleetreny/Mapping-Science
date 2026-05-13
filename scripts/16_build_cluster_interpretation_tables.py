from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cluster_interpretation import (
    SUPPORTED_SPACES,
    agreement_summary_markdown,
    agreement_summary_table,
    build_master_table,
    build_metric_profile_long,
    cluster_composition,
    cluster_interpretation_markdown,
    combined_cluster_size_summary,
    main_cluster_column,
    plot_combined_metric_space_umap,
    plot_combined_pca_scatter,
    plot_contingency_heatmap,
    plot_domain_stacked_bar,
    plot_profile_heatmap,
    readable_representatives,
    representative_fallback_from_master,
    require_existing_paths,
    top_domains_by_cluster,
    top_metric_differences,
    top_profile_metrics_by_cluster,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build thesis-facing interpretation tables and figures for metric-space clusters."
    )
    parser.add_argument(
        "--clustering-dir",
        default="outputs/analysis/metric_clustering",
        help="Directory containing outputs from scripts/15_cluster_metric_spaces.py.",
    )
    parser.add_argument(
        "--metrics-umap",
        default="data/processed/subfield_morphology_metrics.parquet",
        help="Per-subfield projected UMAP morphology metric table.",
    )
    parser.add_argument(
        "--metrics-embedding",
        default="data/processed/subfield_embedding_space_metrics.parquet",
        help="Per-subfield embedding-space metric table.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/cluster_interpretation",
        help="Directory where interpretation outputs will be written.",
    )
    parser.add_argument(
        "--main-space",
        choices=SUPPORTED_SPACES,
        default="combined",
        help="Cluster space treated as the main exploratory typology.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing cluster interpretation outputs without "
            f"--overwrite: {display_path(output_dir)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_space_tables(
    clustering_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, str]]:
    assignments_by_space: dict[str, pd.DataFrame] = {}
    pca_scores_by_space: dict[str, pd.DataFrame] = {}
    cluster_columns_by_space: dict[str, str] = {}

    required_paths: dict[str, Path] = {}
    for space in SUPPORTED_SPACES:
        space_dir = clustering_dir / space
        required_paths[f"{space} cluster_assignments"] = space_dir / "cluster_assignments.csv"
        required_paths[f"{space} pca_scores"] = space_dir / "pca_scores.csv"
    require_existing_paths(required_paths)

    for space in SUPPORTED_SPACES:
        space_dir = clustering_dir / space
        assignments = pd.read_csv(space_dir / "cluster_assignments.csv")
        pca_scores = pd.read_csv(space_dir / "pca_scores.csv")
        summary_path = space_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        assignments_by_space[space] = assignments
        pca_scores_by_space[space] = pca_scores
        cluster_columns_by_space[space] = main_cluster_column(assignments, summary=summary)

    return assignments_by_space, pca_scores_by_space, cluster_columns_by_space


def load_metric_umap_coordinates(clustering_dir: Path) -> dict[str, pd.DataFrame]:
    coordinates: dict[str, pd.DataFrame] = {}
    for space in SUPPORTED_SPACES:
        csv_path = clustering_dir / f"{space}_metric_umap_coordinates.csv"
        parquet_path = clustering_dir / f"{space}_metric_umap_coordinates.parquet"
        if csv_path.exists():
            coordinates[space] = pd.read_csv(csv_path)
        elif parquet_path.exists():
            coordinates[space] = pd.read_parquet(parquet_path)
    return coordinates


def load_optional_representatives(clustering_dir: Path, master: pd.DataFrame) -> pd.DataFrame:
    reps_path = clustering_dir / "combined" / "cluster_representatives.csv"
    if not reps_path.exists():
        return representative_fallback_from_master(master)
    reps = pd.read_csv(reps_path)
    try:
        return readable_representatives(reps, master)
    except ValueError:
        return representative_fallback_from_master(master)


def load_agreement_inputs(clustering_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    comparison_dir = clustering_dir / "comparison"
    agreement_path = comparison_dir / "cluster_partition_agreement.csv"
    require_existing_paths({"cluster partition agreement": agreement_path})
    agreement = pd.read_csv(agreement_path)
    contingency_files = {
        "embedding_only vs combined": comparison_dir / "cluster_contingency_embedding_vs_combined.csv",
        "umap_only vs combined": comparison_dir / "cluster_contingency_umap_vs_combined.csv",
        "embedding_only vs umap_only": comparison_dir / "cluster_contingency_embedding_vs_umap.csv",
    }
    contingencies: dict[str, pd.DataFrame] = {}
    for name, path in contingency_files.items():
        if path.exists():
            contingencies[name] = pd.read_csv(path, index_col=0)
    return agreement, contingencies


def write_table_outputs(
    *,
    output_dir: Path,
    master: pd.DataFrame,
    domain_composition: pd.DataFrame,
    field_composition: pd.DataFrame,
    size_summary: pd.DataFrame,
    representatives: pd.DataFrame,
    profile_long: pd.DataFrame,
    top_differences: pd.DataFrame,
    agreement_summary: pd.DataFrame,
) -> list[str]:
    output_paths = [
        output_dir / "subfield_cluster_master_table.csv",
        output_dir / "subfield_cluster_master_table.parquet",
        output_dir / "combined_cluster_domain_composition.csv",
        output_dir / "combined_cluster_field_composition.csv",
        output_dir / "combined_cluster_size_summary.csv",
        output_dir / "combined_cluster_representatives_readable.csv",
        output_dir / "combined_cluster_metric_profile_long.csv",
        output_dir / "combined_cluster_metric_profile_top_differences.csv",
        output_dir / "cluster_family_agreement_summary.csv",
    ]

    master.to_csv(output_paths[0], index=False)
    master.to_parquet(output_paths[1], index=False)
    domain_composition.to_csv(output_paths[2], index=False)
    field_composition.to_csv(output_paths[3], index=False)
    size_summary.to_csv(output_paths[4], index=False)
    representatives.to_csv(output_paths[5], index=False)
    profile_long.to_csv(output_paths[6], index=False)
    top_differences.to_csv(output_paths[7], index=False)
    agreement_summary.to_csv(output_paths[8], index=False)
    return [display_path(path) for path in output_paths]


def write_markdown_outputs(
    *,
    output_dir: Path,
    master: pd.DataFrame,
    size_summary: pd.DataFrame,
    domain_composition: pd.DataFrame,
    field_composition: pd.DataFrame,
    representatives: pd.DataFrame,
    top_differences: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    contingencies: dict[str, pd.DataFrame],
) -> list[str]:
    interpretation_path = output_dir / "combined_cluster_interpretation_summary.md"
    agreement_path = output_dir / "cluster_family_agreement_summary.md"
    interpretation_path.write_text(
        cluster_interpretation_markdown(
            master=master,
            size_summary=size_summary,
            domain_composition=domain_composition,
            field_composition=field_composition,
            representatives=representatives,
            top_differences=top_differences,
        ),
        encoding="utf-8",
    )
    agreement_path.write_text(
        agreement_summary_markdown(
            agreement=agreement_summary,
            contingencies=contingencies,
        ),
        encoding="utf-8",
    )
    return [display_path(interpretation_path), display_path(agreement_path)]


def write_figures(
    *,
    output_dir: Path,
    master: pd.DataFrame,
    domain_composition: pd.DataFrame,
    representatives: pd.DataFrame,
    profile_long: pd.DataFrame,
    top_differences: pd.DataFrame,
    contingencies: dict[str, pd.DataFrame],
) -> list[str]:
    for stale_path in [
        output_dir / "combined_cluster_profile_heatmap_top_metrics.png",
        output_dir / "combined_pca_scatter_labeled.png",
    ]:
        if stale_path.exists():
            stale_path.unlink()

    paths = [
        output_dir / "combined_cluster_domain_stacked_bar.png",
        output_dir / "combined_metric_space_umap_clusters.png",
        output_dir / "combined_cluster_metric_profile_heatmap.png",
        output_dir / "combined_metric_space_pca_diagnostic.png",
    ]
    plot_domain_stacked_bar(domain_composition, paths[0])
    plot_combined_metric_space_umap(master, representatives, paths[1])
    plot_profile_heatmap(profile_long, top_differences, paths[2])
    plot_combined_pca_scatter(master, representatives, paths[3])

    optional_specs = [
        (
            "embedding_only vs combined",
            output_dir / "cluster_agreement_contingency_heatmap_combined_vs_embedding.png",
            "Embedding-Only Vs Combined Cluster Contingency",
        ),
        (
            "umap_only vs combined",
            output_dir / "cluster_agreement_contingency_heatmap_combined_vs_umap.png",
            "UMAP-Only Vs Combined Cluster Contingency",
        ),
    ]
    for name, path, title in optional_specs:
        if name in contingencies:
            plot_contingency_heatmap(contingencies[name], path, title=title)
            paths.append(path)
    return [display_path(path) for path in paths]


def main() -> None:
    args = parse_args()
    clustering_dir = resolve_path(args.clustering_dir)
    umap_metrics_path = resolve_path(args.metrics_umap)
    embedding_metrics_path = resolve_path(args.metrics_embedding)
    output_dir = resolve_path(args.output_dir)

    require_existing_paths(
        {
            "clustering directory": clustering_dir,
            "UMAP morphology metrics": umap_metrics_path,
            "embedding-space metrics": embedding_metrics_path,
        }
    )
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    assignments_by_space, pca_scores_by_space, cluster_columns_by_space = load_space_tables(
        clustering_dir
    )
    metric_umap_coordinates_by_space = load_metric_umap_coordinates(clustering_dir)
    umap_metrics = pd.read_parquet(umap_metrics_path)
    embedding_metrics = pd.read_parquet(embedding_metrics_path)

    master = build_master_table(
        assignments_by_space=assignments_by_space,
        pca_scores_by_space=pca_scores_by_space,
        cluster_columns_by_space=cluster_columns_by_space,
        umap_metrics=umap_metrics,
        embedding_metrics=embedding_metrics,
        main_space=args.main_space,
        metric_umap_coordinates_by_space=metric_umap_coordinates_by_space,
    )
    domain_composition = cluster_composition(master, level="domain")
    field_composition = cluster_composition(master, level="field")
    size_summary = combined_cluster_size_summary(master)
    representatives = load_optional_representatives(clustering_dir, master)
    profile_long = build_metric_profile_long(
        master=master,
        umap_metrics=umap_metrics,
        embedding_metrics=embedding_metrics,
    )
    top_differences = top_metric_differences(profile_long, top_n=8)
    agreement, contingencies = load_agreement_inputs(clustering_dir)
    agreement_summary = agreement_summary_table(agreement)

    table_outputs = write_table_outputs(
        output_dir=output_dir,
        master=master,
        domain_composition=domain_composition,
        field_composition=field_composition,
        size_summary=size_summary,
        representatives=representatives,
        profile_long=profile_long,
        top_differences=top_differences,
        agreement_summary=agreement_summary,
    )
    markdown_outputs = write_markdown_outputs(
        output_dir=output_dir,
        master=master,
        size_summary=size_summary,
        domain_composition=domain_composition,
        field_composition=field_composition,
        representatives=representatives,
        top_differences=top_differences,
        agreement_summary=agreement_summary,
        contingencies=contingencies,
    )
    figure_outputs = write_figures(
        output_dir=output_dir,
        master=master,
        domain_composition=domain_composition,
        representatives=representatives,
        profile_long=profile_long,
        top_differences=top_differences,
        contingencies=contingencies,
    )

    combined_cluster_sizes = {
        str(row.cluster_id): int(row.n_subfields)
        for row in size_summary.itertuples(index=False)
    }
    warnings = []
    if master["combined_umap_x"].isna().all() or master["combined_umap_y"].isna().all():
        warnings.append(
            "No metric-space UMAP coordinate table was found; combined_umap_x/y were left empty."
        )
    missing_coordinate_spaces = [
        space for space in SUPPORTED_SPACES if space not in metric_umap_coordinates_by_space
    ]
    if missing_coordinate_spaces:
        warnings.append(
            "Missing metric-space UMAP coordinate tables for: "
            + ", ".join(missing_coordinate_spaces)
        )

    summary_path = output_dir / "summary.json"
    output_paths = table_outputs + markdown_outputs + figure_outputs + [display_path(summary_path)]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "clustering_dir": display_path(clustering_dir),
            "metrics_umap": display_path(umap_metrics_path),
            "metrics_embedding": display_path(embedding_metrics_path),
        },
        "output_paths": output_paths,
        "n_subfields": int(master["subfield_id"].nunique()),
        "cluster_spaces_used": list(SUPPORTED_SPACES),
        "cluster_columns_by_space": cluster_columns_by_space,
        "metric_umap_coordinate_spaces": sorted(metric_umap_coordinates_by_space),
        "combined_cluster_sizes": combined_cluster_sizes,
        "dominant_domains_by_cluster": top_domains_by_cluster(domain_composition),
        "top_profile_metrics_by_cluster": top_profile_metrics_by_cluster(top_differences),
        "warnings": warnings,
    }
    write_json(summary_path, summary)

    print(f"Wrote {display_path(output_dir)}")
    print(f"Master table subfields: {summary['n_subfields']}")
    print(f"Combined cluster sizes: {combined_cluster_sizes}")


if __name__ == "__main__":
    main()
