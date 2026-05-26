from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# Set root and adjust path to import src modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.per_subfield_umap_maps import (
    DEFAULT_DENSITY_GRID_SIZE,
    DEFAULT_DENSITY_METHOD,
    DEFAULT_DENSITY_SIGMA,
    DEFAULT_DENSITY_VMAX_PERCENTILE,
    coordinate_limits,
    density_plot_vmax,
    filter_input_window,
    main_analysis_subfields,
    safe_subfield_stem,
    sample_subfield_rows,
    validate_index_columns,
    validate_year_window,
    plot_density_panel,
)
from src.storage import load_parquet, save_parquet

def plot_projection_panels(
    coordinates: np.ndarray,
    *,
    subfield_name: str,
    method: str,
    n_used: int,
    year_min: int,
    year_max: int,
    output_path: Path,
    dpi: int,
) -> str:
    coordinates = np.asarray(coordinates, dtype=float)
    xlim, ylim = coordinate_limits(coordinates)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    try:
        # Panel A: Scatter
        axes[0].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=2,
            alpha=0.45,
            linewidths=0,
            color="#26547c",
            rasterized=True,
        )
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        axes[0].set_xlabel(f"{method} 1")
        axes[0].set_ylabel(f"{method} 2")
        axes[0].set_title(
            "A. Scatter",
            fontsize=10,
        )

        # Panel B: Density
        density_artist, density_method = plot_density_panel(
            axes[1],
            coordinates,
            xlim=xlim,
            ylim=ylim,
            density_method="smooth_hist",
            density_grid_size=150,
            density_sigma=3.0,
            density_vmax_percentile=99.0,
        )
        axes[1].set_xlabel(f"{method} 1")
        axes[1].set_ylabel(f"{method} 2")
        axes[1].set_title("B. Density", fontsize=10)
        
        fig.colorbar(density_artist, ax=axes[1], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.grid(False)

        fig.tight_layout()
        fig.savefig(output_path)
        return density_method
    finally:
        plt.close(fig)

def run_single_projection(
    subfield_id: int,
    subfield_name: str,
    safe_subfield_stem_str: str,
    method: str,
    row_ids: np.ndarray,
    sampled_index_dict: dict[str, list[Any]],
    embeddings_path_str: str,
    output_dir_str: str,
    year_min: int,
    year_max: int,
    random_state: int,
    dpi: int,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    output_dir = Path(output_dir_str)
    fig_path = output_dir / "figures" / "method_comparison" / safe_subfield_stem_str / f"{method}.png"
    coord_path = output_dir / "coordinates" / "method_comparison" / safe_subfield_stem_str / f"{method}.parquet"
    
    n_used = len(row_ids)
    
    try:
        # Load the SPECTER2 embedding matrix
        embeddings_path = Path(embeddings_path_str)
        embeddings = np.load(embeddings_path, mmap_mode="r")
        sub_embeddings = embeddings[row_ids]
        
        # Fit projection
        if method == "PCA":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=random_state)
            coords = pca.fit_transform(sub_embeddings)
        elif method == "t-SNE":
            from sklearn.manifold import TSNE
            perplexity = min(30.0, max(5.0, n_used / 3.0))
            tsne = TSNE(
                n_components=2,
                random_state=random_state,
                perplexity=perplexity,
                n_jobs=1,
                init="pca" if n_used > 30 else "random"
            )
            coords = tsne.fit_transform(sub_embeddings)
        elif method == "PHATE":
            import phate
            phate_op = phate.PHATE(
                n_components=2,
                random_state=random_state,
                n_jobs=1,
                verbose=False
            )
            coords = phate_op.fit_transform(sub_embeddings)
        else:
            raise ValueError(f"Unsupported projection method: {method}")
            
        # Ensure finite coordinates
        if not np.isfinite(coords).all():
            raise ValueError("Computed coordinates contain non-finite values.")
            
        # Create output directories
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        coord_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save coordinates
        sampled_index = pd.DataFrame(sampled_index_dict)
        df_coords = sampled_index.copy()
        df_coords["x"] = coords[:, 0].astype(np.float32)
        df_coords["y"] = coords[:, 1].astype(np.float32)
        df_coords.to_parquet(coord_path, index=False)
        
        # Draw plot
        plot_projection_panels(
            coords,
            subfield_name=subfield_name,
            method=method,
            n_used=n_used,
            year_min=year_min,
            year_max=year_max,
            output_path=fig_path,
            dpi=dpi
        )
        
        status = "completed"
        error_message = ""
    except Exception as e:
        status = "failed"
        error_message = str(e)
        print(f"Error computing {method} for {subfield_name}: {e}")
        
    return {
        "subfield_id": str(subfield_id),
        "subfield_name": subfield_name,
        "method": method,
        "n_used": n_used,
        "year_min": year_min,
        "year_max": year_max,
        "status": status,
        "coordinate_path": str(coord_path) if status == "completed" else "",
        "figure_path": str(fig_path) if status == "completed" else "",
        "error_message": error_message,
        "random_state": random_state,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PCA, t-SNE, and PHATE projections for all 241 ASJC subfields in parallel."
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/analysis_embedding_index.parquet",
        help="Path to the analysis embedding index."
    )
    parser.add_argument(
        "--embeddings-path",
        default="embeddings/specter2_v1_2000_2024_400py/analysis/main_embeddings.float16.npy",
        help="Path to the SPECTER2 embedding matrix."
    )
    parser.add_argument(
        "--typologies-path",
        default="outputs/09_morphological_typologies/subfield_typology_assignments.csv",
        help="Path to ASJC typologies assignments."
    )
    parser.add_argument(
        "--output-dir",
        default="frontend",
        help="Path to the production frontend directory."
    )
    parser.add_argument("--year-min", type=int, default=2000)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--max-papers-per-subfield", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4, 
        help="Number of parallel multiprocessing workers (recommend 4-6 for balanced CPU load)."
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-calculation of all projections, overwriting existing files."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_year_window(args.year_min, args.year_max)

    index_path = ROOT / args.index_path
    embeddings_path = ROOT / args.embeddings_path
    typologies_path = ROOT / args.typologies_path
    output_dir = ROOT / args.output_dir

    print(f"Loading ASJC subfields list from {typologies_path}...")
    if not typologies_path.exists():
        print(f"Error: Typologies file not found at {typologies_path}")
        return
    df_typo = pd.read_csv(typologies_path)
    all_subfields = df_typo[["subfield_id", "subfield_display_name"]].drop_duplicates().to_dict(orient="records")
    print(f"Loaded {len(all_subfields)} ASJC subfields for processing.")

    print(f"Loading analysis index: {index_path}...")
    analysis_index = load_parquet(index_path)
    validate_index_columns(analysis_index)

    print(f"Filtering index to year window {args.year_min}-{args.year_max}...")
    window_index = filter_input_window(
        analysis_index,
        year_min=args.year_min,
        year_max=args.year_max
    )
    
    # We will compute 3 projections (PCA, t-SNE, PHATE) per subfield
    # UMAP is already pre-calculated and copied to the frontend!
    methods_list = ["PCA", "t-SNE", "PHATE"]
    jobs = []
    skipped_count = 0

    print("Checking existing projections for resume capability...")
    # Loop over all subfields and check which projections are already completed
    for sf in all_subfields:
        sf_id = int(sf["subfield_id"])
        sf_name = sf["subfield_display_name"]
        stem = safe_subfield_stem(sf_id, sf_name)
        
        # Slicing and sampling index rows
        sub_df = window_index[window_index["subfield_id"].astype(int) == sf_id].reset_index(drop=True)
        if len(sub_df) == 0:
            print(f"Skipping empty subfield: {sf_name} ({sf_id})")
            continue
            
        sampled = sample_subfield_rows(
            sub_df,
            max_papers=args.max_papers_per_subfield,
            random_state=args.random_state,
            subfield_id=sf_id
        )
        row_ids = sampled["analysis_row_id"].astype(int).to_numpy()
        sampled_dict = sampled.to_dict(orient="list")
        
        for method in methods_list:
            fig_path = output_dir / "figures" / "method_comparison" / stem / f"{method}.png"
            coord_path = output_dir / "coordinates" / "method_comparison" / stem / f"{method}.parquet"
            
            # Resume check: Skip if both coordinates and figures already exist
            if not args.force and fig_path.exists() and coord_path.exists():
                skipped_count += 1
                continue
                
            jobs.append({
                "subfield_id": sf_id,
                "subfield_name": sf_name,
                "safe_subfield_stem_str": stem,
                "method": method,
                "row_ids": row_ids,
                "sampled_index_dict": sampled_dict,
                "embeddings_path_str": str(embeddings_path),
                "output_dir_str": str(output_dir),
                "year_min": args.year_min,
                "year_max": args.year_max,
                "random_state": args.random_state,
                "dpi": args.dpi,
            })

    total_jobs = len(jobs)
    print(f"Resume Status: {skipped_count} runs already completed. {total_jobs} runs remaining to calculate.")
    
    if total_jobs == 0:
        print("All subfield projections are fully completed! Nothing to run.")
        return

    print(f"Starting execution of {total_jobs} projection runs...")
    print(f"Parallelizing with {args.workers} workers across CPU cores...")

    start_time = time.time()
    manifest_rows = []

    # Execute all remaining projection runs in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_projection, **job): job for job in jobs}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            manifest_rows.append(res)
            completed += 1
            if completed % 10 == 0 or completed == total_jobs:
                elapsed = time.time() - start_time
                estimated_total = (elapsed / completed) * total_jobs
                remaining = estimated_total - elapsed
                print(f"Progress: {completed}/{total_jobs} runs finished | Elapsed: {elapsed:.1f}s | Est. Remaining: {remaining:.1f}s")

    # Generate final manifest report
    df_manifest = pd.DataFrame(manifest_rows)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(manifest_dir / "all_subfields_projections_manifest.csv", index=False)

    print("\n" + "="*60)
    print("PROJECTION PROCESSING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    print(f"Completed runs: {completed}")
    print(f"Saved run manifest to {manifest_dir / 'all_subfields_projections_manifest.csv'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
