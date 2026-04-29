# SPECTER2 quickstart

This is a minimal local workflow for a first semantic 2D view of the corpus.

## 1. Install dependencies

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/requirements-specter2.txt
```

## 2. Optional smoke test

Run only 100 papers first:

```powershell
python research/specter2_embed_papers.py --limit 100 --overwrite
python research/umap_specter2_quicklook.py --color-by publication_year
```

Outputs go to:

```text
research/results/specter2/
```

## 3. Full embedding run

CPU-only default:

```powershell
python research/specter2_embed_papers.py --batch-size 8 --overwrite
```

If you later run on a CUDA machine:

```powershell
python research/specter2_embed_papers.py --device cuda --batch-size 32 --overwrite
```

## 4. Quick UMAP visualization

Fast sampled view:

```powershell
python research/umap_specter2_quicklook.py --sample-size 15000 --color-by publication_year
```

Full corpus view:

```powershell
python research/umap_specter2_quicklook.py --color-by publication_year
```

Other useful color columns:

```powershell
python research/umap_specter2_quicklook.py --sample-size 15000 --color-by primary_topic_name
python research/umap_specter2_quicklook.py --sample-size 15000 --color-by cited_by_count
```

The main files created are:

```text
research/results/specter2/specter2_embeddings.npy
research/results/specter2/specter2_paper_metadata.csv
research/results/specter2/specter2_umap_2d.csv
research/results/specter2/specter2_umap_2d_publication_year.png
```
