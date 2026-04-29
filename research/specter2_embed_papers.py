from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "openalex_stats_probability" / "corpus.sqlite"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "research" / "results" / "specter2"


SQL_QUERY = """
SELECT
    work_id,
    COALESCE(display_name, title, '') AS title,
    abstract_text,
    publication_year,
    publication_date,
    cited_by_count,
    primary_topic_name,
    primary_subfield_name,
    primary_field_name,
    primary_domain_name,
    primary_source_name
FROM works
WHERE abstract_text IS NOT NULL
  AND trim(abstract_text) != ''
ORDER BY work_id
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local SPECTER2 embeddings for OpenAlex papers using title + abstract."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default="allenai/specter2_base")
    parser.add_argument("--adapter", default="allenai/specter2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional small run for testing.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_papers(db_path: Path, limit: int | None) -> pd.DataFrame:
    query = SQL_QUERY
    if limit is not None:
        query += f"\nLIMIT {int(limit)}"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_arg)


def require_clean_output(output_dir: Path, overwrite: bool) -> None:
    targets = [
        output_dir / "specter2_embeddings.npy",
        output_dir / "specter2_paper_metadata.csv",
        output_dir / "specter2_run_config.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing)
        raise SystemExit(
            "Output already exists. Re-run with --overwrite or move these files:\n" + names
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_clean_output(args.output_dir, args.overwrite)

    try:
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing SPECTER2 dependencies. Install them with:\n"
            "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "python -m pip install -r research/requirements-specter2.txt"
        ) from exc

    papers = load_papers(args.db_path, args.limit)
    if papers.empty:
        raise SystemExit("No papers with abstracts were found.")

    device = resolve_device(args.device)
    print(f"Loaded {len(papers):,} papers")
    print(f"Device: {device}")
    print(f"Loading tokenizer/model: {args.base_model} + adapter {args.adapter}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoAdapterModel.from_pretrained(args.base_model)
    model.load_adapter(args.adapter, source="hf", load_as="specter2", set_active=True)
    model.to(device)
    model.eval()

    texts = (
        papers["title"].fillna("").astype(str)
        + tokenizer.sep_token
        + papers["abstract_text"].fillna("").astype(str)
    ).tolist()

    start = time.time()
    chunks: list[np.ndarray] = []
    for start_idx in tqdm(range(0, len(texts), args.batch_size), desc="Embedding papers"):
        text_batch = texts[start_idx : start_idx + args.batch_size]
        inputs = tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=args.max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
            batch_embeddings = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
        chunks.append(batch_embeddings.astype(np.float32, copy=False))

    embeddings = np.vstack(chunks)
    elapsed = time.time() - start

    embeddings_path = args.output_dir / "specter2_embeddings.npy"
    metadata_path = args.output_dir / "specter2_paper_metadata.csv"
    config_path = args.output_dir / "specter2_run_config.json"

    np.save(embeddings_path, embeddings)
    papers.drop(columns=["abstract_text"]).to_csv(metadata_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "db_path": str(args.db_path),
                "base_model": args.base_model,
                "adapter": args.adapter,
                "n_papers": int(len(papers)),
                "embedding_shape": list(embeddings.shape),
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "device": str(device),
                "elapsed_seconds": round(elapsed, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved embeddings: {embeddings_path} shape={embeddings.shape}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
