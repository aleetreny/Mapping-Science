#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


DEFAULT_MODEL_BASE = "allenai/specter2_base"
DEFAULT_ADAPTER = "allenai/specter2"

METADATA_COLUMNS = [
    "work_id",
    "subfield_id",
    "subfield_display_name",
    "field_id",
    "field_display_name",
    "domain_id",
    "domain_display_name",
    "primary_topic_id",
    "primary_topic_display_name",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "title_token_count",
    "abstract_token_count",
    "text_token_count",
    "cited_by_count",
    "referenced_works_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SPECTER2 embeddings in shards on Kaggle.")
    parser.add_argument("--input-local", default="/kaggle/working/data/works_text.parquet")
    parser.add_argument("--input-remote", default="gdrive:TFM/openalex_subfields/data/works_text.parquet")
    parser.add_argument("--output-local", default="/kaggle/working/embeddings/specter2_v1_2000_2024_400py")
    parser.add_argument("--output-remote", default="gdrive:TFM/openalex_subfields/embeddings/specter2_v1_2000_2024_400py")
    parser.add_argument("--model-base", default=DEFAULT_MODEL_BASE)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=None)
    parser.add_argument("--delete-local-after-upload", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument(
        "--setup-rclone-from-secret",
        default=None,
        help="Kaggle secret name containing rclone.conf, usually RCLONE_CONF.",
    )
    parser.add_argument(
        "--check-remote-only",
        action="store_true",
        help="Only check rclone remote and exit.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def command_exists(name: str) -> bool:
    return subprocess.run(["bash", "-lc", f"command -v {name} >/dev/null 2>&1"]).returncode == 0


def setup_rclone_from_kaggle_secret(secret_name: str) -> None:
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception as exc:
        raise RuntimeError("kaggle_secrets is not available. Are you running inside Kaggle?") from exc

    secret = UserSecretsClient().get_secret(secret_name).strip()

    # Supports both a valid multiline rclone.conf and the common one-line pasted version.
    if secret.startswith("[gdrive] ") and "\n" not in secret:
        secret = secret.replace("[gdrive] ", "[gdrive]\n", 1)
        secret = re.sub(r"\s+type\s*=", "\ntype =", secret, count=1)
        secret = re.sub(r"\s+scope\s*=", "\nscope =", secret, count=1)
        secret = re.sub(r"\s+token\s*=", "\ntoken =", secret, count=1)
        secret = re.sub(r"\s+team_drive\s*=", "\nteam_drive =", secret, count=1)

    secret = secret.replace("\\n", "\n")

    config_dir = Path("/root/.config/rclone")
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "rclone.conf").write_text(secret, encoding="utf-8")

    safe_lines = []
    for line in secret.splitlines():
        safe_lines.append("token = REDACTED" if line.strip().startswith("token") else line)

    print("rclone.conf written.")
    print("Safe preview:")
    print("\n".join(safe_lines))


def rclone_lsf(remote_dir: str) -> set[str]:
    if not command_exists("rclone"):
        return set()

    result = subprocess.run(
        ["rclone", "lsf", remote_dir],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        return set()

    return {line.strip().rstrip("/") for line in result.stdout.splitlines() if line.strip()}


def ensure_remote_dir(remote_dir: str) -> None:
    if command_exists("rclone"):
        run_cmd(["rclone", "mkdir", remote_dir], check=False)


def copy_remote_to_local(remote_path: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["rclone", "copy", remote_path, str(local_dir), "-P"])


def upload_file(local_path: Path, remote_dir: str) -> None:
    ensure_remote_dir(remote_dir)
    run_cmd(["rclone", "copy", str(local_path), remote_dir, "-P"])


def read_input_table(input_local: Path, input_remote: str) -> pd.DataFrame:
    if not input_local.exists():
        if not command_exists("rclone"):
            raise FileNotFoundError(f"{input_local} does not exist and rclone is not available.")

        print(f"Input not found locally. Downloading from {input_remote}...")
        copy_remote_to_local(input_remote, input_local.parent)

    desired_columns = [
        "work_id",
        "title",
        "abstract",
        "text_for_embedding",
        "publication_year",
        "publication_date",
        "type",
        "language",
        "subfield_id",
        "subfield_display_name",
        "field_id",
        "field_display_name",
        "domain_id",
        "domain_display_name",
        "primary_topic_id",
        "primary_topic_display_name",
        "title_token_count",
        "abstract_token_count",
        "text_token_count",
        "cited_by_count",
        "referenced_works_count",
    ]

    empty = pd.read_parquet(input_local).head(0)
    available_columns = [column for column in desired_columns if column in empty.columns]

    return pd.read_parquet(input_local, columns=available_columns)


def build_texts(df: pd.DataFrame, sep_token: str) -> list[str]:
    title = df["title"].fillna("").astype(str)
    abstract = df["abstract"].fillna("").astype(str)
    return (title + sep_token + abstract).tolist()


@torch.inference_mode()
def embed_batches(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for start in tqdm(range(0, len(texts), batch_size), leave=False):
        batch = texts[start:start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(**inputs)
            emb = output.last_hidden_state[:, 0, :]

        outputs.append(emb.detach().cpu().float().numpy())

    return np.vstack(outputs)


def load_specter2(model_base: str, adapter_name: str, device: torch.device):
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    tokenizer = AutoTokenizer.from_pretrained(model_base)
    model = AutoAdapterModel.from_pretrained(model_base)

    model.load_adapter(adapter_name, source="hf", load_as="specter2")
    model.set_active_adapters("specter2")

    print("Active adapters:", model.active_adapters)

    model.to(device)
    model.eval()

    return tokenizer, model


def shard_paths(output_dir: Path, shard_idx: int) -> tuple[Path, Path, Path]:
    prefix = f"shard_{shard_idx:04d}"
    return (
        output_dir / f"{prefix}_embeddings.npy",
        output_dir / f"{prefix}_metadata.parquet",
        output_dir / f"{prefix}_summary.json",
    )


def remote_shard_complete(remote_files: set[str], shard_idx: int) -> bool:
    prefix = f"shard_{shard_idx:04d}"
    return (
        f"{prefix}_embeddings.npy" in remote_files
        and f"{prefix}_metadata.parquet" in remote_files
        and f"{prefix}_summary.json" in remote_files
    )


def local_shard_complete(output_dir: Path, shard_idx: int) -> bool:
    emb_path, meta_path, summary_path = shard_paths(output_dir, shard_idx)
    return emb_path.exists() and meta_path.exists() and summary_path.exists()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.setup_rclone_from_secret:
        setup_rclone_from_kaggle_secret(args.setup_rclone_from_secret)

    if args.check_remote_only:
        run_cmd(["rclone", "listremotes"], check=False)
        run_cmd(["rclone", "lsd", "gdrive:TFM/openalex_subfields"], check=False)
        return

    output_dir = Path(args.output_local)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_upload:
        ensure_remote_dir(args.output_remote)

    print("Reading input table...")
    df = read_input_table(Path(args.input_local), args.input_remote)

    if args.limit_rows is not None:
        df = df.head(args.limit_rows).copy()

    df["subfield_id"] = df["subfield_id"].astype(str)
    df["work_id"] = df["work_id"].astype(str)

    df = df.sort_values(
        ["subfield_id", "publication_year", "work_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    n_rows = len(df)
    n_shards = int(math.ceil(n_rows / args.shard_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Rows: {n_rows:,}")
    print(f"Shards: {n_shards}")
    print(f"Device: {device}")

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_base": args.model_base,
        "adapter": args.adapter,
        "pooling": "cls_token",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "dtype": args.dtype,
        "n_rows": n_rows,
        "n_shards": n_shards,
        "input_local": args.input_local,
        "input_remote": args.input_remote,
        "output_remote": args.output_remote,
    }

    config_path = output_dir / "embedding_config.json"
    write_json(config_path, config)

    if not args.no_upload:
        upload_file(config_path, args.output_remote)

    remote_files = rclone_lsf(args.output_remote) if (args.resume and not args.no_upload) else set()

    print("Loading SPECTER2...")
    tokenizer, model = load_specter2(args.model_base, args.adapter, device)

    run_records = []
    end_shard = args.end_shard if args.end_shard is not None else n_shards

    for shard_idx in range(args.start_shard, min(end_shard, n_shards)):
        emb_path, meta_path, summary_path = shard_paths(output_dir, shard_idx)

        if not args.force and args.resume:
            if local_shard_complete(output_dir, shard_idx):
                print(f"Skipping shard {shard_idx:04d}: complete locally.")
                continue
            if remote_shard_complete(remote_files, shard_idx):
                print(f"Skipping shard {shard_idx:04d}: complete on Drive.")
                continue

        start_time = time.time()
        start = shard_idx * args.shard_size
        end = min(start + args.shard_size, n_rows)

        shard = df.iloc[start:end].copy()

        print(f"\nEmbedding shard {shard_idx:04d}: rows {start:,}-{end:,} ({len(shard):,})")

        texts = build_texts(shard, tokenizer.sep_token or " [SEP] ")

        embeddings = embed_batches(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        if args.dtype == "float16":
            embeddings = embeddings.astype(np.float16)
        else:
            embeddings = embeddings.astype(np.float32)

        np.save(emb_path, embeddings)

        meta_cols = [column for column in METADATA_COLUMNS if column in shard.columns]
        shard[meta_cols].to_parquet(meta_path, index=False)

        summary = {
            "shard_idx": shard_idx,
            "start_row": int(start),
            "end_row": int(end),
            "n_rows": int(len(shard)),
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.time() - start_time, 2),
            "subfields": sorted(shard["subfield_id"].dropna().astype(str).unique().tolist()),
            "publication_year_min": int(shard["publication_year"].min()),
            "publication_year_max": int(shard["publication_year"].max()),
        }
        write_json(summary_path, summary)

        if not args.no_upload:
            upload_file(emb_path, args.output_remote)
            upload_file(meta_path, args.output_remote)
            upload_file(summary_path, args.output_remote)

        run_records.append(summary)
        manifest = pd.DataFrame(run_records)
        manifest_path = output_dir / "embedding_run_manifest.csv"
        manifest.to_csv(manifest_path, index=False)

        if not args.no_upload:
            upload_file(manifest_path, args.output_remote)

        if args.delete_local_after_upload and not args.no_upload:
            emb_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
