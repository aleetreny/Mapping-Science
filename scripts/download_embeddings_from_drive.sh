#!/usr/bin/env bash
set -euo pipefail

if [ -f ".env" ]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      name="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
      if [ -z "${!name:-}" ]; then
        export "$name=$value"
      fi
    fi
  done < ".env"
fi

RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
DRIVE_EMBEDDINGS_PATH="${DRIVE_EMBEDDINGS_PATH:-TFM/openalex_subfields/embeddings/specter2_v1_2000_2024_400py}"
LOCAL="${LOCAL_EMBEDDINGS_DIR:-embeddings/specter2_v1_2000_2024_400py}"

if [[ "$DRIVE_EMBEDDINGS_PATH" == *:* ]]; then
  REMOTE="$DRIVE_EMBEDDINGS_PATH"
else
  REMOTE="${RCLONE_REMOTE}:${DRIVE_EMBEDDINGS_PATH}"
fi

mkdir -p "$LOCAL"

rclone lsf "$REMOTE" >/dev/null

rclone copy "$REMOTE" "$LOCAL" -P \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 64M

embeddings=$(find "$LOCAL" -maxdepth 1 -type f -name "shard_*_embeddings.npy" | wc -l | tr -d ' ')
metadata=$(find "$LOCAL" -maxdepth 1 -type f -name "shard_*_metadata.parquet" | wc -l | tr -d ' ')
summaries=$(find "$LOCAL" -maxdepth 1 -type f -name "shard_*_summary.json" | wc -l | tr -d ' ')

if [ "$embeddings" -ne 37 ]; then
  echo "Expected 37 embedding shards, found $embeddings" >&2
  exit 1
fi
if [ "$metadata" -ne 37 ]; then
  echo "Expected 37 metadata shards, found $metadata" >&2
  exit 1
fi
if [ "$summaries" -ne 37 ]; then
  echo "Expected 37 summary files, found $summaries" >&2
  exit 1
fi

echo "Embedding download completed successfully."
