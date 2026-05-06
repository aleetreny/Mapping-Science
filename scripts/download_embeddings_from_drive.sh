#!/usr/bin/env bash
set -euo pipefail

REMOTE="gdrive:TFM/openalex_subfields/embeddings/specter2_v1"
LOCAL="embeddings/specter2_v1"

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
