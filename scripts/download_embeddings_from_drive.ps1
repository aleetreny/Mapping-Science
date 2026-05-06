$ErrorActionPreference = "Stop"

$Remote = "gdrive:TFM/openalex_subfields/embeddings/specter2_v1"
$Local = "embeddings/specter2_v1"

New-Item -ItemType Directory -Force -Path $Local | Out-Null

rclone lsf $Remote | Out-Null

rclone copy $Remote $Local -P `
  --transfers 4 `
  --checkers 8 `
  --drive-chunk-size 64M

$files = Get-ChildItem $Local -File

$embeddings = ($files | Where-Object { $_.Name -like "shard_*_embeddings.npy" }).Count
$metadata = ($files | Where-Object { $_.Name -like "shard_*_metadata.parquet" }).Count
$summaries = ($files | Where-Object { $_.Name -like "shard_*_summary.json" }).Count

if ($embeddings -ne 37) { throw "Expected 37 embedding shards, found $embeddings" }
if ($metadata -ne 37) { throw "Expected 37 metadata shards, found $metadata" }
if ($summaries -ne 37) { throw "Expected 37 summary files, found $summaries" }

Write-Host "Embedding download completed successfully."
