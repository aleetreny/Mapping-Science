$ErrorActionPreference = "Stop"

function Import-DotEnv {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return }

  Get-Content -LiteralPath $Path | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#") -or -not $line.Contains("=")) { return }
    $parts = $line.Split("=", 2)
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if ($name -and -not [Environment]::GetEnvironmentVariable($name)) {
      [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
  }
}

function Resolve-RemotePath {
  param([string]$RemoteName, [string]$DrivePath)
  if ($DrivePath -match "^[^:]+:") { return $DrivePath }
  return "${RemoteName}:$DrivePath"
}

$Root = Split-Path -Parent $PSScriptRoot
Import-DotEnv -Path (Join-Path $Root ".env")

$RemoteName = if ($env:RCLONE_REMOTE) { $env:RCLONE_REMOTE } else { "gdrive" }
$DriveEmbeddingsPath = if ($env:DRIVE_EMBEDDINGS_PATH) { $env:DRIVE_EMBEDDINGS_PATH } else { "TFM/openalex_subfields/embeddings/specter2_v1" }
$Remote = Resolve-RemotePath -RemoteName $RemoteName -DrivePath $DriveEmbeddingsPath
$Local = if ($env:LOCAL_EMBEDDINGS_DIR) { $env:LOCAL_EMBEDDINGS_DIR } else { "embeddings/specter2_v1" }

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
