#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_DIR="/home/ec2-user/ai-voice-train-data/backend/data/_downloads"
OUT_BASE="${DOWNLOAD_DIR}/commonvoice"
LOG="/home/ec2-user/ai-voice-train-data/PROGRESS_LOG.md"

mkdir -p "$OUT_BASE"

log() {
  local msg="$1"
  printf "%s — %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" >> "$LOG"
}

extract_one() {
  local tarfile="$1"
  local base
  base=$(basename "$tarfile")
  local name="${base%.tar.gz}"
  local outdir="${OUT_BASE}/${name}"
  if [ -d "$outdir" ]; then
    return 0
  fi
  log "Auto-extract start: ${base} -> ${outdir}"
  mkdir -p "$outdir"
  tar -xf "$tarfile" -C "$outdir"
  log "Auto-extract complete: ${base}"
}

while true; do
  shopt -s nullglob
  for tarfile in "$DOWNLOAD_DIR"/*.tar.gz; do
    case "$(basename "$tarfile")" in
      mcv-*.tar.gz|Common*Voice*Speech*.tar.gz|*common*voice*.tar.gz)
        extract_one "$tarfile"
        ;;
    esac
  done
  shopt -u nullglob
  sleep 60
 done
