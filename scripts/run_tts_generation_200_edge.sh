#!/usr/bin/env bash
set -euo pipefail

VENV_PY="/home/ec2-user/ai-voice-train-data/backend/deep/.venv/bin/python"
GEN="/home/ec2-user/ai-voice-train-data/backend/deep/generate_ai_tts.py"
TEXT_DIR="/home/ec2-user/ai-voice-train-data/backend/data/_downloads/texts"
OUT_BASE="/home/ec2-user/ai-voice-train-data/backend/data/train/ai"
LOG="/home/ec2-user/ai-voice-train-data/PROGRESS_LOG.md"

log() {
  printf "%s — %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG"
}

run_lang() {
  local lang="$1"
  local text_file="${TEXT_DIR}/${lang}.txt"
  local out_dir="${OUT_BASE}/${lang}/generated_edge_200"
  if [ ! -f "$text_file" ]; then
    log "Edge TTS generation skipped: ${lang} text file missing (${text_file})"
    return 0
  fi
  log "Edge TTS generation start: ${lang} (200 samples, multi-voice)"
  taskset -c 0-3 nice -n 10 "$VENV_PY" "$GEN" \
    --text "$text_file" \
    --out "$out_dir" \
    --lang "$lang" \
    --max 200 \
    --backend edge \
    --refs 4 >> "/tmp/edge_tts_${lang}.log" 2>&1
  log "Edge TTS generation complete: ${lang}"
}

run_lang Hindi
run_lang Tamil
run_lang Malayalam
run_lang Telugu
