#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/ai-voice-train-data"
VENV_PY="${ROOT}/backend/deep/.venv/bin/python"
TRAIN_PY="${ROOT}/backend/deep/train_multitask.py"
DATA_DIR="${ROOT}/backend/data"
LOG="${ROOT}/PROGRESS_LOG.md"

LANGS=(Hindi Tamil Malayalam Telugu)
AI_TARGET=200
EPOCHS=10
BATCH=48
WORKERS=2

log() {
  printf "%s — %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG"
}

count_files() {
  local path="$1"
  if [ -d "$path" ]; then
    find "$path" -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" -o -iname "*.m4a" \) | wc -l
  else
    echo 0
  fi
}

for lang in "${LANGS[@]}"; do
  log "Auto-train waiting for AI generation: ${lang} (target ${AI_TARGET})"
  while true; do
    ai_count=$(count_files "${DATA_DIR}/train/ai/${lang}/generated_edge_200")
    human_count=$(count_files "${DATA_DIR}/train/human/${lang}")
    if [ "$ai_count" -ge "$AI_TARGET" ] && [ "$human_count" -gt 0 ]; then
      break
    fi
    sleep 60
  done

  out_model="${ROOT}/backend/deep/multitask_${lang}.pt"
  out_log="${ROOT}/backend/deep/train_multitask_${lang}.log"
  log "Auto-train start: ${lang} (AI ${AI_TARGET}+, human ${human_count})"
  nohup "$VENV_PY" -u "$TRAIN_PY" \
    --data "$DATA_DIR" \
    --out "$out_model" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --languages "$lang" \
    >> "$out_log" 2>&1
  log "Auto-train finished: ${lang} (log: ${out_log})"

done

log "Auto-train pipeline complete (Hindi->Tamil->Malayalam->Telugu)"
