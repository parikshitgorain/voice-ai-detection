#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:3000}"
API_KEY="${VOICE_DETECT_API_KEY:-change-me}"

header_args=("-H" "Content-Type: application/json")
if [ -n "$API_KEY" ]; then
  header_args+=("-H" "x-api-key: ${API_KEY}")
fi

echo "== Health check =="
curl -sS "${BASE_URL}/health" | cat

echo

echo "== Voice detection (expected 400 on dummy payload) =="
set +e
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE_URL}/api/voice-detection" \
  "${header_args[@]}" \
  -d '{"language":"English","audioFormat":"mp3","audioBase64":"AAA"}' | cat
set -e
