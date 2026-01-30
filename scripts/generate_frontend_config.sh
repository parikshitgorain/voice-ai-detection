#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/frontend/config.js"
ENV_FILE="${VOICE_AI_ENV_FILE:-/etc/voice-ai.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

API_BASE_URL="${VOICE_AI_API_BASE_URL:-}"
API_KEY="${VOICE_AI_API_KEY:-}"

cat > "${OUT}" <<EOF2
window.VOICE_AI_CONFIG = {
  // Optional base URL for API (empty = same origin)
  apiBaseUrl: "${API_BASE_URL}",
  // API key required by backend
  apiKey: "${API_KEY}",
};
EOF2

echo "Wrote ${OUT}"
