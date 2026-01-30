# Voice AI Detection (Human vs AI)

Maintainer: Parikshit Gorain
Contact: parikshitgorain@yahoo.com

Production-grade system to classify uploaded audio as **HUMAN** or **AI_GENERATED** with multilingual support.

## Features
- Multilingual detection: English, Hindi, Tamil, Malayalam, Telugu
- Deep model inference (multitask: AI vs Human + language detection)
- Language-gate safety (warn/soft/block on mismatch)
- Voice activity detection (rejects non-speech)
- Privacy-first: audio is processed transiently and not stored

## Architecture
- `frontend/`: static single-page UI
- `backend/`: Node.js API + audio pipeline + deep model integration
- `backend/deep/`: Python inference code + model weights

## Requirements
- Node.js 18+
- Python 3.9+
- ffmpeg + ffprobe available on PATH

## Quick Start (Local)
```bash
cd backend
npm install

# Python venv for deep model
python3 -m venv deep/.venv
./deep/.venv/bin/pip install -r deep/requirements.txt

# Run API
node server.js
```

Serve frontend:
```bash
python3 -m http.server 5173 --directory frontend
```

Open: `http://localhost:5173`

Frontend runtime config:
- `frontend/config.js` is loaded at runtime.
- Edit it to set `apiBaseUrl` and `apiKey` if needed.

## Production Deploy (Nginx + API)
1) Copy frontend to nginx web root:
```bash
sudo mkdir -p /var/www/voice-ai-detection
sudo rsync -av --delete frontend/ /var/www/voice-ai-detection/
```

2) Nginx site config (example):
```nginx
server {
    listen 80;
    server_name _;

    root /var/www/voice-ai-detection;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location / {
        try_files $uri /index.html;
    }
}
```

3) Start backend (Node). For production, use a process manager (systemd, PM2, etc.).

## Environment Variables
Backend config (see `backend/config.js`):
- `VOICE_DETECT_API_KEY` (required; API requests must include `x-api-key`)
- `DEEP_MODEL_PATH` (single multilingual model)
- `DEEP_MODEL_DEVICE` (default: `cuda`)
- `DEEP_MODEL_PYTHON` (default: auto-detects `backend/deep/.venv/bin/python`, else `python3`)
- Optional per-language models:
  - `DEEP_MODEL_PATH_ENGLISH`, `DEEP_MODEL_PATH_HINDI`, `DEEP_MODEL_PATH_TAMIL`,
    `DEEP_MODEL_PATH_MALAYALAM`, `DEEP_MODEL_PATH_TELUGU`
- Optional language detector:
  - `DEEP_LANG_MODEL_PATH`, `DEEP_LANG_MODEL_DEVICE`
 - `CORS_ORIGINS` (comma-separated list for non-same-origin clients; default allows localhost in dev)

## API
`POST /api/voice-detection`

Headers:
```
Content-Type: application/json
x-api-key: <key>
```

Body:
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64>"
}
```

Audio formats: `mp3` only.

Success response:
```json
{
  "classification": "AI_GENERATED",
  "confidenceScore": 0.93,
  "explanation": "Deep model estimated an AI probability of 93%.",
  "languageWarning": false,
  "languageWarningReason": null,
  "detectedLanguage": "English",
  "languageConfidence": 0.86
}
```

Error response:
```json
{ "status": "error", "message": "Audio could not be processed." }
```

## Health Check
`GET /health` returns `{ "status": "ok" }`.

## Smoke Test
```
./scripts/smoke_test.sh http://localhost:3000
```

## Model Weights
Model weights live in `backend/deep/*.pt` and are tracked via Git LFS.
If you remove LFS, store weights externally and document download steps.

## Privacy
Audio is processed transiently and deleted after analysis. No audio is stored or tied to user identity.

## Sponsorship
If you want to sponsor development or request enterprise support, contact:
parikshitgorain@yahoo.com

## License
MIT License. See `LICENSE`.
