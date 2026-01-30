# Voice AI Detection (Human vs AI)

Maintainer: Parikshit Gorain
Contact: parikshitgorain@yahoo.com

Production-grade system to classify uploaded audio as **HUMAN** or **AI_GENERATED** with multilingual support.

## Features
- Multilingual detection: English, Hindi, Tamil, Malayalam, Telugu
- AI vs Human classification from base64-encoded MP3
- API key authentication (`x-api-key`)
- Privacy-first: audio is processed transiently and not stored
- Concurrency control with queue and rate limits

## Architecture
- `frontend/`: static single-page UI
- `backend/`: Node.js API + deep model integration
- `backend/deep/`: Python inference code + model weights

## Requirements (CPU VPS)
- Node.js 18+
- Python 3.9+ (tested with 3.12)
- ffmpeg + ffprobe available on PATH
- Git LFS (for model weights)

## Quick Start (Local)
```bash
cd backend
npm install

# Python venv for deep model (CPU)
python3 -m venv deep/.venv
./deep/.venv/bin/pip install -r deep/requirements.txt
# If torch/torchaudio need CPU wheels explicitly:
# ./deep/.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

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
- Set `apiBaseUrl` if backend is on another origin.
- API key is NOT stored or prefilled in the UI.

## Production Deploy (Nginx + systemd)

### 1) Backend environment
Create `/etc/voice-ai-detection.env`:
```
VOICE_DETECT_API_KEY=your-secret-key
HOST=127.0.0.1
DEEP_MODEL_DEVICE=cpu
DEEP_MODEL_PATH_ENGLISH=/var/www/voice-ai-detection/backend/deep/multitask_English.pt
DEEP_MODEL_PATH_HINDI=/var/www/voice-ai-detection/backend/deep/multitask_Hindi.pt
DEEP_MODEL_PATH_TAMIL=/var/www/voice-ai-detection/backend/deep/multitask_Tamil.pt
DEEP_MODEL_PATH_MALAYALAM=/var/www/voice-ai-detection/backend/deep/multitask_Malayalam.pt
DEEP_MODEL_PATH_TELUGU=/var/www/voice-ai-detection/backend/deep/multitask_Telugu.pt
QUEUE_MAX_CONCURRENT=3
QUEUE_MAX_LENGTH=10
```

### 2) Systemd service
Example `/etc/systemd/system/voice-ai-detection.service`:
```
[Unit]
Description=Voice AI Detection API
After=network.target

[Service]
Type=simple
WorkingDirectory=/var/www/voice-ai-detection/backend
EnvironmentFile=/etc/voice-ai-detection.env
ExecStart=/usr/bin/node server.js
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voice-ai-detection.service
```

### 3) Nginx (SSL + API proxy)
Example server block:
```nginx
server {
    server_name voiceai.example.com;

    root /var/www/voice-ai-detection/frontend;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        try_files $uri /index.html;
    }

    # Optional: disable caching
    add_header Cache-Control "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0" always;
    add_header Pragma "no-cache" always;
    add_header Expires "0" always;

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/voiceai.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voiceai.example.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
}
```

### 4) Health
`GET /health` returns `{ "status": "ok" }`.

## API

### POST /api/voice-detection
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

Success response:
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.93,
  "explanation": "Deep model estimated an AI probability of 93%."
}
```

Error responses:
- `404` for unauthorized or unknown routes (intentionally hides API)
- `400` for malformed requests
- `413` if body exceeds limits
- `429` when rate limit is exceeded
- `503` when the queue is full
- `500` for internal failures

### GET /api/queue
- Requires `x-api-key`
- Returns queue status `{ active, queued, maxConcurrent, maxQueue }`
- Returns `404` if key is missing/invalid

## Queue and Rate Limiting
- Max concurrent requests: `QUEUE_MAX_CONCURRENT` (default 3)
- Queue size: `QUEUE_MAX_LENGTH` (default 10)
- Rate limit (token bucket):
  - `maxTokens: 12`
  - `refillPerSecond: 0.2` (about 1 request per 5 seconds after burst)

## Logging
- Request log file: `/var/log/voice-ai-detection.log`
- Rotated daily (`/etc/logrotate.d/voice-ai-detection`)

## Model Weights
Model weights live in `backend/deep/*.pt` and are tracked via Git LFS.

## Training Details (Truthful)
- Training was performed on a VPS (cloud server).
- VPS specs: 4 vCPU (Intel Xeon Platinum 8259CL), 15 GiB RAM, NVIDIA Tesla T4 (16 GB VRAM).
- Data source: Hugging Face datasets.
- Total audio used: ~100 GB combined (AI + human).
- Per language: ~10 GB AI + ~10 GB human for Tamil, English, Hindi, Malayalam, Telugu.
- Per-language models were trained.
- Training time: ~1 day (~24 hours).

## Privacy
Audio is processed transiently and deleted after analysis. No audio is stored or tied to user identity.

## License
MIT License. See `LICENSE`.
