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

## Requirements
- Node.js 18+
- Python 3.9+ (tested with 3.12)
- ffmpeg + ffprobe available on PATH
- Git LFS (for model weights)

**GPU Support (Optional):**
- NVIDIA GPU (T4, A10, A100, or any CUDA-compatible)
- System auto-detects GPU and falls back to CPU if unavailable
- See [GPU_CONFIGURATION.md](GPU_CONFIGURATION.md) for GPU setup

## Quick Start (Local)
```bash
cd backend
npm install

# Automated GPU/CPU setup (recommended)
cd ..
./scripts/setup_gpu.sh

# OR manual setup:
# python3 -m venv backend/deep/.venv
# backend/deep/.venv/bin/pip install -r backend/deep/requirements.txt

# Run API
cd backend
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

## Production Deployment

This project uses **systemd** for process management, which provides:
- Automatic restart on crash
- Start on system boot
- Integrated logging
- Resource limits
- Security hardening

### 1) Environment Configuration
Create `.env` file in project root or set environment variables:
```bash
VOICE_DETECT_API_KEY=your-secret-key
HOST=127.0.0.1
CORS_ORIGINS=https://yourdomain.com
DEEP_MODEL_DEVICE=auto
DEEP_MODEL_PATH_ENGLISH=/path/to/backend/deep/multitask_English.pt
DEEP_MODEL_PATH_HINDI=/path/to/backend/deep/multitask_Hindi.pt
DEEP_MODEL_PATH_TAMIL=/path/to/backend/deep/multitask_Tamil.pt
DEEP_MODEL_PATH_MALAYALAM=/path/to/backend/deep/multitask_Malayalam.pt
DEEP_MODEL_PATH_TELUGU=/path/to/backend/deep/multitask_Telugu.pt
QUEUE_MAX_CONCURRENT=8
QUEUE_MAX_LENGTH=20
```

### 2) Systemd Service (Auto-restart on Crash)

Copy the service file to systemd:
```bash
sudo cp voice-ai-detection.service /etc/systemd/system/
sudo systemctl daemon-reload
```

Enable and start the service:
```bash
sudo systemctl enable voice-ai-detection
sudo systemctl start voice-ai-detection
sudo systemctl status voice-ai-detection
```

The service will:
- ✓ Start automatically on system boot
- ✓ Restart automatically if it crashes (within 5 seconds)
- ✓ Log to system journal
- ✓ Run with proper security restrictions

Manage the service:
```bash
# Check status
sudo systemctl status voice-ai-detection

# View logs
sudo journalctl -u voice-ai-detection -f

# Restart
sudo systemctl restart voice-ai-detection

# Stop
sudo systemctl stop voice-ai-detection

# Disable auto-start
sudo systemctl disable voice-ai-detection
```

### 3) Nginx Configuration (SSL + API Proxy)
Example server block:
```nginx
server {
    server_name yourdomain.com;

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

    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### 4) Health Check
`GET /health` returns `{ "status": "ok" }`.

Test your deployment:
```bash
curl http://localhost:3000/health
```

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
