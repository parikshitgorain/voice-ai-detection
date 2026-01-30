# Backend

Node.js API server and audio analysis pipeline.

## Responsibilities
- HTTP routing, auth, and input validation
- Queue + rate limiting (see `config.js`)
- Orchestration of deep model inference via Python

## Key files
- `server.js`: HTTP server, auth, queue, rate limit
- `config.js`: runtime configuration and limits
- `api/`: request handlers
- `services/`: processing pipeline and inference bridge
- `deep/`: Python inference code and model weights
- `utils/`: helpers (auth, queue, logger)

## Run locally
```bash
cd backend
npm install
node server.js
```

## Environment variables
See top-level `README.md` for full env list. Most important:
- `VOICE_DETECT_API_KEY`
- `HOST` (default: 127.0.0.1)
- `QUEUE_MAX_CONCURRENT`, `QUEUE_MAX_LENGTH`

## Health
`GET /health`

## Notes
- Unauthorized requests return `404` to hide the API surface.
- `POST /api/voice-detection` requires `x-api-key` header.
