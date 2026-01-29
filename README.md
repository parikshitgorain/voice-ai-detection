# Project Structure (Locked)

This repository has a finalized structure. Changes to layout or service boundaries
should only happen with an explicit design review.

Key directories:
- `backend/` API + audio pipeline + classifier services
- `frontend/` Single-page UI

For backend service boundaries, see `backend/services/README.md`.

## Audio Handling & Privacy (API Contract)
- Audio is processed transiently and deleted after analysis completes.
- No raw audio, Base64 payloads, or PCM are stored or persisted.
- No audio is associated with user identity.

## Setup (Local)
Prereqs:
- Node.js (backend)
- ffmpeg + ffprobe available on PATH

Install backend dependencies:
```
cd backend
npm install
```

Run backend:
```
node server.js
```

Run frontend (static):
```
python -m http.server 5173 --directory frontend
```

Note: the frontend posts to `http://localhost:3000/api/voice-detection`, so the
backend must be running. CORS is enabled for `http://localhost:5173` in dev.

## API Contract (JSON Only)
Endpoint:
```
POST /api/voice-detection
```

Headers:
```
x-api-key: <key>
Content-Type: application/json
```

Body:
```
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64>"
}
```

Success response:
```
{
  "classification": "AI_GENERATED" | "HUMAN",
  "confidenceScore": 0.0,
  "explanation": "Short technical explanation",
  "languageWarning": false,
  "languageWarningReason": "Selected language \"Hindi\" may be incorrect. Detected \"Tamil\".",
  "detectedLanguage": "Hindi",
  "languageConfidence": 0.82
}
```

Error response (always JSON):
```
{
  "status": "error",
  "message": "Audio could not be processed. Please try another file."
}
```

## Speech-Presence Gate (VAD)
- Non-speech audio is rejected before classification.
- VAD is implemented with WebRTC VAD compiled to WebAssembly
  (`@ennuicastr/webrtcvad.js`) to avoid native build issues.

## Deep Model (Multitask)
The backend can fuse a deep model score into the final decision. The default
deep model path can be overridden via `DEEP_MODEL_PATH`.

Trained model artifacts (included in this repo):
- `backend/deep/multitask_English.pt`
- `backend/deep/multitask_Hindi.pt`
- `backend/deep/multitask_Tamil.pt`
- `backend/deep/multitask_Malayalam.pt`
- `backend/deep/multitask_Telugu.pt`

Runtime requirements (for inference):
- Python 3.9+
- `backend/deep/.venv` created and installed via
  `pip install -r backend/deep/requirements.txt`
