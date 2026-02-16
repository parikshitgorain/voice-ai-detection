# Voice AI Detection - Project Structure

## Overview
Production-grade system to classify audio as HUMAN or AI_GENERATED with multilingual support.

## Directory Structure

```
voice-ai-detection/
├── backend/                    # Node.js API server
│   ├── admin/                  # Admin panel UI
│   ├── api/                    # API route handlers
│   ├── config.js               # Configuration management
│   ├── data/                   # Runtime data (API keys, usage)
│   ├── deep/                   # Python inference models
│   │   ├── detect_device.py    # GPU/CPU detection
│   │   ├── inference_server.py # Persistent inference server
│   │   ├── infer_multitask.py  # Multitask inference script
│   │   ├── requirements.txt    # Python dependencies
│   │   └── multitask_*.pt      # Model weights (5 languages)
│   ├── logs/                   # Application logs
│   ├── server.js               # Main server entry point
│   ├── services/               # Core business logic
│   │   ├── audio_loader/       # MP3 decoding
│   │   ├── audio_pipeline.js   # Audio processing pipeline
│   │   ├── deep_model/         # Deep learning integration
│   │   ├── feature_extractor/  # Audio feature extraction
│   │   ├── vad/                # Voice activity detection
│   │   └── voice_detection_service.js
│   └── utils/                  # Utility functions
│       ├── admin.js
│       ├── authentication.js
│       ├── client_ip.js
│       ├── gpu_helper.js
│       ├── logger.js
│       ├── rate_limiter.js
│       ├── replay_cache.js
│       ├── request_queue.js
│       └── validation.js
│
├── frontend/                   # Static web UI
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   ├── config.js               # Runtime configuration
│   └── index.html
│
├── scripts/                    # Setup and utility scripts
│   ├── generate_frontend_config.sh
│   ├── install_gpu_deps.sh     # GPU dependency installer
│   ├── setup_gpu.sh            # Automated GPU/CPU setup
│   ├── smoke_test.sh           # API smoke tests
│   └── test_admin.sh           # Admin panel tests
│
├── .env.example                # Environment template
├── .gitignore
├── CLEANUP_SUMMARY.md          # Cleanup report
├── CONTRIBUTING.md             # Contribution guidelines
├── DEPLOYMENT.md               # Deployment & systemd guide
├── DESIGN.md                   # System design document
├── GPU_CONFIGURATION.md        # GPU setup guide
├── LICENSE
├── PROJECT_STRUCTURE.md        # This file
├── README.md                   # Main documentation
├── SECURITY.md                 # Security policy
└── voice-ai-detection.service  # systemd service file

```

## Key Files

### Backend
- `backend/server.js` - Main HTTP server with CORS, rate limiting, queue management
- `backend/config.js` - Centralized configuration with GPU auto-detection
- `backend/api/voice_detection.js` - Voice detection API handler
- `backend/services/audio_pipeline.js` - Audio processing orchestration
- `backend/services/deep_model/persistent_server.js` - Fast GPU inference server

### Frontend
- `frontend/index.html` - Single-page application
- `frontend/js/app.js` - Client-side logic
- `frontend/config.js` - Runtime API configuration

### Python Models
- `backend/deep/infer_multitask.py` - Inference script for all languages
- `backend/deep/inference_server.py` - Persistent server for fast inference
- `backend/deep/multitask_*.pt` - Trained models (English, Hindi, Tamil, Malayalam, Telugu)

### Scripts
- `scripts/setup_gpu.sh` - Automated GPU/CPU setup with PyTorch
- `scripts/smoke_test.sh` - Quick API validation
- `scripts/test_admin.sh` - Admin panel system test

## Technology Stack

### Backend
- Node.js 18+
- Express-like HTTP server (native http module)
- JWT authentication
- bcrypt for password hashing

### Python/ML
- PyTorch (GPU/CPU)
- torchaudio
- librosa (audio processing)
- CUDA support (optional)

### Frontend
- Vanilla JavaScript
- CSS3
- No build tools required

## Deployment

### Requirements
- Node.js 18+
- Python 3.9+
- ffmpeg + ffprobe
- Optional: NVIDIA GPU with CUDA

### Quick Start
```bash
# Backend setup
cd backend
npm install

# Python setup (automated)
cd ..
./scripts/setup_gpu.sh

# Start server
cd backend
node server.js
```

### Production
- Use systemd for process management
- Nginx for reverse proxy and SSL
- See README.md for detailed instructions

## Features

- Multilingual support (5 languages)
- GPU acceleration with CPU fallback
- API key authentication
- Rate limiting and request queuing
- Admin panel for API key management
- Privacy-first (no audio storage)
- Persistent inference server for fast responses

## Documentation

- `README.md` - Setup and deployment guide
- `DEPLOYMENT.md` - systemd process management guide
- `DESIGN.md` - System architecture and design decisions
- `GPU_CONFIGURATION.md` - GPU setup and optimization
- `SECURITY.md` - Security policy and reporting
- `CONTRIBUTING.md` - Contribution guidelines
- `PROJECT_STRUCTURE.md` - Project overview (this file)
- `CLEANUP_SUMMARY.md` - Cleanup report
