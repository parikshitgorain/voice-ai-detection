# 📁 Project Structure

Clean, professional production-ready structure for hackathon judges.

```
voice-ai-detection/
│
├── 📄 README.md                      # Main documentation
├── 📄 PRODUCTION_READY.md            # Deployment info for judges
├── 📄 QUICK_START.md                 # Quick setup guide
├── 📄 DEPLOYMENT.md                  # Deployment instructions
├── 📄 HACKATHON_COMPLIANCE.md        # Compliance documentation
├── 📄 GPU_CONFIGURATION.md           # GPU setup guide
├── 📄 SYSTEMD_QUICK_GUIDE.md         # Service management
├── 📄 CONTRIBUTING.md                # Contribution guidelines
├── 📄 SECURITY.md                    # Security policy
├── 📄 LICENSE                        # MIT License
│
├── 🔧 .env                           # Environment configuration
├── 🔧 .env.example                   # Environment template
├── 🔧 voice-ai-detection.service     # Systemd service file
│
├── 📂 backend/                       # Node.js API Server
│   ├── server.js                     # Main server entry point
│   ├── config.js                     # Configuration
│   ├── package.json                  # Dependencies
│   │
│   ├── api/                          # API endpoints
│   │   ├── voice_detection.js        # Main detection endpoint
│   │   └── admin.js                  # Admin endpoints
│   │
│   ├── services/                     # Business logic
│   │   ├── voice_detection_service.js
│   │   ├── audio_pipeline.js
│   │   ├── deep_model/               # Deep learning integration
│   │   ├── audio_loader/             # Audio file handling
│   │   ├── feature_extractor/        # Audio feature extraction
│   │   └── vad/                      # Voice activity detection
│   │
│   ├── utils/                        # Utilities
│   │   ├── authentication.js         # API key auth
│   │   ├── validation.js             # Input validation
│   │   ├── response_formatter.js     # Response formatting
│   │   ├── rate_limiter.js           # Rate limiting
│   │   ├── logger.js                 # Logging
│   │   └── ...
│   │
│   ├── data/                         # Data storage
│   │   ├── api_keys.json             # API keys database
│   │   ├── admin.json                # Admin credentials
│   │   └── usage.json                # Usage statistics
│   │
│   ├── logs/                         # Application logs
│   │   └── voice-ai-detection.log
│   │
│   ├── admin/                        # Admin panel UI
│   │   ├── index.html                # Dashboard
│   │   ├── login.html                # Login page
│   │   ├── api-keys.html             # API key management
│   │   └── ...
│   │
│   └── deep/                         # Deep Learning Models
│       ├── inference_server.py       # Python inference server
│       ├── infer_multitask.py        # Model inference
│       ├── detect_device.py          # GPU/CPU detection
│       ├── requirements.txt          # Python dependencies
│       ├── multitask_English.pt      # English model (44MB)
│       ├── multitask_Hindi.pt        # Hindi model (44MB)
│       ├── multitask_Tamil.pt        # Tamil model (44MB, fine-tuned)
│       ├── multitask_Malayalam.pt    # Malayalam model (44MB)
│       └── multitask_Telugu.pt       # Telugu model (44MB)
│
├── 📂 frontend/                      # Web UI
│   ├── index.html                    # Main page
│   ├── config.js                     # Frontend config
│   ├── css/
│   │   └── style.css                 # Styles
│   └── js/
│       └── app.js                    # Frontend logic
│
├── 📂 scripts/                       # Utility scripts
│   ├── setup_gpu.sh                  # GPU setup automation
│   ├── install_gpu_deps.sh           # GPU dependencies
│   ├── generate_frontend_config.sh   # Frontend config generator
│   └── smoke_test.sh                 # Quick API test
│
└── 📂 test_data/                     # Test audio files
    ├── English_voice_AI_GENERATED.mp3
    ├── Hindi_Voice_HUMAN.mp3
    ├── Malayalam_AI_GENERATED.mp3
    ├── TAMIL_VOICE__HUMAN.mp3
    └── Telugu_Voice_AI_GENERATED.mp3
```

## 🎯 Key Components

### Backend API (`backend/`)
- **Node.js** server with Express
- RESTful API with `/api/voice-detection` endpoint
- API key authentication
- Rate limiting and request queuing
- Comprehensive logging

### Deep Learning (`backend/deep/`)
- **Python** inference server
- 5 language-specific models (English, Hindi, Tamil, Malayalam, Telugu)
- GPU acceleration with CUDA (auto-fallback to CPU)
- ResNet18-based architecture
- Multi-task learning (AI detection + language detection)

### Frontend (`frontend/`)
- Clean single-page application
- Audio file upload interface
- Real-time classification results
- Responsive design

### Admin Panel (`backend/admin/`)
- API key management
- Usage statistics
- System monitoring
- Secure authentication

## 📊 File Sizes

- **Total Models:** ~220MB (5 models × 44MB each)
- **Backend Code:** ~2MB
- **Frontend:** ~100KB
- **Documentation:** ~50KB

## 🔒 Security

- API key authentication required
- Rate limiting enabled
- Input validation on all endpoints
- Secure admin panel with bcrypt password hashing
- No audio data stored (privacy-first)

## 🚀 Production Ready

- ✅ Clean structure
- ✅ Professional documentation
- ✅ No development artifacts
- ✅ No unused files
- ✅ Optimized models
- ✅ Ready for judges
