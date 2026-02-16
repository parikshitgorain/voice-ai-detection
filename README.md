# 🎙️ Voice AI Detection API

**Production-grade AI voice detection system with multilingual support**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org/)
[![Score](https://img.shields.io/badge/Score-90%2F100-brightgreen.svg)](PRODUCTION_READY.md)

Classify audio as **HUMAN** or **AI_GENERATED** with high accuracy across 5 languages.

---

## ✨ Features

- 🌍 **Multilingual**: English, Hindi, Tamil, Malayalam, Telugu
- 🎯 **High Accuracy**: 90/100 score (Grade A)
- ⚡ **Fast**: 0.2-0.4s response time (GPU) or 2-5s (CPU)
- 🔒 **Secure**: API key authentication
- 🔄 **Auto-Fallback**: GPU → CPU automatic fallback
- 📊 **Production Ready**: Rate limiting, logging, monitoring

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/parikshitgorain/voice-ai-detection.git
cd voice-ai-detection
git checkout production-ready

# Install dependencies
cd backend && npm install

# Setup Python environment (auto-detects GPU/CPU)
cd .. && ./scripts/setup_gpu.sh

# Start server
cd backend && node server.js
```

Server runs at `http://localhost:3000`

**See [Quick Start Guide](docs/QUICK_START.md) for detailed instructions.**

---

## 📡 API Usage

```bash
# Classify audio
curl -X POST http://localhost:3000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "audioBase64": "base64_encoded_audio",
    "language": "English",
    "audioFormat": "mp3"
  }'

# Response
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.97
}
```

**See [API Reference](docs/API_REFERENCE.md) for complete documentation.**

---

## 📚 Documentation

### Getting Started
- [Quick Start Guide](docs/QUICK_START.md) - Setup in 5 minutes
- [API Reference](docs/API_REFERENCE.md) - Complete API docs
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment

### Advanced
- [Model Training Guide](docs/MODEL_TRAINING.md) - Train custom models
- [GPU Configuration](docs/GPU_CONFIGURATION.md) - GPU optimization
- [Deployment Checklist](docs/DEPLOYMENT_CHECKLIST.md) - Pre-deployment verification

### Reference
- [Production Ready Info](PRODUCTION_READY.md) - Hackathon submission details
- [Project Structure](PROJECT_STRUCTURE.md) - File organization
- [Contributing](CONTRIBUTING.md) - Contribution guidelines
- [Security](SECURITY.md) - Security best practices

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Frontend  │─────▶│  Node.js API │─────▶│  Python Models  │
│   (Web UI)  │      │   (Express)  │      │   (PyTorch)     │
└─────────────┘      └──────────────┘      └─────────────────┘
                            │
                            ├─ Authentication
                            ├─ Rate Limiting
                            ├─ Request Queue
                            └─ Logging
```

**Components:**
- **Frontend**: Single-page web UI for audio upload
- **Backend API**: Node.js/Express REST API
- **Deep Models**: PyTorch ResNet18-based models (5 languages)
- **Admin Panel**: API key management and monitoring

---

## 🎯 Performance

| Metric | GPU Mode | CPU Mode |
|--------|----------|----------|
| Response Time | 0.2-0.4s | 2-5s |
| Accuracy | 100% (5/5) | 100% (5/5) |
| Score | 90/100 | 90/100 |
| Throughput | 100-200 req/min | 10-20 req/min |

---

## 🛠️ Requirements

**System:**
- Node.js 18+
- Python 3.9+
- 4GB RAM (8GB recommended)
- Optional: NVIDIA GPU with CUDA 11.8+

**Software:**
- ffmpeg (audio processing)
- Git LFS (model weights)
- PM2 (production deployment)

---

## 📦 Installation

### Option 1: Automated Setup (Recommended)

```bash
git clone https://github.com/parikshitgorain/voice-ai-detection.git
cd voice-ai-detection
./scripts/setup_gpu.sh  # Auto-detects GPU/CPU
cd backend && npm install && node server.js
```

### Option 2: Manual Setup

```bash
# Backend
cd backend
npm install

# Python environment
python3 -m venv backend/deep/.venv
backend/deep/.venv/bin/pip install -r backend/deep/requirements.txt

# Start server
node server.js
```

---

## 🌐 Deployment

### Production Deployment

```bash
# Using PM2 (recommended)
cd backend
pm2 start server.js --name voice-ai-detection
pm2 save
pm2 startup

# Using Systemd
sudo cp voice-ai-detection.service /etc/systemd/system/
sudo systemctl enable voice-ai-detection
sudo systemctl start voice-ai-detection
```

**See [Deployment Guide](docs/DEPLOYMENT.md) for detailed instructions.**

---

## 🔑 API Keys

Generate API keys from the admin panel:

1. Navigate to `http://localhost:3000/admin`
2. Login with admin credentials
3. Go to "API Keys" section
4. Click "Generate New Key"
5. Copy and save the key (shown only once)

---

## 🧪 Testing

```bash
# Health check
curl http://localhost:3000/health

# Test classification
./scripts/smoke_test.sh
```

---

## 📊 Monitoring

**Admin Panel:** `http://localhost:3000/admin`

Features:
- API key management
- Usage statistics
- System logs
- Performance metrics

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Maintainer

**Parikshit Gorain**  
Email: parikshitgorain@yahoo.com  
GitHub: [@parikshitgorain](https://github.com/parikshitgorain)

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- ResNet architecture by Microsoft Research
- Open-source community for various tools and libraries

---

## 📈 Project Status

- ✅ Production Ready (Grade A - 90/100)
- ✅ GPU/CPU Auto-Fallback
- ✅ 5 Languages Supported
- ✅ Complete Documentation
- ✅ Hackathon Submission Ready

**Last Updated:** February 16, 2026
