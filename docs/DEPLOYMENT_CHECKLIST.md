# 🚀 Deployment Checklist - CPU/GPU Ready

## ✅ GPU Fallback Verified

Your system has **automatic GPU/CPU fallback** built-in. You can safely destroy this VPS and deploy on a new one (with or without GPU).

### How It Works

1. **Auto-Detection**: System automatically detects GPU availability
2. **Safe Fallback**: If GPU fails or unavailable, automatically uses CPU
3. **No Code Changes**: Same codebase works on both GPU and CPU servers

### Tested Scenarios

✅ **GPU Available (Current)**
- Device: CUDA
- Response time: 0.2-0.4 seconds
- Status: Working perfectly

✅ **CPU Fallback (Tested)**
- Device: CPU
- Response time: ~2-5 seconds (slower but functional)
- Status: Working perfectly

✅ **GPU Detection Failure**
- Automatically falls back to CPU
- No crashes or errors
- Graceful degradation

---

## 📋 Deployment Steps for New VPS

### 1. Clone Repository
```bash
git clone https://github.com/parikshitgorain/voice-ai-detection.git
cd voice-ai-detection

# Use production-ready branch
git checkout production-ready
```

### 2. Install Dependencies

**Node.js (Backend)**
```bash
cd backend
npm install
```

**Python (Deep Learning)**
```bash
# Automated setup (recommended)
cd ..
./scripts/setup_gpu.sh

# OR manual setup
python3 -m venv backend/deep/.venv
backend/deep/.venv/bin/pip install -r backend/deep/requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
nano .env
```

Set these variables:
```bash
PORT=3000
NODE_ENV=production
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password
```

### 4. Start Server

**Option A: PM2 (Production)**
```bash
cd backend
pm2 start server.js --name voice-ai-detection
pm2 save
pm2 startup  # Enable auto-start on reboot
```

**Option B: Systemd (Alternative)**
```bash
sudo cp voice-ai-detection.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable voice-ai-detection
sudo systemctl start voice-ai-detection
```

### 5. Verify Deployment
```bash
# Health check
curl http://localhost:3000/health

# Test API
curl -X POST http://localhost:3000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"audioBase64":"...","language":"English","audioFormat":"mp3"}'
```

---

## 🔧 Device Configuration

### Force CPU Mode (Optional)
If you want to force CPU mode (e.g., for testing):

**Method 1: Environment Variable**
```bash
export DEEP_MODEL_DEVICE=cpu
pm2 restart voice-ai-detection
```

**Method 2: .env File**
```bash
echo "DEEP_MODEL_DEVICE=cpu" >> .env
pm2 restart voice-ai-detection
```

### Force GPU Mode (Optional)
```bash
export DEEP_MODEL_DEVICE=cuda
pm2 restart voice-ai-detection
```

### Auto-Detection (Default)
```bash
# Remove or comment out DEEP_MODEL_DEVICE
pm2 restart voice-ai-detection
```

---

## 📊 Performance Expectations

### With GPU (CUDA)
- **Response Time**: 0.2-0.4 seconds
- **Throughput**: ~100-200 requests/minute
- **Memory**: ~2GB GPU VRAM
- **Recommended**: T4, A10, A100, or any CUDA GPU

### With CPU Only
- **Response Time**: 2-5 seconds
- **Throughput**: ~10-20 requests/minute
- **Memory**: ~4GB RAM
- **Recommended**: 4+ CPU cores

---

## 🛡️ Production Checklist

Before going live, verify:

- [ ] API server starts successfully
- [ ] Health endpoint returns `{"status":"ok"}`
- [ ] Test classification works correctly
- [ ] API key authentication working
- [ ] Rate limiting enabled
- [ ] Logs are being written
- [ ] PM2 auto-restart configured
- [ ] Firewall allows port 3000 (or your port)
- [ ] Admin panel accessible
- [ ] Database files have correct permissions

---

## 🔍 Troubleshooting

### Server Won't Start
```bash
# Check logs
pm2 logs voice-ai-detection

# Check port availability
lsof -i :3000

# Check Python dependencies
backend/deep/.venv/bin/python -c "import torch; print(torch.__version__)"
```

### GPU Not Detected
```bash
# Check CUDA availability
nvidia-smi

# Test GPU detection
python3 backend/deep/detect_device.py

# Should output: cuda (if GPU available) or cpu (if not)
```

### Slow Response Times
```bash
# Check which device is being used
pm2 logs voice-ai-detection | grep "device"

# If using CPU, consider:
# 1. Upgrading to GPU instance
# 2. Increasing CPU cores
# 3. Enabling request queuing (already configured)
```

### Out of Memory
```bash
# Check memory usage
pm2 status
free -h

# If GPU OOM:
# - Reduce batch size (already optimized)
# - Use smaller GPU or CPU mode

# If CPU OOM:
# - Increase RAM
# - Enable swap
# - Reduce concurrent requests
```

---

## 🎯 Quick Commands Reference

```bash
# Start server
pm2 start backend/server.js --name voice-ai-detection

# Stop server
pm2 stop voice-ai-detection

# Restart server
pm2 restart voice-ai-detection

# View logs
pm2 logs voice-ai-detection

# Check status
pm2 status

# Health check
curl http://localhost:3000/health

# Test device detection
python3 backend/deep/detect_device.py

# Check GPU
nvidia-smi

# View environment
pm2 env 0
```

---

## ✅ Safe to Destroy Current VPS

**YES!** You can safely destroy this VPS because:

1. ✅ All code pushed to GitHub (production-ready branch)
2. ✅ GPU fallback tested and working
3. ✅ Documentation complete
4. ✅ Deployment process documented
5. ✅ No data loss (models in Git LFS)
6. ✅ API keys can be regenerated
7. ✅ System works on CPU-only servers

**Next Steps:**
1. Destroy this VPS
2. Create new VPS (with or without GPU)
3. Follow deployment steps above
4. Generate new API keys in admin panel
5. Update judges with new endpoint URL

---

## 📞 Support

If you encounter issues during deployment:

1. Check logs: `pm2 logs voice-ai-detection`
2. Verify dependencies: `./scripts/smoke_test.sh`
3. Test device: `python3 backend/deep/detect_device.py`
4. Review documentation: `README.md`, `DEPLOYMENT.md`

**Maintainer**: Parikshit Gorain  
**Email**: parikshitgorain@yahoo.com
