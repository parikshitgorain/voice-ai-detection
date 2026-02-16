# Docker Setup Guide for Voice AI Detection

## 🎯 What to Choose in Your PNG Image

Since I cannot view your image directly, here's what to look for:

### If You're Looking at Docker Settings/Options:

#### 🔥 For GPU Support:
```
✅ CHOOSE THESE:
□ Runtime: nvidia / NVIDIA Container Runtime
□ GPU: Enabled / Available
□ Device: /dev/nvidia0 (or similar)
□ Capabilities: compute, utility
□ GPU Count: 1 (or "all")
□ Driver: nvidia
```

#### 💻 For CPU-Only:
```
✅ CHOOSE THESE:
□ Runtime: runc / default
□ No GPU options needed
□ Memory: 2GB+ recommended (4GB better)
□ CPU: 2+ cores recommended
```

#### ⚙️ Common Docker Settings:
```
✅ RECOMMENDED:
Network Mode: bridge (default)
Restart Policy: unless-stopped
Health Check: enabled
Ports: 3000:3000
Volumes: enabled (for logs/data persistence)
Security: no privileged mode needed
```

---

## 🚀 Quick Start

### Option 1: GPU-Enabled (Recommended if you have GPU)

```bash
# 1. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 2. Build and run
cd /var/www/voice-ai-detection
cp .env.docker .env
# Edit .env and set your API key!

docker-compose up -d

# 3. Verify GPU usage
docker exec voice-ai-detection nvidia-smi
```

### Option 2: CPU-Only (For VPS without GPU)

```bash
cd /var/www/voice-ai-detection
cp .env.docker .env
# Edit .env and set your API key!

docker-compose -f docker-compose.cpu.yml up -d
```

---

## 📋 Docker Commands Cheat Sheet

### Build & Run
```bash
# GPU version
docker-compose up -d --build

# CPU version
docker-compose -f docker-compose.cpu.yml up -d --build

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d
```

### Monitor & Logs
```bash
# View logs
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100

# Check status
docker-compose ps

# Check GPU usage (if GPU version)
docker exec voice-ai-detection nvidia-smi
docker exec voice-ai-detection python3 /app/backend/deep/detect_device.py
```

### Manage Containers
```bash
# Stop
docker-compose down

# Restart
docker-compose restart

# Remove and clean
docker-compose down -v
docker system prune -a
```

### Execute Commands Inside Container
```bash
# Check device
docker exec voice-ai-detection node -e "console.log(require('./backend/utils/gpu_helper').getDevice())"

# Shell access
docker exec -it voice-ai-detection bash

# Check Python environment
docker exec voice-ai-detection /app/backend/deep/.venv/bin/python --version
```

---

## 🎮 Your PNG Image - What to Look For

### If it shows **Docker Desktop Settings**:
1. **General Tab**:
   - ✅ Enable: "Use the WSL 2 based engine" (if on Windows)
   - ✅ Enable: "Expose daemon on tcp://localhost:2375"

2. **Resources Tab**:
   - CPUs: 2+ (4 recommended)
   - Memory: 4GB+ (8GB recommended)
   - Swap: 1GB+
   - Disk: 20GB+

### If it shows **Container Creation UI**:
1. **Image**: `voice-ai-detection:latest`
2. **Ports**: `3000:3000`
3. **Volumes**:
   - `/var/www/voice-ai-detection/backend/logs` → `/app/backend/logs`
   - `/var/www/voice-ai-detection/backend/data` → `/app/backend/data`
4. **Environment Variables**: (from .env file)
5. **Restart**: `unless-stopped`
6. **Network**: `bridge`

### If it shows **Docker Compose Options**:
1. **File**: `docker-compose.yml` (GPU) or `docker-compose.cpu.yml` (CPU)
2. **Project Name**: `voice-ai-detection`
3. **Environment File**: `.env`
4. **Build**: Rebuild images

### If it shows **Cloud Provider Docker Options**:
Look at [QUICK_GPU_GUIDE.md](QUICK_GPU_GUIDE.md) - same GPU selection applies!

---

## 🔍 Specific Settings Explained

### GPU Runtime Settings (if available in image):
```yaml
# What each setting means:
runtime: nvidia              # Use NVIDIA GPU runtime
gpu: enabled                 # Enable GPU passthrough
device: /dev/nvidia0         # GPU device file
count: 1                     # Number of GPUs (1 is fine)
capabilities: [gpu]          # What GPU can do (compute, graphics, etc.)
```

### Memory Settings:
```
Minimum: 2GB    # Will work but slow
Recommended: 4GB # Good performance
Optimal: 8GB+    # Best performance, handles multiple requests
```

### Network Settings:
```
bridge (default)    # ✅ Recommended - isolated network
host                # ⚠️ Use only if you know what you're doing
none                # ❌ Won't work - needs network access
```

### Restart Policies:
```
unless-stopped  # ✅ Recommended - auto-restart except manual stop
always          # ✅ Also fine - always restart
on-failure      # ⚠️ OK but might not restart after manual stop
no              # ❌ No auto-restart
```

---

## 📊 Which Docker Setup to Choose?

| Scenario | Use | Command |
|----------|-----|---------|
| Have GPU, want speed | GPU version | `docker-compose up -d` |
| No GPU, budget VPS | CPU version | `docker-compose -f docker-compose.cpu.yml up -d` |
| Testing locally | CPU version | `docker-compose -f docker-compose.cpu.yml up -d` |
| Production with GPU | GPU version | `docker-compose up -d` |
| Production without GPU | CPU version | `docker-compose -f docker-compose.cpu.yml up -d` |

---

## 🆘 Troubleshooting

### "Could not select device driver 'nvidia'"
**Problem**: Docker can't find NVIDIA runtime  
**Solution**:
```bash
# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# OR use CPU version instead:
docker-compose -f docker-compose.cpu.yml up -d
```

### "Port 3000 already in use"
**Problem**: Another service using port 3000  
**Solution**:
```bash
# Stop existing service
sudo systemctl stop voice-ai-detection
# OR
pm2 stop voice-ai-detection

# Then start Docker
docker-compose up -d
```

### "Out of memory"
**Problem**: Not enough RAM for PyTorch models  
**Solution**:
1. Increase Docker memory limit (Docker Desktop → Resources)
2. Or reduce concurrent requests: `QUEUE_MAX_CONCURRENT=1`

### Container exits immediately
**Problem**: Configuration error  
**Solution**:
```bash
# Check logs
docker-compose logs

# Common issues:
# - API key not set (.env file)
# - Model files not found (check paths)
# - Python dependencies failed
```

---

## 🎯 Summary: What to Select

### In Your PNG/Screenshot:

**If it's asking about GPU:**
- ✅ Enable GPU: Yes
- ✅ Runtime: nvidia
- ✅ Device: any NVIDIA GPU
- ✅ Count: 1

**If it's asking about resources:**
- CPUs: 2-4
- Memory: 4-8GB
- Storage: 20GB+

**If it's asking about networking:**
- Network: bridge
- Ports: 3000:3000
- Expose: Yes

**If it's asking about restart:**
- Policy: unless-stopped
- Auto-restart: Yes

**If unsure:**
- Use CPU version (safer, works everywhere)
- Can always switch to GPU later

---

## 📸 Can't See Your Image?

Tell me what you see! Options might include:
1. "Docker Desktop Settings" → Tell me what tab/section
2. "Container Configuration" → Tell me what fields you see
3. "Cloud Provider Docker Options" → Tell me the provider name
4. "Docker Compose UI" → Tell me what options are shown

I'll guide you exactly! 🎯
