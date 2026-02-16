# GPU Configuration Guide

This guide explains how to add GPU support to your Voice AI Detection system with automatic CPU fallback.

## Overview

The system now supports **automatic GPU detection** with seamless CPU fallback:
- If GPU is available → uses GPU (faster)
- If GPU is not available → uses CPU (slower but works)
- GPU can be attached/removed at any time (like a removable extension)

## What GPU to Choose

### Cloud Providers (AWS, Azure, GCP, etc.)

When creating a VM/instance, look for these GPU options:

#### Budget-Friendly (Good for Inference)
- **NVIDIA Tesla T4** - ~$0.35/hour
  - 16GB memory
  - Great for AI inference
  - Best value for money

#### Mid-Range
- **NVIDIA A10** - ~$1.00/hour
  - 24GB memory
  - Good for larger models

#### High-Performance
- **NVIDIA A100** - ~$3.00/hour
  - 40GB/80GB memory
  - Overkill for inference unless high volume

### What to Look For

In your cloud provider dashboard (the PNG image you mentioned), choose:
- **Instance Type**: Look for "GPU", "Accelerated Computing", or "GPU-enabled"
- **GPU Type**: Select any of:
  - T4 (recommended for cost)
  - A10 / A10G
  - V100
  - A100 (if budget allows)
- **Driver**: Choose "NVIDIA" or "CUDA-enabled"

### Keywords to Search
- "GPU instances"
- "CUDA"
- "NVIDIA"
- "Accelerated compute"
- "Graphics-optimized"

## Setup Instructions

### 1. Initial Setup (First Time)

Run the automated setup script:
```bash
cd /var/www/voice-ai-detection
./scripts/setup_gpu.sh
```

This script will:
- Detect if GPU is present
- Install correct PyTorch version (GPU or CPU)
- Configure automatic device detection
- Test GPU functionality

### 2. Check Current Device

Test what device your system will use:
```bash
cd /var/www/voice-ai-detection/backend
./deep/.venv/bin/python deep/detect_device.py
```

Output:
- `cuda` = GPU will be used ✓
- `cpu` = CPU will be used ⚠

### 3. Manual Control (Optional)

#### Force GPU Usage
```bash
export DEEP_MODEL_DEVICE=cuda
```

#### Force CPU Usage
```bash
export DEEP_MODEL_DEVICE=cpu
```

#### Auto-Detect (Default)
```bash
export DEEP_MODEL_DEVICE=auto
# OR simply unset:
unset DEEP_MODEL_DEVICE
```

## GPU as "Removable Extension"

### Scenario 1: Start with CPU, Add GPU Later

1. **Initial setup (CPU-only VPS)**:
   ```bash
   ./scripts/setup_gpu.sh
   # Installs CPU version
   node backend/server.js
   # Runs on CPU
   ```

2. **Later: Attach GPU** (e.g., add GPU to VM):
   ```bash
   # Reinstall with GPU support
   ./scripts/setup_gpu.sh
   # Auto-detects GPU, installs CUDA version
   
   # Restart server
   pm2 restart voice-ai-detection
   # Now uses GPU!
   ```

### Scenario 2: Start with GPU, Remove Later

1. **With GPU**:
   ```bash
   ./scripts/setup_gpu.sh
   # Detects GPU, uses CUDA
   ```

2. **Remove GPU** (e.g., downgrade instance):
   ```bash
   # System automatically falls back to CPU
   # No reinstall needed!
   
   # Or explicitly force CPU:
   export DEEP_MODEL_DEVICE=cpu
   pm2 restart voice-ai-detection
   ```

## Production Deployment

### Update Environment File

Edit `/etc/voice-ai-detection.env`:

```bash
# Auto-detect (recommended)
DEEP_MODEL_DEVICE=auto

# Or force specific device:
# DEEP_MODEL_DEVICE=cuda  # Force GPU
# DEEP_MODEL_DEVICE=cpu   # Force CPU
```

### Restart Service

```bash
sudo systemctl restart voice-ai-detection
# OR
pm2 restart voice-ai-detection
```

## Verifying GPU Usage

### Method 1: Check Logs
```bash
# System logs show detected device at startup
pm2 logs voice-ai-detection | grep -i device
```

### Method 2: Monitor GPU
```bash
# If GPU is being used, you'll see activity:
watch -n 1 nvidia-smi
```

Look for:
- **GPU Memory Usage**: Should increase during inference
- **GPU Utilization**: Should spike during API calls

### Method 3: API Test
```bash
# Make API call and check logs
curl -X POST http://localhost:3000/api/detect \
  -H "x-api-key: your-key" \
  -F "audio=@test.mp3"

# Check which device was used in logs
```

## Performance Comparison

| Device | Speed | Cost | When to Use |
|--------|-------|------|-------------|
| CPU | 1x (baseline) | Low | Low traffic, budget hosting |
| T4 GPU | ~5-10x faster | Medium | Production, moderate traffic |
| A10/A100 GPU | ~10-20x faster | High | High traffic, real-time needs |

## Troubleshooting

### GPU not detected but nvidia-smi works

```bash
# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA
cd /var/www/voice-ai-detection
./scripts/setup_gpu.sh
```

### "CUDA out of memory" error

```bash
# Force CPU mode temporarily
export DEEP_MODEL_DEVICE=cpu
pm2 restart voice-ai-detection
```

### GPU too slow / not working

```bash
# Test directly
cd /var/www/voice-ai-detection/backend
./deep/.venv/bin/python deep/infer_multitask.py \
  --audio test.mp3 \
  --model deep/multitask_English.pt \
  --device cuda
```

## Cost Optimization

### Strategy 1: Time-based GPU
- Use GPU during peak hours
- Switch to CPU during off-hours
- Save ~50-70% on GPU costs

### Strategy 2: Hybrid Setup
- GPU instance for API
- CPU instance as backup
- Load balancer switches based on traffic

### Strategy 3: On-demand GPU
- Default CPU instance
- Attach GPU when traffic increases
- Detach GPU when traffic drops

## Summary

✓ **System now auto-detects GPU/CPU**  
✓ **GPU works as removable extension**  
✓ **No code changes needed to switch**  
✓ **Automatic fallback to CPU if GPU unavailable**  

Choose GPU based on:
- **Budget**: T4 is best value
- **Speed**: A10/A100 for high performance
- **Flexibility**: All GPUs work, system adapts automatically
