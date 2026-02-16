# Quick GPU Setup Reference

## What to Choose from Your Cloud Provider Image/Dashboard

### 🎯 RECOMMENDED GPU (Best Value)
**NVIDIA Tesla T4**
- Cost: ~$0.30-0.50/hour
- Memory: 16GB
- Perfect for AI inference
- 5-10x faster than CPU

### 🔍 What to Look For in the Image/Menu:

| Look For | Select |
|----------|--------|
| **Instance Type** | "GPU", "Accelerated", "GPU-Optimized" |
| **GPU Model** | T4, A10, V100, or A100 |
| **Driver** | "NVIDIA", "CUDA-enabled" |
| **OS** | Ubuntu 20.04+ (you have this ✓) |

### ❌ Avoid These:
- "CPU optimized" (no GPU)
- "Memory optimized" (no GPU)
- "Graphics" without NVIDIA (might be AMD/integrated)

---

## 🚀 After Creating GPU Instance

### Step 1: Verify GPU
```bash
nvidia-smi
```
Should show your GPU details.

### Step 2: Run Setup Script
```bash
cd /var/www/voice-ai-detection
./scripts/setup_gpu.sh
```

### Step 3: Restart Server
```bash
pm2 restart voice-ai-detection
# OR
sudo systemctl restart voice-ai-detection
```

### Step 4: Verify Usage
```bash
# Check what device is being used:
node -e "console.log(require('./backend/utils/gpu_helper').getDevice())"
# Should show: cuda

# Monitor GPU during API calls:
watch -n 1 nvidia-smi
```

---

## 💡 How It Works (Removable GPU)

### Current System Behavior:
```
Your VPS → Check GPU → Found? → Use GPU (fast)
                     → Not Found? → Use CPU (slower but works)
```

### You Can:
1. ✅ Start with CPU → Add GPU later → System auto-uses GPU
2. ✅ Start with GPU → Remove GPU → System auto-falls back to CPU
3. ✅ Force CPU even with GPU available: `export DEEP_MODEL_DEVICE=cpu`
4. ✅ Force GPU: `export DEEP_MODEL_DEVICE=cuda`
5. ✅ Auto-detect (default): No environment variable needed

### No Reinstall Needed!
The system adapts automatically when you:
- Attach/detach GPU
- Switch instances
- Upgrade/downgrade server

---

## 📊 Current Status

**Your System Right Now:**
- Device: `CPU` (no GPU detected)
- Works: ✅ Yes
- Speed: Normal (baseline)

**If You Add GPU:**
- Device: `CUDA` (auto-detected)
- Works: ✅ Yes  
- Speed: 5-10x faster

---

## 🎮 Cloud Provider Examples

### AWS EC2
1. Choose instance: `g4dn.xlarge` (has T4 GPU)
2. AMI: Ubuntu 20.04+
3. Instance has nvidia-smi pre-installed
4. Run setup script → Done!

### Google Cloud
1. Machine type: n1-standard-4
2. ✅ Add: "1 x NVIDIA Tesla T4"
3. Install GPU drivers: `sudo /opt/google/cuda-installer/cuda-installer`
4. Run setup script → Done!

### Azure
1. VM size: NC4as T4 v3
2. Has T4 GPU included
3. Run setup script → Done!

### Hetzner Cloud  
1. Server type: "GPU-enabled"
2. Select: Nvidia GPU option
3. Run setup script → Done!

---

## 🆘 Can't Find the Screenshot?

Send me details about:
1. What cloud provider? (AWS/GCP/Azure/other)
2. What page are you on? (Instance creation/VM settings)
3. What options do you see?

I'll tell you exactly what to select!

---

## 🎯 Summary

**For Your PNG Image:**
- Look for **"GPU"** or **"NVIDIA"** keywords
- Choose **T4** if available (best value)
- If no T4, choose **A10** or **any NVIDIA GPU**
- Avoid non-GPU instances

**After Setup:**
- GPU = Removable extension ✅
- Auto-falls back to CPU ✅
- No code changes needed ✅
- Works on any VPS ✅

**Ready to use!** 🚀
