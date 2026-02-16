#!/bin/bash
# Install GPU dependencies for fast inference

set -e

echo "=== Voice AI Detection - GPU Setup ==="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "⚠️  Please don't run as root (use regular user)"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found"
    echo "Please install NVIDIA drivers first:"
    echo "  sudo apt update"
    echo "  sudo apt install nvidia-driver-535"
    echo "  sudo reboot"
    exit 1
fi

echo "✓ NVIDIA GPU detected"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

echo ""
echo "CUDA Version: $CUDA_VERSION"

# Determine PyTorch index URL
if [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    echo "Using PyTorch for CUDA 12.x"
elif [ "$CUDA_MAJOR" -ge 11 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    echo "Using PyTorch for CUDA 11.x"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "⚠️  CUDA version too old, using CPU version"
fi

echo ""
echo "=== Installing System Dependencies ==="
sudo apt update
sudo apt install -y ffmpeg libsndfile1 python3-pip

echo ""
echo "=== Installing PyTorch ==="
pip3 install --upgrade pip
pip3 install torch torchaudio torchvision --index-url $TORCH_INDEX

echo ""
echo "=== Installing Audio Processing Libraries ==="
pip3 install librosa soundfile ffmpeg-python pydub

echo ""
echo "=== Verifying Installation ==="

# Test PyTorch
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch $TORCH_VERSION"
else
    echo "❌ PyTorch import failed"
    exit 1
fi

# Test CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✓ CUDA available: $GPU_NAME"
else
    echo "⚠️  CUDA not available (will use CPU)"
fi

# Test librosa
if python3 -c "import librosa" 2>/dev/null; then
    LIBROSA_VERSION=$(python3 -c "import librosa; print(librosa.__version__)")
    echo "✓ Librosa $LIBROSA_VERSION"
else
    echo "❌ Librosa import failed"
    exit 1
fi

echo ""
echo "=== Testing GPU Detection ==="
DEVICE=$(python3 backend/deep/detect_device.py 2>/dev/null)
echo "Detected device: $DEVICE"

if [ "$DEVICE" = "cuda" ]; then
    echo "✓ GPU ready for inference"
else
    echo "⚠️  GPU not detected, will use CPU"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Start server: cd backend && node server.js"
echo "2. Or use systemd: sudo systemctl start voice-ai-detection"
echo ""
echo "Configuration:"
echo "  DEEP_MODEL_DEVICE=cuda (in .env)"
echo "  USE_PERSISTENT_SERVER=true (in .env)"
echo "  QUEUE_MAX_CONCURRENT=8 (in .env)"
echo ""
