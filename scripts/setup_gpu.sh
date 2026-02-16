#!/bin/bash
# GPU Setup Script for Voice AI Detection
# This script helps set up PyTorch with GPU (CUDA) support or CPU fallback

set -e

echo "=================================="
echo "GPU/CPU Setup for Voice AI Detection"
echo "=================================="
echo

# Detect if running in virtual environment
if [ -d "backend/deep/.venv" ]; then
    PYTHON="backend/deep/.venv/bin/python"
    PIP="backend/deep/.venv/bin/pip"
    echo "✓ Using existing virtual environment"
else
    echo "Creating Python virtual environment..."
    python3 -m venv backend/deep/.venv
    PYTHON="backend/deep/.venv/bin/python"
    PIP="backend/deep/.venv/bin/pip"
    echo "✓ Virtual environment created"
fi

echo

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "=================================="
    echo "GPU DETECTED"
    echo "=================================="
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo
    
    # Detect CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        echo "CUDA Version: $CUDA_VERSION"
    else
        echo "WARNING: nvcc not found, defaulting to CUDA 12.1"
        CUDA_VERSION="12.1"
    fi
    
    # Convert to PyTorch index format (e.g., 12.1 -> cu121)
    CUDA_INDEX=$(echo $CUDA_VERSION | tr -d '.')
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_INDEX}"
    
    echo
    echo "Installing PyTorch with CUDA ${CUDA_VERSION} support..."
    echo "This will enable GPU acceleration for faster inference."
    echo
    
    $PIP install torch torchaudio torchvision --index-url $TORCH_INDEX_URL
    
    echo
    echo "✓ GPU-enabled PyTorch installed"
    echo "  Device will be set to: cuda"
    
else
    echo "=================================="
    echo "NO GPU DETECTED"
    echo "=================================="
    echo "Installing CPU-only PyTorch..."
    echo "This is fine for production but slower than GPU."
    echo
    
    $PIP install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
    
    echo
    echo "✓ CPU-only PyTorch installed"
    echo "  Device will be set to: cpu"
fi

# Install other requirements
echo
echo "Installing additional dependencies..."
$PIP install -r backend/deep/requirements.txt --no-deps

echo
echo "=================================="
echo "Testing Device Detection"
echo "=================================="
DETECTED_DEVICE=$($PYTHON backend/deep/detect_device.py)
echo "Auto-detected device: $DETECTED_DEVICE"

if [ "$DETECTED_DEVICE" = "cuda" ]; then
    echo
    echo "✓✓✓ GPU IS READY ✓✓✓"
    echo "Your system will use GPU acceleration for inference."
else
    echo
    echo "⚠ CPU MODE"
    echo "Your system will use CPU for inference (slower but functional)."
fi

echo
echo "=================================="
echo "Configuration"
echo "=================================="
echo "The system automatically detects GPU/CPU."
echo "To force a specific device, set environment variable:"
echo "  export DEEP_MODEL_DEVICE=cuda   # Force GPU"
echo "  export DEEP_MODEL_DEVICE=cpu    # Force CPU"
echo "  export DEEP_MODEL_DEVICE=auto   # Auto-detect (default)"
echo
echo "✓ Setup complete!"
