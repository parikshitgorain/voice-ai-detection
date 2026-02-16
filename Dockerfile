# Multi-stage Dockerfile for Voice AI Detection
# Supports both CPU and GPU with automatic detection

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base-gpu
# GPU base image - will fall back to CPU if GPU not available

FROM ubuntu:22.04 AS base-cpu
# CPU-only fallback

# Choose base based on build argument
ARG USE_GPU=true
FROM ${USE_GPU:+base-gpu}${USE_GPU:-base-cpu} AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    software-properties-common \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy package files
COPY backend/package*.json ./backend/

# Install Node dependencies
RUN cd backend && npm ci --only=production

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# Create Python virtual environment and install dependencies
RUN python3.11 -m venv /app/backend/deep/.venv

# Install PyTorch based on GPU availability
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN /app/backend/deep/.venv/bin/pip install --no-cache-dir \
    torch>=2.2.0 \
    torchaudio>=2.2.0 \
    torchvision>=0.17.0 \
    --index-url ${PYTORCH_INDEX_URL}

# Install other Python dependencies
RUN /app/backend/deep/.venv/bin/pip install --no-cache-dir \
    soundfile>=0.12.1 \
    ffmpeg-python==0.2.0 \
    librosa==0.11.0 \
    pydub==0.25.1

# Create necessary directories
RUN mkdir -p /app/backend/logs /app/backend/data

# Set proper permissions
RUN chmod +x /app/scripts/*.sh /app/backend/deep/detect_device.py

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Set environment variables
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3000 \
    DEEP_MODEL_DEVICE=auto

# Run the application
CMD ["node", "/app/backend/server.js"]
