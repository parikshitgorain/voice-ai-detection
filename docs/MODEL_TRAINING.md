# 🎓 Model Training Guide

Complete guide for training and fine-tuning voice detection models.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [Training Process](#training-process)
5. [Model Evaluation](#model-evaluation)
6. [Deployment](#deployment)
7. [Best Practices](#best-practices)

---

## Overview

The Voice AI Detection system uses ResNet18-based multi-task models for:
- AI vs HUMAN classification
- Language detection (English, Hindi, Tamil, Malayalam, Telugu)
- Multi-speaker detection

### Model Architecture

```
Input: Audio (MP3/WAV) → Mel Spectrogram → ResNet18 Backbone → Multi-Task Heads
                                                                  ├─ AI Detection Head
                                                                  ├─ Language Head
                                                                  └─ Multi-Speaker Head
```

---

## Prerequisites

### System Requirements

**Minimum (CPU Training)**
- 8GB RAM
- 4 CPU cores
- 10GB disk space

**Recommended (GPU Training)**
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (T4, A10, A100)
- CUDA 11.8+
- 20GB disk space

### Software Requirements

```bash
# Python 3.9+
python3 --version

# PyTorch with CUDA (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Required packages
pip install librosa soundfile numpy tqdm
```

---

## Data Preparation

### 1. Directory Structure

Create training data directories:

```bash
training_data/
├── english/
│   ├── human/          # Human voice samples
│   │   ├── sample_0000.mp3
│   │   ├── sample_0001.mp3
│   │   └── ...
│   └── ai/             # AI-generated samples
│       ├── ai_sample_0000.mp3
│       ├── ai_sample_0001.mp3
│       └── ...
├── hindi/
│   ├── human/
│   └── ai/
├── tamil/
│   ├── human/
│   └── ai/
├── malayalam/
│   ├── human/
│   └── ai/
└── telugu/
    ├── human/
    └── ai/
```

### 2. Data Requirements

**Audio Format**
- Format: MP3, WAV, FLAC, OGG
- Sample Rate: 16kHz (will be resampled automatically)
- Duration: 3-30 seconds per sample
- Quality: Clear speech, minimal background noise

**Dataset Size (Recommended)**
- Minimum: 50 samples per class (25 HUMAN + 25 AI)
- Good: 200 samples per class (100 HUMAN + 100 AI)
- Excellent: 1000+ samples per class

**Balance**
- Keep HUMAN and AI samples balanced (50/50 split)
- Diverse speakers for HUMAN samples
- Diverse AI voices (different TTS engines)

### 3. Collecting Training Data

**HUMAN Voices**
```bash
# Option 1: Record your own
# Use any recording app, speak naturally for 5-10 seconds

# Option 2: Public datasets
# - Common Voice (Mozilla)
# - LibriSpeech
# - VoxCeleb
# - Indic TTS datasets
```

**AI-Generated Voices**
```bash
# Option 1: Use gTTS (Google TTS)
pip install gtts
python3 << 'EOF'
from gtts import gTTS
tts = gTTS("Hello, this is AI generated voice", lang='en')
tts.save("ai_sample.mp3")
EOF

# Option 2: Use other TTS engines
# - ElevenLabs
# - Azure TTS
# - Amazon Polly
# - Coqui TTS
```

---

## Training Process

### Method 1: Fine-Tuning (Recommended)

Fine-tune existing models with new data. This is faster and requires less data.

#### Step 1: Prepare Data

```bash
# Create directory structure
mkdir -p training_data/tamil/{human,ai}

# Add your audio files
cp /path/to/human/*.mp3 training_data/tamil/human/
cp /path/to/ai/*.mp3 training_data/tamil/ai/
```

#### Step 2: Create Fine-Tuning Script

Create `scripts/finetune.py`:

```python
#!/usr/bin/env python3
"""Fine-tune existing model with new data"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
from pathlib import Path

class VoiceDataset(Dataset):
    def __init__(self, data_dir, sr=16000, segment=3.0, n_mels=80, n_fft=400, hop=160):
        self.sr = sr
        self.segment_samples = int(segment * sr)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop = hop
        self.samples = []
        
        # Load HUMAN samples (label = 0)
        human_dir = Path(data_dir) / "human"
        if human_dir.exists():
            for f in human_dir.glob("*.mp3"):
                self.samples.append((str(f), 0))
        
        # Load AI samples (label = 1)
        ai_dir = Path(data_dir) / "ai"
        if ai_dir.exists():
            for f in ai_dir.glob("*.mp3"):
                self.samples.append((str(f), 1))
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"  HUMAN: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"  AI: {sum(1 for _, l in self.samples if l == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        # Load audio
        wav_np, orig_sr = librosa.load(audio_path, sr=None, mono=True)
        wav = torch.from_numpy(wav_np).unsqueeze(0)
        
        # Resample if needed
        if orig_sr != self.sr:
            wav = torchaudio.functional.resample(wav, orig_sr, self.sr)
        
        # Extract random segment
        if wav.shape[1] > self.segment_samples:
            start = np.random.randint(0, wav.shape[1] - self.segment_samples)
            wav = wav[:, start:start + self.segment_samples]
        elif wav.shape[1] < self.segment_samples:
            pad = self.segment_samples - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        
        # Compute mel spectrogram
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            f_min=40,
            f_max=self.sr // 2,
            power=2.0,
        )(wav)
        
        mel_db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        
        return mel_db, torch.tensor(label, dtype=torch.float32)

def load_model(model_path, device):
    """Load existing model"""
    from backend.deep.infer_multitask import build_model
    
    state = torch.load(model_path, map_location="cpu")
    arch = state.get("arch", "resnet18")
    languages = state.get("languages", ["Tamil", "English", "Hindi", "Malayalam", "Telugu"])
    
    model = build_model(arch, len(languages))
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device)
    
    return model, state

def train(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    model, state = load_model(args.model, device)
    
    # Freeze backbone (optional)
    if args.freeze_backbone:
        print("Freezing backbone...")
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    # Load data
    print(f"Loading data: {args.data_dir}")
    dataset = VoiceDataset(
        args.data_dir,
        sr=int(state.get("sr", 16000)),
        segment=float(state.get("segment", 3.0)),
        n_mels=int(state.get("mels", 80)),
        n_fft=int(state.get("nfft", 400)),
        hop=int(state.get("hop", 160))
    )
    
    if len(dataset) == 0:
        print("ERROR: No training data found!")
        return
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Training loop
    model.train()
    print(f"Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for mel, labels in dataloader:
            mel = mel.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            _, _, ai_logits, _, _ = model(mel)
            ai_logits = ai_logits.squeeze()
            
            loss = criterion(ai_logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(ai_logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    # Save model
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"model_finetuned_{timestamp}.pt"
    
    state["model_state"] = model.state_dict()
    state["finetuned_date"] = timestamp
    torch.save(state, output_path)
    
    print(f"✅ Model saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Existing model path")
    parser.add_argument("--data-dir", required=True, help="Training data directory")
    parser.add_argument("--output", help="Output model path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone")
    
    args = parser.parse_args()
    train(args)
```

#### Step 3: Run Fine-Tuning

```bash
# Fine-tune Tamil model
python3 scripts/finetune.py \
  --model backend/deep/multitask_Tamil.pt \
  --data-dir training_data/tamil \
  --epochs 50 \
  --batch-size 8 \
  --device cuda \
  --freeze-backbone

# Output: model_finetuned_YYYYMMDD_HHMMSS.pt
```

#### Step 4: Test New Model

```bash
# Test on sample audio
python3 backend/deep/infer_multitask.py \
  --audio test_audio.mp3 \
  --model model_finetuned_20260216_120000.pt \
  --device cuda

# Expected output:
# {"aiScore": 0.95, "languageDistribution": {...}, ...}
```

#### Step 5: Deploy New Model

```bash
# Backup original
cp backend/deep/multitask_Tamil.pt backend/deep/multitask_Tamil.pt.backup

# Replace with new model
cp model_finetuned_20260216_120000.pt backend/deep/multitask_Tamil.pt

# Restart server
pm2 restart voice-ai-detection
```

---

## Model Evaluation

### 1. Create Test Set

```bash
# Separate test data (not used in training)
test_data/
├── human_test_001.mp3
├── human_test_002.mp3
├── ai_test_001.mp3
└── ai_test_002.mp3
```

### 2. Evaluate Model

Create `scripts/evaluate.py`:

```python
#!/usr/bin/env python3
"""Evaluate model accuracy"""
import sys
import json
from pathlib import Path

# Test files with ground truth
test_cases = [
    ("test_data/human_test_001.mp3", "HUMAN"),
    ("test_data/human_test_002.mp3", "HUMAN"),
    ("test_data/ai_test_001.mp3", "AI_GENERATED"),
    ("test_data/ai_test_002.mp3", "AI_GENERATED"),
]

correct = 0
total = len(test_cases)

for audio_path, expected in test_cases:
    # Run inference
    result = subprocess.check_output([
        "python3", "backend/deep/infer_multitask.py",
        "--audio", audio_path,
        "--model", "backend/deep/multitask_Tamil.pt"
    ])
    
    data = json.loads(result)
    ai_score = data["aiScore"]
    predicted = "AI_GENERATED" if ai_score >= 0.5 else "HUMAN"
    
    if predicted == expected:
        correct += 1
        print(f"✅ {audio_path}: {predicted} (confidence: {ai_score:.2f})")
    else:
        print(f"❌ {audio_path}: {predicted} (expected: {expected})")

accuracy = 100 * correct / total
print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{total})")
```

Run evaluation:

```bash
python3 scripts/evaluate.py
```

### 3. Performance Metrics

**Target Metrics**
- Accuracy: >90%
- Confidence (correct predictions): >0.8
- False Positive Rate: <10%
- False Negative Rate: <10%

---

## Deployment

### 1. Backup Original Model

```bash
cp backend/deep/multitask_Tamil.pt backend/deep/multitask_Tamil.pt.backup
```

### 2. Deploy New Model

```bash
# Copy fine-tuned model
cp model_finetuned_20260216_120000.pt backend/deep/multitask_Tamil.pt

# Restart API server
pm2 restart voice-ai-detection
```

### 3. Verify Deployment

```bash
# Test API endpoint
curl -X POST http://localhost:3000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "audioBase64": "...",
    "language": "Tamil",
    "audioFormat": "mp3"
  }'
```

### 4. Rollback (if needed)

```bash
# Restore backup
cp backend/deep/multitask_Tamil.pt.backup backend/deep/multitask_Tamil.pt

# Restart
pm2 restart voice-ai-detection
```

---

## Best Practices

### Data Quality

✅ **DO:**
- Use high-quality audio (clear speech, minimal noise)
- Balance HUMAN and AI samples (50/50)
- Include diverse speakers and accents
- Use multiple AI TTS engines for AI samples
- Validate audio files before training

❌ **DON'T:**
- Use low-quality or corrupted audio
- Create imbalanced datasets (e.g., 90% AI, 10% HUMAN)
- Use only one speaker for HUMAN samples
- Use only one TTS engine for AI samples
- Mix languages in single-language models

### Training Strategy

✅ **DO:**
- Start with fine-tuning (faster, less data needed)
- Use 50 epochs for deep training
- Freeze backbone to preserve learned features
- Monitor training loss and accuracy
- Test on separate validation set
- Save checkpoints regularly

❌ **DON'T:**
- Train from scratch (requires massive dataset)
- Use too few epochs (<10)
- Unfreeze all layers (risks catastrophic forgetting)
- Overfit on training data
- Skip validation testing
- Overwrite original models without backup

### Model Management

✅ **DO:**
- Version your models (use timestamps)
- Keep backups of working models
- Document training parameters
- Test thoroughly before deployment
- Monitor production performance
- Keep training data organized

❌ **DON'T:**
- Overwrite models without backup
- Deploy untested models
- Lose track of model versions
- Delete training data immediately
- Skip performance monitoring
- Mix training data from different sources

---

## Troubleshooting

### Low Accuracy (<70%)

**Possible Causes:**
- Insufficient training data
- Imbalanced dataset
- Poor audio quality
- Wrong hyperparameters

**Solutions:**
- Collect more training samples (aim for 200+ per class)
- Balance HUMAN/AI samples
- Clean audio data (remove noise, normalize volume)
- Increase epochs to 100
- Try different learning rates (0.001, 0.0001, 0.00001)

### Overfitting

**Symptoms:**
- High training accuracy (>95%)
- Low validation accuracy (<70%)

**Solutions:**
- Add more diverse training data
- Use data augmentation (pitch shift, time stretch)
- Reduce training epochs
- Add dropout layers
- Use early stopping

### Out of Memory

**Solutions:**
- Reduce batch size (try 4 or 2)
- Use CPU instead of GPU
- Process shorter audio segments
- Clear GPU cache: `torch.cuda.empty_cache()`

### Slow Training

**Solutions:**
- Use GPU instead of CPU
- Increase batch size (if memory allows)
- Reduce number of epochs
- Use mixed precision training
- Freeze more layers

---

## Advanced Topics

### Data Augmentation

```python
# Add to VoiceDataset.__getitem__()
import torchaudio.transforms as T

# Pitch shift
if np.random.random() > 0.5:
    pitch_shift = T.PitchShift(self.sr, n_steps=np.random.randint(-2, 3))
    wav = pitch_shift(wav)

# Time stretch
if np.random.random() > 0.5:
    rate = np.random.uniform(0.9, 1.1)
    wav = torchaudio.functional.resample(wav, self.sr, int(self.sr * rate))
```

### Transfer Learning

```python
# Fine-tune from different language model
# Example: Use English model as base for Hindi
model, state = load_model("backend/deep/multitask_English.pt", device)
# Train on Hindi data
# Save as multitask_Hindi.pt
```

### Ensemble Models

```python
# Use multiple models for better accuracy
models = [
    load_model("multitask_Tamil_v1.pt"),
    load_model("multitask_Tamil_v2.pt"),
    load_model("multitask_Tamil_v3.pt"),
]

# Average predictions
ai_scores = [model.predict(audio) for model in models]
final_score = np.mean(ai_scores)
```

---

## Resources

### Datasets
- [Common Voice](https://commonvoice.mozilla.org/) - Multilingual speech dataset
- [LibriSpeech](http://www.openslr.org/12/) - English audiobook dataset
- [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) - Speaker recognition dataset
- [Indic TTS](https://www.iitm.ac.in/donlab/tts/) - Indian language TTS

### Tools
- [Audacity](https://www.audacityteam.org/) - Audio editing
- [FFmpeg](https://ffmpeg.org/) - Audio conversion
- [Librosa](https://librosa.org/) - Audio analysis
- [PyTorch](https://pytorch.org/) - Deep learning framework

### Papers
- ResNet: "Deep Residual Learning for Image Recognition"
- Mel Spectrograms: "Mel Frequency Cepstral Coefficients"
- Transfer Learning: "A Survey on Transfer Learning"

---

## Support

For training issues or questions:
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [GitHub Issues](https://github.com/parikshitgorain/voice-ai-detection/issues)
- Contact: parikshitgorain@yahoo.com
