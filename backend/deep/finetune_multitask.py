import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import torchvision
from torch.utils.data import Dataset, DataLoader


def build_model(arch, lang_count):
    arch = arch.lower()
    if arch == "resnet18":
        backbone = torchvision.models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        feat_dim = 512
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    class MultiHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.project = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            self.ai_head = nn.Linear(feat_dim, 1)
            self.lang_head = nn.Linear(feat_dim, lang_count)
            self.multi_head = nn.Linear(feat_dim, 1)

        def forward(self, x):
            feats = self.backbone(x)
            proj = self.project(feats)
            return feats, proj, self.ai_head(feats), self.lang_head(feats), self.multi_head(feats)

    return MultiHead()


def load_audio(path, sr):
    try:
        wav, orig_sr = torchaudio.load(path)
    except Exception:
        data, orig_sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav


def compute_logmel(wav, sr, n_mels, n_fft, hop_length):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=40,
        f_max=sr // 2,
        power=2.0,
    )(wav)
    mel_db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db


def add_noise(wav, noise_level=0.01):
    """Add Gaussian noise to audio waveform"""
    noise = torch.randn_like(wav) * noise_level
    return wav + noise


def time_stretch(wav, rate=1.0):
    """Simple time stretching by resampling"""
    if rate == 1.0:
        return wav
    # Resample to simulate time stretching
    stretched = torchaudio.functional.resample(wav, 1000, int(1000 * rate))
    return stretched


def pitch_shift(spec, shift_amount=0):
    """Shift spectrogram along frequency axis"""
    if shift_amount == 0:
        return spec
    return torch.roll(spec, shift_amount, dims=2)


class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, sr, segment_samples, n_mels, nfft, hop, 
                 noise_level=0.0, augment_count=10):
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.segment_samples = segment_samples
        self.n_mels = n_mels
        self.nfft = nfft
        self.hop = hop
        self.noise_level = noise_level
        self.augment_count = augment_count
        
    def __len__(self):
        return len(self.audio_files) * self.augment_count
    
    def __getitem__(self, idx):
        audio_idx = idx // self.augment_count
        aug_idx = idx % self.augment_count
        
        # Load audio
        wav = load_audio(self.audio_files[audio_idx], self.sr)
        label = self.labels[audio_idx]
        
        # Random segment
        if wav.shape[1] > self.segment_samples:
            start = random.randint(0, wav.shape[1] - self.segment_samples)
            wav = wav[:, start:start + self.segment_samples]
        elif wav.shape[1] < self.segment_samples:
            pad = self.segment_samples - wav.shape[1]
            wav = F.pad(wav, (0, pad))
        
        # Apply augmentation
        if aug_idx > 0 and self.noise_level > 0:
            # Add noise with varying levels
            noise_scale = self.noise_level * random.uniform(0.5, 1.5)
            wav = add_noise(wav, noise_scale)
            
            # Random time stretch
            if random.random() > 0.5:
                rate = random.uniform(0.9, 1.1)
                wav = time_stretch(wav, rate)
                if wav.shape[1] > self.segment_samples:
                    wav = wav[:, :self.segment_samples]
                elif wav.shape[1] < self.segment_samples:
                    pad = self.segment_samples - wav.shape[1]
                    wav = F.pad(wav, (0, pad))
        
        # Compute mel spectrogram
        mel = compute_logmel(wav, self.sr, self.n_mels, self.nfft, self.hop)
        
        # Random pitch shift in spectrogram
        if aug_idx > 0 and random.random() > 0.5:
            shift = random.randint(-2, 2)
            mel = pitch_shift(mel, shift)
        
        return mel, torch.tensor([label], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AI voice detection model with balanced dataset")
    parser.add_argument("--ai-audio", nargs="+", required=True, help="AI-generated audio file(s)")
    parser.add_argument("--human-audio", nargs="+", required=True, help="Human audio file(s)")
    parser.add_argument("--model", required=True, help="Base model to fine-tune")
    parser.add_argument("--output", default=None, help="Output model path (default: overwrite input)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise augmentation level (0-1)")
    parser.add_argument("--augment", type=int, default=20, help="Augmentation multiplier per sample")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate (default: very low to preserve existing knowledge)")
    args = parser.parse_args()
    
    output_path = args.output if args.output else args.model
    
    # Validate AI audio files
    for audio_file in args.ai_audio:
        if not os.path.exists(audio_file):
            raise SystemExit(f"AI audio file not found: {audio_file}")
    
    # Validate human audio files
    for audio_file in args.human_audio:
        if not os.path.exists(audio_file):
            raise SystemExit(f"Human audio file not found: {audio_file}")
    
    if not os.path.exists(args.model):
        raise SystemExit("Model file missing")
    
    print(f"Loading base model: {args.model}")
    state = torch.load(args.model, map_location="cpu")
    arch = state.get("arch", "resnet18")
    n_mels = int(state.get("mels", 80))
    sr = int(state.get("sr", 16000))
    segment = float(state.get("segment", 3.0))
    nfft = int(state.get("nfft", 400))
    hop = int(state.get("hop", 160))
    languages = state.get("languages", ["Tamil", "English", "Hindi", "Malayalam", "Telugu"])
    
    print(f"Model config: arch={arch}, sr={sr}, segment={segment}s, mels={n_mels}")
    
    model = build_model(arch, len(languages))
    model.load_state_dict(state["model_state"], strict=True)
    
    device = torch.device(args.device)
    model.to(device)
    model.train()
    
    # Prepare BALANCED dataset with both AI and human samples
    segment_samples = int(segment * sr)
    
    # Combine AI samples (label=1) and human samples (label=0)
    all_audio_files = list(args.ai_audio) + list(args.human_audio)
    all_labels = [1] * len(args.ai_audio) + [0] * len(args.human_audio)
    
    dataset = AudioDataset(
        all_audio_files, all_labels, sr, segment_samples, n_mels, nfft, hop,
        noise_level=args.noise, augment_count=args.augment
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"\n=== BALANCED TRAINING ===")
    print(f"AI samples: {len(args.ai_audio)} files (label=1)")
    print(f"Human samples: {len(args.human_audio)} files (label=0)")
    print(f"Total audio files: {len(all_audio_files)}")
    print(f"Total samples (with augmentation): {len(dataset)}")
    print(f"Noise augmentation: {args.noise}")
    print(f"Augmentation multiplier: {args.augment}x")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr} (low to preserve existing knowledge)")
    print("=" * 30)
    
    # Optimizer - only train AI head with very low learning rate
    optimizer = torch.optim.Adam(model.ai_head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        ai_correct = 0
        ai_total = 0
        human_correct = 0
        human_total = 0
        
        for batch_idx, (mel, label) in enumerate(dataloader):
            mel = mel.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            
            _, _, ai_logits, _, _ = model(mel)
            loss = criterion(ai_logits, label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy per class
            pred = (torch.sigmoid(ai_logits) > 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            # Track per-class accuracy
            ai_mask = label == 1
            human_mask = label == 0
            if ai_mask.any():
                ai_correct += (pred[ai_mask] == label[ai_mask]).sum().item()
                ai_total += ai_mask.sum().item()
            if human_mask.any():
                human_correct += (pred[human_mask] == label[human_mask]).sum().item()
                human_total += human_mask.sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        ai_acc = 100.0 * ai_correct / ai_total if ai_total > 0 else 0
        human_acc = 100.0 * human_correct / human_total if human_total > 0 else 0
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} - Loss: {avg_loss:.4f}, "
                  f"Total: {accuracy:.1f}%, AI: {ai_acc:.1f}%, Human: {human_acc:.1f}%")
    
    # Save fine-tuned model
    print(f"\nSaving fine-tuned model to: {output_path}")
    state["model_state"] = model.state_dict()
    torch.save(state, output_path)
    
    print("\n✅ Fine-tuning complete!")
    print(f"Model saved: {output_path}")
    print("\nTest your audio samples again to verify balanced detection.")


if __name__ == "__main__":
    main()
