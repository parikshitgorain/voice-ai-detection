"""
Incremental Training with Knowledge Distillation
Adds new training samples WITHOUT forgetting the original 140GB training.

Uses Learning without Forgetting (LwF) technique:
1. Keep original model as "teacher" 
2. Train new model to match teacher on general audio
3. While also learning to detect new AI samples
"""

import argparse
import json
import os
import random
import copy

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
    noise = torch.randn_like(wav) * noise_level
    return wav + noise


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
            noise_scale = self.noise_level * random.uniform(0.3, 1.5)
            wav = add_noise(wav, noise_scale)
        
        mel = compute_logmel(wav, self.sr, self.n_mels, self.nfft, self.hop)
        return mel, torch.tensor([label], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Incremental training with knowledge distillation")
    parser.add_argument("--ai-audio", nargs="+", required=True, help="New AI audio files to learn")
    parser.add_argument("--human-audio", nargs="+", required=True, help="Human audio files (anchor)")
    parser.add_argument("--model", required=True, help="Original trained model")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--augment", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--distill-weight", type=float, default=0.7, 
                        help="Weight for preserving original model behavior (0-1)")
    args = parser.parse_args()
    
    # Validate files
    for f in args.ai_audio + args.human_audio:
        if not os.path.exists(f):
            raise SystemExit(f"File not found: {f}")
    
    if not os.path.exists(args.model):
        raise SystemExit("Model file missing")
    
    print(f"Loading original model: {args.model}")
    state = torch.load(args.model, map_location="cpu")
    arch = state.get("arch", "resnet18")
    n_mels = int(state.get("mels", 80))
    sr = int(state.get("sr", 16000))
    segment = float(state.get("segment", 3.0))
    nfft = int(state.get("nfft", 400))
    hop = int(state.get("hop", 160))
    languages = state.get("languages", ["Tamil", "English", "Hindi", "Malayalam", "Telugu"])
    
    # Create TEACHER model (frozen original)
    teacher = build_model(arch, len(languages))
    teacher.load_state_dict(state["model_state"], strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Create STUDENT model (will be trained)
    student = build_model(arch, len(languages))
    student.load_state_dict(state["model_state"], strict=True)
    
    device = torch.device(args.device)
    teacher.to(device)
    student.to(device)
    student.train()
    
    # Prepare dataset
    segment_samples = int(segment * sr)
    all_files = list(args.ai_audio) + list(args.human_audio)
    all_labels = [1] * len(args.ai_audio) + [0] * len(args.human_audio)
    
    dataset = AudioDataset(
        all_files, all_labels, sr, segment_samples, n_mels, nfft, hop,
        noise_level=args.noise, augment_count=args.augment
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"\n{'='*50}")
    print("INCREMENTAL TRAINING WITH KNOWLEDGE DISTILLATION")
    print(f"{'='*50}")
    print(f"New AI samples: {len(args.ai_audio)}")
    print(f"Human anchors: {len(args.human_audio)}")
    print(f"Augmentation: {args.augment}x → {len(dataset)} total samples")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Distillation weight: {args.distill_weight} (preservation strength)")
    print(f"{'='*50}\n")
    
    # Train FULL model (backbone + heads) with very careful learning
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    
    # Learning rate scheduler - reduce over time
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    ce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    
    best_combined_acc = 0
    best_state = None
    
    for epoch in range(args.epochs):
        total_ce = 0
        total_distill = 0
        ai_correct = 0
        ai_total = 0
        human_correct = 0
        human_total = 0
        
        for mel, label in dataloader:
            mel = mel.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            
            # Get teacher predictions (original model behavior)
            with torch.no_grad():
                t_feats, _, t_ai_logits, t_lang_logits, t_multi_logits = teacher(mel)
            
            # Get student predictions
            s_feats, _, s_ai_logits, s_lang_logits, s_multi_logits = student(mel)
            
            # Loss 1: Learn new samples correctly
            loss_ce = ce_loss(s_ai_logits, label)
            
            # Loss 2: Knowledge distillation - preserve behavior on ALL samples
            # Match feature representations
            loss_feat = mse_loss(s_feats, t_feats)
            # Match AI predictions (soft targets)
            loss_ai_dist = mse_loss(torch.sigmoid(s_ai_logits), torch.sigmoid(t_ai_logits))
            # Match language predictions
            loss_lang = mse_loss(s_lang_logits, t_lang_logits)
            # Match multi-speaker predictions
            loss_multi = mse_loss(s_multi_logits, t_multi_logits)
            
            loss_distill = loss_feat + loss_ai_dist + loss_lang + loss_multi
            
            # Combined loss: balance learning new vs preserving old
            # Higher distill_weight = more preservation
            loss = (1 - args.distill_weight) * loss_ce + args.distill_weight * loss_distill
            
            loss.backward()
            
            # Gradient clipping to prevent catastrophic changes
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_ce += loss_ce.item()
            total_distill += loss_distill.item()
            
            # Track accuracy per class
            pred = (torch.sigmoid(s_ai_logits) > 0.5).float()
            ai_mask = label == 1
            human_mask = label == 0
            if ai_mask.any():
                ai_correct += (pred[ai_mask] == label[ai_mask]).sum().item()
                ai_total += ai_mask.sum().item()
            if human_mask.any():
                human_correct += (pred[human_mask] == label[human_mask]).sum().item()
                human_total += human_mask.sum().item()
        
        scheduler.step()
        
        avg_ce = total_ce / len(dataloader)
        avg_distill = total_distill / len(dataloader)
        ai_acc = 100.0 * ai_correct / ai_total if ai_total > 0 else 0
        human_acc = 100.0 * human_correct / human_total if human_total > 0 else 0
        combined_acc = (ai_acc + human_acc) / 2
        
        # Save best model
        if combined_acc > best_combined_acc:
            best_combined_acc = combined_acc
            best_state = copy.deepcopy(student.state_dict())
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:3d}/{args.epochs} | CE: {avg_ce:.4f} | Distill: {avg_distill:.4f} | "
                  f"AI: {ai_acc:.1f}% | Human: {human_acc:.1f}% | LR: {lr:.6f}")
    
    # Use best model
    if best_state:
        student.load_state_dict(best_state)
        print(f"\nRestored best model with {best_combined_acc:.1f}% combined accuracy")
    
    # Save
    state["model_state"] = student.state_dict()
    torch.save(state, args.output)
    
    print(f"\n✅ Incremental training complete!")
    print(f"Model saved: {args.output}")
    print("\nTest with: python infer_multitask.py --audio <file> --model <output>")


if __name__ == "__main__":
    main()
