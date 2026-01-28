import argparse
import json
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision

ALLOWED_EXTS = {".mp3", ".wav", ".flac", ".m4a"}


def build_model(arch, n_mels):
    arch = arch.lower()
    if arch == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "smallcnn":
        class SmallCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.head = nn.Linear(64, 1)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.head(x)
                return x

        return SmallCNN()
    raise ValueError(f"Unsupported arch: {arch}")


def load_audio(path, sr):
    wav, orig_sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav


def build_windows(total_samples, segment_samples):
    if total_samples <= segment_samples:
        return [0]
    starts = [0]
    middle = max(0, (total_samples - segment_samples) // 2)
    end = max(0, total_samples - segment_samples)
    starts.extend([middle, end])
    offsets = [0.25, 0.5, 0.75]
    for offset in offsets:
        start = int((total_samples - segment_samples) * offset)
        starts.append(max(0, min(start, end)))
    seen = set()
    uniq = []
    for s in starts:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        raise SystemExit("audio file missing")
    if not os.path.exists(args.model):
        raise SystemExit("model file missing")

    state = torch.load(args.model, map_location="cpu")
    arch = state.get("arch", "resnet18")
    n_mels = int(state.get("mels", 80))
    sr = int(state.get("sr", 16000))
    segment = float(state.get("segment", 3.0))
    nfft = int(state.get("nfft", 400))
    hop = int(state.get("hop", 160))

    model = build_model(arch, n_mels)
    model.load_state_dict(state["model_state"])
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    wav = load_audio(args.audio, sr)
    segment_samples = int(segment * sr)
    starts = build_windows(wav.shape[1], segment_samples)

    probs = []
    with torch.no_grad():
        for start in starts:
            chunk = wav[:, start : start + segment_samples]
            if chunk.shape[1] < segment_samples:
                pad = segment_samples - chunk.shape[1]
                chunk = F.pad(chunk, (0, pad))
            mel = compute_logmel(chunk, sr, n_mels, nfft, hop)
            mel = mel.unsqueeze(0).to(device)
            logit = model(mel)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

    score = sum(probs) / len(probs) if probs else 0.5
    print(json.dumps({"deepScore": score}))


if __name__ == "__main__":
    main()
