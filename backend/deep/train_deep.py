import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import torchvision
from torch.utils.data import Dataset, DataLoader

ALLOWED_EXTS = {".mp3", ".wav", ".flac", ".m4a"}


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_audio_files(root):
    files = []
    for base, _, names in os.walk(root):
        for name in names:
            ext = os.path.splitext(name)[1].lower()
            if ext in ALLOWED_EXTS:
                files.append(os.path.join(base, name))
    return files


class AudioDataset(Dataset):
    def __init__(self, files, label, sample_rate, segment_seconds, n_mels, n_fft, hop_length):
        self.files = files
        self.label = label
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=40,
            f_max=sample_rate // 2,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T)
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def _segment(self, wav):
        total = wav.shape[1]
        if total == self.segment_samples:
            return wav
        if total < self.segment_samples:
            pad = self.segment_samples - total
            return F.pad(wav, (0, pad))
        max_start = total - self.segment_samples
        start = random.randint(0, max_start)
        return wav[:, start : start + self.segment_samples]

    def __getitem__(self, idx):
        attempts = 0
        last_err = None
        while attempts < 3:
            path = self.files[idx]
            try:
                wav = self._load(path)
                wav = self._segment(wav)
                mel = self.mel(wav)
                mel_db = self.amplitude_to_db(mel)
                mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
                return mel_db, torch.tensor(self.label, dtype=torch.float32)
            except Exception as exc:
                last_err = exc
                attempts += 1
                idx = random.randint(0, len(self.files) - 1)
        # Fallback to a zero sample to avoid crashing on bad files.
        wav = torch.zeros(1, self.segment_samples, dtype=torch.float32)
        mel = self.mel(wav)
        mel_db = self.amplitude_to_db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db, torch.tensor(self.label, dtype=torch.float32)


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


def build_loaders(data_root, sample_rate, segment_seconds, n_mels, n_fft, hop_length, batch_size, num_workers):
    train_ai = list_audio_files(os.path.join(data_root, "train", "ai"))
    train_human = list_audio_files(os.path.join(data_root, "train", "human"))
    val_ai = list_audio_files(os.path.join(data_root, "val", "ai"))
    val_human = list_audio_files(os.path.join(data_root, "val", "human"))

    if not train_ai or not train_human:
        raise RuntimeError("Training data missing. Add files to backend/data/train/ai and backend/data/train/human.")

    train_ds = torch.utils.data.ConcatDataset([
        AudioDataset(train_ai, 1, sample_rate, segment_seconds, n_mels, n_fft, hop_length),
        AudioDataset(train_human, 0, sample_rate, segment_seconds, n_mels, n_fft, hop_length),
    ])

    val_ds = None
    if val_ai or val_human:
        val_ds = torch.utils.data.ConcatDataset([
            AudioDataset(val_ai, 1, sample_rate, segment_seconds, n_mels, n_fft, hop_length),
            AudioDataset(val_human, 0, sample_rate, segment_seconds, n_mels, n_fft, hop_length),
        ])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def evaluate(model, loader, device):
    if loader is None:
        return None
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mel, label in loader:
            mel = mel.to(device)
            label = label.to(device)
            logits = model(mel)
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).float()
            correct += (preds == label).sum().item()
            total += label.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "model.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--segment", type=float, default=3.0)
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--mels", type=int, default=80)
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=240)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arch", default="resnet18")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader = build_loaders(
        args.data,
        args.sr,
        args.segment,
        args.mels,
        args.nfft,
        args.hop,
        args.batch,
        args.workers,
    )

    model = build_model(args.arch, args.mels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0
        for mel, label in train_loader:
            mel = mel.to(device)
            label = label.to(device)
            opt.zero_grad()
            logits = model(mel).squeeze(1)
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            running += loss.item()
        avg_loss = running / max(len(train_loader), 1)
        val_acc = evaluate(model, val_loader, device)
        if val_acc is None:
            val_acc = -1
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "arch": args.arch,
                    "model_state": model.state_dict(),
                    "mels": args.mels,
                    "sr": args.sr,
                    "segment": args.segment,
                    "nfft": args.nfft,
                    "hop": args.hop,
                },
                args.out,
            )
        print(f"epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    elapsed = time.time() - start
    print(json.dumps({"status": "ok", "best_val": best_val, "seconds": elapsed}))


if __name__ == "__main__":
    main()
