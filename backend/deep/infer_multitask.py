import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import torchvision


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
    languages = state.get("languages", ["Tamil", "English", "Hindi", "Malayalam", "Telugu"])

    model = build_model(arch, len(languages))
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    wav = load_audio(args.audio, sr)
    segment_samples = int(segment * sr)
    starts = build_windows(wav.shape[1], segment_samples)

    ai_probs = []
    lang_probs = []
    multi_probs = []
    with torch.no_grad():
        for start in starts:
            chunk = wav[:, start : start + segment_samples]
            if chunk.shape[1] < segment_samples:
                pad = segment_samples - chunk.shape[1]
                chunk = F.pad(chunk, (0, pad))
            mel = compute_logmel(chunk, sr, n_mels, nfft, hop)
            mel = mel.unsqueeze(0).to(device)
            _, _, ai_logits, lang_logits, multi_logits = model(mel)
            ai_probs.append(torch.sigmoid(ai_logits).item())
            lang_probs.append(torch.softmax(lang_logits, dim=1).squeeze(0))
            multi_probs.append(torch.sigmoid(multi_logits).item())

    ai_score = sum(ai_probs) / len(ai_probs) if ai_probs else 0.5
    multi_score = sum(multi_probs) / len(multi_probs) if multi_probs else 0.0
    lang_avg = torch.stack(lang_probs).mean(dim=0) if lang_probs else torch.zeros(len(languages))

    # Build language distribution dictionary
    lang_distribution = {}
    for idx, lang in enumerate(languages):
        lang_distribution[lang] = float(lang_avg[idx].item())
    
    output = {
        "aiScore": float(ai_score),
        "languageDistribution": lang_distribution,
        "multiSpeakerScore": float(multi_score),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
