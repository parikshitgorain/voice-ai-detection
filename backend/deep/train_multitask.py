import argparse
import json
import os
import random
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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


def infer_language_from_path(path, languages):
    parts = [p.lower() for p in path.split(os.sep)]
    for lang in languages:
        if lang.lower() in parts:
            return lang
    return None


def infer_multi_label(path):
    parts = [p.lower() for p in path.split(os.sep)]
    if "multi" in parts or "multi_speaker" in parts or "multispeaker" in parts:
        return 1
    if "single" in parts:
        return 0
    return -1


def summarize_records(records, languages):
    counts = {lang: {"human": 0, "ai": 0} for lang in languages}
    unknown = 0
    for _, label, lang_id, _ in records:
        if lang_id is None or lang_id < 0 or lang_id >= len(languages):
            unknown += 1
            continue
        lang = languages[lang_id]
        if label == 1:
            counts[lang]["ai"] += 1
        else:
            counts[lang]["human"] += 1
    return counts, unknown


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        languages,
        sample_rate,
        segment_seconds,
        n_mels,
        n_fft,
        hop_length,
        filter_languages=False,
    ):
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.languages = languages
        self.lang_to_id = {lang: idx for idx, lang in enumerate(languages)}

        ai_root = os.path.join(data_root, split, "ai")
        human_root = os.path.join(data_root, split, "human")

        self.records = []
        for label, root in [(1, ai_root), (0, human_root)]:
            files = list_audio_files(root)
            print(f"scanned {len(files)} files under {root}")
            for path in files:
                lang = infer_language_from_path(path, languages)
                if filter_languages and not lang:
                    continue
                if filter_languages and lang not in self.lang_to_id:
                    continue
                lang_id = self.lang_to_id.get(lang) if lang else -1
                multi_label = infer_multi_label(path)
                self.records.append((path, label, lang_id, multi_label))

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=40,
            f_max=sample_rate // 2,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.records)

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
        if total <= self.segment_samples:
            pad = self.segment_samples - total
            if pad > 0:
                wav = F.pad(wav, (0, pad))
            return wav
        max_start = total - self.segment_samples
        start = random.randint(0, max_start)
        return wav[:, start : start + self.segment_samples]

    def _make_mel(self, wav):
        mel = self.mel(wav)
        mel_db = self.to_db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db

    def __getitem__(self, idx):
        attempts = 0
        last_err = None
        while attempts < 3:
            path, ai_label, lang_id, multi_label = self.records[idx]
            try:
                wav = self._load(path)
                seg_a = self._segment(wav)
                seg_b = self._segment(wav)
                mel_a = self._make_mel(seg_a)
                mel_b = self._make_mel(seg_b)
                return (
                    mel_a,
                    mel_b,
                    torch.tensor(ai_label, dtype=torch.float32),
                    torch.tensor(lang_id, dtype=torch.long),
                    torch.tensor(multi_label, dtype=torch.float32),
                )
            except Exception as exc:
                last_err = exc
                attempts += 1
                idx = random.randint(0, len(self.records) - 1)
        # Fallback to a zero sample to avoid crashing on bad files.
        mel_shape = (1, self.n_mels, max(1, self.segment_samples // self.hop_length + 1))
        mel_a = torch.zeros(mel_shape, dtype=torch.float32)
        mel_b = torch.zeros(mel_shape, dtype=torch.float32)
        return (
            mel_a,
            mel_b,
            torch.tensor(ai_label, dtype=torch.float32),
            torch.tensor(lang_id, dtype=torch.long),
            torch.tensor(multi_label, dtype=torch.float32),
        )


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


def nt_xent_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1).float()
    z2 = F.normalize(z2, dim=1).float()
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim.float()
    batch = z1.size(0)

    mask = torch.eye(2 * batch, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e4)

    positives = torch.cat([torch.arange(batch, 2 * batch), torch.arange(0, batch)]).to(z.device)
    loss = F.cross_entropy(sim, positives)
    return loss


def apply_spec_augment(mel, time_mask, freq_mask):
    if time_mask > 0:
        mel = torchaudio.transforms.TimeMasking(time_mask)(mel)
    if freq_mask > 0:
        mel = torchaudio.transforms.FrequencyMasking(freq_mask)(mel)
    return mel


class AugmentScheduler:
    def __init__(self, max_time_mask=60, max_freq_mask=12):
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask

    def strength(self, epoch, total_epochs):
        if total_epochs <= 1:
            return 0.0
        return min(1.0, epoch / max(1, total_epochs - 1))

    def params(self, epoch, total_epochs):
        s = self.strength(epoch, total_epochs)
        time_mask = int(self.max_time_mask * s)
        freq_mask = int(self.max_freq_mask * s)
        return time_mask, freq_mask


def evaluate(model, loader, device):
    if loader is None:
        return {}
    model.eval()
    total = 0
    ai_correct = 0
    lang_correct = 0
    lang_total = 0
    multi_correct = 0
    multi_total = 0

    with torch.no_grad():
        for mel_a, mel_b, ai_label, lang_id, multi_label in loader:
            mel_a = mel_a.to(device)
            ai_label = ai_label.to(device)
            lang_id = lang_id.to(device)
            multi_label = multi_label.to(device)

            feats, proj, ai_logits, lang_logits, multi_logits = model(mel_a)
            ai_pred = (torch.sigmoid(ai_logits).squeeze(1) >= 0.5).float()
            ai_correct += (ai_pred == ai_label).sum().item()
            total += ai_label.numel()

            valid_lang = lang_id >= 0
            if valid_lang.any():
                lang_pred = torch.argmax(lang_logits, dim=1)
                lang_correct += (lang_pred[valid_lang] == lang_id[valid_lang]).sum().item()
                lang_total += valid_lang.sum().item()

            valid_multi = multi_label >= 0
            if valid_multi.any():
                multi_pred = (torch.sigmoid(multi_logits).squeeze(1) >= 0.5).float()
                multi_correct += (multi_pred[valid_multi] == multi_label[valid_multi]).sum().item()
                multi_total += valid_multi.sum().item()

    return {
        "ai_acc": ai_correct / max(total, 1),
        "lang_acc": lang_correct / max(lang_total, 1),
        "multi_acc": multi_correct / max(multi_total, 1),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "multitask.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--segment", type=float, default=3.0)
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--mels", type=int, default=80)
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=240)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--languages", default="", help="Comma-separated language list to include.")
    parser.add_argument("--balance", action="store_true", help="Enable class-balanced sampling.")
    parser.add_argument("--resume", default="", help="Path to a checkpoint to resume from.")
    parser.add_argument(
        "--min-ai-per-lang",
        type=int,
        default=0,
        help="Fail training if any language has fewer than this many AI samples.",
    )
    parser.add_argument(
        "--min-human-per-lang",
        type=int,
        default=0,
        help="Fail training if any language has fewer than this many human samples.",
    )
    parser.add_argument(
        "--require-language",
        action="store_true",
        help="Fail training if any samples lack a language label in the path.",
    )
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)

    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu")
        args.arch = resume_state.get("arch", args.arch)
        args.mels = int(resume_state.get("mels", args.mels))
        args.sr = int(resume_state.get("sr", args.sr))
        args.segment = float(resume_state.get("segment", args.segment))
        args.nfft = int(resume_state.get("nfft", args.nfft))
        args.hop = int(resume_state.get("hop", args.hop))

    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
        filter_languages = True
    elif resume_state and resume_state.get("languages"):
        languages = list(resume_state.get("languages"))
        filter_languages = False
    else:
        languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        filter_languages = False

    if resume_state and resume_state.get("languages"):
        resume_langs = list(resume_state.get("languages"))
        if set(resume_langs) != set(languages):
            raise RuntimeError(
                f"Resume checkpoint languages {resume_langs} do not match requested {languages}."
            )

    train_ds = MultiTaskDataset(
        args.data,
        "train",
        languages,
        args.sr,
        args.segment,
        args.mels,
        args.nfft,
        args.hop,
        filter_languages=filter_languages,
    )
    val_ds = MultiTaskDataset(
        args.data,
        "val",
        languages,
        args.sr,
        args.segment,
        args.mels,
        args.nfft,
        args.hop,
        filter_languages=filter_languages,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Training data missing or not found.")

    train_counts, train_unknown = summarize_records(train_ds.records, languages)
    val_counts, val_unknown = summarize_records(val_ds.records, languages)
    print(
        json.dumps(
            {
                "status": "data_counts",
                "split": "train",
                "total": len(train_ds),
                "unknown_language": train_unknown,
                "counts": train_counts,
            }
        )
    )
    print(
        json.dumps(
            {
                "status": "data_counts",
                "split": "val",
                "total": len(val_ds),
                "unknown_language": val_unknown,
                "counts": val_counts,
            }
        )
    )
    if args.require_language and train_unknown > 0:
        raise RuntimeError(f"Found {train_unknown} training samples without language labels.")
    for lang in languages:
        ai_count = train_counts[lang]["ai"]
        human_count = train_counts[lang]["human"]
        if args.min_ai_per_lang and ai_count < args.min_ai_per_lang:
            raise RuntimeError(f"{lang} AI samples too low: {ai_count} < {args.min_ai_per_lang}")
        if args.min_human_per_lang and human_count < args.min_human_per_lang:
            raise RuntimeError(
                f"{lang} human samples too low: {human_count} < {args.min_human_per_lang}"
            )

    sampler = None
    if args.balance:
        counts = {0: 0, 1: 0}
        for _, label, _, _ in train_ds.records:
            counts[label] += 1
        weights = []
        for _, label, _, _ in train_ds.records:
            denom = counts.get(label, 1) or 1
            weights.append(1.0 / denom)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loader_kwargs = {"num_workers": args.workers}
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True
        if args.workers > 0:
            loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **loader_kwargs)

    model = build_model(args.arch, len(languages)).to(device)
    if resume_state and resume_state.get("model_state"):
        model.load_state_dict(resume_state["model_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    scheduler = AugmentScheduler()

    print(
        json.dumps(
            {
                "status": "dataset_ready",
                "train_records": len(train_ds),
                "val_records": len(val_ds),
                "batch_size": args.batch,
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "languages": languages,
            }
        )
    )

    best_val = -1
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        time_mask, freq_mask = scheduler.params(epoch, args.epochs)
        epoch_start = time.time()
        total_batches = max(len(train_loader), 1)

        for step, (mel_a, mel_b, ai_label, lang_id, multi_label) in enumerate(train_loader, 1):
            mel_a = mel_a.to(device)
            mel_b = mel_b.to(device)
            ai_label = ai_label.to(device)
            lang_id = lang_id.to(device)
            multi_label = multi_label.to(device)

            mel_a = apply_spec_augment(mel_a, time_mask, freq_mask)
            mel_b = apply_spec_augment(mel_b, time_mask, freq_mask)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                feats_a, proj_a, ai_logits, lang_logits, multi_logits = model(mel_a)
                feats_b, proj_b, _, _, _ = model(mel_b)

                ai_loss = F.binary_cross_entropy_with_logits(ai_logits.squeeze(1), ai_label)

                valid_lang = lang_id >= 0
                if valid_lang.any():
                    lang_loss = F.cross_entropy(lang_logits[valid_lang], lang_id[valid_lang])
                else:
                    lang_loss = torch.tensor(0.0, device=device)

                valid_multi = multi_label >= 0
                if valid_multi.any():
                    multi_loss = F.binary_cross_entropy_with_logits(
                        multi_logits.squeeze(1)[valid_multi], multi_label[valid_multi]
                    )
                else:
                    multi_loss = torch.tensor(0.0, device=device)

                contrastive = nt_xent_loss(proj_a, proj_b, temperature=0.2)
                temporal_consistency = 1 - F.cosine_similarity(feats_a, feats_b, dim=1).mean()

                loss = (
                    ai_loss
                    + 0.6 * lang_loss
                    + 0.4 * multi_loss
                    + 0.3 * contrastive
                    + 0.2 * temporal_consistency
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if args.log_interval > 0 and step % args.log_interval == 0:
                pct = round(100 * step / total_batches, 2)
                elapsed = time.time() - epoch_start
                rate = step / max(elapsed, 1e-6)
                eta = (total_batches - step) / max(rate, 1e-6)
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "progress_pct": pct,
                            "batches_done": int(step),
                            "batches_total": int(total_batches),
                            "eta_seconds": round(eta, 1),
                        }
                    )
                )

        avg_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, device)
        val_score = metrics.get("ai_acc", 0)

        if val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "arch": args.arch,
                    "model_state": model.state_dict(),
                    "mels": args.mels,
                    "sr": args.sr,
                    "segment": args.segment,
                    "nfft": args.nfft,
                    "hop": args.hop,
                    "languages": languages,
                },
                args.out,
            )

        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": round(avg_loss, 4),
                    "ai_acc": round(metrics.get("ai_acc", 0), 4),
                    "lang_acc": round(metrics.get("lang_acc", 0), 4),
                    "multi_acc": round(metrics.get("multi_acc", 0), 4),
                }
            )
        )

    elapsed = time.time() - start
    print(json.dumps({"status": "ok", "best_val": best_val, "seconds": elapsed}))


if __name__ == "__main__":
    main()
