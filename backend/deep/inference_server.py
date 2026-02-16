#!/usr/bin/env python3
"""
Persistent inference server - keeps models in GPU memory for instant responses
Eliminates model loading overhead by keeping models warm
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
import librosa


class MultiTaskModel(nn.Module):
    """Multi-task model with language detection"""
    def __init__(self, arch, lang_count):
        super().__init__()
        if arch.lower() == "resnet18":
            backbone = torchvision.models.resnet18(weights=None)
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            backbone.fc = nn.Identity()
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported arch: {arch}")
        
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


class InferenceServer:
    """Persistent inference server with model caching"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.models = {}  # Cache loaded models
        self.model_configs = {}  # Store model metadata
        print(f"[Server] Initialized on device: {self.device}", file=sys.stderr)
    
    def load_model(self, model_path):
        """Load model into memory (cached)"""
        if model_path in self.models:
            return self.models[model_path], self.model_configs[model_path]
        
        start = time.time()
        state = torch.load(model_path, map_location='cpu')
        
        arch = state.get('arch', 'resnet18')
        n_mels = int(state.get('mels', 80))
        sr = int(state.get('sr', 16000))
        segment = float(state.get('segment', 3.0))
        nfft = int(state.get('nfft', 400))
        hop = int(state.get('hop', 160))
        languages = state.get('languages', ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu'])
        
        model = MultiTaskModel(arch, len(languages))
        model.load_state_dict(state['model_state'], strict=True)
        model.eval()
        model.to(self.device)
        
        # Warm up model with dummy input
        with torch.no_grad():
            dummy = torch.randn(1, 1, n_mels, 100).to(self.device)
            _ = model(dummy)
        
        config = {
            'arch': arch,
            'n_mels': n_mels,
            'sr': sr,
            'segment': segment,
            'nfft': nfft,
            'hop': hop,
            'languages': languages
        }
        
        self.models[model_path] = model
        self.model_configs[model_path] = config
        
        elapsed = time.time() - start
        print(f"[Server] Loaded model {Path(model_path).name} in {elapsed:.2f}s", file=sys.stderr)
        
        return model, config
    
    def load_audio(self, path, sr):
        """Load audio file"""
        wav_np, orig_sr = librosa.load(path, sr=None, mono=True)
        wav = torch.from_numpy(wav_np).unsqueeze(0)
        
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        
        return wav
    
    def compute_logmel(self, wav, sr, n_mels, n_fft, hop_length):
        """Compute log-mel spectrogram"""
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
    
    def build_windows(self, total_samples, segment_samples):
        """Build analysis windows"""
        if total_samples <= segment_samples:
            return [0]
        
        starts = [0]
        middle = max(0, (total_samples - segment_samples) // 2)
        end = max(0, total_samples - segment_samples)
        starts.extend([middle, end])
        
        # Add intermediate windows
        offsets = [0.25, 0.5, 0.75]
        for offset in offsets:
            start = int((total_samples - segment_samples) * offset)
            starts.append(max(0, min(start, end)))
        
        # Remove duplicates
        seen = set()
        uniq = []
        for s in starts:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        
        return uniq
    
    @torch.no_grad()
    def infer(self, audio_path, model_path):
        """Run inference on audio file"""
        start_time = time.time()
        
        # Load model (cached)
        model, config = self.load_model(model_path)
        
        # Load audio
        wav = self.load_audio(audio_path, config['sr'])
        segment_samples = int(config['segment'] * config['sr'])
        starts = self.build_windows(wav.shape[1], segment_samples)
        
        # Process windows
        ai_probs = []
        lang_probs = []
        multi_probs = []
        
        for start in starts:
            chunk = wav[:, start : start + segment_samples]
            if chunk.shape[1] < segment_samples:
                pad = segment_samples - chunk.shape[1]
                chunk = F.pad(chunk, (0, pad))
            
            mel = self.compute_logmel(chunk, config['sr'], config['n_mels'], 
                                     config['nfft'], config['hop'])
            mel = mel.unsqueeze(0).to(self.device)
            
            _, _, ai_logits, lang_logits, multi_logits = model(mel)
            
            ai_probs.append(torch.sigmoid(ai_logits).item())
            lang_probs.append(torch.softmax(lang_logits, dim=1).squeeze(0))
            multi_probs.append(torch.sigmoid(multi_logits).item())
        
        # Aggregate results
        ai_score = sum(ai_probs) / len(ai_probs) if ai_probs else 0.5
        multi_score = sum(multi_probs) / len(multi_probs) if multi_probs else 0.0
        lang_avg = torch.stack(lang_probs).mean(dim=0) if lang_probs else torch.zeros(len(config['languages']))
        
        # Build language distribution
        lang_distribution = {}
        for idx, lang in enumerate(config['languages']):
            lang_distribution[lang] = float(lang_avg[idx].item())
        
        elapsed = time.time() - start_time
        
        return {
            'aiScore': float(ai_score),
            'languageDistribution': lang_distribution,
            'multiSpeakerScore': float(multi_score),
            'inferenceTimeMs': int(elapsed * 1000)
        }
    
    def run(self):
        """Run server loop - read commands from stdin"""
        print("[Server] Ready for requests", file=sys.stderr)
        sys.stderr.flush()
        
        for line in sys.stdin:
            try:
                line = line.strip()
                if not line:
                    continue
                
                cmd = json.loads(line)
                
                if cmd.get('action') == 'infer':
                    audio_path = cmd.get('audio')
                    model_path = cmd.get('model')
                    
                    if not audio_path or not model_path:
                        result = {'error': 'Missing audio or model path'}
                    elif not os.path.exists(audio_path):
                        result = {'error': f'Audio file not found: {audio_path}'}
                    elif not os.path.exists(model_path):
                        result = {'error': f'Model file not found: {model_path}'}
                    else:
                        result = self.infer(audio_path, model_path)
                    
                    print(json.dumps(result))
                    sys.stdout.flush()
                
                elif cmd.get('action') == 'ping':
                    print(json.dumps({'status': 'ok'}))
                    sys.stdout.flush()
                
                elif cmd.get('action') == 'exit':
                    break
                
            except Exception as e:
                error_result = {
                    'error': str(e),
                    'type': type(e).__name__
                }
                print(json.dumps(error_result))
                sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='Persistent inference server')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')
    args = parser.parse_args()
    
    server = InferenceServer(device=args.device)
    server.run()


if __name__ == '__main__':
    main()
