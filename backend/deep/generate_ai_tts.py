import argparse
import os
import random
import time


def read_texts(path, max_items):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(line)
            if max_items and len(texts) >= max_items:
                break
    return texts


def save_wav(audio, sample_rate, out_path):
    import numpy as np
    import soundfile as sf

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = audio.squeeze()
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(out_path, audio, sample_rate)


def infer_model_id(lang):
    mapping = {
        "Hindi": "facebook/mms-tts-hin",
        "Tamil": "facebook/mms-tts-tam",
        "Malayalam": "facebook/mms-tts-mal",
        "Telugu": "facebook/mms-tts-tel",
    }
    return mapping.get(lang)


def infer_lang_code(lang):
    mapping = {
        "Hindi": "hi",
        "Tamil": "ta",
        "Malayalam": "ml",
        "Telugu": "te",
        "English": "en",
    }
    return mapping.get(lang)


def infer_edge_locale(lang):
    mapping = {
        "Hindi": "hi-IN",
        "Tamil": "ta-IN",
        "Malayalam": "ml-IN",
        "Telugu": "te-IN",
        "English": "en-US",
    }
    return mapping.get(lang)


def list_ref_audio(ref_dir, max_refs):
    exts = {".wav", ".mp3", ".flac", ".m4a"}
    refs = []
    for base, _, names in os.walk(ref_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() in exts:
                refs.append(os.path.join(base, name))
    random.shuffle(refs)
    if max_refs:
        refs = refs[:max_refs]
    return refs


def ensure_wav(path, tmp_dir):
    if path.lower().endswith(".wav"):
        return path
    os.makedirs(tmp_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(tmp_dir, f"{base}.wav")
    if os.path.exists(out_path):
        return out_path
    cmd = f'ffmpeg -y -i "{path}" -ac 1 -ar 24000 "{out_path}"'
    if os.system(cmd) != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    return out_path


def list_edge_voices(locale, max_voices):
    try:
        import asyncio
        import edge_tts
    except Exception as exc:
        raise SystemExit(f"Missing dependencies: {exc}. Install with: pip install edge-tts")

    async def _fetch():
        voices = await edge_tts.list_voices()
        return [v for v in voices if v.get("Locale") == locale]

    voices = asyncio.run(_fetch())
    shortnames = [v.get("ShortName") for v in voices if v.get("ShortName")]
    random.shuffle(shortnames)
    if max_voices:
        shortnames = shortnames[:max_voices]
    return shortnames


def generate_edge_tts(text, voice, out_path):
    import asyncio
    import edge_tts

    async def _run():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mp3_path = out_path.replace(".wav", ".mp3")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(mp3_path)
        cmd = f'ffmpeg -y -i "{mp3_path}" -ac 1 -ar 24000 "{out_path}"'
        if os.system(cmd) != 0:
            raise RuntimeError(f"ffmpeg failed for {mp3_path}")
        try:
            os.remove(mp3_path)
        except OSError:
            pass

    asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Path to text file (one sentence per line)")
    parser.add_argument("--out", required=True, help="Output directory for wav files")
    parser.add_argument("--lang", default="English", help="Language label (for logging only)")
    parser.add_argument("--max", type=int, default=100, help="Max samples to generate")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model", default="", help="Override model id (e.g. facebook/mms-tts-hin)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="auto", choices=["auto", "mms", "xtts", "edge"])
    parser.add_argument("--ref-dir", default="", help="Reference audio directory for XTTS voices")
    parser.add_argument("--refs", type=int, default=5, help="Number of reference voices to sample (XTTS)")
    args = parser.parse_args()

    random.seed(args.seed)
    texts = read_texts(args.text, args.max)
    if not texts:
        raise SystemExit("No texts found.")

    backend = args.backend
    if backend == "auto":
        backend = "edge"

    if backend == "xtts":
        try:
            from TTS.api import TTS
        except Exception as exc:
            raise SystemExit(f"Missing dependencies: {exc}. Install with: pip install TTS")
        lang_code = infer_lang_code(args.lang)
        if not lang_code:
            raise SystemExit(f"No language code mapping for {args.lang}.")
        if not args.ref_dir:
            raise SystemExit("XTTS requires --ref-dir.")
        refs = list_ref_audio(args.ref_dir, args.refs)
        if not refs:
            raise SystemExit(f"No reference audio found in {args.ref_dir}")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=args.device.startswith("cuda"))
        tmp_dir = "/tmp/xtts_ref_wavs"

        for idx, text in enumerate(texts, 1):
            ref_path = refs[(idx - 1) % len(refs)]
            try:
                wav_ref = ensure_wav(ref_path, tmp_dir)
            except Exception as exc:
                print(f"ref convert failed: {ref_path} -> {exc}")
                continue
            out_path = os.path.join(args.out, f"tts_{args.lang}_{idx:06d}.wav")
            tts.tts_to_file(text=text, file_path=out_path, speaker_wav=wav_ref, language=lang_code)
            if idx % 10 == 0:
                print(f"generated {idx}/{len(texts)} for {args.lang}")
            time.sleep(0.01)
        return

    if backend == "edge":
        locale = infer_edge_locale(args.lang)
        if not locale:
            raise SystemExit(f"No locale mapping for {args.lang}.")
        voices = list_edge_voices(locale, args.refs)
        if not voices:
            raise SystemExit(f"No voices found for locale {locale}.")
        for idx, text in enumerate(texts, 1):
            voice = voices[(idx - 1) % len(voices)]
            out_path = os.path.join(args.out, f"tts_{args.lang}_{idx:06d}.wav")
            try:
                generate_edge_tts(text, voice, out_path)
            except Exception as exc:
                print(f"failed {idx}/{len(texts)} for {args.lang}: {exc}")
                continue
            if idx % 10 == 0:
                print(f"generated {idx}/{len(texts)} for {args.lang}")
            time.sleep(0.01)
        return

    try:
        import torch
        from transformers import AutoTokenizer, VitsModel
    except Exception as exc:
        raise SystemExit(f"Missing dependencies: {exc}. Install with: pip install transformers torch soundfile")

    model_id = args.model.strip() or infer_model_id(args.lang)
    if not model_id:
        raise SystemExit(f"No model mapping for language: {args.lang}. Use --model to override.")

    device = torch.device(args.device)
    model = VitsModel.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for idx, text in enumerate(texts, 1):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).waveform
        out_path = os.path.join(args.out, f"tts_{args.lang}_{idx:06d}.wav")
        save_wav(output, model.config.sampling_rate, out_path)
        if idx % 10 == 0:
            print(f"generated {idx}/{len(texts)} for {args.lang}")
        time.sleep(0.01)


if __name__ == "__main__":
    main()
