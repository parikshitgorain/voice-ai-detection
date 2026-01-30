# Deep Model (Inference)

Python model code used by the API for inference.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# If torch/torchaudio need CPU wheels explicitly:
# .venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Run inference
```bash
.venv/bin/python infer_multitask.py --audio /path/to/audio.mp3 --model multitask_English.pt --device cpu
```

## Model files
- `multitask_English.pt`
- `multitask_Hindi.pt`
- `multitask_Tamil.pt`
- `multitask_Malayalam.pt`
- `multitask_Telugu.pt`

Weights are tracked by Git LFS or should be downloaded separately for deployments.
