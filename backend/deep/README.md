# Deep Model (Inference)

This folder contains the Python model code used by the API for inference.

## Inference
Create venv and install requirements:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# If torch/torchaudio need CPU wheels explicitly:
# .venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Run inference on a file:
```bash
.venv/bin/python infer_multitask.py --audio /path/to/audio.wav --model multitask_multilingual.pt
```

## Model Files
- `multitask_multilingual.pt` (primary multilingual model)
- Optional per-language models:
  - `multitask_English.pt`, `multitask_Hindi.pt`, `multitask_Tamil.pt`,
    `multitask_Malayalam.pt`, `multitask_Telugu.pt`

For public GitHub repos, store weights with Git LFS or provide download instructions.

Training scripts are intentionally excluded from this deployment.
