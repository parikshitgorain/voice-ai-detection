Dataset Links and Folder Mapping (100 GB Plan)

Goal
- Total budget ~100 GB.
- Per language target: ~5 GB HUMAN + ~5 GB AI.
- Languages: Tamil, English, Hindi, Malayalam, Telugu.

Folder mapping (class-first, language-nested)
Use these folders for both train and validation splits.

```
backend/data/train/ai/<Language>/
backend/data/train/human/<Language>/
backend/data/val/ai/<Language>/
backend/data/val/human/<Language>/
```

Language mapping rule
- The training system infers language from the folder name in the path (case-insensitive).
- Example: ai/Tamil/, human/English/, ai/Hindi/, human/Malayalam/, ai/Telugu/.

Human audio sources (direct links)
English (target ~5 GB)
- LibriSpeech train-clean-100 (5.9 GB). citeturn6search0
```
https://www.openslr.org/resources/12/train-clean-100.tar.gz
```

Hindi (target ~5 GB)
- OpenSLR SLR103 Hindi train (4.4 GB) + Hindi test (258 MB). citeturn8view0
```
https://openslr.trmal.net/resources/103/Hindi_train.tar.gz
https://openslr.trmal.net/resources/103/Hindi_test.tar.gz
```
- Optional top-up: Common Voice Scripted Speech Hindi 23.0 (464.59 MB). citeturn1search2
```
https://datacollective.mozillafoundation.org/datasets/cmflnuzw5hbe47u0fvrugjyb6
```

Tamil (target ~5 GB)
- OpenSLR SLR65 Tamil female+male (1.37 GB total). citeturn7view0
```
https://www.openslr.org/resources/65/ta_in_female.zip
https://www.openslr.org/resources/65/ta_in_male.zip
```
- Common Voice Scripted Speech Tamil 23.0 (8.56 GB; sample down to 3.6–4 GB). citeturn2view0
```
https://datacollective.mozillafoundation.org/datasets/cmflnuzw73r9g1avrbu6bwkfx
```

Malayalam (target ~5 GB)
- OpenSLR SLR63 Malayalam female+male (1.35 GB total). citeturn7view1
```
https://openslr.trmal.net/resources/63/ml_in_female.zip
https://openslr.trmal.net/resources/63/ml_in_male.zip
```
- Common Voice Scripted Speech Malayalam 23.0 (217.58 MB). citeturn1search0
```
https://datacollective.mozillafoundation.org/datasets/cmflnuzw6tk5drnf28rwq5e0m
```
Note: Public human-only Malayalam audio at 5 GB is limited; plan to add additional public corpora if available or increase collection via curated, licensed sources.

Telugu (target ~5 GB)
- OpenSLR SLR66 Telugu female+male (1.03 GB total). citeturn7view2
```
https://openslr.trmal.net/resources/66/te_in_female.zip
https://openslr.trmal.net/resources/66/te_in_male.zip
```
- Common Voice Scripted Speech Telugu 23.0 (58.24 MB). citeturn1search3
```
https://datacollective.mozillafoundation.org/datasets/cmflnuzw7wkvsk4grx6kr93oq
```
Note: Public human-only Telugu audio at 5 GB is limited; plan to add additional public corpora if available or increase collection via curated, licensed sources.

AI audio sources (direct links where available)
English (recommended primary AI set)
- ASVspoof 2021 LA eval (AI + human, 7.8 GB). citeturn5search5
```
https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1
```
- ASVspoof 2021 DF eval (deepfake audio, 34.5 GB; use subset to 5 GB). citeturn5search4
```
https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1
https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz?download=1
https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz?download=1
https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz?download=1
```
Keys/labels (required for ASVspoof):
```
https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz
```

Multilingual AI audio (request/approval required)
- SpoofCeleb (AI + human, hosted on Hugging Face; access by request). citeturn5search0
```
https://www.jungjee.com/spoofceleb/
```
- TITW (human speech, used as bona fide for SpoofCeleb; access by request). citeturn5search2
```
https://www.jungjee.com/titw/
```
- FakeAVCeleb (includes synthesized audio; access by request). citeturn5search3
```
https://github.com/DASH-Lab/FakeAVCeleb
```

AI audio for Tamil/Hindi/Malayalam/Telugu (generation approach)
Public, language-specific AI audio at 5 GB per language is limited. To reach the 5 GB AI target per language, generate synthetic audio offline using open-source TTS models and language-specific text corpora.

Open-source multilingual TTS models (support Tamil, Hindi, Malayalam, Telugu, English):
- AI4Bharat Indic Parler‑TTS (21 languages including Tamil/Hindi/Malayalam/Telugu/English). citeturn10search0
```
https://huggingface.co/ai4bharat/indic-parler-tts
```
- AI4Bharat IndicF5 (supports Hindi, Malayalam, Tamil, Telugu). citeturn10search2
```
https://github.com/AI4Bharat/IndicF5
```

Recommended AI folder placement for generated audio
- Place synthesized files under:
```
backend/data/train/ai/English/
backend/data/train/ai/Hindi/
backend/data/train/ai/Tamil/
backend/data/train/ai/Malayalam/
backend/data/train/ai/Telugu/
```

Notes on size targets
- Some public human corpora for Malayalam and Telugu are below 5 GB; use additional public sources or expand via licensed collection to meet the 5 GB target.
- For AI audio, use ASVspoof as a base and generate language-specific synthetic speech with open-source TTS to reach per-language targets.
