# Services Structure (Locked)

This directory structure is final and should remain stable.

- `audio_loader/` Base64 decode + MP3 to PCM decoding + duration
- `feature_extractor/` Frame-based feature extraction + aggregation
- `audio_types/` Feature schema helpers
- `audio_pipeline.js` Orchestration of loading + feature extraction
- `orchestration/` Public orchestration entrypoint(s)
- `classifier/` Legacy feature-based classifier (not used when deep-only mode is enabled)
- `explanation/` Reserved for future explanation layer
- `vad/` Speech presence detection gate (WebRTC VAD)
- `deep_model/` Deep-model inference bridge (Python)
- `language_warning.js` Language mismatch warning (selected vs detected)

Do not move files or merge responsibilities without an explicit design change.
