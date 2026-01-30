# Services

Core processing pipeline for voice detection.

## Active modules
- `deep_model/`: Python inference bridge
- `voice_detection_service.js`: orchestrates model inference

## Supporting modules
- `audio_loader/`, `feature_extractor/`, `audio_pipeline.js`, `vad/`

Keep structure stable unless you change the architecture.
