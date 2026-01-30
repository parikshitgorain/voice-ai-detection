# Services Structure (Locked)

This directory structure is final and should remain stable.

Active in the strict evaluation path:
- `deep_model/` Deep-model inference bridge (Python)
- `voice_detection_service.js` Orchestration for AI vs Human scoring

Present but not used in the strict evaluation path:
- `audio_loader/`, `feature_extractor/`, `audio_types/`, `audio_pipeline.js`, `vad/`,
  `language_warning.js`

Do not move files or merge responsibilities without an explicit design change.
