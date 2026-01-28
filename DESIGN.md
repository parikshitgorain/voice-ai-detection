# Voice AI Detection System - Production Design Blueprint

## 1) System Goal
Detect whether a single MP3 voice sample is AI_GENERATED or HUMAN, and return a calibrated confidence score with a short technical explanation. The system uses feature-based, explainable ML without transcription or audio enhancement.

Supported languages (metadata only): Tamil, English, Hindi, Malayalam, Telugu.

## 2) Architecture Overview (Strict Separation)

### Frontend (Desktop-only Internal Console)
- Compact UI for internal analysts.
- Validates inputs (file type, size, metadata) before upload.
- Sends API request to backend.
- Displays: classification, confidenceScore, explanation.
- No animations or flashy UI.

### Backend API (REST)
- Endpoint: `POST /api/voice-detection`
- Auth: `x-api-key` header.
- Strict request validation, fast-fail on errors.
- Orchestrates audio pipeline, feature extraction, classification, explanation.

### Audio Pipeline (Deterministic)
- Base64 decode -> MP3 -> PCM.
- No enhancement, filtering, or normalization.
- Duration and sample rate recorded.
- Deterministic processing (fixed windowing, fixed statistics).

### Feature Extraction (Feature-only ML)
- Duration, sample rate.
- RMS energy stats.
- Pitch (F0) stats + stability.
- MFCC summary stats.
- Spectral flatness.
- Zero-crossing rate.

### Classifier (Explainable Classical ML)
- Feature vector only.
- Model: Logistic Regression or Linear SVM (preferred for interpretability).
- Output:
  - classification: `AI_GENERATED` or `HUMAN`
  - confidenceScore in [0.0, 1.0]
- Confidence is calibrated, never extreme.

### Explanation Layer (Deterministic)
- Maps feature signals to short technical explanations.
- 1-2 lines, aligned with classification.
- No dramatic wording.

## 3) API Contract (Strict)

### Request
```
POST /api/voice-detection
Headers:
  x-api-key: <key>
Body (JSON):
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "..."
}
```

### Response
```
{
  "classification": "AI_GENERATED" | "HUMAN",
  "confidenceScore": 0.0 - 1.0,
  "explanation": "Short technical explanation"
}
```

### Error Responses (Explicit)
- Invalid API key -> 401
- Unsupported format -> 400
- File size > 50 MB -> 413
- Duration < 10 sec -> 400
- Duration > 5 min -> 400 (or partial analysis with clear flag)
- Malformed request / Base64 -> 400

## 4) Input Limits & Safeguards

### Limits
- Max file size: 50 MB
- Min duration: 10 sec
- Max duration: 300 sec

### Behavior
- < 10 sec: reject.
- > 300 sec: reject or analyze representative windows only.
- Large but valid: do not increase confidence automatically.

### Safeguards
- Soft rate limiting per API key (token bucket).
- Reject malformed Base64 and invalid JSON.
- Optional hash-based replay detection (short-lived cache).
- Fail fast on oversized request body.
- No data retention beyond request processing.
- Audio storage is transient only; see `AUDIO_STORAGE_POLICY.md`.
- Speech-presence gate (VAD) rejects non-speech audio before classification.

## 5) Audio Pipeline

### Steps
1) Base64 decode to MP3 bytes.
2) Parse MP3 -> PCM using deterministic decoder settings.
3) Validate duration and format.
4) Segment into analysis windows (see below).
5) Extract features for each window.
6) Aggregate feature stats.
7) Classify and generate explanation.

### Determinism
- Fixed window size, hop size, FFT parameters.
- No randomization or audio manipulation.

## 6) Multi-Window Analysis (Long Audio)

### Window Selection
- Short audio (<= 30 sec): single full window.
- Long audio (> 30 sec): analyze windows at start, middle, end.

### Aggregation
- Per-window feature vectors -> aggregate with:
  - Mean
  - Median
  - Interquartile range
  - Drift metrics (see below)
- Classification uses aggregated features.
- If windows disagree, reduce confidence.

## 7) Feature Extraction Details

### Global
- Duration (sec), sample rate (Hz).

### RMS Energy
- Frame-based RMS.
- Stats: mean, std, min, max, interquartile range.

### Pitch (F0)
- Frame-based F0 estimate.
- Stats: mean, std, range.
- Stability: % frames with stable pitch, F0 drift.

### MFCC
- 13 MFCCs (or 20 if available).
- Stats: mean + std for each coefficient.

### Spectral Flatness
- Frame-based spectral flatness.
- Stats: mean, std, skewness.

### Zero-Crossing Rate (ZCR)
- Frame-based ZCR.
- Stats: mean, std.

## 8) Feature Stability Analysis

### Drift Metrics
- Per-window feature variance across windows.
- Low drift (very stable) => stronger AI signal.
- Natural variability (moderate drift) => stronger HUMAN signal.

### Stability Score
- Normalized score combining:
  - Pitch stability
  - RMS stability
  - Spectral flatness stability
- Used as a feature to classifier and explanation layer.

## 9) Classifier & Calibration

### Model
- Classical ML (Logistic Regression or Linear SVM).
- Feature-only input, deterministic.
- Trained offline with balanced AI/human datasets.

### Calibration
- Use Platt scaling or isotonic regression offline.
- Runtime confidence is calibrated probability.
- Cap confidence at 0.95.
- Decrease confidence when:
  - Duration is near minimum.
  - Window disagreement is high.
  - Noise level is high.

## 10) Explanation Layer

### Rules (Examples)
- AI_GENERATED:
  - "Low pitch drift and highly uniform spectral flatness suggest synthetic stability."
- HUMAN:
  - "Natural pitch variability and uneven energy contours align with human speech."

### Constraints
- 1-2 short technical lines.
- Deterministic mapping from feature signals.
- Always aligned with classification.

## 11) Frontend Console (Desktop-only)

### Inputs
- MP3 file picker.
- Language selector (metadata only).
- Upload button (disabled until validation passes).

### Output
- Classification label.
- Confidence score.
- Short explanation.
- Clear error messages.

## 12) Backend Orchestration Flow

1) Validate API key.
2) Validate request schema.
3) Enforce size and duration limits.
4) Decode Base64, parse MP3 -> PCM.
5) Extract features (multi-window if needed).
6) Aggregate and compute stability metrics.
7) Predict classification.
8) Calibrate confidence and apply caps.
9) Build explanation.
10) Return response.

## 13) Ethics & Transparency
- No transcription-based detection.
- No audio enhancement or manipulation.
- No personal data retention.
- Explainability prioritized over aggressive claims.
- Honest uncertainty handling.
- Communicate that audio is processed transiently and not retained after analysis.

## 14) Limitations (Mandatory)
- Accuracy depends on audio quality.
- Edge cases exist for highly expressive human speech.
- AI voice generation evolves rapidly.
- System is probabilistic, not absolute.

## 15) Future-Ready Improvements (Within Constraints)
- Add more AI model families to training data.
- Expand per-language calibration sets (metadata only).
- Improve stability metrics with better window selection.
- Continuous evaluation pipeline (offline, no user data retention).
