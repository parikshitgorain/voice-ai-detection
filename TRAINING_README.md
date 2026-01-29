Training System Overview (VPS)

What this training system learns
- Human physiology: chaotic phase noise, sub-phonetic timing jitter, breath-energy-formant coupling, and vocal-tract drift.
- Neural synthesis artifacts: over-regular prosody planning, bounded phase coherence, excessive temporal consistency, and spectral smoothness limits.
- Language identity without speech-to-text: rhythm, stress, vowel-consonant balance, phoneme duration patterns, and pitch contour tendencies.
- Long-range temporal memory: carry-over vs reset behavior across segments and persistence of prosodic signatures.

Representation-first philosophy
- The system learns a shared acoustic representation before decision rules are applied.
- Multiple objectives are trained together so the model captures physics, phonetics, and temporal behavior rather than a single decision boundary.
- Decisions remain conservative and are governed by explicit rules, not a single end-to-end score.

Multi-branch learning structure
- A shared encoder produces a stable representation of each segment.
- Separate heads learn: AI vs HUMAN, language identity, and multi-speaker risk.
- Contrastive objectives align segments from the same source and separate unrelated speech, improving robustness to unseen synthesis systems.
- Curriculum learning progressively increases augmentation difficulty to harden invariants.

Data organization and label inference
- Training data is organized by class first (AI vs HUMAN), and optionally by language under each class.
- Language identity is inferred from the folder name in the path, so the system can learn phonetic identity without speech-to-text.
- Valid language labels are: Tamil, English, Hindi, Malayalam, Telugu.
- Example layout concept (names are matched by substring, case-insensitive):
  - ai/Tamil/..., ai/English/..., ai/Hindi/..., ai/Malayalam/..., ai/Telugu/...
  - human/Tamil/..., human/English/..., human/Hindi/..., human/Malayalam/..., human/Telugu/...
- Multi-speaker labels are optional; if a path contains “multi” or “multispeaker”, it is treated as multi-speaker for the safety head.

How learning objectives are combined
- AI vs HUMAN head: learns synthesis artifacts and physics-consistency constraints.
- Language head: learns rhythm, stress, vowel-consonant balance, and pitch contour tendencies without any text decoding.
- Multi-speaker head: learns speaker-boundary cues, pitch/formant regime shifts, and inconsistent breath signatures.
- Contrastive head: enforces temporal memory by aligning two segments from the same source while separating different sources.

How large datasets are handled
- The system avoids storing raw tensors or frame-level dumps; only compact, aggregated summaries are emitted for analysis.
- Training consumes audio in short segments and never relies on container metadata as labels.
- Scaling is achieved by streaming data and by using balanced sampling across languages and classes.

Why the system may return HUMAN with low confidence
- Short clips often lack enough temporal evidence.
- Multi-speaker recordings degrade reliability by mixing competing vocal regimes.
- Conflicting signals are treated as uncertainty rather than forcing an AI label.

How to interpret outputs
- Evidence indicates which signal groups support or contradict AI behavior.
- Confidence reflects agreement between independent signal groups, not certainty.
- Classification follows strict governance: multiple independent AI signals must align, otherwise the outcome remains conservative.
- Uncertainty is a feature: it protects against false accusations and remains honest under adversarial audio.

Expected behavior on modern TTS
- High-quality synthetic speech can appear human-like; the system will report low confidence when signals conflict.
- Robustness improves as more diverse synthetic families are included in training without naming vendors or using fingerprints.

How to use this training system (conceptual)
- Place AI and HUMAN audio into the class-first folder structure, optionally nested by language.
- Start the training pipeline on a machine with adequate GPU memory for the chosen model size.
- Training produces a compact model artifact that is loaded by the backend at inference time.
- After training, datasets can be removed; only the trained model is required for VPS deployment.

Trained model artifacts (per-language)
- English: `backend/deep/multitask_English.pt`
- Hindi: `backend/deep/multitask_Hindi.pt`
- Tamil: `backend/deep/multitask_Tamil.pt`
- Malayalam: `backend/deep/multitask_Malayalam.pt`
- Telugu: `backend/deep/multitask_Telugu.pt`
