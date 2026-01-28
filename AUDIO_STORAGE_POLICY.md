# Audio Storage Policy (Strict)

This system processes audio transiently and must never retain audio or raw PCM
beyond the scope of a single request.

## Non-Negotiable Rules
- Audio must never be saved permanently.
- Audio must not be written to long-term storage.
- Audio must not be associated with user identity.
- No database persistence of audio or raw PCM.

## Temporary Handling Rules
- Audio may exist only in memory (preferred) or in the OS temp directory.
- Temporary storage must be isolated per request.
- Temporary files must have random, non-identifiable names.

## Processing Lifecycle (Required)
1) Receive Base64 MP3 in request.
2) Decode and process audio.
3) Extract features.
4) Run classification.
5) Generate explanation.
6) Return response.
7) Immediately delete any temporary audio data.

## Auto-Deletion Guarantee
- Temporary audio must be deleted immediately after result generation or on
  request completion (success or failure).
- Cleanup must also occur on errors, timeouts, or interrupted processing.

## Fallback Safety
- If in-memory processing is used, ensure no references persist after the
  request scope ends.
- If temp files are used, only OS-managed temp directories are allowed and
  automatic cleanup is mandatory.

## Logging & Privacy
- Do not log raw audio or Base64 content.
- Logs may include only: duration, file size, processing time, and (optionally)
  the final classification.

## Compliance Statement (Required)
The system must clearly state that audio is processed transiently, no audio is
stored after analysis, and no personal voice data is retained.
