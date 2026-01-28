const { isValidApiKey } = require("../utils/authentication");
const { validateRequest } = require("../utils/validation");
const { createRateLimiter } = require("../utils/rate_limiter");
const { createReplayCache } = require("../utils/replay_cache");
const { detectVoiceSource } = require("../services/voice_detection_service");

let limiter = null;
let replayCache = null;

const sendError = (res, statusCode, message) => {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ status: "error", message }));
};

const mapErrorMessage = (code) => {
  switch (code) {
    case "INVALID_BASE64":
    case "BASE64_DECODE_FAILED":
    case "EMPTY_AUDIO":
    case "DECODE_FAILED":
    case "UNSUPPORTED_FORMAT":
      return "Audio decode failed or unsupported format";
    case "FORMAT_REQUIRED":
    case "AUDIO_REQUIRED":
    case "INVALID_SAMPLE_RATE":
    case "DURATION_TOO_SHORT":
    case "DURATION_TOO_LONG":
    case "FRAME_ERROR":
    case "NON_SPEECH_AUDIO":
    case "REPLAY_DETECTED":
      return "Invalid request or unsupported audio.";
    case "DECODER_MISSING":
    case "VAD_UNAVAILABLE":
    case "VAD_UNSUPPORTED":
    case "PITCH_ESTIMATOR_MISSING":
    case "MFCC_EXTRACTOR_MISSING":
    case "SPECTRAL_FLATNESS_MISSING":
      return "Processing failed. Please retry.";
    default:
      return "Processing failed. Please retry.";
  }
};

const handleVoiceDetection = async (req, res, payload, config) => {
  if (!isValidApiKey(req.headers, config)) {
    sendError(res, 401, "Processing failed. Please retry.");
    return;
  }

  if (!limiter) limiter = createRateLimiter(config.rateLimit);
  if (!replayCache) replayCache = createReplayCache(config.replayCache);

  const apiKey = req.headers["x-api-key"];
  if (!limiter.allow(apiKey)) {
    sendError(res, 429, "Processing failed. Please retry.");
    return;
  }

  const validationError = validateRequest(payload, config);
  if (validationError) {
    sendError(res, 400, mapErrorMessage(validationError.code));
    return;
  }

  const replayCheck = replayCache.checkAndStore(payload.audioBase64);
  if (!replayCheck.ok) {
    sendError(res, 409, mapErrorMessage("REPLAY_DETECTED"));
    return;
  }

  try {
    const result = await detectVoiceSource(payload, config);
    if (!result.ok) {
      sendError(
        res,
        result.statusCode ?? 400,
        mapErrorMessage(result.error?.code)
      );
      return;
    }

    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(result.data));
  } finally {
    replayCache.release(replayCheck.hash);
  }
};

module.exports = {
  handleVoiceDetection,
};
