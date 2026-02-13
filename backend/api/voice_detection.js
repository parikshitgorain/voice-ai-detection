const { isValidApiKey } = require("../utils/authentication");
const { validateRequest } = require("../utils/validation");
const { detectVoiceSource } = require("../services/voice_detection_service");
const { logDetection } = require("../utils/logger");
const { getClientIp } = require("../utils/client_ip");

const sendError = (res, statusCode, message) => {
  if (res.headersSent || res.destroyed) return;
  try {
    res.statusCode = statusCode;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ status: "error", message }));
  } catch (err) {
    console.error('Error sending error response:', err.message);
  }
};

const buildQueueInfo = (req) => {
  const meta = req.queueMeta || {};
  if (!meta.queued) return null;
  const waitMs =
    Number.isFinite(meta.startedAt) && Number.isFinite(meta.queuedAt)
      ? Math.max(0, meta.startedAt - meta.queuedAt)
      : null;
  return {
    position: meta.position,
    waitMs,
  };
};

const buildLogBase = (req, payload) => {
  const meta = req.queueMeta || {};
  const queueWaitMs =
    Number.isFinite(meta.startedAt) && Number.isFinite(meta.queuedAt)
      ? Math.max(0, meta.startedAt - meta.queuedAt)
      : 0;
  const processingMs = Number.isFinite(meta.startedAt)
    ? Math.max(0, Date.now() - meta.startedAt)
    : null;

  return {
    requestId: req.requestId || null,
    ip: req.clientIp || getClientIp(req),
    language: payload && payload.language ? payload.language : null,
    queued: Boolean(meta.queued),
    queuePosition: meta.position || 0,
    queueWaitMs,
    processingMs,
  };
};

const handleVoiceDetection = async (req, res, payload, config) => {
  const logBase = buildLogBase(req, payload);

  const authResult = isValidApiKey(req.headers, config);
  if (!authResult || !authResult.valid) {
    sendError(res, 401, "Invalid API key.");
    logDetection({
      ...logBase,
      status: "error",
      reason: "INVALID_API_KEY",
    });
    return;
  }

  const validationError = validateRequest(payload, config);
  if (validationError) {
    sendError(res, 400, validationError.message || "Malformed request.");
    logDetection({
      ...logBase,
      status: "error",
      reason: validationError.code || "INVALID_REQUEST",
    });
    return;
  }

  try {
    const result = await detectVoiceSource(payload, config);
    if (!result.ok) {
      const errorMessage = result.error?.message || "Voice detection failed.";
      sendError(res, result.statusCode ?? 400, errorMessage);
      logDetection({
        ...logBase,
        status: "error",
        reason: result.error?.code || "DETECTION_FAILED",
      });
      return;
    }

    const responsePayload = {
      status: "success",
      language: payload.language,
      classification: result.data.classification,
      confidenceScore: result.data.confidenceScore,
      explanation: result.data.explanation,
    };

    const queueInfo = buildQueueInfo(req);
    if (queueInfo) {
      responsePayload.queue = queueInfo;
    }

    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(responsePayload));

    logDetection({
      ...logBase,
      status: "success",
      classification: result.data.classification,
      confidenceScore: result.data.confidenceScore,
    });
  } catch (err) {
    sendError(res, 500, "Internal server error. Please try again.");
    logDetection({
      ...logBase,
      status: "error",
      reason: "UNHANDLED_EXCEPTION",
      detail: err && err.message ? err.message : "unknown",
    });
  }
};

module.exports = {
  handleVoiceDetection,
};
