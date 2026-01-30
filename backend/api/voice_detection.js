const { isValidApiKey } = require("../utils/authentication");
const { validateRequest } = require("../utils/validation");
const { detectVoiceSource } = require("../services/voice_detection_service");

const STRICT_ERROR_MESSAGE = "Invalid API key or malformed request";

const sendError = (res, statusCode, message = STRICT_ERROR_MESSAGE) => {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ status: "error", message }));
};

const handleVoiceDetection = async (req, res, payload, config) => {
  if (!isValidApiKey(req.headers, config)) {
    sendError(res, 401);
    return;
  }

  const validationError = validateRequest(payload, config);
  if (validationError) {
    sendError(res, 400);
    return;
  }

  try {
    const result = await detectVoiceSource(payload, config);
    if (!result.ok) {
      sendError(res, result.statusCode ?? 400);
      return;
    }

    const responsePayload = {
      status: "success",
      language: payload.language,
      classification: result.data.classification,
      confidenceScore: result.data.confidenceScore,
      explanation: result.data.explanation,
    };

    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(responsePayload));
  } catch (err) {
    sendError(res, 500);
  }
};

module.exports = {
  handleVoiceDetection,
};
