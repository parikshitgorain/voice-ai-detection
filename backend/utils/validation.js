const isNonEmptyString = (value) => typeof value === "string" && value.trim().length > 0;

const estimateBase64Bytes = (base64String) => {
  if (!isNonEmptyString(base64String)) return 0;
  const sanitized = base64String.trim().replace(/\s+/g, "");
  const padding = sanitized.endsWith("==") ? 2 : sanitized.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((sanitized.length * 3) / 4) - padding);
};

const validateRequest = (payload, config) => {
  if (!payload || typeof payload !== "object") {
    return { code: "MALFORMED_REQUEST", message: "Malformed request body." };
  }
  if (!isNonEmptyString(payload.language)) {
    return { code: "LANGUAGE_REQUIRED", message: "Language is required." };
  }
  if (!config.supportedLanguages.includes(payload.language)) {
    return { code: "UNSUPPORTED_LANGUAGE", message: "Unsupported language." };
  }
  if (!isNonEmptyString(payload.audioFormat)) {
    return { code: "FORMAT_REQUIRED", message: "audioFormat is required." };
  }
  if (payload.audioFormat !== config.audioFormat) {
    return { code: "UNSUPPORTED_FORMAT", message: "Unsupported audio format." };
  }
  if (!isNonEmptyString(payload.audioBase64)) {
    return { code: "AUDIO_REQUIRED", message: "audioBase64 is required." };
  }

  const estimatedBytes = estimateBase64Bytes(payload.audioBase64);
  if (estimatedBytes > config.limits.maxFileBytes) {
    return { code: "FILE_TOO_LARGE", message: "File size exceeds 50 MB limit." };
  }
  return null;
};

module.exports = {
  validateRequest,
  estimateBase64Bytes,
};
