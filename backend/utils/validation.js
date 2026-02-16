const isNonEmptyString = (value) => typeof value === "string" && value.trim().length > 0;

/**
 * Validate base64 string format
 * @param {string} base64String - Base64 encoded string
 * @returns {boolean} - True if valid base64
 */
const isValidBase64 = (base64String) => {
  if (!isNonEmptyString(base64String)) return false;
  const sanitized = base64String.trim().replace(/\s+/g, "");
  // Base64 regex pattern
  const base64Pattern = /^[A-Za-z0-9+/]*={0,2}$/;
  return base64Pattern.test(sanitized);
};

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
  
  // Validate language field (case-sensitive exact match)
  if (!isNonEmptyString(payload.language)) {
    return { code: "LANGUAGE_REQUIRED", message: "Language is required." };
  }
  
  // Case-sensitive language validation - exact match required
  const validLanguages = ["English", "Hindi", "Tamil", "Malayalam", "Telugu"];
  if (!validLanguages.includes(payload.language)) {
    return { 
      code: "UNSUPPORTED_LANGUAGE", 
      message: `Unsupported language. Must be one of: ${validLanguages.join(", ")}` 
    };
  }
  
  // Validate audioFormat field (case-sensitive exact match)
  if (!isNonEmptyString(payload.audioFormat)) {
    return { code: "FORMAT_REQUIRED", message: "audioFormat is required." };
  }
  
  // Case-sensitive format validation - must be exactly "mp3"
  if (payload.audioFormat !== "mp3") {
    return { 
      code: "UNSUPPORTED_FORMAT", 
      message: 'Unsupported audio format. Must be exactly "mp3".' 
    };
  }
  
  // Validate audioBase64 field
  if (!isNonEmptyString(payload.audioBase64)) {
    return { code: "AUDIO_REQUIRED", message: "audioBase64 is required." };
  }
  
  // Validate base64 format
  if (!isValidBase64(payload.audioBase64)) {
    return { 
      code: "INVALID_BASE64", 
      message: "audioBase64 contains invalid base64 encoding." 
    };
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
  isValidBase64,
};
