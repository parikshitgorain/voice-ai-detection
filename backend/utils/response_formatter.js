/**
 * Response Formatter Utility
 * Ensures API responses match exact hackathon evaluation format
 */

/**
 * Format a successful detection response according to evaluation spec
 * @param {string} classification - "HUMAN" or "AI_GENERATED"
 * @param {number} confidenceScore - Float between 0.0 and 1.0
 * @returns {object} - Exactly 3 fields: status, classification, confidenceScore
 */
function formatSuccessResponse(classification, confidenceScore) {
  // Validate classification
  if (classification !== "HUMAN" && classification !== "AI_GENERATED") {
    throw new Error(`Invalid classification: ${classification}. Must be "HUMAN" or "AI_GENERATED"`);
  }

  // Validate confidenceScore
  if (typeof confidenceScore !== "number" || confidenceScore < 0 || confidenceScore > 1) {
    throw new Error(`Invalid confidenceScore: ${confidenceScore}. Must be a number between 0.0 and 1.0`);
  }

  const response = {
    status: "success",
    classification: classification,
    confidenceScore: confidenceScore
  };

  // Verify exactly 3 fields
  const keys = Object.keys(response);
  if (keys.length !== 3) {
    throw new Error(`Response must have exactly 3 fields, got ${keys.length}`);
  }

  return response;
}

/**
 * Format an error response
 * @param {string} message - Error description
 * @returns {object} - Error response with status and message
 */
function formatErrorResponse(message) {
  if (typeof message !== "string" || message.length === 0) {
    message = "An error occurred";
  }

  return {
    status: "error",
    message: message
  };
}

module.exports = {
  formatSuccessResponse,
  formatErrorResponse
};
