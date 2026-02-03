// FIX: Updated to handle new validation response with limit enforcement
const isValidApiKey = (headers, config) => {
  // First check new admin-managed API keys with limit enforcement
  const apiKey = headers["x-api-key"];
  if (apiKey) {
    try {
      const adminModule = require("./admin");
      if (adminModule && adminModule.validateAndTrackApiKey) {
        const result = adminModule.validateAndTrackApiKey(apiKey);
        
        // Return result object for proper error handling
        // { valid: true/false, error: string, code: number }
        if (result && typeof result === 'object') {
          return result;
        }
      }
    } catch (err) {
      // Admin module not available or error, continue to legacy check
      console.error("Admin module error:", err.message);
    }
  }
  
  // Fallback to legacy config-based API key (unlimited)
  if (config && config.apiKey && apiKey === config.apiKey) {
    return { valid: true };
  }
  
  return { valid: false, error: "Invalid API key" };
};

module.exports = {
  isValidApiKey,
};
