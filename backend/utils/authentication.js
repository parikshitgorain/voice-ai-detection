// Authentication with admin-managed API keys (sk_*) and legacy config key
const isValidApiKey = (headers, config) => {
  const apiKey = headers["x-api-key"];
  
  if (!apiKey) {
    return false;
  }
  
  // Check admin-managed API keys (sk_*) with limit enforcement
  if (apiKey.startsWith("sk_")) {
    try {
      const adminModule = require("./admin");
      if (adminModule && adminModule.validateAndTrackApiKey) {
        const result = adminModule.validateAndTrackApiKey(apiKey);
        
        // Result: { valid: true/false, error: string, code: number }
        if (result && typeof result === 'object') {
          return result.valid === true;
        }
      }
    } catch (err) {
      // Admin module error - reject admin keys if module fails
      console.error("Admin module error:", err.message);
      return false;
    }
  }
  
  // Fallback to legacy config-based API key (unlimited, no sk_ prefix)
  if (config && config.apiKey && apiKey === config.apiKey) {
    return true;
  }
  
  return false;
};

module.exports = {
  isValidApiKey,
};
