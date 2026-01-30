const isValidApiKey = (headers, config) => {
  if (!config || !config.apiKey) return false;
  const apiKey = headers["x-api-key"];
  return apiKey && apiKey === config.apiKey;
};

module.exports = {
  isValidApiKey,
};
