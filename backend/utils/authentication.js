const isValidApiKey = (headers, config) => {
  const apiKey = headers["x-api-key"];
  return apiKey && apiKey === config.apiKey;
};

module.exports = {
  isValidApiKey,
};
