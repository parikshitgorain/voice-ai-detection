const createRateLimiter = (config) => {
  const buckets = new Map();

  const refill = (bucket, now) => {
    const elapsed = (now - bucket.lastRefill) / 1000;
    bucket.tokens = Math.min(
      config.maxTokens,
      bucket.tokens + elapsed * config.refillPerSecond
    );
    bucket.lastRefill = now;
  };

  const allow = (key) => {
    const now = Date.now();
    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = { tokens: config.maxTokens, lastRefill: now };
      buckets.set(key, bucket);
    }

    refill(bucket, now);
    if (bucket.tokens < 1) return false;
    bucket.tokens -= 1;
    return true;
  };

  return { allow };
};

module.exports = {
  createRateLimiter,
};
