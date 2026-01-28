const crypto = require("crypto");

const createReplayCache = (config) => {
  const entries = new Map();

  const prune = () => {
    const now = Date.now();
    for (const [key, value] of entries.entries()) {
      if (now - value.timestamp > config.ttlMs) {
        entries.delete(key);
      }
    }

    if (entries.size > config.maxEntries) {
      const overflow = entries.size - config.maxEntries;
      const keys = entries.keys();
      for (let i = 0; i < overflow; i += 1) {
        const next = keys.next();
        if (next.done) break;
        entries.delete(next.value);
      }
    }
  };

  const hashPayload = (payload) =>
    crypto.createHash("sha256").update(payload).digest("hex");

  const checkAndStore = (payload) => {
    if (!payload) return { ok: true };
    prune();
    const hash = hashPayload(payload);
    if (entries.has(hash)) return { ok: false, hash };
    entries.set(hash, { timestamp: Date.now() });
    return { ok: true, hash };
  };

  return {
    checkAndStore,
  };
};

module.exports = {
  createReplayCache,
};
