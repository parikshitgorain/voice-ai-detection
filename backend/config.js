const config = {
  apiKey: process.env.VOICE_DETECT_API_KEY || "change-me",
  supportedLanguages: ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
  audioFormat: "mp3",
  limits: {
    maxFileBytes: 50 * 1024 * 1024,
    maxBodyBytes: 80 * 1024 * 1024,
    minDurationSeconds: 10,
    maxDurationSeconds: 300,
    longAudioThresholdSeconds: 30,
    windowDurationSeconds: 20,
    maxWindowCount: 3,
  },
  rateLimit: {
    maxTokens: 12,
    refillPerSecond: 0.2,
  },
  replayCache: {
    ttlMs: 5 * 60 * 1000,
    maxEntries: 2000,
  },
  vad: {
    enabled: true,
    aggressiveness: 2,
    frameMs: 20,
    minSpeechRatio: 0.2,
    minSpeechFrames: 5,
    targetSampleRate: 16000,
  },
};

module.exports = config;
