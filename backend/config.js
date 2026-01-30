const fs = require("fs");
const path = require("path");

const defaultPythonPath = (() => {
  const venvPath = path.join(__dirname, "deep", ".venv", "bin", "python");
  if (fs.existsSync(venvPath)) return venvPath;
  return "python3";
})();

const languageModelPaths = {
  English: process.env.DEEP_MODEL_PATH_ENGLISH,
  Hindi: process.env.DEEP_MODEL_PATH_HINDI,
  Tamil: process.env.DEEP_MODEL_PATH_TAMIL,
  Malayalam: process.env.DEEP_MODEL_PATH_MALAYALAM,
  Telugu: process.env.DEEP_MODEL_PATH_TELUGU,
};
const modelByLanguage = Object.fromEntries(
  Object.entries(languageModelPaths).filter(([, value]) => value)
);
const config = {
  apiKey: process.env.VOICE_DETECT_API_KEY || "",
  supportedLanguages: ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
  audioFormat: "mp3",
  audioFormats: ["mp3"],
  limits: {
    maxFileBytes: 50 * 1024 * 1024,
    maxBodyBytes: 80 * 1024 * 1024,
    minDurationSeconds: 2,
    maxDurationSeconds: 300,
    longAudioThresholdSeconds: 15,
    windowDurationSeconds: 9,
    maxWindowCount: 6,
    windowOverlapRatio: 0.5,
    fixedWindowOffsets: [0.2, 0.4, 0.6, 0.8],
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
  analysis: {
    frameSize: 1024,
    hopSize: 512,
    mfccCoefficients: 13,
    mfccFilters: 26,
    phaseAnalysisStride: 2,
    phaseEntropyBins: 24,
    phaseSampleBins: 18,
    phaseDeltaStdScale: 0.35,
    entropyBins: 24,
    peakMinDistanceMs: 120,
    intonationSmoothnessScale: 35,
    contourWindowRatio: 0.25,
    spectralFluxScale: 0.5,
    spectralRolloffScale: 0.4,
    rolloffPercent: 0.85,
    breath: {
      thresholdRatio: 0.4,
      minMs: 150,
      maxMs: 800,
      preFrames: 5,
      postFrames: 5,
      minEventsForCoupling: 2,
      baseline: {
        couplingMean: 0.55,
        couplingStd: 0.18,
      },
    },
    sentenceReset: {
      minPauseMs: 120,
      preFrames: 5,
      postFrames: 5,
      voiceThresholdRatio: 0.3,
    },
  },
  compressionTest: {
    enabled: true,
    bitratesKbps: [64, 96],
  },
  deepModel: {
    enabled: true,
    useMultitask: true,
    pythonPath: process.env.DEEP_MODEL_PYTHON || defaultPythonPath,
    scriptPath: process.env.DEEP_MODEL_SCRIPT || null,
    modelPath: process.env.DEEP_MODEL_PATH || null,
    device: process.env.DEEP_MODEL_DEVICE || "cuda",
    timeoutMs: 30000,
    classifyThreshold: 0.5,
    fusionWeight: 0.45,
    evidenceThreshold: 0.65,
    modelByLanguage: Object.keys(modelByLanguage).length ? modelByLanguage : null,
    languageDetector: {
      enabled: false,
      scriptPath: process.env.DEEP_LANG_MODEL_SCRIPT || null,
      modelPath: process.env.DEEP_LANG_MODEL_PATH || null,
      device: process.env.DEEP_LANG_MODEL_DEVICE || null,
    },
    languageGate: {
      enabled: true,
      mode: "warn", // warn | soft | block
      minConfidence: 0.7,
      softPenalty: 0.2,
    },
  },
};

module.exports = config;
