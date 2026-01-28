const safeNumber = (value) => (Number.isFinite(value) ? value : null);

const scoreLow = (value, low, high) => {
  if (value === null) return 0;
  if (value <= low) return 1;
  if (value >= high) return 0;
  return (high - value) / (high - low);
};

const scoreHigh = (value, low, high) => {
  if (value === null) return 0;
  if (value <= low) return 0;
  if (value >= high) return 1;
  return (value - low) / (high - low);
};

const mean = (values) => {
  if (!Array.isArray(values) || values.length === 0) return null;
  const total = values.reduce((sum, value) => sum + value, 0);
  return total / values.length;
};

const getFeatures = (features) => {
  return {
    rmsMean: safeNumber(features?.rms?.mean),
    rmsStd: safeNumber(features?.rms?.std),
    pitchStd: safeNumber(features?.pitch?.std),
    pitchStability: safeNumber(features?.pitch?.stability),
    spectralMean: safeNumber(features?.spectralFlatness?.mean),
    spectralStd: safeNumber(features?.spectralFlatness?.std),
    mfccStdMean: safeNumber(mean(features?.mfcc?.std)),
    zcrStd: safeNumber(features?.zcr?.std),
    stabilityOverall:
      features?.windowCount && features.windowCount > 1
        ? safeNumber(features?.stability?.overall)
        : null,
  };
};

const mapSignals = (features) => {
  const {
    rmsMean,
    rmsStd,
    pitchStd,
    pitchStability,
    spectralMean,
    spectralStd,
    mfccStdMean,
    zcrStd,
    stabilityOverall,
  } = getFeatures(features);

  const eps = 1e-6;
  const rmsCv = rmsMean ? rmsStd / Math.max(rmsMean, eps) : null;

  // Heuristic bands are for explanation ranking only; tune with data later.
  const signals = [
    {
      id: "pitch_stable",
      aligns: "AI_GENERATED",
      strength: Math.max(scoreLow(pitchStd, 20, 55), scoreHigh(pitchStability, 0.7, 0.9)),
      phrase: "consistent pitch patterns with minimal drift",
    },
    {
      id: "pitch_variable",
      aligns: "HUMAN",
      strength: Math.max(scoreHigh(pitchStd, 35, 80), scoreLow(pitchStability, 0.55, 0.75)),
      phrase: "variable pitch patterns with natural drift",
    },
    {
      id: "spectral_smooth",
      aligns: "AI_GENERATED",
      strength: Math.max(scoreLow(spectralMean, 0.3, 0.55), scoreLow(spectralStd, 0.06, 0.12)),
      phrase: "low spectral noise",
    },
    {
      id: "spectral_varied",
      aligns: "HUMAN",
      strength: Math.max(scoreHigh(spectralMean, 0.45, 0.7), scoreHigh(spectralStd, 0.08, 0.14)),
      phrase: "higher spectral variability",
    },
    {
      id: "energy_stable",
      aligns: "AI_GENERATED",
      strength: scoreLow(rmsCv, 0.35, 0.75),
      phrase: "steady energy profile across frames",
    },
    {
      id: "energy_dynamic",
      aligns: "HUMAN",
      strength: scoreHigh(rmsCv, 0.5, 0.95),
      phrase: "natural energy dynamics across frames",
    },
    {
      id: "mfcc_stable",
      aligns: "AI_GENERATED",
      strength: scoreLow(mfccStdMean, 6, 14),
      phrase: "stable spectral envelope",
    },
    {
      id: "mfcc_varied",
      aligns: "HUMAN",
      strength: scoreHigh(mfccStdMean, 10, 18),
      phrase: "varying spectral envelope",
    },
    {
      id: "zcr_stable",
      aligns: "AI_GENERATED",
      strength: scoreLow(zcrStd, 0.01, 0.03),
      phrase: "consistent voicing transitions",
    },
    {
      id: "zcr_varied",
      aligns: "HUMAN",
      strength: scoreHigh(zcrStd, 0.02, 0.05),
      phrase: "variable voicing transitions",
    },
    {
      id: "window_stable",
      aligns: "AI_GENERATED",
      strength: scoreLow(stabilityOverall, 0.15, 0.35),
      phrase: "highly consistent patterns across windows",
    },
    {
      id: "window_varied",
      aligns: "HUMAN",
      strength: scoreHigh(stabilityOverall, 0.25, 0.5),
      phrase: "natural variability across windows",
    },
  ];

  return signals;
};

module.exports = {
  mapSignals,
};
