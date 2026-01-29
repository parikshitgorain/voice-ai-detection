const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const clamp01 = (value) => clamp(value, 0, 1);

const LANGUAGE_PROFILES = {
  English: {
    zcr: { target: 0.12, scale: 0.08 },
    spectral: { target: 0.08, scale: 0.05 },
    stress: { target: 0.38, scale: 0.22 },
    pitchStd: { target: 45, scale: 25 },
    emphasis: { target: 0.45, scale: 0.25 },
  },
  Hindi: {
    zcr: { target: 0.09, scale: 0.08 },
    spectral: { target: 0.06, scale: 0.05 },
    stress: { target: 0.6, scale: 0.24 },
    pitchStd: { target: 38, scale: 22 },
    emphasis: { target: 0.55, scale: 0.25 },
  },
  Malayalam: {
    zcr: { target: 0.08, scale: 0.08 },
    spectral: { target: 0.05, scale: 0.05 },
    stress: { target: 0.62, scale: 0.24 },
    pitchStd: { target: 36, scale: 22 },
    emphasis: { target: 0.58, scale: 0.25 },
  },
  Tamil: {
    zcr: { target: 0.075, scale: 0.08 },
    spectral: { target: 0.05, scale: 0.05 },
    stress: { target: 0.64, scale: 0.24 },
    pitchStd: { target: 36, scale: 22 },
    emphasis: { target: 0.6, scale: 0.25 },
  },
  Telugu: {
    zcr: { target: 0.08, scale: 0.08 },
    spectral: { target: 0.055, scale: 0.05 },
    stress: { target: 0.63, scale: 0.24 },
    pitchStd: { target: 37, scale: 22 },
    emphasis: { target: 0.58, scale: 0.25 },
  },
};

const scoreTarget = (value, target, scale) => {
  if (!Number.isFinite(value)) return null;
  const diff = Math.abs(value - target);
  return clamp01(1 - diff / scale);
};

const meanScore = (scores) => {
  const finite = scores.filter(Number.isFinite);
  if (finite.length < 3) return null;
  const total = finite.reduce((sum, value) => sum + value, 0);
  return total / finite.length;
};

const estimateLanguage = (features) => {
  const zcr = Number.isFinite(features?.zcr?.mean) ? features.zcr.mean : null;
  const spectral = Number.isFinite(features?.spectralFlatness?.mean)
    ? features.spectralFlatness.mean
    : null;
  const stress = Number.isFinite(features?.prosodyPlanning?.stressSymmetry)
    ? features.prosodyPlanning.stressSymmetry
    : null;
  const pitchStd = Number.isFinite(features?.pitch?.std) ? features.pitch.std : null;
  const emphasis = Number.isFinite(features?.prosodyPlanning?.emphasisRegularity)
    ? features.prosodyPlanning.emphasisRegularity
    : null;

  const scored = [];
  for (const [language, profile] of Object.entries(LANGUAGE_PROFILES)) {
    const score = meanScore([
      scoreTarget(zcr, profile.zcr.target, profile.zcr.scale),
      scoreTarget(spectral, profile.spectral.target, profile.spectral.scale),
      scoreTarget(stress, profile.stress.target, profile.stress.scale),
      scoreTarget(pitchStd, profile.pitchStd.target, profile.pitchStd.scale),
      scoreTarget(emphasis, profile.emphasis.target, profile.emphasis.scale),
    ]);
    if (Number.isFinite(score)) scored.push({ language, score });
  }

  if (!scored.length) return { language: null, confidence: 0 };
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const second = scored[1] ?? { score: 0 };
  const separation = clamp01((best.score - second.score) / 0.25);
  const confidence = clamp01(best.score * 0.6 + separation * 0.4);
  if (confidence < 0.6) return { language: null, confidence };

  return {
    language: best.language,
    confidence,
  };
};

const estimateLanguageDistribution = (features) => {
  const zcr = Number.isFinite(features?.zcr?.mean) ? features.zcr.mean : null;
  const spectral = Number.isFinite(features?.spectralFlatness?.mean)
    ? features.spectralFlatness.mean
    : null;
  const stress = Number.isFinite(features?.prosodyPlanning?.stressSymmetry)
    ? features.prosodyPlanning.stressSymmetry
    : null;
  const pitchStd = Number.isFinite(features?.pitch?.std) ? features.pitch.std : null;
  const emphasis = Number.isFinite(features?.prosodyPlanning?.emphasisRegularity)
    ? features.prosodyPlanning.emphasisRegularity
    : null;

  const scored = [];
  for (const [language, profile] of Object.entries(LANGUAGE_PROFILES)) {
    const score = meanScore([
      scoreTarget(zcr, profile.zcr.target, profile.zcr.scale),
      scoreTarget(spectral, profile.spectral.target, profile.spectral.scale),
      scoreTarget(stress, profile.stress.target, profile.stress.scale),
      scoreTarget(pitchStd, profile.pitchStd.target, profile.pitchStd.scale),
      scoreTarget(emphasis, profile.emphasis.target, profile.emphasis.scale),
    ]);
    if (Number.isFinite(score)) scored.push({ language, score });
  }

  if (!scored.length) return { distribution: null };
  const maxScore = Math.max(...scored.map((item) => item.score));
  const expScores = scored.map((item) => Math.exp(item.score - maxScore));
  const sum = expScores.reduce((a, b) => a + b, 0);
  const distribution = {};
  for (let i = 0; i < scored.length; i += 1) {
    distribution[scored[i].language] = clamp01(expScores[i] / sum);
  }
  return { distribution };
};

const computeLanguageWarning = (
  features,
  selectedLanguage,
  detectedLanguage = null,
  detectedConfidence = null
) => {
  if (!selectedLanguage) return { languageWarning: false };

  if (
    detectedLanguage &&
    detectedLanguage !== selectedLanguage &&
    Number.isFinite(detectedConfidence) &&
    detectedConfidence >= 0.7
  ) {
    return {
      languageWarning: true,
      reason: `Selected language "${selectedLanguage}" may be incorrect. Detected "${detectedLanguage}".`,
    };
  }

  const estimate = estimateLanguage(features);
  if (!estimate.language || estimate.confidence < 0.7) {
    return { languageWarning: false };
  }
  if (estimate.language !== selectedLanguage) {
    return {
      languageWarning: true,
      reason: `Selected language "${selectedLanguage}" may be incorrect. Detected "${estimate.language}".`,
    };
  }
  return { languageWarning: false };
};

module.exports = {
  computeLanguageWarning,
  estimateLanguageDistribution,
};
