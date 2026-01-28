const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const applyDurationPenalty = (probability, durationSeconds) => {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return probability * 0.5;

  if (durationSeconds < 10) return probability * 0.65;
  if (durationSeconds < 20) return probability * 0.8;
  return probability;
};

const applyAgreementPenalty = (probability, agreementScore) => {
  if (!Number.isFinite(agreementScore)) return probability * 0.9;
  if (agreementScore < 0.5) return probability * 0.75;
  if (agreementScore < 0.7) return probability * 0.9;
  return probability;
};

const applyNoisePenalty = (probability, noiseScore) => {
  if (!Number.isFinite(noiseScore)) return probability;
  if (noiseScore > 0.7) return probability * 0.8;
  if (noiseScore > 0.5) return probability * 0.9;
  return probability;
};

const calibrateConfidence = (probability, durationSeconds, agreementScore, noiseScore) => {
  const penalized = applyNoisePenalty(
    applyAgreementPenalty(applyDurationPenalty(probability, durationSeconds), agreementScore),
    noiseScore
  );

  const softened = 0.1 + penalized * 0.8;
  return clamp(softened, 0.05, 0.95);
};

module.exports = {
  calibrateConfidence,
};
