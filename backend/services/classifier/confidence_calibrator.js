const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const shrinkToMid = (probability, strength) => {
  const bounded = clamp(strength, 0, 1);
  return 0.5 + (probability - 0.5) * (1 - bounded);
};

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

const applyDisagreementPenalty = (probability, groupDisagreement, windowDisagreement) => {
  let result = probability;
  if (Number.isFinite(groupDisagreement) && groupDisagreement > 0.3) {
    const strength = clamp((groupDisagreement - 0.3) / 0.5, 0, 0.6);
    result = shrinkToMid(result, strength);
  }
  if (Number.isFinite(windowDisagreement) && windowDisagreement > 0.35) {
    const strength = clamp((windowDisagreement - 0.35) / 0.5, 0, 0.6);
    result = shrinkToMid(result, strength);
  }
  return result;
};

const applyLowSignalGovernance = (probability, lowSignalScore) => {
  if (!Number.isFinite(lowSignalScore)) return probability;
  let result = probability;
  if (lowSignalScore > 0.35) {
    const strength = clamp((lowSignalScore - 0.35) / 0.5, 0, 0.6);
    result = shrinkToMid(result, strength);
  }
  return result;
};

const capForLowSignal = (value, lowSignalScore) => {
  if (!Number.isFinite(lowSignalScore)) return value;
  if (lowSignalScore > 0.6) return clamp(value, 0.05, 0.6);
  if (lowSignalScore > 0.4) return clamp(value, 0.05, 0.7);
  return value;
};

const calibrateConfidence = (probability, context = {}) => {
  const durationSeconds = context.durationSeconds;
  const agreementScore = context.agreementScore;
  const noiseScore = context.noiseScore;
  const groupDisagreement = context.groupDisagreement;
  const windowDisagreement = context.windowDisagreement;
  const lowSignalScore = context.lowSignalScore;

  let adjusted = applyNoisePenalty(
    applyAgreementPenalty(applyDurationPenalty(probability, durationSeconds), agreementScore),
    noiseScore
  );

  adjusted = applyDisagreementPenalty(adjusted, groupDisagreement, windowDisagreement);
  adjusted = applyLowSignalGovernance(adjusted, lowSignalScore);

  const softened = 0.1 + adjusted * 0.8;
  const capped = capForLowSignal(softened, lowSignalScore);
  return clamp(capped, 0.05, 0.95);
};

module.exports = {
  calibrateConfidence,
};
