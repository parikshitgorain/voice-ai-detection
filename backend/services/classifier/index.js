const { loadModel } = require("./model_loader");
const { toFeatureVector } = require("./feature_adapter");
const { predictProbability } = require("./predictor");
const { calibrateConfidence } = require("./confidence_calibrator");
const config = require("../../config");

const model = loadModel();

const clamp01 = (value) => Math.min(Math.max(value, 0), 1);

const meanOf = (values) => {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
};

const weightedMean = (pairs) => {
  let sum = 0;
  let weightTotal = 0;
  for (const pair of pairs) {
    if (!pair) continue;
    const value = pair.value;
    const weight = pair.weight;
    if (!Number.isFinite(value) || !Number.isFinite(weight)) continue;
    sum += value * weight;
    weightTotal += weight;
  }
  if (!weightTotal) return null;
  return sum / weightTotal;
};

const computeLongRangeCoverage = (features) => {
  const longRange = features?.longRange ?? {};
  const values = [
    longRange.varianceStability,
    longRange.prosodyEntropyStability,
    longRange.pitchEnergyCouplingStability,
    longRange.emotionResetScore,
    longRange.breathPeriodicityRegularity,
    longRange.sentenceResetSharpness,
  ];
  const total = values.length;
  if (!total) return 0;
  const finiteCount = values.filter(Number.isFinite).length;
  return finiteCount / total;
};

const computeFallbackScore = (features) => {
  const spectral = features?.spectralConsistency ?? {};
  const advanced = features?.advanced ?? {};

  const phaseDeltaStability = spectral.phaseDeltaStability;
  const spectralConsistency = meanOf([
    spectral.hfDecayRegularity,
    spectral.spectralWobbleStability,
    spectral.microPhaseStability,
    spectral.fluxStability,
    spectral.rolloffStability,
  ]);
  const compressionConsistency = advanced.compressionConsistency;

  return weightedMean([
    { value: phaseDeltaStability, weight: 1.2 },
    { value: spectralConsistency, weight: 1.0 },
    { value: compressionConsistency, weight: 1.1 },
  ]);
};

const countAiEvidenceGroups = (features) => {
  if (!features) return 0;
  const longRange = features.longRange || {};
  const prosody = features.prosodyPlanning || {};
  const spectral = features.spectralConsistency || {};
  const advanced = features.advanced || {};
  const metadata = features.metadata || {};
  const deepScore = features.deepScore;

  const groupFlags = {
    longRange:
      Number.isFinite(longRange.varianceStability) && longRange.varianceStability > 0.65 ||
      Number.isFinite(longRange.prosodyEntropyStability) && longRange.prosodyEntropyStability > 0.65 ||
      Number.isFinite(longRange.pitchEnergyCouplingStability) && longRange.pitchEnergyCouplingStability > 0.65,
    prosody:
      Number.isFinite(prosody.stressSymmetry) && prosody.stressSymmetry > 0.6 ||
      Number.isFinite(prosody.intonationSmoothness) && prosody.intonationSmoothness > 0.6 ||
      Number.isFinite(prosody.emphasisRegularity) && prosody.emphasisRegularity > 0.6 ||
      Number.isFinite(prosody.microProsodyVariability) && prosody.microProsodyVariability < 0.3,
    spectral:
      Number.isFinite(spectral.phaseDeltaStability) && spectral.phaseDeltaStability > 0.6 ||
      Number.isFinite(spectral.phaseEntropyStability) && spectral.phaseEntropyStability > 0.6 ||
      Number.isFinite(spectral.microPhaseStability) && spectral.microPhaseStability > 0.6 ||
      Number.isFinite(spectral.hfDecayRegularity) && spectral.hfDecayRegularity > 0.6 ||
      Number.isFinite(spectral.spectralWobbleStability) && spectral.spectralWobbleStability > 0.6,
    compression:
      Number.isFinite(advanced.compressionConsistency) && advanced.compressionConsistency > 0.75,
    breath:
      Number.isFinite(advanced.breathCouplingAnomaly) && advanced.breathCouplingAnomaly > 0.6 ||
      Number.isFinite(advanced.breathCouplingStability) && advanced.breathCouplingStability > 0.6,
    phase:
      Number.isFinite(advanced.phaseCoherence) && advanced.phaseCoherence > 0.6,
    planning:
      Number.isFinite(advanced.pmciWeakness) && advanced.pmciWeakness > 0.6,
    metadata:
      Number.isFinite(metadata.suspicionScore) && metadata.suspicionScore > 0.6,
    deep:
      Number.isFinite(deepScore) &&
      deepScore > (config?.deepModel?.evidenceThreshold ?? 0.75),
  };

  return Object.values(groupFlags).filter(Boolean).length;
};

const classifyFeatures = (featureObject) => {
  if (!featureObject || !featureObject.features) {
    return {
      ok: false,
      error: { code: "INVALID_FEATURE_OBJECT", message: "Feature object is missing." },
    };
  }

  const { vector, missing } = toFeatureVector(featureObject, model.featureOrder);
  if (missing.length === model.featureOrder.length) {
    return {
      ok: false,
      error: { code: "MISSING_FEATURES", message: "No usable features found." },
    };
  }

  const rawProbability = predictProbability(vector, model);
  const longRangeCoverage = computeLongRangeCoverage(featureObject.features);
  const fallbackScore = computeFallbackScore(featureObject.features);

  let fusedProbability = rawProbability;
  if (Number.isFinite(fallbackScore) && longRangeCoverage < 0.5) {
    const missingness = 1 - longRangeCoverage;
    const weight = clamp01((missingness - 0.5) / 0.5) * 0.2;
    if (weight > 0) {
      fusedProbability = clamp01(rawProbability * (1 - weight) + fallbackScore * weight);
    }
  }
  const deepScore = featureObject.features?.deepScore;
  if (Number.isFinite(deepScore)) {
    const weight = clamp01(config?.deepModel?.fusionWeight ?? 0.2);
    if (weight > 0) {
      fusedProbability = clamp01(fusedProbability * (1 - weight) + deepScore * weight);
    }
  }

  const aiEvidenceCount = countAiEvidenceGroups(featureObject.features);
  if (fusedProbability >= 0.5 && aiEvidenceCount < 2) {
    fusedProbability = Math.min(fusedProbability, 0.49);
  }

  if (featureObject.features?.multiSpeaker?.detected) {
    fusedProbability = Math.min(fusedProbability, 0.4);
  }

  const agreementScore = featureObject.features?.agreementScore;
  const noiseScore = featureObject.features?.noiseScore;
  const governance = featureObject.features?.governance ?? {};

  const confidenceScore = calibrateConfidence(fusedProbability, {
    durationSeconds: featureObject.duration,
    agreementScore,
    noiseScore,
    groupDisagreement: governance.groupDisagreement,
    windowDisagreement: governance.windowDisagreement,
    lowSignalScore: governance.lowSignalScore,
  });
  const classification = confidenceScore >= 0.5 ? "AI_GENERATED" : "HUMAN";

  return {
    ok: true,
    data: {
      classification,
      confidenceScore,
    },
  };
};

module.exports = {
  classifyFeatures,
};
