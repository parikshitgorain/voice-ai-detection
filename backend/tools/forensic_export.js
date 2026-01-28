const fs = require("fs");
const path = require("path");
const { buildFeatureSet } = require("../services/audio_pipeline");
const { classifyFeatures } = require("../services/classifier");
const { mapSignals } = require("../services/explanation/signal_mapper");
const { estimateLanguageDistribution } = require("../services/language_warning");
const config = require("../config");

const file = process.argv[2];
if (!file) {
  console.error("Missing audio file path.");
  process.exit(1);
}

const readBase64 = (filePath) => fs.readFileSync(filePath).toString("base64");

const clamp01 = (value) => Math.min(Math.max(value, 0), 1);

const groupForSignal = (id) => {
  if (id.startsWith("metadata")) return "metadata";
  if (id.startsWith("prosody_entropy") || id.startsWith("pitch_energy") || id.startsWith("emotion_reset") || id.startsWith("breath_regular") || id.startsWith("sentence_reset")) {
    return "longRange";
  }
  if (id.startsWith("stress_") || id.startsWith("intonation_") || id.startsWith("emphasis_") || id.startsWith("contour_") || id.startsWith("micro_prosody")) {
    return "prosody";
  }
  if (id.startsWith("spectral_") || id.startsWith("phase_") || id.startsWith("hf_decay") || id.startsWith("micro_phase")) {
    return "spectral";
  }
  if (id.startsWith("pmci_") || id.startsWith("breath_coupling") || id.startsWith("compression_") || id.startsWith("phase_coherent")) {
    return "advanced";
  }
  if (id.startsWith("window_")) return "temporal";
  if (id.startsWith("deep_model")) return "deep";
  return "other";
};

const summarizeSignals = (signals) => {
  const aiGroups = {};
  const humanGroups = {};
  const agreement = [];

  for (const signal of signals) {
    if (!signal || !Number.isFinite(signal.strength)) continue;
    const group = groupForSignal(signal.id);
    if (signal.aligns === "AI_GENERATED") {
      aiGroups[group] = aiGroups[group] || [];
      aiGroups[group].push(signal.strength);
    } else if (signal.aligns === "HUMAN") {
      humanGroups[group] = humanGroups[group] || [];
      humanGroups[group].push(signal.strength);
    } else if (signal.aligns === "AMBIGUOUS") {
      agreement.push(signal.strength);
    }
  }

  const mean = (values) => {
    const finite = values.filter(Number.isFinite);
    if (!finite.length) return null;
    return finite.reduce((sum, v) => sum + v, 0) / finite.length;
  };

  const aiEvidence = {};
  for (const [group, values] of Object.entries(aiGroups)) {
    const score = mean(values);
    if (Number.isFinite(score)) aiEvidence[group] = clamp01(score);
  }

  const humanEvidence = {};
  for (const [group, values] of Object.entries(humanGroups)) {
    const score = mean(values);
    if (Number.isFinite(score)) humanEvidence[group] = clamp01(score);
  }

  return {
    aiEvidence,
    humanEvidence,
    ambiguity: mean(agreement),
  };
};

(async () => {
  const audioBase64 = readBase64(file);
  const featureResult = await buildFeatureSet(audioBase64, {}, config);
  if (!featureResult.ok) {
    console.error(JSON.stringify({ ok: false, error: featureResult.error }));
    process.exit(1);
  }

  const classificationResult = classifyFeatures(featureResult.data);
  if (!classificationResult.ok) {
    console.error(JSON.stringify({ ok: false, error: classificationResult.error }));
    process.exit(1);
  }

  const features = featureResult.data.features;
  const signals = mapSignals(features);
  const signalSummary = summarizeSignals(signals);
  const languageDistribution = estimateLanguageDistribution(features).distribution;

  const output = {
    classification: classificationResult.data.classification,
    confidenceScore: classificationResult.data.confidenceScore,
    aiEvidence: signalSummary.aiEvidence,
    humanEvidence: signalSummary.humanEvidence,
    languageLikelihood: languageDistribution,
    temporalConsistency: {
      windowCount: features.windowCount ?? null,
      agreementScore: features.agreementScore ?? null,
      windowDisagreement: features?.governance?.windowDisagreement ?? null,
      longRange: features.longRange ?? null,
    },
    phaseEntropy: {
      phaseEntropyStability: features?.spectralConsistency?.phaseEntropyStability ?? null,
      phaseDeltaStability: features?.spectralConsistency?.phaseDeltaStability ?? null,
      microPhaseStability: features?.spectralConsistency?.microPhaseStability ?? null,
    },
    prosodyMemory: {
      pmciWeakness: features?.advanced?.pmciWeakness ?? null,
      emotionResetScore: features?.longRange?.emotionResetScore ?? null,
      prosodyEntropyStability: features?.longRange?.prosodyEntropyStability ?? null,
    },
    compressionStress: {
      compressionConsistency: features?.advanced?.compressionConsistency ?? null,
      compressionConsistencySpread: features?.advanced?.compressionConsistencySpread ?? null,
    },
    multiSpeaker: {
      likelihood: features?.multiSpeaker?.score ?? null,
      detected: features?.multiSpeaker?.detected ?? false,
    },
    evidenceAgreement: {
      groupDisagreement: features?.governance?.groupDisagreement ?? null,
      lowSignalScore: features?.governance?.lowSignalScore ?? null,
      ambiguity: signalSummary.ambiguity ?? null,
    },
    metadataSummary: features?.metadata ?? null,
  };

  console.log(JSON.stringify(output, null, 2));
})();
