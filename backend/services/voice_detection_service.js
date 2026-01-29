const { buildFeatureSet } = require("./audio_pipeline");
const { classifyFeatures } = require("./classifier");
const { buildExplanation } = require("./explanation");
const { computeLanguageWarning } = require("./language_warning");
const { inferDeepScore } = require("./deep_model");

const detectVoiceSource = async (payload, config) => {
  const featureResult = await buildFeatureSet(payload.audioBase64, {}, config);
  if (!featureResult.ok) {
    return {
      ok: false,
      error: featureResult.error,
      statusCode: featureResult.statusCode,
    };
  }

  const deepResult = await inferDeepScore(payload.audioBase64, config);
  if (deepResult.ok && Number.isFinite(deepResult.score)) {
    featureResult.data.features.deepScore = deepResult.score;
  }

  const classificationResult = classifyFeatures(featureResult.data);
  if (!classificationResult.ok) {
    return {
      ok: false,
      error: classificationResult.error,
      statusCode: 500,
    };
  }

  const explanationResult = buildExplanation({
    classification: classificationResult.data.classification,
    confidenceScore: classificationResult.data.confidenceScore,
    features: featureResult.data.features,
  });
  const languageWarningResult = computeLanguageWarning(
    featureResult.data.features,
    payload.language,
    deepResult.ok ? deepResult.detectedLanguage : null,
    deepResult.ok ? deepResult.languageConfidence : null
  );

  return {
    ok: true,
    data: {
      classification: classificationResult.data.classification,
      confidenceScore: classificationResult.data.confidenceScore,
      explanation: explanationResult.explanation,
      languageWarning: languageWarningResult.languageWarning,
      languageWarningReason: languageWarningResult.reason || null,
      deepScore: deepResult.ok ? deepResult.score : null,
      detectedLanguage: deepResult.ok ? deepResult.detectedLanguage : null,
      languageConfidence: deepResult.ok ? deepResult.languageConfidence : null,
    },
  };
};

module.exports = {
  detectVoiceSource,
};
