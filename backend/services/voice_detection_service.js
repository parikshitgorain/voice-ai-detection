const { buildFeatureSet } = require("./audio_pipeline");
const { classifyFeatures } = require("./classifier");
const { buildExplanation } = require("./explanation");

const detectVoiceSource = async (payload, config) => {
  const featureResult = await buildFeatureSet(payload.audioBase64, {}, config);
  if (!featureResult.ok) {
    return {
      ok: false,
      error: featureResult.error,
      statusCode: featureResult.statusCode,
    };
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

  return {
    ok: true,
    data: {
      classification: classificationResult.data.classification,
      confidenceScore: classificationResult.data.confidenceScore,
      explanation: explanationResult.explanation,
    },
  };
};

module.exports = {
  detectVoiceSource,
};
