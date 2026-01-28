const { loadModel } = require("./model_loader");
const { toFeatureVector } = require("./feature_adapter");
const { predictProbability } = require("./predictor");
const { calibrateConfidence } = require("./confidence_calibrator");

const model = loadModel();

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
  const agreementScore = featureObject.features?.agreementScore;
  const noiseScore = featureObject.features?.noiseScore;
  const confidenceScore = calibrateConfidence(
    rawProbability,
    featureObject.duration,
    agreementScore,
    noiseScore
  );
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
