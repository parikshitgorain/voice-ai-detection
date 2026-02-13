const { inferDeepScore } = require("./deep_model");

const buildDeepExplanation = (score) => {
  if (!Number.isFinite(score)) {
    return "Deep model output unavailable.";
  }
  const pct = (score * 100).toFixed(1);
  if (score >= 0.5) {
    return `Deep model estimated an AI probability of ${pct}%.`;
  }
  return `Deep model estimated an AI probability of ${pct}%, leaning human.`;
};

const detectVoiceSource = async (payload, config) => {
  try {
    const deepResult = await inferDeepScore(
      payload.audioBase64,
      config,
      payload.language,
      payload.audioFormat
    );
    
    if (!deepResult.ok || !Number.isFinite(deepResult.score)) {
      console.error('Deep model failed:', deepResult.error);
      return {
        ok: false,
        error: deepResult.error || { code: "DEEP_MODEL_FAILED", message: "Deep model failed." },
        statusCode: 500,
      };
    }

  const threshold =
    Number.isFinite(config?.deepModel?.classifyThreshold) ? config.deepModel.classifyThreshold : 0.5;
  const classification = deepResult.score >= threshold ? "AI_GENERATED" : "HUMAN";
  let confidenceScore = deepResult.score;
  const explanation = buildDeepExplanation(deepResult.score);
  const languageGate = config?.deepModel?.languageGate || {};
  const selectedLanguage = payload.language || null;
  const mismatchDetected =
    languageGate.enabled &&
    selectedLanguage &&
    deepResult.detectedLanguage &&
    deepResult.detectedLanguage !== selectedLanguage &&
    Number.isFinite(deepResult.languageConfidence) &&
    deepResult.languageConfidence >= (languageGate.minConfidence ?? 0.7);

  if (mismatchDetected && languageGate.mode === "block") {
    return {
      ok: false,
      statusCode: 422,
      error: {
        code: "LANGUAGE_MISMATCH",
        message: `Selected language "${selectedLanguage}" does not match detected "${deepResult.detectedLanguage}".`,
      },
    };
  }

  if (mismatchDetected && languageGate.mode === "soft") {
    const penalty = Math.max(0, Math.min(1, languageGate.softPenalty ?? 0.2));
    confidenceScore = Math.max(0, confidenceScore - penalty);
  }
  return {
    ok: true,
    data: {
      classification,
      confidenceScore,
      explanation,
      deepScore: deepResult.ok ? deepResult.score : null,
      detectedLanguage: deepResult.ok ? deepResult.detectedLanguage : null,
      languageConfidence: deepResult.ok ? deepResult.languageConfidence : null,
    },
  };
  } catch (err) {
    console.error('Voice detection service error:', err);
    return {
      ok: false,
      error: { code: "SERVICE_ERROR", message: err.message || "Voice detection failed." },
      statusCode: 500,
    };
  }
};

module.exports = {
  detectVoiceSource,
};
