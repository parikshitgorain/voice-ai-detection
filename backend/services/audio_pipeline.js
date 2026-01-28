const { loadAudioFromBase64 } = require("./audio_loader");
const { extractFeatures } = require("./feature_extractor");
const { buildFeatureObject } = require("./audio_types/schemas");
const { decodeMp3, encodePcmToMp3 } = require("./audio_loader/mp3_decoder");
const { detectSpeechPresence } = require("./vad");

const defaultDeps = {
  mp3Decoder: decodeMp3,
  mp3Encoder: encodePcmToMp3,
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const clamp01 = (value) => clamp(value, 0, 1);

const sliceWindow = (pcm, startSample, windowSamples) =>
  pcm.slice(startSample, startSample + windowSamples);

const selectWindows = (pcm, sampleRate, durationSeconds, config) => {
  const windows = [];
  const windowDuration = Math.min(config.windowDurationSeconds, durationSeconds);
  const windowSamples = Math.max(1, Math.floor(windowDuration * sampleRate));

  if (durationSeconds <= config.longAudioThresholdSeconds) {
    windows.push({ label: "full", pcm: pcm.slice(0) });
    return windows;
  }

  const lastStart = Math.max(0, pcm.length - windowSamples);
  const middleStart = Math.max(0, Math.floor(pcm.length / 2 - windowSamples / 2));
  const seen = new Set();

  const addWindow = (label, start) => {
    const clamped = Math.max(0, Math.min(start, lastStart));
    const key = Math.floor(clamped);
    if (seen.has(key)) return;
    seen.add(key);
    windows.push({ label, pcm: sliceWindow(pcm, key, windowSamples) });
  };

  addWindow("start", 0);
  addWindow("middle", middleStart);
  addWindow("end", lastStart);

  if (Array.isArray(config.fixedWindowOffsets)) {
    for (const offset of config.fixedWindowOffsets) {
      if (!Number.isFinite(offset)) continue;
      const normalized = clamp01(offset);
      const start = Math.floor((pcm.length - windowSamples) * normalized);
      addWindow(`offset_${Math.round(normalized * 100)}`, start);
    }
  }

  return windows.slice(0, config.maxWindowCount);
};

const aggregateNumeric = (values) => {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return null;
  const total = finite.reduce((sum, value) => sum + value, 0);
  return total / finite.length;
};

const aggregateObjects = (objects) => {
  if (!objects.length) return null;
  if (typeof objects[0] === "number") return aggregateNumeric(objects);
  if (Array.isArray(objects[0])) {
    const length = objects[0].length;
    const result = [];
    for (let i = 0; i < length; i += 1) {
      const slice = objects.map((item) => item[i]).filter(Number.isFinite);
      result.push(aggregateNumeric(slice));
    }
    return result;
  }

  const result = {};
  for (const key of Object.keys(objects[0])) {
    const slice = objects
      .map((item) => item[key])
      .filter((value) => value !== undefined && value !== null);
    result[key] = aggregateObjects(slice);
  }
  return result;
};

const stdDev = (values, avg) => {
  const finite = values.filter(Number.isFinite);
  if (finite.length < 2) return null;
  const mean = avg ?? aggregateNumeric(finite);
  const variance =
    finite.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / finite.length;
  return Math.sqrt(variance);
};

const relativeStd = (values) => {
  const mean = aggregateNumeric(values);
  if (!Number.isFinite(mean) || Math.abs(mean) < 1e-6) return null;
  const std = stdDev(values, mean);
  if (!Number.isFinite(std)) return null;
  return std / Math.abs(mean);
};

const stabilityFromRelativeStd = (values, scale) => {
  const rel = relativeStd(values);
  if (!Number.isFinite(rel)) return null;
  return clamp01(1 - rel / scale);
};

const stabilityFromStd = (values, scale) => {
  const std = stdDev(values);
  if (!Number.isFinite(std)) return null;
  return clamp01(1 - std / scale);
};

const computeDrift = (windowFeatures) => {
  if (windowFeatures.length < 2) {
    return {
      pitch: 0,
      rms: 0,
      spectralFlatness: 0,
      overall: 0,
    };
  }

  const pick = (featurePath) =>
    windowFeatures.map((features) => {
      const parts = featurePath.split(".");
      return parts.reduce((acc, key) => (acc ? acc[key] : null), features);
    });

  const calcDrift = (values) => {
    const finite = values.filter(Number.isFinite);
    if (finite.length < 2) return 0;
    const mean = aggregateNumeric(finite);
    const variance =
      finite.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / finite.length;
    const std = Math.sqrt(variance);
    const denom = Math.max(Math.abs(mean), 1e-4);
    return Math.min(1, std / denom);
  };

  const pitchDrift = calcDrift(pick("pitch.mean"));
  const rmsDrift = calcDrift(pick("rms.mean"));
  const spectralDrift = calcDrift(pick("spectralFlatness.mean"));

  const overall = Math.min(1, (pitchDrift + rmsDrift + spectralDrift) / 3);
  return {
    pitch: pitchDrift,
    rms: rmsDrift,
    spectralFlatness: spectralDrift,
    overall,
  };
};

const computeWindowDisagreement = (windowFeatures) => {
  if (windowFeatures.length < 2) return 0;
  const featurePaths = [
    "rms.mean",
    "rms.std",
    "pitch.mean",
    "pitch.std",
    "spectralFlatness.mean",
    "mfcc.mean.0",
    "zcr.mean",
  ];

  const scores = [];
  for (const path of featurePaths) {
    const values = windowFeatures
      .map((features) => path.split(".").reduce((acc, key) => (acc ? acc[key] : null), features))
      .filter(Number.isFinite);
    if (values.length < 2) continue;
    const rel = relativeStd(values);
    if (Number.isFinite(rel)) {
      scores.push(clamp01(rel / 0.6));
    }
  }

  if (!scores.length) return 0;
  return clamp01(scores.reduce((sum, value) => sum + value, 0) / scores.length);
};

const getPathValue = (obj, path) => {
  return path.split(".").reduce((acc, key) => (acc ? acc[key] : undefined), obj);
};

const normalizeDifference = (a, b) => {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.abs(a - b) / (Math.abs(a) + Math.abs(b) + 1e-6);
};

const analyzeMetadata = (metadata, actualDuration, actualSampleRate) => {
  if (!metadata) {
    return {
      suspicionScore: null,
      tagCompleteness: null,
      bitrateConsistency: null,
      durationConsistency: null,
      sampleRateCommon: null,
      formatMissing: null,
      codecMissing: null,
      codecTagMissing: null,
      frameRateMissing: null,
    };
  }

  const encoderValue = typeof metadata.encoder === "string" ? metadata.encoder.trim() : "";
  const encoderMissing = !encoderValue || encoderValue.toLowerCase() === "unknown";
  const tagCount = Number.isFinite(metadata.tagCount) ? metadata.tagCount : 0;
  const tagCompleteness = clamp01(tagCount / 8);
  const tagSparseScore = clamp01(1 - tagCompleteness);

  let bitrateConsistency = null;
  let bitrateMismatchScore = 0;
  if (
    Number.isFinite(metadata.bitRate) &&
    Number.isFinite(metadata.duration) &&
    metadata.duration > 0 &&
    Number.isFinite(metadata.size)
  ) {
    const expected = (metadata.size * 8) / metadata.duration;
    const diff = Math.abs(expected - metadata.bitRate) / Math.max(metadata.bitRate, 1);
    bitrateConsistency = clamp01(1 - diff / 0.12);
    bitrateMismatchScore = clamp01(1 - bitrateConsistency);
  }

  let durationConsistency = null;
  let durationMismatchScore = 0;
  if (Number.isFinite(metadata.duration) && Number.isFinite(actualDuration) && actualDuration > 0) {
    const diff = Math.abs(metadata.duration - actualDuration) / actualDuration;
    durationConsistency = clamp01(1 - diff / 0.03);
    durationMismatchScore = clamp01(1 - durationConsistency);
  }

  const commonRates = new Set([8000, 16000, 22050, 24000, 32000, 44100, 48000]);
  const sampleRateToCheck = Number.isFinite(actualSampleRate)
    ? actualSampleRate
    : metadata.sampleRate;
  const sampleRateCommon = commonRates.has(sampleRateToCheck) ? 1 : 0;
  const sampleRateUncommon = clamp01(1 - sampleRateCommon);

  const formatName = metadata.formatName ? String(metadata.formatName).toLowerCase() : "";
  const codecName = metadata.codecName ? String(metadata.codecName).toLowerCase() : "";
  const formatMissing = formatName ? 0 : 1;
  const codecMissing = codecName ? 0 : 1;
  const codecTagMissing = metadata.codecTag ? 0 : 1;
  const frameRateMissing = Number.isFinite(metadata.avgFrameRate) ? 0 : 1;

  const suspicionScore = clamp01(
    0.2 * (encoderMissing ? 1 : 0) +
      0.15 * tagSparseScore +
      0.15 * bitrateMismatchScore +
      0.15 * durationMismatchScore +
      0.1 * formatMissing +
      0.1 * codecMissing +
      0.05 * codecTagMissing +
      0.05 * sampleRateUncommon +
      0.05 * frameRateMissing
  );

  return {
    suspicionScore,
    tagCompleteness,
    bitrateConsistency,
    durationConsistency,
    sampleRateCommon,
    formatMissing,
    codecMissing,
    codecTagMissing,
    frameRateMissing,
  };
};

const computeLongRangeSignals = (windowFeatures, windowAnalyses) => {
  const pitchStd = windowFeatures.map((item) => item?.pitch?.std).filter(Number.isFinite);
  const rmsStd = windowFeatures.map((item) => item?.rms?.std).filter(Number.isFinite);
  const spectralStd = windowFeatures
    .map((item) => item?.spectralFlatness?.std)
    .filter(Number.isFinite);

  const varianceStabilityScores = [
    stabilityFromRelativeStd(pitchStd, 0.6),
    stabilityFromRelativeStd(rmsStd, 0.6),
    stabilityFromRelativeStd(spectralStd, 0.6),
  ].filter(Number.isFinite);
  const varianceStability = aggregateNumeric(varianceStabilityScores);

  const pitchEntropy = windowAnalyses.map((item) => item?.pitchEntropy).filter(Number.isFinite);
  const energyEntropy = windowAnalyses.map((item) => item?.energyEntropy).filter(Number.isFinite);
  const entropyStabilityScores = [
    stabilityFromStd(pitchEntropy, 0.2),
    stabilityFromStd(energyEntropy, 0.2),
  ].filter(Number.isFinite);
  const prosodyEntropyStability = aggregateNumeric(entropyStabilityScores);

  const pitchEnergyCoupling = windowAnalyses
    .map((item) => item?.pitchEnergyCorrelation)
    .filter(Number.isFinite);
  const pitchEnergyCouplingStability = stabilityFromStd(pitchEnergyCoupling, 0.4);

  const sentenceResetSharpness = aggregateNumeric(
    windowAnalyses.map((item) => item?.sentenceResetSharpness).filter(Number.isFinite)
  );

  const breathPeriodicityRegularity = aggregateNumeric(
    windowAnalyses.map((item) => item?.breath?.intervalRegularity).filter(Number.isFinite)
  );

  const emotionResetScore = (() => {
    if (windowFeatures.length < 2) return null;
    const diffs = [];
    for (let i = 1; i < windowFeatures.length; i += 1) {
      const prev = windowFeatures[i - 1];
      const curr = windowFeatures[i];
      const scores = [];
      if (Number.isFinite(prev?.pitch?.mean) && Number.isFinite(curr?.pitch?.mean)) {
        scores.push(Math.abs(curr.pitch.mean - prev.pitch.mean) / 80);
      }
      if (Number.isFinite(prev?.rms?.mean) && Number.isFinite(curr?.rms?.mean)) {
        scores.push(Math.abs(curr.rms.mean - prev.rms.mean) / 0.05);
      }
      if (scores.length) {
        diffs.push(clamp01(scores.reduce((sum, value) => sum + value, 0) / scores.length));
      }
    }
    return aggregateNumeric(diffs);
  })();

  return {
    varianceStability,
    prosodyEntropyStability,
    pitchEnergyCouplingStability,
    emotionResetScore,
    breathPeriodicityRegularity,
    sentenceResetSharpness,
  };
};

const computeProsodyPlanningSignals = (windowAnalyses) => {
  const stressSymmetry = aggregateNumeric(
    windowAnalyses.map((item) => item?.stressSymmetry).filter(Number.isFinite)
  );
  const emphasisRegularity = aggregateNumeric(
    windowAnalyses.map((item) => item?.emphasisRegularity).filter(Number.isFinite)
  );
  const intonationSmoothness = aggregateNumeric(
    windowAnalyses.map((item) => item?.intonationSmoothness).filter(Number.isFinite)
  );
  const contourPredictability = (() => {
    const slopes = windowAnalyses.map((item) => item?.contourSlope).filter(Number.isFinite);
    return stabilityFromStd(slopes, 15);
  })();

  return {
    stressSymmetry,
    intonationSmoothness,
    emphasisRegularity,
    contourPredictability,
  };
};

const computeSpectralConsistency = (windowAnalyses) => {
  const hfDecayRegularity = aggregateNumeric(
    windowAnalyses
      .map((item) => item?.spectral?.hfDecayRegularity)
      .filter(Number.isFinite)
  );
  const spectralWobbleStability = aggregateNumeric(
    windowAnalyses
      .map((item) => item?.spectral?.spectralWobbleStability)
      .filter(Number.isFinite)
  );
  const microPhaseStability = aggregateNumeric(
    windowAnalyses
      .map((item) => item?.spectral?.microPhaseStability)
      .filter(Number.isFinite)
  );
  const phaseDeltaStd = aggregateNumeric(
    windowAnalyses
      .map((item) => item?.spectral?.phaseDeltaStd)
      .filter(Number.isFinite)
  );
  const phaseDeltaStability = Number.isFinite(phaseDeltaStd)
    ? clamp01(1 - phaseDeltaStd)
    : null;
  const fluxStability = aggregateNumeric(
    windowAnalyses
      .map((item) => item?.spectral?.fluxStability)
      .filter(Number.isFinite)
  );
  const rolloffStability = aggregateNumeric(
    windowAnalyses
      .map((item) => item?.spectral?.rolloffStability)
      .filter(Number.isFinite)
  );

  return {
    hfDecayRegularity,
    spectralWobbleStability,
    microPhaseStability,
    phaseDeltaStability,
    fluxStability,
    rolloffStability,
  };
};

const computePmciWeakness = (windowFeatures) => {
  if (windowFeatures.length < 2) return null;

  const vectorForWindow = (window) => {
    const values = [
      window?.pitch?.mean,
      window?.pitch?.range,
      window?.rms?.mean,
      window?.rms?.std,
    ];
    if (!values.every(Number.isFinite)) return null;
    return values;
  };

  const cosineSimilarity = (a, b) => {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i += 1) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA === 0 || normB === 0) return null;
    return dot / Math.sqrt(normA * normB);
  };

  const sims = [];
  for (let i = 1; i < windowFeatures.length; i += 1) {
    const prev = vectorForWindow(windowFeatures[i - 1]);
    const curr = vectorForWindow(windowFeatures[i]);
    if (!prev || !curr) continue;
    const sim = cosineSimilarity(prev, curr);
    if (Number.isFinite(sim)) sims.push(sim);
  }
  if (!sims.length) return null;
  const avgSim = aggregateNumeric(sims);
  const normalized = clamp01((avgSim + 1) / 2);
  return clamp01(1 - normalized);
};

const computeBreathCouplingAnomaly = (windowAnalyses, baseline) => {
  let weightedSum = 0;
  let weightTotal = 0;

  const meanBaseline = baseline?.couplingMean ?? 0.55;
  const stdBaseline = baseline?.couplingStd ?? 0.18;

  for (const analysis of windowAnalyses) {
    const breath = analysis?.breath;
    if (!breath || breath.eventCount < 2 || !Number.isFinite(breath.couplingScore)) continue;
    const couplingNormalized = clamp01((breath.couplingScore + 1) / 2);
    const normalized = clamp01(
      (couplingNormalized - meanBaseline) / (stdBaseline * 2) + 0.5
    );
    const anomaly = 1 - normalized;
    const weight = Math.min(5, breath.eventCount);
    weightedSum += anomaly * weight;
    weightTotal += weight;
  }

  if (!weightTotal) return null;
  return clamp01(weightedSum / weightTotal);
};

const computePhaseCoherence = (windowAnalyses) => {
  const entropies = windowAnalyses
    .map((item) => item?.spectral?.phaseEntropy)
    .filter(Number.isFinite);
  if (!entropies.length) return null;
  const avgEntropy = aggregateNumeric(entropies);
  return clamp01(1 - avgEntropy);
};

const computeSignalGovernance = (metadata, longRange, prosodyPlanning, spectralConsistency, advanced, stability, windowDisagreement) => {
  const scores = [];
  const addScore = (value) => {
    if (Number.isFinite(value)) scores.push(clamp01(value));
  };

  const meanOf = (values) => {
    const finite = values.filter(Number.isFinite);
    return finite.length ? aggregateNumeric(finite) : null;
  };

  addScore(metadata?.suspicionScore);
  addScore(meanOf([
    longRange?.varianceStability,
    longRange?.prosodyEntropyStability,
    longRange?.pitchEnergyCouplingStability,
    longRange?.emotionResetScore,
    longRange?.breathPeriodicityRegularity,
    longRange?.sentenceResetSharpness,
  ]));
  addScore(meanOf([
    prosodyPlanning?.stressSymmetry,
    prosodyPlanning?.intonationSmoothness,
    prosodyPlanning?.emphasisRegularity,
    prosodyPlanning?.contourPredictability,
  ]));
  addScore(meanOf([
    spectralConsistency?.hfDecayRegularity,
    spectralConsistency?.spectralWobbleStability,
    spectralConsistency?.microPhaseStability,
    spectralConsistency?.phaseDeltaStability,
    spectralConsistency?.fluxStability,
    spectralConsistency?.rolloffStability,
  ]));
  addScore(meanOf([
    advanced?.pmciWeakness,
    advanced?.breathCouplingAnomaly,
    advanced?.phaseCoherence,
    advanced?.compressionConsistency,
  ]));
  addScore(stability?.overall);

  let groupDisagreement = 0;
  if (scores.length >= 2) {
    const avg = aggregateNumeric(scores);
    const variance = scores.reduce((sum, value) => sum + Math.pow(value - avg, 2), 0) / scores.length;
    const std = Math.sqrt(variance);
    groupDisagreement = clamp01(std / 0.25);
  }

  const signalPool = [
    metadata?.suspicionScore,
    longRange?.varianceStability,
    longRange?.prosodyEntropyStability,
    longRange?.pitchEnergyCouplingStability,
    longRange?.emotionResetScore,
    longRange?.breathPeriodicityRegularity,
    longRange?.sentenceResetSharpness,
    prosodyPlanning?.stressSymmetry,
    prosodyPlanning?.intonationSmoothness,
    prosodyPlanning?.emphasisRegularity,
    prosodyPlanning?.contourPredictability,
    spectralConsistency?.hfDecayRegularity,
    spectralConsistency?.spectralWobbleStability,
    spectralConsistency?.microPhaseStability,
    spectralConsistency?.phaseDeltaStability,
    spectralConsistency?.fluxStability,
    spectralConsistency?.rolloffStability,
    advanced?.pmciWeakness,
    advanced?.breathCouplingAnomaly,
    advanced?.phaseCoherence,
    advanced?.compressionConsistency,
    stability?.overall,
  ];

  const expectedCount = signalPool.length;
  const finiteCount = signalPool.filter(Number.isFinite).length;
  const coverage = expectedCount ? finiteCount / expectedCount : 0;
  const lowSignalScore = clamp01(1 - coverage);

  return {
    groupDisagreement,
    lowSignalScore,
    windowDisagreement: Number.isFinite(windowDisagreement) ? windowDisagreement : 0,
  };
};

const extractAggregatedFeaturesForPcm = async (pcm, sampleRate, config) => {
  const duration = pcm.length / sampleRate;
  const windows = selectWindows(pcm, sampleRate, duration, config.limits);
  const results = [];

  for (const window of windows) {
    const featureResult = await extractFeatures(window.pcm, sampleRate, {
      config: config.analysis,
      analysisMode: "lite",
    });
    if (!featureResult.ok) return null;
    results.push(featureResult.features);
  }

  return aggregateObjects(results);
};

const computeCompressionConsistency = (originalFeatures, compressedFeatures) => {
  if (!originalFeatures || !compressedFeatures) return null;
  const comparePaths = [
    "rms.mean",
    "rms.std",
    "pitch.mean",
    "pitch.std",
    "spectralFlatness.mean",
    "spectralFlatness.std",
    "zcr.mean",
    "zcr.std",
    "mfcc.mean.0",
    "mfcc.mean.1",
    "mfcc.std.0",
  ];

  const diffs = [];
  for (const path of comparePaths) {
    const originalValue = getPathValue(originalFeatures, path);
    const compressedValue = getPathValue(compressedFeatures, path);
    const diff = normalizeDifference(originalValue, compressedValue);
    if (Number.isFinite(diff)) diffs.push(diff);
  }

  const drift = aggregateNumeric(diffs);
  if (!Number.isFinite(drift)) return null;
  return clamp01(1 - drift / 0.35);
};

const buildFeatureSet = async (audioBase64, deps = {}, config) => {
  const mergedDeps = { ...defaultDeps, ...deps };
  let loadResult = null;
  let windows = [];
  const windowResults = [];
  const windowAnalyses = [];

  try {
    loadResult = await loadAudioFromBase64(
      audioBase64,
      mergedDeps.mp3Decoder,
      config?.limits?.maxFileBytes
    );
    if (!loadResult.ok) {
      const statusCode = loadResult.error?.code === "DECODER_MISSING" ? 500 : 400;
      return { ...loadResult, statusCode };
    }

    if (loadResult.duration < config.limits.minDurationSeconds) {
      return {
        ok: false,
        error: { code: "DURATION_TOO_SHORT", message: "Audio duration is under 10 seconds." },
        statusCode: 400,
      };
    }

    if (loadResult.duration > config.limits.maxDurationSeconds) {
      return {
        ok: false,
        error: { code: "DURATION_TOO_LONG", message: "Audio exceeds 5 minute limit." },
        statusCode: 400,
      };
    }

    if (config.vad?.enabled) {
      const vadResult = await detectSpeechPresence(
        loadResult.pcm,
        loadResult.sampleRate,
        config.vad
      );
      if (!vadResult.ok) {
        return {
          ok: false,
          error: vadResult.error,
          statusCode: vadResult.statusCode ?? 500,
        };
      }
      if (!vadResult.speechDetected) {
        return {
          ok: false,
          error: {
            code: "NON_SPEECH_AUDIO",
            message: "Audio does not contain sufficient speech for analysis.",
          },
          statusCode: 400,
        };
      }
    }

    windows = selectWindows(
      loadResult.pcm,
      loadResult.sampleRate,
      loadResult.duration,
      config.limits
    );

    for (const window of windows) {
      const featureResult = await extractFeatures(window.pcm, loadResult.sampleRate, {
        config: config.analysis,
        analysisMode: "full",
      });
      if (!featureResult.ok) {
        const statusCode = featureResult.error?.code === "FRAME_ERROR" ? 400 : 500;
        return { ...featureResult, statusCode };
      }
      windowResults.push(featureResult.features);
      if (featureResult.analysis) {
        windowAnalyses.push(featureResult.analysis);
      }
    }

    const aggregated = aggregateObjects(windowResults);
    const drift = computeDrift(windowResults);
    const windowDisagreement = computeWindowDisagreement(windowResults);
    const agreementScore = Math.max(0, Math.min(1, 1 - drift.overall));
    const spectralMean = aggregated?.spectralFlatness?.mean ?? 0;
    const noiseScore = Math.max(0, Math.min(1, (spectralMean - 0.3) / 0.4));

    const metadata = analyzeMetadata(loadResult.metadata, loadResult.duration, loadResult.sampleRate);
    const longRange = computeLongRangeSignals(windowResults, windowAnalyses);
    const prosodyPlanning = computeProsodyPlanningSignals(windowAnalyses);
    const spectralConsistency = computeSpectralConsistency(windowAnalyses);

    const pmciWeakness = computePmciWeakness(windowResults);
    const breathCouplingAnomaly = computeBreathCouplingAnomaly(
      windowAnalyses,
      config.analysis?.breath?.baseline
    );
    const phaseCoherence = computePhaseCoherence(windowAnalyses);

    let compressionConsistency = null;
    let compressionConsistencySpread = null;
    if (config.compressionTest?.enabled && mergedDeps.mp3Encoder && mergedDeps.mp3Decoder) {
      try {
        const bitrates = Array.isArray(config.compressionTest.bitratesKbps)
          ? config.compressionTest.bitratesKbps
          : [config.compressionTest.bitrateKbps];
        const scores = [];
        for (const bitrate of bitrates) {
          if (!Number.isFinite(bitrate)) continue;
          const compressedBuffer = await mergedDeps.mp3Encoder(
            loadResult.pcm,
            loadResult.sampleRate,
            bitrate
          );
          const decoded = await mergedDeps.mp3Decoder(compressedBuffer);
          if (decoded && decoded.pcm && decoded.sampleRate) {
            const compressedAggregated = await extractAggregatedFeaturesForPcm(
              decoded.pcm,
              decoded.sampleRate,
              config
            );
            const score = computeCompressionConsistency(aggregated, compressedAggregated);
            if (Number.isFinite(score)) scores.push(score);
            decoded.pcm = null;
          }
        }
        compressionConsistency = aggregateNumeric(scores);
        if (scores.length > 1) {
          compressionConsistencySpread = stdDev(scores, compressionConsistency);
        }
      } catch (err) {
        compressionConsistency = null;
      }
    }

    const advanced = {
      pmciWeakness,
      breathCouplingAnomaly,
      phaseCoherence,
      compressionConsistency,
      compressionConsistencySpread,
    };

    const governance = computeSignalGovernance(
      metadata,
      longRange,
      prosodyPlanning,
      spectralConsistency,
      advanced,
      drift,
      windowDisagreement
    );

    return {
      ok: true,
      data: buildFeatureObject({
        duration: loadResult.duration,
        sampleRate: loadResult.sampleRate,
        rms: aggregated.rms,
        pitch: aggregated.pitch,
        mfcc: aggregated.mfcc,
        spectralFlatness: aggregated.spectralFlatness,
        zcr: aggregated.zcr,
        stability: drift,
        windowCount: windows.length,
        agreementScore,
        noiseScore,
        metadata,
        longRange,
        prosodyPlanning,
        spectralConsistency,
        advanced,
        governance,
      }),
      windows: windowResults,
    };
  } finally {
    if (loadResult && loadResult.pcm) {
      loadResult.pcm = null;
    }
    if (Array.isArray(windows)) {
      for (const window of windows) {
        if (window && window.pcm) {
          window.pcm = null;
        }
      }
      windows.length = 0;
    }
  }
};

module.exports = {
  buildFeatureSet,
};
