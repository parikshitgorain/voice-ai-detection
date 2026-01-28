const { loadAudioFromBase64 } = require("./audio_loader");
const { extractFeatures } = require("./feature_extractor");
const { buildFeatureObject } = require("./audio_types/schemas");
const { decodeMp3 } = require("./audio_loader/mp3_decoder");
const { detectSpeechPresence } = require("./vad");

const defaultDeps = {
  mp3Decoder: decodeMp3,
};

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

  windows.push({ label: "start", pcm: sliceWindow(pcm, 0, windowSamples) });
  windows.push({ label: "middle", pcm: sliceWindow(pcm, middleStart, windowSamples) });
  windows.push({ label: "end", pcm: sliceWindow(pcm, lastStart, windowSamples) });

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

const buildFeatureSet = async (audioBase64, deps = {}, config) => {
  const mergedDeps = { ...defaultDeps, ...deps };
  let loadResult = null;
  let windows = [];
  const windowResults = [];

  try {
    loadResult = await loadAudioFromBase64(
      audioBase64,
      mergedDeps.mp3Decoder,
      config?.limits?.maxFileBytes
    );
    if (!loadResult.ok) {
      const statusCode =
        loadResult.error?.code === "DECODER_MISSING" ? 500 : 400;
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
      const featureResult = await extractFeatures(window.pcm, loadResult.sampleRate, mergedDeps);
      if (!featureResult.ok) {
        const statusCode =
          featureResult.error?.code === "FRAME_ERROR" ? 400 : 500;
        return { ...featureResult, statusCode };
      }
      windowResults.push(featureResult.features);
    }

    const aggregated = aggregateObjects(windowResults);
    const drift = computeDrift(windowResults);
    const agreementScore = Math.max(0, Math.min(1, 1 - drift.overall));
    const spectralMean = aggregated?.spectralFlatness?.mean ?? 0;
    const noiseScore = Math.max(0, Math.min(1, (spectralMean - 0.3) / 0.4));

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
