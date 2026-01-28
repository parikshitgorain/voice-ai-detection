let VadModulePromise = null;

const SUPPORTED_RATES = new Set([8000, 16000, 32000, 48000]);

const loadVadModule = async () => {
  if (VadModulePromise) return VadModulePromise;
  try {
    // WebRTC VAD compiled to WebAssembly (no native build).
    // eslint-disable-next-line global-require, import/no-extraneous-dependencies
    const VadFactory = require("@ennuicastr/webrtcvad.js");
    VadModulePromise = VadFactory();
    return VadModulePromise;
  } catch (err) {
    return null;
  }
};

const resampleLinear = (pcm, fromRate, toRate) => {
  if (fromRate === toRate) return pcm;
  const ratio = toRate / fromRate;
  const outLength = Math.max(1, Math.floor(pcm.length * ratio));
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i += 1) {
    const srcIndex = i / ratio;
    const left = Math.floor(srcIndex);
    const right = Math.min(left + 1, pcm.length - 1);
    const frac = srcIndex - left;
    out[i] = pcm[left] * (1 - frac) + pcm[right] * frac;
  }
  return out;
};

const floatToInt16 = (pcm) => {
  const out = new Int16Array(pcm.length);
  for (let i = 0; i < pcm.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, pcm[i]));
    out[i] = Math.round(clamped * 32767);
  }
  return out;
};

const detectSpeechPresence = async (pcm, sampleRate, config = {}) => {
  if (!pcm || !pcm.length) {
    return { ok: false, error: { code: "EMPTY_PCM", message: "PCM buffer is empty." } };
  }
  if (!sampleRate || sampleRate <= 0) {
    return { ok: false, error: { code: "INVALID_SAMPLE_RATE", message: "Sample rate is invalid." } };
  }

  const vadModule = await loadVadModule();
  if (!vadModule) {
    return {
      ok: false,
      error: { code: "VAD_UNAVAILABLE", message: "VAD dependency is not installed." },
      statusCode: 500,
    };
  }

  const targetRate = config.targetSampleRate ?? 16000;
  const useRate = SUPPORTED_RATES.has(sampleRate) ? sampleRate : targetRate;
  const resampled = sampleRate === useRate ? pcm : resampleLinear(pcm, sampleRate, useRate);

  const frameMs = config.frameMs ?? 20;
  const frameSize = Math.floor((useRate * frameMs) / 1000);
  if (!frameSize) {
    return {
      ok: false,
      error: { code: "INVALID_VAD_FRAME", message: "Invalid VAD frame size." },
      statusCode: 500,
    };
  }

  const pcmInt16 = floatToInt16(resampled);
  let speechFrames = 0;
  let totalFrames = 0;
  let handle = 0;
  let framePtr = 0;

  try {
    handle = vadModule.Create();
    if (!handle) {
      return {
        ok: false,
        error: { code: "VAD_UNSUPPORTED", message: "Failed to initialize VAD." },
        statusCode: 500,
      };
    }
    vadModule.Init(handle);
    const level = Number.isFinite(config.aggressiveness) ? config.aggressiveness : 2;
    vadModule.set_mode(handle, level);

    const heap = vadModule.HEAPU8 ?? vadModule.heap;
    if (!heap || typeof heap.set !== "function") {
      return {
        ok: false,
        error: { code: "VAD_UNSUPPORTED", message: "VAD heap is unavailable." },
        statusCode: 500,
      };
    }

    const byteLength = frameSize * 2;
    framePtr = vadModule.malloc(byteLength);
    if (!framePtr) {
      return {
        ok: false,
        error: { code: "VAD_UNSUPPORTED", message: "Failed to allocate VAD buffer." },
        statusCode: 500,
      };
    }

    for (let offset = 0; offset + frameSize <= pcmInt16.length; offset += frameSize) {
      const slice = pcmInt16.subarray(offset, offset + frameSize);
      const view = new Uint8Array(slice.buffer, slice.byteOffset, slice.byteLength);
      heap.set(view, framePtr);
      const result = vadModule.Process(handle, useRate, framePtr, frameSize);
      totalFrames += 1;
      if (result === 1) speechFrames += 1;
    }
  } finally {
    if (framePtr) vadModule.free(framePtr);
    if (handle) vadModule.Free(handle);
  }

  const speechRatio = totalFrames ? speechFrames / totalFrames : 0;
  const minRatio = config.minSpeechRatio ?? 0.2;
  const minFrames = config.minSpeechFrames ?? 5;
  const speechDetected = speechFrames >= minFrames && speechRatio >= minRatio;

  return {
    ok: true,
    speechDetected,
    speechFrames,
    totalFrames,
    speechRatio,
    sampleRate: useRate,
  };
};

module.exports = {
  detectSpeechPresence,
};
