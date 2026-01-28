const mean = (values) => {
  if (!values.length) return 0;
  const total = values.reduce((sum, value) => sum + value, 0);
  return total / values.length;
};

const variance = (values, average) => {
  if (!values.length) return 0;
  const meanValue = average ?? mean(values);
  const total = values.reduce((sum, value) => sum + Math.pow(value - meanValue, 2), 0);
  return total / values.length;
};

const stdDev = (values, average) => Math.sqrt(variance(values, average));

const percentile = (values, p) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (sorted.length - 1) * p;
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (idx - lower);
};

const iqr = (values) => {
  if (!values.length) return 0;
  return percentile(values, 0.75) - percentile(values, 0.25);
};

const skewness = (values) => {
  if (!values.length) return 0;
  const avg = mean(values);
  const std = stdDev(values, avg);
  if (std === 0) return 0;
  const total = values.reduce((sum, value) => sum + Math.pow((value - avg) / std, 3), 0);
  return total / values.length;
};

const frameSignal = (pcm, frameSize, hopSize) => {
  const frames = [];
  for (let start = 0; start + frameSize <= pcm.length; start += hopSize) {
    frames.push(pcm.slice(start, start + frameSize));
  }
  return frames;
};

const rmsForFrame = (frame) => {
  const energy = frame.reduce((sum, sample) => sum + sample * sample, 0);
  return Math.sqrt(energy / frame.length);
};

const zcrForFrame = (frame) => {
  let crossings = 0;
  for (let i = 1; i < frame.length; i += 1) {
    if ((frame[i - 1] >= 0 && frame[i] < 0) || (frame[i - 1] < 0 && frame[i] >= 0)) {
      crossings += 1;
    }
  }
  return crossings / frame.length;
};

const extractRmsStats = (frames) => {
  const values = frames.map(rmsForFrame);
  const avg = mean(values);
  return {
    mean: avg,
    std: stdDev(values, avg),
    min: Math.min(...values),
    max: Math.max(...values),
    iqr: iqr(values),
  };
};

const extractZcrStats = (frames) => {
  const values = frames.map(zcrForFrame);
  const avg = mean(values);
  return { mean: avg, std: stdDev(values, avg) };
};

const extractPitchStats = async (frames, pitchEstimator) => {
  if (!pitchEstimator) {
    return {
      ok: false,
      error: { code: "PITCH_ESTIMATOR_MISSING", message: "Pitch estimator not configured." },
    };
  }

  const values = [];
  for (const frame of frames) {
    const pitch = await pitchEstimator(frame);
    if (Number.isFinite(pitch)) values.push(pitch);
  }

  const avg = mean(values);
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 0;
  const range = maxValue - minValue;
  let stableCount = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (Math.abs(values[i] - values[i - 1]) <= 20) stableCount += 1;
  }
  const stability = values.length > 1 ? stableCount / (values.length - 1) : 0;

  return {
    ok: true,
    stats: {
      mean: avg,
      std: stdDev(values, avg),
      min: minValue,
      max: maxValue,
      range,
      stability,
    },
  };
};

const extractMfccStats = async (frames, mfccExtractor) => {
  if (!mfccExtractor) {
    return {
      ok: false,
      error: { code: "MFCC_EXTRACTOR_MISSING", message: "MFCC extractor not configured." },
    };
  }

  const coefficients = [];
  for (const frame of frames) {
    const mfcc = await mfccExtractor(frame);
    if (Array.isArray(mfcc)) coefficients.push(mfcc);
  }

  const dims = coefficients[0] ? coefficients[0].length : 0;
  const means = [];
  const stds = [];

  for (let i = 0; i < dims; i += 1) {
    const values = coefficients.map((coeff) => coeff[i]);
    const avg = mean(values);
    means.push(avg);
    stds.push(stdDev(values, avg));
  }

  return { ok: true, stats: { mean: means, std: stds } };
};

const extractSpectralFlatnessStats = async (frames, spectralFlatnessFn) => {
  if (!spectralFlatnessFn) {
    return {
      ok: false,
      error: { code: "SPECTRAL_FLATNESS_MISSING", message: "Spectral flatness function not configured." },
    };
  }

  const values = [];
  for (const frame of frames) {
    const flatness = await spectralFlatnessFn(frame);
    if (Number.isFinite(flatness)) values.push(flatness);
  }

  const avg = mean(values);
  return { ok: true, stats: { mean: avg, std: stdDev(values, avg), skewness: skewness(values) } };
};

const applyHann = (frame) => {
  const windowed = new Float32Array(frame.length);
  for (let i = 0; i < frame.length; i += 1) {
    const multiplier = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (frame.length - 1));
    windowed[i] = frame[i] * multiplier;
  }
  return windowed;
};

const fftReal = (signal) => {
  const n = signal.length;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let i = 0; i < n; i += 1) re[i] = signal[i];

  for (let i = 0, j = 0; i < n; i += 1) {
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size / 2;
    const delta = -2 * Math.PI / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k += 1) {
        const angle = delta * k;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const evenIndex = start + k;
        const oddIndex = evenIndex + half;
        const tre = cos * re[oddIndex] - sin * im[oddIndex];
        const tim = sin * re[oddIndex] + cos * im[oddIndex];
        re[oddIndex] = re[evenIndex] - tre;
        im[oddIndex] = im[evenIndex] - tim;
        re[evenIndex] += tre;
        im[evenIndex] += tim;
      }
    }
  }

  return { re, im };
};

const powerSpectrum = (frame) => {
  const windowed = applyHann(frame);
  const { re, im } = fftReal(windowed);
  const half = Math.floor(re.length / 2);
  const spectrum = new Float32Array(half);
  for (let i = 0; i < half; i += 1) {
    spectrum[i] = re[i] * re[i] + im[i] * im[i];
  }
  return spectrum;
};

const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
const melToHz = (mel) => 700 * (10 ** (mel / 2595) - 1);

const createMelFilterBank = (sampleRate, fftSize, filterCount = 26) => {
  const nyquist = sampleRate / 2;
  const lowMel = hzToMel(0);
  const highMel = hzToMel(nyquist);
  const melPoints = [];
  for (let i = 0; i < filterCount + 2; i += 1) {
    melPoints.push(lowMel + (i / (filterCount + 1)) * (highMel - lowMel));
  }

  const hzPoints = melPoints.map(melToHz);
  const bin = hzPoints.map((hz) => Math.floor((fftSize + 1) * hz / sampleRate));

  const filters = [];
  for (let i = 1; i <= filterCount; i += 1) {
    const filter = new Float32Array(Math.floor(fftSize / 2));
    for (let j = bin[i - 1]; j < bin[i]; j += 1) {
      filter[j] = (j - bin[i - 1]) / (bin[i] - bin[i - 1]);
    }
    for (let j = bin[i]; j < bin[i + 1]; j += 1) {
      filter[j] = (bin[i + 1] - j) / (bin[i + 1] - bin[i]);
    }
    filters.push(filter);
  }
  return filters;
};

const dct = (vector, coeffCount) => {
  const result = [];
  const n = vector.length;
  for (let k = 0; k < coeffCount; k += 1) {
    let sum = 0;
    for (let i = 0; i < n; i += 1) {
      sum += vector[i] * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
    }
    result.push(sum);
  }
  return result;
};

const extractMfcc = (frame, sampleRate, fftSize, filterBank, coeffCount = 13) => {
  const spectrum = powerSpectrum(frame);
  const energies = filterBank.map((filter) => {
    let total = 0;
    for (let i = 0; i < spectrum.length; i += 1) {
      total += spectrum[i] * filter[i];
    }
    return Math.log10(total + 1e-9);
  });

  return dct(energies, coeffCount);
};

const estimatePitch = (frame, sampleRate, minHz = 70, maxHz = 400) => {
  const size = frame.length;
  const meanValue = mean(frame);
  const centered = new Float32Array(size);
  for (let i = 0; i < size; i += 1) centered[i] = frame[i] - meanValue;

  const minLag = Math.floor(sampleRate / maxHz);
  const maxLag = Math.floor(sampleRate / minHz);
  let bestLag = 0;
  let bestCorr = 0;
  let energy = 0;

  for (let i = 0; i < size; i += 1) {
    energy += centered[i] * centered[i];
  }
  if (energy === 0) return null;

  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let corr = 0;
    for (let i = 0; i < size - lag; i += 1) {
      corr += centered[i] * centered[i + lag];
    }
    if (corr > bestCorr) {
      bestCorr = corr;
      bestLag = lag;
    }
  }

  if (bestCorr / energy < 0.3 || bestLag === 0) return null;
  return sampleRate / bestLag;
};

const spectralFlatness = (frame) => {
  const spectrum = powerSpectrum(frame);
  const eps = 1e-12;
  let logSum = 0;
  let linearSum = 0;
  for (const value of spectrum) {
    logSum += Math.log(value + eps);
    linearSum += value;
  }
  const geoMean = Math.exp(logSum / spectrum.length);
  const arithMean = linearSum / spectrum.length;
  return geoMean / (arithMean + eps);
};

const extractFeatures = async (pcm, sampleRate, deps = {}) => {
  if (!pcm || !pcm.length) {
    return { ok: false, error: { code: "EMPTY_PCM", message: "PCM buffer is empty." } };
  }
  if (!sampleRate || sampleRate <= 0) {
    return { ok: false, error: { code: "INVALID_SAMPLE_RATE", message: "Sample rate is invalid." } };
  }

  // Typical analysis settings for speech; pass overrides in deps.config when needed.
  const config = {
    frameSize: deps.config?.frameSize ?? 1024,
    hopSize: deps.config?.hopSize ?? 512,
    mfccCoefficients: deps.config?.mfccCoefficients ?? 13,
    mfccFilters: deps.config?.mfccFilters ?? 26,
  };

  if ((config.frameSize & (config.frameSize - 1)) !== 0) {
    return {
      ok: false,
      error: { code: "INVALID_FRAME_SIZE", message: "Frame size must be a power of two." },
    };
  }

  const frames = frameSignal(pcm, config.frameSize, config.hopSize);
  if (!frames.length) {
    return { ok: false, error: { code: "FRAME_ERROR", message: "Audio too short for framing." } };
  }

  const rms = extractRmsStats(frames);
  const zcr = extractZcrStats(frames);

  const pitchEstimator =
    deps.pitchEstimator ?? ((frame) => estimatePitch(frame, sampleRate));
  const pitchResult = await extractPitchStats(frames, pitchEstimator);
  if (!pitchResult.ok) return pitchResult;

  const filterBank =
    deps.mfccFilterBank ??
    createMelFilterBank(sampleRate, config.frameSize, config.mfccFilters);
  const mfccExtractor =
    deps.mfccExtractor ??
    ((frame) =>
      extractMfcc(frame, sampleRate, config.frameSize, filterBank, config.mfccCoefficients));
  const mfccResult = await extractMfccStats(frames, mfccExtractor);
  if (!mfccResult.ok) return mfccResult;

  const spectralFn = deps.spectralFlatnessFn ?? spectralFlatness;
  const spectralResult = await extractSpectralFlatnessStats(frames, spectralFn);
  if (!spectralResult.ok) return spectralResult;

  return {
    ok: true,
    features: {
      rms,
      pitch: pitchResult.stats,
      mfcc: mfccResult.stats,
      spectralFlatness: spectralResult.stats,
      zcr,
    },
  };
};

module.exports = {
  extractFeatures,
};
