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

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const clamp01 = (value) => clamp(value, 0, 1);

const filterFinite = (values) => values.filter((value) => Number.isFinite(value));

const createRunningStats = () => ({ count: 0, mean: 0, m2: 0 });

const updateRunningStats = (stats, value) => {
  if (!Number.isFinite(value)) return;
  stats.count += 1;
  const delta = value - stats.mean;
  stats.mean += delta / stats.count;
  const delta2 = value - stats.mean;
  stats.m2 += delta * delta2;
};

const finalizeRunningStats = (stats) => {
  if (!stats.count) return { mean: null, std: null };
  const variance = stats.count > 1 ? stats.m2 / stats.count : 0;
  return { mean: stats.mean, std: Math.sqrt(variance) };
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

const extractRmsStatsFromValues = (values) => {
  const finite = filterFinite(values);
  if (!finite.length) {
    return { mean: 0, std: 0, min: 0, max: 0, iqr: 0 };
  }
  const avg = mean(finite);
  return {
    mean: avg,
    std: stdDev(finite, avg),
    min: Math.min(...finite),
    max: Math.max(...finite),
    iqr: iqr(finite),
  };
};

const extractZcrStatsFromValues = (values) => {
  const finite = filterFinite(values);
  if (!finite.length) return { mean: 0, std: 0 };
  const avg = mean(finite);
  return { mean: avg, std: stdDev(finite, avg) };
};

const extractPitchStatsFromValues = (values) => {
  const finite = filterFinite(values);
  if (!finite.length) {
    return { mean: 0, std: 0, min: 0, max: 0, range: 0, stability: 0 };
  }
  const avg = mean(finite);
  const minValue = Math.min(...finite);
  const maxValue = Math.max(...finite);
  const range = maxValue - minValue;
  let stableCount = 0;
  for (let i = 1; i < finite.length; i += 1) {
    if (Math.abs(finite[i] - finite[i - 1]) <= 20) stableCount += 1;
  }
  const stability = finite.length > 1 ? stableCount / (finite.length - 1) : 0;
  return {
    mean: avg,
    std: stdDev(finite, avg),
    min: minValue,
    max: maxValue,
    range,
    stability,
  };
};

const extractSpectralStatsFromValues = (values) => {
  const finite = filterFinite(values);
  if (!finite.length) return { mean: 0, std: 0, skewness: 0 };
  const avg = mean(finite);
  return { mean: avg, std: stdDev(finite, avg), skewness: skewness(finite) };
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

const spectralFlatnessFromSpectrum = (spectrum) => {
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

const spectralCentroidFromSpectrum = (spectrum, sampleRate) => {
  let weighted = 0;
  let total = 0;
  const binCount = spectrum.length;
  for (let i = 0; i < binCount; i += 1) {
    const frequency = (i * sampleRate) / (2 * binCount);
    const magnitude = spectrum[i];
    weighted += frequency * magnitude;
    total += magnitude;
  }
  return total > 0 ? weighted / total : 0;
};

const spectralRolloffFromSpectrum = (spectrum, sampleRate, rolloffPercent) => {
  const total = spectrum.reduce((sum, value) => sum + value, 0);
  if (!total) return 0;
  const threshold = total * rolloffPercent;
  let cumulative = 0;
  for (let i = 0; i < spectrum.length; i += 1) {
    cumulative += spectrum[i];
    if (cumulative >= threshold) {
      return (i * sampleRate) / (2 * spectrum.length);
    }
  }
  return sampleRate / 2;
};

const spectralFluxFromSpectra = (prevSpectrum, spectrum) => {
  if (!prevSpectrum || !spectrum) return null;
  let diffSum = 0;
  let magSum = 0;
  for (let i = 0; i < spectrum.length; i += 1) {
    const diff = spectrum[i] - prevSpectrum[i];
    if (diff > 0) diffSum += diff;
    magSum += spectrum[i];
  }
  if (magSum === 0) return 0;
  return diffSum / magSum;
};

const highFreqRatioFromSpectrum = (spectrum) => {
  const total = spectrum.reduce((sum, value) => sum + value, 0);
  if (!total) return 0;
  const start = Math.floor(spectrum.length * 0.8);
  let high = 0;
  for (let i = start; i < spectrum.length; i += 1) {
    high += spectrum[i];
  }
  return high / total;
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
  const bin = hzPoints.map((hz) => Math.floor(((fftSize + 1) * hz) / sampleRate));

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

const entropy = (values, binCount = 20) => {
  if (!values.length) return null;
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  if (maxValue === minValue) return 0;
  const bins = new Array(binCount).fill(0);
  const range = maxValue - minValue;
  for (const value of values) {
    const idx = Math.min(binCount - 1, Math.floor(((value - minValue) / range) * binCount));
    bins[idx] += 1;
  }
  const total = values.length;
  let ent = 0;
  for (const count of bins) {
    if (!count) continue;
    const p = count / total;
    ent -= p * Math.log(p);
  }
  return ent / Math.log(binCount);
};

const correlation = (valuesA, valuesB) => {
  if (!Array.isArray(valuesA) || !Array.isArray(valuesB)) return null;
  const count = Math.min(valuesA.length, valuesB.length);
  const pairs = [];
  for (let i = 0; i < count; i += 1) {
    const a = valuesA[i];
    const b = valuesB[i];
    if (Number.isFinite(a) && Number.isFinite(b)) {
      pairs.push([a, b]);
    }
  }
  if (pairs.length < 2) return null;
  const xs = pairs.map((pair) => pair[0]);
  const ys = pairs.map((pair) => pair[1]);
  const meanX = mean(xs);
  const meanY = mean(ys);
  let cov = 0;
  for (let i = 0; i < pairs.length; i += 1) {
    cov += (xs[i] - meanX) * (ys[i] - meanY);
  }
  cov /= pairs.length;
  const stdX = stdDev(xs, meanX);
  const stdY = stdDev(ys, meanY);
  if (stdX === 0 || stdY === 0) return 0;
  return cov / (stdX * stdY);
};

const linearRegressionSlope = (values, startIndex, hopSeconds) => {
  const points = [];
  for (let i = startIndex; i < values.length; i += 1) {
    const value = values[i];
    if (!Number.isFinite(value)) continue;
    points.push({ x: i * hopSeconds, y: value });
  }
  if (points.length < 2) return null;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const meanX = mean(xs);
  const meanY = mean(ys);
  let cov = 0;
  let varX = 0;
  for (let i = 0; i < points.length; i += 1) {
    cov += (xs[i] - meanX) * (ys[i] - meanY);
    varX += Math.pow(xs[i] - meanX, 2);
  }
  if (varX === 0) return null;
  return cov / varX;
};

const secondDerivativeStd = (values) => {
  const derivatives = [];
  for (let i = 1; i < values.length - 1; i += 1) {
    const prev = values[i - 1];
    const current = values[i];
    const next = values[i + 1];
    if (!Number.isFinite(prev) || !Number.isFinite(current) || !Number.isFinite(next)) continue;
    derivatives.push(next - 2 * current + prev);
  }
  if (derivatives.length < 2) return null;
  const avg = mean(derivatives);
  return stdDev(derivatives, avg);
};

const findPeaks = (values, threshold, minDistance) => {
  const peaks = [];
  let lastPeak = -Infinity;
  for (let i = 1; i < values.length - 1; i += 1) {
    const value = values[i];
    const prev = values[i - 1];
    const next = values[i + 1];
    if (!Number.isFinite(value) || value < threshold) continue;
    if (!Number.isFinite(prev) || !Number.isFinite(next)) continue;
    if (value >= prev && value > next) {
      if (i - lastPeak >= minDistance) {
        peaks.push(i);
        lastPeak = i;
      }
    }
  }
  return peaks;
};

const intervalRegularity = (indices) => {
  if (!indices || indices.length < 3) return null;
  const intervals = [];
  for (let i = 1; i < indices.length; i += 1) {
    intervals.push(indices[i] - indices[i - 1]);
  }
  const avg = mean(intervals);
  if (!avg) return null;
  const cv = stdDev(intervals, avg) / avg;
  return clamp01(1 - Math.min(cv, 1));
};

const wrapToPi = (value) => {
  let wrapped = value;
  while (wrapped > Math.PI) wrapped -= 2 * Math.PI;
  while (wrapped < -Math.PI) wrapped += 2 * Math.PI;
  return wrapped;
};

const entropyFromHistogram = (histogram) => {
  const total = histogram.reduce((sum, value) => sum + value, 0);
  if (!total) return null;
  let ent = 0;
  for (const count of histogram) {
    if (!count) continue;
    const p = count / total;
    ent -= p * Math.log(p);
  }
  return ent / Math.log(histogram.length);
};

const computeSpectralWobbleStability = (centroidValues) => {
  const finite = filterFinite(centroidValues);
  if (finite.length < 2) return null;
  const avg = mean(finite);
  if (!avg) return null;
  const cv = stdDev(finite, avg) / avg;
  return clamp01(1 - cv / 0.6);
};

const computeHfDecayRegularity = (ratios) => {
  const finite = filterFinite(ratios);
  if (finite.length < 2) return null;
  const avg = mean(finite);
  const std = stdDev(finite, avg);
  return clamp01(1 - std / 0.05);
};

const computeFluxStability = (values, scale) => {
  const finite = filterFinite(values);
  if (finite.length < 2) return null;
  const avg = mean(finite);
  const std = stdDev(finite, avg);
  if (!avg) return clamp01(1 - std / scale);
  const cv = std / avg;
  return clamp01(1 - cv / scale);
};

const computeRolloffStability = (values, scale) => {
  const finite = filterFinite(values);
  if (finite.length < 2) return null;
  const avg = mean(finite);
  if (!avg) return null;
  const cv = stdDev(finite, avg) / avg;
  return clamp01(1 - cv / scale);
};

const computeIntonationSmoothness = (pitchValues, scale) => {
  const curvatureStd = secondDerivativeStd(pitchValues);
  if (!Number.isFinite(curvatureStd)) return null;
  return clamp01(1 - curvatureStd / scale);
};

const computeContourSlope = (pitchValues, hopSeconds, ratio) => {
  if (!pitchValues.length) return null;
  const startIndex = Math.floor(pitchValues.length * (1 - ratio));
  return linearRegressionSlope(pitchValues, startIndex, hopSeconds);
};

const analyzeBreathEvents = (rmsValues, centroidValues, config, hopMs) => {
  if (!rmsValues.length) {
    return { eventCount: 0, intervalRegularity: null, couplingScore: null };
  }

  const medianRms = percentile(rmsValues, 0.5);
  const threshold = medianRms * config.thresholdRatio;
  const minFrames = Math.max(1, Math.floor(config.minMs / hopMs));
  const maxFrames = Math.max(minFrames, Math.floor(config.maxMs / hopMs));
  const preFrames = config.preFrames;
  const postFrames = config.postFrames;

  const events = [];
  let index = 0;
  while (index < rmsValues.length) {
    if (rmsValues[index] < threshold) {
      const start = index;
      while (index < rmsValues.length && rmsValues[index] < threshold) index += 1;
      const end = index - 1;
      const length = end - start + 1;
      if (length >= minFrames && length <= maxFrames) {
        events.push({ start, end });
      }
    } else {
      index += 1;
    }
  }

  if (!events.length) {
    return { eventCount: 0, intervalRegularity: null, couplingScore: null };
  }

  const energyDrops = [];
  const formantShifts = [];
  const centers = [];

  for (const event of events) {
    centers.push(Math.floor((event.start + event.end) / 2));
    const preStart = Math.max(0, event.start - preFrames);
    const preEnd = Math.max(0, event.start - 1);
    const postStart = Math.min(rmsValues.length - 1, event.end + 1);
    const postEnd = Math.min(rmsValues.length - 1, event.end + postFrames);

    const preRms = mean(rmsValues.slice(preStart, preEnd + 1));
    const duringRms = mean(rmsValues.slice(event.start, event.end + 1));
    const preCentroid = centroidValues.length
      ? mean(centroidValues.slice(preStart, preEnd + 1))
      : null;
    const duringCentroid = centroidValues.length
      ? mean(centroidValues.slice(event.start, event.end + 1))
      : null;

    if (Number.isFinite(preRms) && Number.isFinite(duringRms)) {
      const drop = Math.max(0, preRms - duringRms);
      if (Number.isFinite(preCentroid) && Number.isFinite(duringCentroid)) {
        const shift = Math.abs(duringCentroid - preCentroid) / Math.max(preCentroid, 1e-6);
        energyDrops.push(drop);
        formantShifts.push(shift);
      }
    }
  }

  const intervalRegularityScore = intervalRegularity(centers);
  const coupling = correlation(energyDrops, formantShifts);

  return {
    eventCount: events.length,
    intervalRegularity: intervalRegularityScore,
    couplingScore: coupling,
  };
};

const computeSentenceResetSharpness = (pitchValues, rmsValues, config, hopMs) => {
  if (!pitchValues.length) return null;
  const medianRms = percentile(rmsValues, 0.5);
  const voicedThreshold = medianRms * config.voiceThresholdRatio;
  const voicedMask = pitchValues.map(
    (pitch, index) => Number.isFinite(pitch) && rmsValues[index] > voicedThreshold
  );
  const minPauseFrames = Math.max(3, Math.floor(config.minPauseMs / hopMs));
  const drops = [];

  let index = 0;
  while (index < voicedMask.length) {
    if (!voicedMask[index]) {
      const pauseStart = index;
      while (index < voicedMask.length && !voicedMask[index]) index += 1;
      const pauseEnd = index - 1;
      if (pauseEnd - pauseStart + 1 >= minPauseFrames) {
        const preStart = Math.max(0, pauseStart - config.preFrames);
        const preEnd = pauseStart - 1;
        const postStart = pauseEnd + 1;
        const postEnd = Math.min(voicedMask.length - 1, pauseEnd + config.postFrames);
        const pre = [];
        const post = [];
        for (let i = preStart; i <= preEnd; i += 1) {
          if (voicedMask[i] && Number.isFinite(pitchValues[i])) pre.push(pitchValues[i]);
        }
        for (let i = postStart; i <= postEnd; i += 1) {
          if (voicedMask[i] && Number.isFinite(pitchValues[i])) post.push(pitchValues[i]);
        }
        if (pre.length >= 2 && post.length >= 2) {
          const preMean = mean(pre);
          const postMean = mean(post);
          const drop = preMean - postMean;
          if (drop > 0) drops.push(drop);
        }
      }
    } else {
      index += 1;
    }
  }

  if (drops.length < 2) return null;
  const avgDrop = mean(drops);
  const spread = stdDev(drops, avgDrop);
  const cv = avgDrop > 0 ? spread / avgDrop : 1;
  const normalizedDrop = clamp01(avgDrop / 80);
  return clamp01(normalizedDrop * (1 - clamp01(cv)));
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

const extractFeatures = async (pcm, sampleRate, deps = {}) => {
  if (!pcm || !pcm.length) {
    return { ok: false, error: { code: "EMPTY_PCM", message: "PCM buffer is empty." } };
  }
  if (!sampleRate || sampleRate <= 0) {
    return { ok: false, error: { code: "INVALID_SAMPLE_RATE", message: "Sample rate is invalid." } };
  }

  const config = {
    frameSize: deps.config?.frameSize ?? 1024,
    hopSize: deps.config?.hopSize ?? 512,
    mfccCoefficients: deps.config?.mfccCoefficients ?? 13,
    mfccFilters: deps.config?.mfccFilters ?? 26,
    phaseAnalysisStride: deps.config?.phaseAnalysisStride ?? 2,
    phaseEntropyBins: deps.config?.phaseEntropyBins ?? 24,
    phaseSampleBins: deps.config?.phaseSampleBins ?? 18,
    phaseDeltaStdScale: deps.config?.phaseDeltaStdScale ?? 0.35,
    entropyBins: deps.config?.entropyBins ?? 24,
    peakMinDistanceMs: deps.config?.peakMinDistanceMs ?? 120,
    intonationSmoothnessScale: deps.config?.intonationSmoothnessScale ?? 35,
    contourWindowRatio: deps.config?.contourWindowRatio ?? 0.25,
    spectralFluxScale: deps.config?.spectralFluxScale ?? 0.5,
    spectralRolloffScale: deps.config?.spectralRolloffScale ?? 0.4,
    rolloffPercent: deps.config?.rolloffPercent ?? 0.85,
    breath: {
      thresholdRatio: deps.config?.breath?.thresholdRatio ?? 0.4,
      minMs: deps.config?.breath?.minMs ?? 150,
      maxMs: deps.config?.breath?.maxMs ?? 800,
      preFrames: deps.config?.breath?.preFrames ?? 5,
      postFrames: deps.config?.breath?.postFrames ?? 5,
    },
    sentenceReset: {
      minPauseMs: deps.config?.sentenceReset?.minPauseMs ?? 120,
      preFrames: deps.config?.sentenceReset?.preFrames ?? 5,
      postFrames: deps.config?.sentenceReset?.postFrames ?? 5,
      voiceThresholdRatio: deps.config?.sentenceReset?.voiceThresholdRatio ?? 0.3,
    },
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

  const includeAnalysis = deps.analysisMode !== "lite";
  const rmsValues = [];
  const zcrValues = [];
  const pitchValues = [];
  const spectralFlatnessValues = [];
  const centroidValues = [];
  const highFreqRatioValues = [];
  const fluxValues = [];
  const rolloffValues = [];

  const pitchEstimator = deps.pitchEstimator ?? ((frame) => estimatePitch(frame, sampleRate));

  const phaseBins = config.phaseEntropyBins;
  const phaseHistogram = new Array(phaseBins).fill(0);
  const phaseDeltaStats = createRunningStats();
  let phaseAbsDeltaSum = 0;
  let phaseDeltaCount = 0;
  let prevPhase = null;
  let phaseBinIndices = null;
  let prevSpectrum = null;

  for (let frameIndex = 0; frameIndex < frames.length; frameIndex += 1) {
    const frame = frames[frameIndex];
    const rms = rmsForFrame(frame);
    rmsValues.push(rms);
    zcrValues.push(zcrForFrame(frame));

    const pitch = await pitchEstimator(frame);
    pitchValues.push(Number.isFinite(pitch) ? pitch : null);

    const spectrum = powerSpectrum(frame);
    const flatness = spectralFlatnessFromSpectrum(spectrum);
    spectralFlatnessValues.push(flatness);

    if (includeAnalysis) {
      centroidValues.push(spectralCentroidFromSpectrum(spectrum, sampleRate));
      highFreqRatioValues.push(highFreqRatioFromSpectrum(spectrum));
      rolloffValues.push(
        spectralRolloffFromSpectrum(spectrum, sampleRate, config.rolloffPercent)
      );
      const flux = spectralFluxFromSpectra(prevSpectrum, spectrum);
      if (Number.isFinite(flux)) fluxValues.push(flux);
      prevSpectrum = spectrum;
    }

    if (includeAnalysis && config.phaseAnalysisStride > 0 && frameIndex % config.phaseAnalysisStride === 0) {
      const { re, im } = fftReal(applyHann(frame));
      const half = Math.floor(re.length / 2);
      if (!phaseBinIndices) {
        const bins = Math.max(4, Math.min(config.phaseSampleBins, half - 1));
        phaseBinIndices = [];
        for (let i = 1; i <= bins; i += 1) {
          phaseBinIndices.push(Math.floor((i * half) / (bins + 1)));
        }
      }
      const phases = new Float32Array(phaseBinIndices.length);
      for (let i = 0; i < phaseBinIndices.length; i += 1) {
        const idx = phaseBinIndices[i];
        phases[i] = Math.atan2(im[idx], re[idx]);
      }
      if (prevPhase) {
        for (let i = 0; i < phases.length; i += 1) {
          const delta = wrapToPi(phases[i] - prevPhase[i]);
          const bin = Math.min(
            phaseBins - 1,
            Math.floor(((delta + Math.PI) / (2 * Math.PI)) * phaseBins)
          );
          phaseHistogram[bin] += 1;
          const absDelta = Math.abs(delta);
          phaseAbsDeltaSum += absDelta;
          phaseDeltaCount += 1;
          updateRunningStats(phaseDeltaStats, absDelta);
        }
      }
      prevPhase = phases;
    }
  }

  const rms = extractRmsStatsFromValues(rmsValues);
  const zcr = extractZcrStatsFromValues(zcrValues);
  const pitch = extractPitchStatsFromValues(pitchValues);
  const spectralFlatness = extractSpectralStatsFromValues(spectralFlatnessValues);

  const filterBank =
    deps.mfccFilterBank ??
    createMelFilterBank(sampleRate, config.frameSize, config.mfccFilters);
  const mfccExtractor =
    deps.mfccExtractor ??
    ((frame) =>
      extractMfcc(frame, sampleRate, config.frameSize, filterBank, config.mfccCoefficients));
  const mfccResult = await extractMfccStats(frames, mfccExtractor);
  if (!mfccResult.ok) return mfccResult;

  let analysis = null;
  if (includeAnalysis) {
    const hopSeconds = config.hopSize / sampleRate;
    const hopMs = hopSeconds * 1000;
    const pitchEntropy = entropy(filterFinite(pitchValues), config.entropyBins);
    const energyEntropy = entropy(rmsValues, config.entropyBins);
    const pitchEnergyCorrelation = correlation(pitchValues, rmsValues);

    const peakDistanceFrames = Math.max(1, Math.floor(config.peakMinDistanceMs / hopMs));
    const stressPeaks = findPeaks(
      rmsValues,
      rms.mean + rms.std * 0.5,
      peakDistanceFrames
    );
    const emphasisPeaks = findPeaks(
      pitchValues,
      pitch.mean + pitch.std * 0.5,
      peakDistanceFrames
    );

    const stressSymmetry = intervalRegularity(stressPeaks);
    const emphasisRegularity = intervalRegularity(emphasisPeaks);
    const intonationSmoothness = computeIntonationSmoothness(
      pitchValues,
      config.intonationSmoothnessScale
    );
    const contourSlope = computeContourSlope(
      pitchValues,
      hopSeconds,
      config.contourWindowRatio
    );

    const breath = analyzeBreathEvents(rmsValues, centroidValues, config.breath, hopMs);
    const sentenceResetSharpness = computeSentenceResetSharpness(
      pitchValues,
      rmsValues,
      config.sentenceReset,
      hopMs
    );

    const spectralWobbleStability = computeSpectralWobbleStability(centroidValues);
    const hfDecayRegularity = computeHfDecayRegularity(highFreqRatioValues);
    const phaseEntropy = entropyFromHistogram(phaseHistogram);
    const microPhaseStability = phaseDeltaCount
      ? clamp01(1 - (phaseAbsDeltaSum / phaseDeltaCount) / Math.PI)
      : null;

    const phaseStats = finalizeRunningStats(phaseDeltaStats);
    const phaseDeltaMean = Number.isFinite(phaseStats.mean)
      ? clamp01(phaseStats.mean / Math.PI)
      : null;
    const phaseDeltaStd = Number.isFinite(phaseStats.std)
      ? clamp01(phaseStats.std / (config.phaseDeltaStdScale * Math.PI))
      : null;

    const fluxStability = computeFluxStability(fluxValues, config.spectralFluxScale);
    const rolloffStability = computeRolloffStability(
      rolloffValues,
      config.spectralRolloffScale
    );

    analysis = {
      pitchEntropy,
      energyEntropy,
      pitchEnergyCorrelation,
      stressSymmetry,
      emphasisRegularity,
      intonationSmoothness,
      contourSlope,
      breath,
      sentenceResetSharpness,
      spectral: {
        hfDecayRegularity,
        spectralWobbleStability,
        microPhaseStability,
        phaseEntropy,
        phaseDeltaMean,
        phaseDeltaStd,
        fluxStability,
        rolloffStability,
      },
    };
  }

  return {
    ok: true,
    features: {
      rms,
      pitch,
      mfcc: mfccResult.stats,
      spectralFlatness,
      zcr,
    },
    analysis,
  };
};

module.exports = {
  extractFeatures,
};
