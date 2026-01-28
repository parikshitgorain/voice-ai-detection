const safeNumber = (value) => (Number.isFinite(value) ? value : null);

const scoreLow = (value, low, high) => {
  if (value === null) return 0;
  if (value <= low) return 1;
  if (value >= high) return 0;
  return (high - value) / (high - low);
};

const scoreHigh = (value, low, high) => {
  if (value === null) return 0;
  if (value <= low) return 0;
  if (value >= high) return 1;
  return (value - low) / (high - low);
};

const mean = (values) => {
  if (!Array.isArray(values) || values.length === 0) return null;
  const total = values.reduce((sum, value) => sum + value, 0);
  return total / values.length;
};

const formatValue = (value) => {
  if (!Number.isFinite(value)) return null;
  return value.toFixed(3);
};

const buildPhrase = (label, value) => {
  const formatted = formatValue(value);
  if (formatted === null) return null;
  return `${label} (${formatted})`;
};

const getFeatures = (features) => {
  return {
    rmsMean: safeNumber(features?.rms?.mean),
    rmsStd: safeNumber(features?.rms?.std),
    pitchStd: safeNumber(features?.pitch?.std),
    pitchStability: safeNumber(features?.pitch?.stability),
    spectralMean: safeNumber(features?.spectralFlatness?.mean),
    spectralStd: safeNumber(features?.spectralFlatness?.std),
    mfccStdMean: safeNumber(mean(features?.mfcc?.std)),
    zcrStd: safeNumber(features?.zcr?.std),
    stabilityOverall:
      features?.windowCount && features.windowCount > 1
        ? safeNumber(features?.stability?.overall)
        : null,
    metadataSuspicion: safeNumber(features?.metadata?.suspicionScore),
    metadataDurationConsistency: safeNumber(features?.metadata?.durationConsistency),
    varianceStability: safeNumber(features?.longRange?.varianceStability),
    prosodyEntropyStability: safeNumber(features?.longRange?.prosodyEntropyStability),
    pitchEnergyCouplingStability: safeNumber(features?.longRange?.pitchEnergyCouplingStability),
    emotionResetScore: safeNumber(features?.longRange?.emotionResetScore),
    breathRegularity: safeNumber(features?.longRange?.breathPeriodicityRegularity),
    sentenceResetSharpness: safeNumber(features?.longRange?.sentenceResetSharpness),
    stressSymmetry: safeNumber(features?.prosodyPlanning?.stressSymmetry),
    intonationSmoothness: safeNumber(features?.prosodyPlanning?.intonationSmoothness),
    emphasisRegularity: safeNumber(features?.prosodyPlanning?.emphasisRegularity),
    contourPredictability: safeNumber(features?.prosodyPlanning?.contourPredictability),
    hfDecayRegularity: safeNumber(features?.spectralConsistency?.hfDecayRegularity),
    spectralWobbleStability: safeNumber(features?.spectralConsistency?.spectralWobbleStability),
    microPhaseStability: safeNumber(features?.spectralConsistency?.microPhaseStability),
    phaseDeltaStability: safeNumber(features?.spectralConsistency?.phaseDeltaStability),
    fluxStability: safeNumber(features?.spectralConsistency?.fluxStability),
    rolloffStability: safeNumber(features?.spectralConsistency?.rolloffStability),
    pmciWeakness: safeNumber(features?.advanced?.pmciWeakness),
    breathCouplingAnomaly: safeNumber(features?.advanced?.breathCouplingAnomaly),
    phaseCoherence: safeNumber(features?.advanced?.phaseCoherence),
    compressionConsistency: safeNumber(features?.advanced?.compressionConsistency),
    compressionConsistencySpread: safeNumber(features?.advanced?.compressionConsistencySpread),
    groupDisagreement: safeNumber(features?.governance?.groupDisagreement),
    windowDisagreement: safeNumber(features?.governance?.windowDisagreement),
    lowSignalScore: safeNumber(features?.governance?.lowSignalScore),
  };
};

const mapSignals = (features) => {
  const {
    rmsMean,
    rmsStd,
    pitchStd,
    pitchStability,
    spectralMean,
    spectralStd,
    mfccStdMean,
    zcrStd,
    stabilityOverall,
    metadataSuspicion,
    metadataDurationConsistency,
    varianceStability,
    prosodyEntropyStability,
    pitchEnergyCouplingStability,
    emotionResetScore,
    breathRegularity,
    sentenceResetSharpness,
    stressSymmetry,
    intonationSmoothness,
    emphasisRegularity,
    contourPredictability,
    hfDecayRegularity,
    spectralWobbleStability,
    microPhaseStability,
    phaseDeltaStability,
    fluxStability,
    rolloffStability,
    pmciWeakness,
    breathCouplingAnomaly,
    phaseCoherence,
    compressionConsistency,
    compressionConsistencySpread,
    groupDisagreement,
    windowDisagreement,
    lowSignalScore,
  } = getFeatures(features);

  const eps = 1e-6;
  const rmsCv = rmsMean ? rmsStd / Math.max(rmsMean, eps) : null;

  const signals = [
    {
      id: "pitch_stable",
      aligns: "AI_GENERATED",
      strength: Math.max(scoreLow(pitchStd, 20, 55), scoreHigh(pitchStability, 0.7, 0.9)),
      phrase: Number.isFinite(pitchStd)
        ? buildPhrase("pitch variance was low, std", pitchStd)
        : buildPhrase("pitch stability index was high", pitchStability),
      value: Number.isFinite(pitchStd) ? pitchStd : pitchStability,
    },
    {
      id: "pitch_variable",
      aligns: "HUMAN",
      strength: Math.max(scoreHigh(pitchStd, 35, 80), scoreLow(pitchStability, 0.55, 0.75)),
      phrase: Number.isFinite(pitchStd)
        ? buildPhrase("pitch variance was high, std", pitchStd)
        : buildPhrase("pitch stability index was low", pitchStability),
      value: Number.isFinite(pitchStd) ? pitchStd : pitchStability,
    },
    {
      id: "spectral_smooth",
      aligns: "AI_GENERATED",
      strength: Math.max(scoreLow(spectralMean, 0.3, 0.55), scoreLow(spectralStd, 0.06, 0.12)),
      phrase: Number.isFinite(spectralStd)
        ? buildPhrase("spectral flatness variance was low, std", spectralStd)
        : buildPhrase("spectral flatness was low", spectralMean),
      value: Number.isFinite(spectralStd) ? spectralStd : spectralMean,
    },
    {
      id: "spectral_varied",
      aligns: "HUMAN",
      strength: Math.max(scoreHigh(spectralMean, 0.45, 0.7), scoreHigh(spectralStd, 0.08, 0.14)),
      phrase: Number.isFinite(spectralStd)
        ? buildPhrase("spectral flatness variance was high, std", spectralStd)
        : buildPhrase("spectral flatness was high", spectralMean),
      value: Number.isFinite(spectralStd) ? spectralStd : spectralMean,
    },
    {
      id: "energy_stable",
      aligns: "AI_GENERATED",
      strength: scoreLow(rmsCv, 0.35, 0.75),
      phrase: buildPhrase("energy coefficient of variation was low, cv", rmsCv),
      value: rmsCv,
    },
    {
      id: "energy_dynamic",
      aligns: "HUMAN",
      strength: scoreHigh(rmsCv, 0.5, 0.95),
      phrase: buildPhrase("energy coefficient of variation was high, cv", rmsCv),
      value: rmsCv,
    },
    {
      id: "mfcc_stable",
      aligns: "AI_GENERATED",
      strength: scoreLow(mfccStdMean, 6, 14),
      phrase: buildPhrase("MFCC variance was low, mean std", mfccStdMean),
      value: mfccStdMean,
    },
    {
      id: "mfcc_varied",
      aligns: "HUMAN",
      strength: scoreHigh(mfccStdMean, 10, 18),
      phrase: buildPhrase("MFCC variance was high, mean std", mfccStdMean),
      value: mfccStdMean,
    },
    {
      id: "zcr_stable",
      aligns: "AI_GENERATED",
      strength: scoreLow(zcrStd, 0.01, 0.03),
      phrase: buildPhrase("voicing transition variability was low, std", zcrStd),
      value: zcrStd,
    },
    {
      id: "zcr_varied",
      aligns: "HUMAN",
      strength: scoreHigh(zcrStd, 0.02, 0.05),
      phrase: buildPhrase("voicing transition variability was high, std", zcrStd),
      value: zcrStd,
    },
    {
      id: "window_stable",
      aligns: "AI_GENERATED",
      strength: scoreLow(stabilityOverall, 0.15, 0.35),
      phrase: buildPhrase("inter-window drift score was low", stabilityOverall),
      value: stabilityOverall,
    },
    {
      id: "window_varied",
      aligns: "HUMAN",
      strength: scoreHigh(stabilityOverall, 0.25, 0.5),
      phrase: buildPhrase("inter-window drift score was high", stabilityOverall),
      value: stabilityOverall,
    },
    {
      id: "window_disagreement_high",
      aligns: "HUMAN",
      strength: scoreHigh(windowDisagreement, 0.35, 0.6),
      phrase: buildPhrase("inter-window disagreement was high", windowDisagreement),
      value: windowDisagreement,
    },
    {
      id: "metadata_inconsistent",
      aligns: "AI_GENERATED",
      strength: Math.max(
        scoreHigh(metadataSuspicion, 0.5, 0.75),
        scoreLow(metadataDurationConsistency, 0.7, 0.9)
      ),
      phrase: buildPhrase("metadata anomaly score was elevated", metadataSuspicion),
      value: metadataSuspicion,
    },
    {
      id: "prosody_entropy_stable",
      aligns: "AI_GENERATED",
      strength: scoreHigh(prosodyEntropyStability, 0.6, 0.8),
      phrase: buildPhrase("prosody entropy stability was high", prosodyEntropyStability),
      value: prosodyEntropyStability,
    },
    {
      id: "prosody_entropy_varied",
      aligns: "HUMAN",
      strength: scoreLow(prosodyEntropyStability, 0.35, 0.55),
      phrase: buildPhrase("prosody entropy stability was low", prosodyEntropyStability),
      value: prosodyEntropyStability,
    },
    {
      id: "pitch_energy_coupling",
      aligns: "AI_GENERATED",
      strength: scoreHigh(pitchEnergyCouplingStability, 0.6, 0.8),
      phrase: buildPhrase("pitch-energy coupling stability was high", pitchEnergyCouplingStability),
      value: pitchEnergyCouplingStability,
    },
    {
      id: "pitch_energy_decoupled",
      aligns: "HUMAN",
      strength: scoreLow(pitchEnergyCouplingStability, 0.35, 0.55),
      phrase: buildPhrase("pitch-energy coupling stability was low", pitchEnergyCouplingStability),
      value: pitchEnergyCouplingStability,
    },
    {
      id: "emotion_reset",
      aligns: "AI_GENERATED",
      strength: scoreHigh(emotionResetScore, 0.5, 0.75),
      phrase: buildPhrase("segment reset magnitude was high", emotionResetScore),
      value: emotionResetScore,
    },
    {
      id: "breath_regular",
      aligns: "AI_GENERATED",
      strength: scoreHigh(breathRegularity, 0.6, 0.8),
      phrase: buildPhrase("breath interval regularity was high", breathRegularity),
      value: breathRegularity,
    },
    {
      id: "breath_irregular",
      aligns: "HUMAN",
      strength: scoreLow(breathRegularity, 0.35, 0.55),
      phrase: buildPhrase("breath interval regularity was low", breathRegularity),
      value: breathRegularity,
    },
    {
      id: "sentence_reset_sharp",
      aligns: "AI_GENERATED",
      strength: scoreHigh(sentenceResetSharpness, 0.6, 0.8),
      phrase: buildPhrase("sentence-boundary reset sharpness was high", sentenceResetSharpness),
      value: sentenceResetSharpness,
    },
    {
      id: "stress_symmetry",
      aligns: "AI_GENERATED",
      strength: scoreHigh(stressSymmetry, 0.6, 0.8),
      phrase: buildPhrase("stress symmetry index was high", stressSymmetry),
      value: stressSymmetry,
    },
    {
      id: "intonation_smooth",
      aligns: "AI_GENERATED",
      strength: scoreHigh(intonationSmoothness, 0.6, 0.8),
      phrase: buildPhrase("intonation smoothness index was high", intonationSmoothness),
      value: intonationSmoothness,
    },
    {
      id: "emphasis_regular",
      aligns: "AI_GENERATED",
      strength: scoreHigh(emphasisRegularity, 0.6, 0.8),
      phrase: buildPhrase("emphasis regularity index was high", emphasisRegularity),
      value: emphasisRegularity,
    },
    {
      id: "contour_predictable",
      aligns: "AI_GENERATED",
      strength: scoreHigh(contourPredictability, 0.6, 0.8),
      phrase: buildPhrase("contour predictability index was high", contourPredictability),
      value: contourPredictability,
    },
    {
      id: "hf_decay_regular",
      aligns: "AI_GENERATED",
      strength: scoreHigh(hfDecayRegularity, 0.6, 0.8),
      phrase: buildPhrase("high-frequency decay regularity was high", hfDecayRegularity),
      value: hfDecayRegularity,
    },
    {
      id: "spectral_wobble_stable",
      aligns: "AI_GENERATED",
      strength: scoreHigh(spectralWobbleStability, 0.6, 0.8),
      phrase: buildPhrase("spectral wobble stability was high", spectralWobbleStability),
      value: spectralWobbleStability,
    },
    {
      id: "micro_phase_stable",
      aligns: "AI_GENERATED",
      strength: scoreHigh(microPhaseStability, 0.6, 0.8),
      phrase: buildPhrase("micro phase stability was high", microPhaseStability),
      value: microPhaseStability,
    },
    {
      id: "phase_delta_stable",
      aligns: "AI_GENERATED",
      strength: scoreHigh(phaseDeltaStability, 0.6, 0.8),
      phrase: buildPhrase("phase-delta stability was high", phaseDeltaStability),
      value: phaseDeltaStability,
    },
    {
      id: "phase_delta_irregular",
      aligns: "HUMAN",
      strength: scoreLow(phaseDeltaStability, 0.35, 0.55),
      phrase: buildPhrase("phase-delta stability was low", phaseDeltaStability),
      value: phaseDeltaStability,
    },
    {
      id: "spectral_flux_stable",
      aligns: "AI_GENERATED",
      strength: scoreHigh(fluxStability, 0.6, 0.8),
      phrase: buildPhrase("spectral flux stability was high", fluxStability),
      value: fluxStability,
    },
    {
      id: "spectral_flux_varied",
      aligns: "HUMAN",
      strength: scoreLow(fluxStability, 0.35, 0.55),
      phrase: buildPhrase("spectral flux stability was low", fluxStability),
      value: fluxStability,
    },
    {
      id: "rolloff_stable",
      aligns: "AI_GENERATED",
      strength: scoreHigh(rolloffStability, 0.6, 0.8),
      phrase: buildPhrase("spectral rolloff stability was high", rolloffStability),
      value: rolloffStability,
    },
    {
      id: "pmci_weak",
      aligns: "AI_GENERATED",
      strength: scoreHigh(pmciWeakness, 0.55, 0.75),
      phrase: buildPhrase("prosody memory carry-over was weak", pmciWeakness),
      value: pmciWeakness,
    },
    {
      id: "breath_coupling_anomaly",
      aligns: "AI_GENERATED",
      strength: scoreHigh(breathCouplingAnomaly, 0.55, 0.75),
      phrase: buildPhrase("breath-energy coupling anomaly was high", breathCouplingAnomaly),
      value: breathCouplingAnomaly,
    },
    {
      id: "breath_coupling_natural",
      aligns: "HUMAN",
      strength: scoreLow(breathCouplingAnomaly, 0.35, 0.55),
      phrase: buildPhrase("breath-energy coupling anomaly was low", breathCouplingAnomaly),
      value: breathCouplingAnomaly,
    },
    {
      id: "phase_coherent",
      aligns: "AI_GENERATED",
      strength: scoreHigh(phaseCoherence, 0.6, 0.8),
      phrase: buildPhrase("phase coherence score was high", phaseCoherence),
      value: phaseCoherence,
    },
    {
      id: "compression_consistent",
      aligns: "AI_GENERATED",
      strength: scoreHigh(compressionConsistency, 0.6, 0.8),
      phrase: buildPhrase("compression drift resistance was high", compressionConsistency),
      value: compressionConsistency,
    },
    {
      id: "compression_sensitive",
      aligns: "HUMAN",
      strength: scoreLow(compressionConsistency, 0.35, 0.55),
      phrase: buildPhrase("compression drift resistance was low", compressionConsistency),
      value: compressionConsistency,
    },
    {
      id: "compression_spread",
      aligns: "HUMAN",
      strength: scoreHigh(compressionConsistencySpread, 0.08, 0.18),
      phrase: buildPhrase("compression drift spread was high", compressionConsistencySpread),
      value: compressionConsistencySpread,
    },
    {
      id: "phase_irregular",
      aligns: "HUMAN",
      strength: scoreLow(phaseCoherence, 0.35, 0.55),
      phrase: buildPhrase("phase coherence score was low", phaseCoherence),
      value: phaseCoherence,
    },
    {
      id: "group_disagreement",
      aligns: "AMBIGUOUS",
      strength: scoreHigh(groupDisagreement, 0.35, 0.6),
      phrase: buildPhrase("signal group disagreement was high", groupDisagreement),
      value: groupDisagreement,
    },
    {
      id: "low_signal",
      aligns: "AMBIGUOUS",
      strength: scoreHigh(lowSignalScore, 0.35, 0.6),
      phrase: buildPhrase("long-range signal coverage was limited", lowSignalScore),
      value: lowSignalScore,
    },
  ];

  return signals;
};

module.exports = {
  mapSignals,
};
