const buildFeatureObject = ({
  duration,
  sampleRate,
  rms,
  pitch,
  mfcc,
  spectralFlatness,
  zcr,
  stability,
  windowCount,
  agreementScore,
  noiseScore,
  metadata,
  longRange,
  prosodyPlanning,
  spectralConsistency,
  advanced,
  governance,
}) => {
  return {
    duration,
    sampleRate,
    features: {
      rms,
      pitch,
      mfcc,
      spectralFlatness,
      zcr,
      stability,
      windowCount,
      agreementScore,
      noiseScore,
      metadata,
      longRange,
      prosodyPlanning,
      spectralConsistency,
      advanced,
      governance,
    },
  };
};

module.exports = {
  buildFeatureObject,
};
