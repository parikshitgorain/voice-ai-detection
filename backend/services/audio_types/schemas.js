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
    },
  };
};

module.exports = {
  buildFeatureObject,
};
