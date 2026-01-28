const getPathValue = (obj, path) => {
  return path.split(".").reduce((acc, key) => (acc ? acc[key] : undefined), obj);
};

const toFeatureVector = (featureObject, featureOrder) => {
  const vector = [];
  const missing = [];

  for (const path of featureOrder) {
    const value = getPathValue(featureObject.features, path);
    if (Number.isFinite(value)) {
      vector.push(value);
    } else {
      vector.push(0);
      missing.push(path);
    }
  }

  return { vector, missing };
};

module.exports = {
  toFeatureVector,
};
