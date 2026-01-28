const sigmoid = (x) => 1 / (1 + Math.exp(-x));

const normalizeVector = (vector, means, stds) => {
  return vector.map((value, index) => {
    const mean = means?.[index] ?? 0;
    const std = stds?.[index] ?? 1;
    if (std === 0) return 0;
    return (value - mean) / std;
  });
};

const linearScore = (vector, weights, bias) => {
  let score = bias ?? 0;
  for (let i = 0; i < vector.length; i += 1) {
    score += vector[i] * (weights[i] ?? 0);
  }
  return score;
};

const predictProbability = (vector, model) => {
  const normalized = normalizeVector(vector, model.means, model.stds);
  const score = linearScore(normalized, model.weights, model.bias);
  return sigmoid(score);
};

module.exports = {
  predictProbability,
};
