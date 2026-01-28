const fs = require("fs");
const path = require("path");

const loadModel = (modelPath = path.join(__dirname, "model.json")) => {
  const raw = fs.readFileSync(modelPath, "utf8");
  const parsed = JSON.parse(raw);

  if (!Array.isArray(parsed.featureOrder) || !Array.isArray(parsed.weights)) {
    throw new Error("Model format invalid.");
  }
  if (parsed.featureOrder.length !== parsed.weights.length) {
    throw new Error("Model feature/weight length mismatch.");
  }

  return parsed;
};

module.exports = {
  loadModel,
};
