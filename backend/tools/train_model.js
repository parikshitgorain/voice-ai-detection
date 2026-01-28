const fs = require("fs");
const path = require("path");
const { buildFeatureSet } = require("../services/audio_pipeline");
const { toFeatureVector } = require("../services/classifier/feature_adapter");
const config = require("../config");
const baseModel = require("../services/classifier/model.json");

const DATA_ROOT = path.resolve(__dirname, "..", "data");
const TRAIN_AI = path.join(DATA_ROOT, "train", "ai");
const TRAIN_HUMAN = path.join(DATA_ROOT, "train", "human");
const VAL_AI = path.join(DATA_ROOT, "val", "ai");
const VAL_HUMAN = path.join(DATA_ROOT, "val", "human");

const ALLOWED_EXTS = new Set([".mp3", ".wav", ".flac", ".m4a"]);

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const idx = args.indexOf(name);
  if (idx >= 0 && idx < args.length - 1) return args[idx + 1];
  return fallback;
};

const EPOCHS = Number(getArg("--epochs", "200"));
const LR = Number(getArg("--lr", "0.05"));
const L2 = Number(getArg("--l2", "0.0005"));

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

const listFiles = (dir) => {
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...listFiles(full));
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name).toLowerCase();
      if (ALLOWED_EXTS.has(ext)) files.push(full);
    }
  }
  return files;
};

const trainingConfig = {
  ...config,
  limits: {
    ...config.limits,
    disableDurationChecks: true,
  },
};

const loadFeatureVector = async (filePath) => {
  const b64 = fs.readFileSync(filePath).toString("base64");
  const result = await buildFeatureSet(b64, {}, trainingConfig);
  if (!result.ok) {
    return { ok: false, error: result.error };
  }
  const { vector } = toFeatureVector(result.data, baseModel.featureOrder);
  return { ok: true, vector };
};

const buildDataset = async (aiFiles, humanFiles) => {
  const X = [];
  const y = [];
  let skipped = 0;

  for (const file of aiFiles) {
    const res = await loadFeatureVector(file);
    if (!res.ok) {
      skipped += 1;
      continue;
    }
    X.push(res.vector);
    y.push(1);
  }

  for (const file of humanFiles) {
    const res = await loadFeatureVector(file);
    if (!res.ok) {
      skipped += 1;
      continue;
    }
    X.push(res.vector);
    y.push(0);
  }

  return { X, y, skipped };
};

const computeMeansStds = (X) => {
  const dims = X[0].length;
  const means = new Array(dims).fill(0);
  const stds = new Array(dims).fill(0);
  const n = X.length;

  for (const row of X) {
    for (let j = 0; j < dims; j += 1) {
      means[j] += row[j];
    }
  }
  for (let j = 0; j < dims; j += 1) {
    means[j] /= n;
  }

  for (const row of X) {
    for (let j = 0; j < dims; j += 1) {
      const diff = row[j] - means[j];
      stds[j] += diff * diff;
    }
  }
  for (let j = 0; j < dims; j += 1) {
    stds[j] = Math.sqrt(stds[j] / n);
    if (!Number.isFinite(stds[j]) || stds[j] < 1e-6) stds[j] = 1;
  }

  return { means, stds };
};

const normalize = (X, means, stds) => {
  return X.map((row) => row.map((v, i) => (v - means[i]) / stds[i]));
};

const trainLogReg = (X, y) => {
  const n = X.length;
  const d = X[0].length;
  const w = new Array(d).fill(0);
  let b = 0;

  for (let epoch = 0; epoch < EPOCHS; epoch += 1) {
    const gradW = new Array(d).fill(0);
    let gradB = 0;

    for (let i = 0; i < n; i += 1) {
      let score = b;
      const row = X[i];
      for (let j = 0; j < d; j += 1) {
        score += w[j] * row[j];
      }
      const p = sigmoid(score);
      const diff = p - y[i];
      gradB += diff;
      for (let j = 0; j < d; j += 1) {
        gradW[j] += diff * row[j];
      }
    }

    const invN = 1 / n;
    for (let j = 0; j < d; j += 1) {
      gradW[j] = gradW[j] * invN + L2 * w[j];
      w[j] -= LR * gradW[j];
    }
    gradB *= invN;
    b -= LR * gradB;

    if ((epoch + 1) % 25 === 0 || epoch === 0 || epoch === EPOCHS - 1) {
      let loss = 0;
      for (let i = 0; i < n; i += 1) {
        let score = b;
        const row = X[i];
        for (let j = 0; j < d; j += 1) score += w[j] * row[j];
        const p = Math.min(Math.max(sigmoid(score), 1e-8), 1 - 1e-8);
        loss += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
      }
      loss = loss / n;
      console.log(`epoch ${epoch + 1}/${EPOCHS} loss=${loss.toFixed(4)}`);
    }
  }

  return { weights: w, bias: b };
};

const evaluate = (X, y, weights, bias) => {
  let correct = 0;
  for (let i = 0; i < X.length; i += 1) {
    let score = bias;
    const row = X[i];
    for (let j = 0; j < row.length; j += 1) {
      score += weights[j] * row[j];
    }
    const p = sigmoid(score);
    const pred = p >= 0.5 ? 1 : 0;
    if (pred === y[i]) correct += 1;
  }
  return correct / X.length;
};

const ensureData = () => {
  const required = [TRAIN_AI, TRAIN_HUMAN];
  for (const dir of required) {
    if (!fs.existsSync(dir)) {
      throw new Error(`Missing folder: ${dir}`);
    }
  }
};

(async () => {
  ensureData();
  const trainAi = listFiles(TRAIN_AI);
  const trainHuman = listFiles(TRAIN_HUMAN);
  const valAi = listFiles(VAL_AI);
  const valHuman = listFiles(VAL_HUMAN);

  if (!trainAi.length || !trainHuman.length) {
    throw new Error("Training data missing. Place files in backend/data/train/ai and backend/data/train/human.");
  }

  console.log(`train ai: ${trainAi.length} | train human: ${trainHuman.length}`);
  console.log(`val ai: ${valAi.length} | val human: ${valHuman.length}`);

  const trainData = await buildDataset(trainAi, trainHuman);
  if (!trainData.X.length) {
    throw new Error("No usable training samples (all were rejected). Check duration/format.");
  }

  const { means, stds } = computeMeansStds(trainData.X);
  const trainX = normalize(trainData.X, means, stds);
  const model = trainLogReg(trainX, trainData.y);

  let valAccuracy = null;
  if (valAi.length || valHuman.length) {
    const valData = await buildDataset(valAi, valHuman);
    if (valData.X.length) {
      const valX = normalize(valData.X, means, stds);
      valAccuracy = evaluate(valX, valData.y, model.weights, model.bias);
      console.log(`val accuracy: ${valAccuracy.toFixed(4)}`);
    }
  }

  const outputModel = {
    version: "0.2",
    featureOrder: baseModel.featureOrder,
    weights: model.weights,
    bias: model.bias,
    means,
    stds,
  };

  const outPath = path.join(__dirname, "..", "services", "classifier", "model.json");
  fs.writeFileSync(outPath, JSON.stringify(outputModel, null, 2));
  console.log(`model saved: ${outPath}`);

  if (trainData.skipped) {
    console.log(`skipped samples: ${trainData.skipped}`);
  }
})();
