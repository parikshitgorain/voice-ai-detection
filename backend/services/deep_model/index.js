const crypto = require("crypto");
const fs = require("fs/promises");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

const runPython = (pythonPath, scriptPath, args, timeoutMs = 30000) =>
  new Promise((resolve, reject) => {
    const proc = spawn(pythonPath, [scriptPath, ...args], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout = [];
    const stderr = [];
    let finished = false;

    const timer = setTimeout(() => {
      if (finished) return;
      finished = true;
      proc.kill("SIGKILL");
      reject(new Error("Deep model inference timed out."));
    }, timeoutMs);

    proc.stdout.on("data", (chunk) => stdout.push(chunk));
    proc.stderr.on("data", (chunk) => stderr.push(chunk));
    proc.on("error", (err) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      reject(err);
    });
    proc.on("close", (code) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      if (code !== 0) {
        const errMsg = Buffer.concat(stderr).toString("utf8");
        reject(new Error(errMsg || "Deep model inference failed."));
        return;
      }
      const output = Buffer.concat(stdout).toString("utf8").trim();
      resolve(output);
    });
  });

const inferDeepScore = async (audioBase64, config, selectedLanguage = null, audioFormat = null) => {
  const settings = config?.deepModel || {};
  if (!settings.enabled) {
    return {
      ok: true,
      score: null,
      detectedLanguage: null,
      languageConfidence: null,
      languageDistribution: null,
      multiSpeakerScore: null,
    };
  }

  const pythonPath = settings.pythonPath || "python";
  const scriptPath =
    settings.scriptPath ||
    path.join(
      __dirname,
      "..",
      "..",
      "deep",
      settings.useMultitask ? "infer_multitask.py" : "infer_deep.py"
    );
  let modelPath =
    settings.modelPath ||
    path.join(
      __dirname,
      "..",
      "..",
      "deep",
      settings.useMultitask ? "multitask_multilingual.pt" : "model.pt"
    );
  if (settings.modelByLanguage) {
    if (!selectedLanguage) {
      return {
        ok: false,
        error: new Error("Language is required when using per-language models."),
      };
    }
    const mapped = settings.modelByLanguage[selectedLanguage];
    if (!mapped) {
      return {
        ok: false,
        error: new Error(`No model configured for language "${selectedLanguage}".`),
      };
    }
    modelPath = mapped;
  }
  const device = settings.device || "cpu";
  const timeoutMs = settings.timeoutMs || 30000;

  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "voice-deep-"));
  const safeFormat =
    typeof audioFormat === "string" ? audioFormat.trim().toLowerCase() : null;
  const ext =
    safeFormat && safeFormat !== "auto" && /^[a-z0-9]+$/.test(safeFormat)
      ? `.${safeFormat}`
      : ".mp3";
  const tempFile = path.join(tempDir, `audio-${crypto.randomBytes(8).toString("hex")}${ext}`);

  try {
    const buffer = Buffer.from(audioBase64, "base64");
    await fs.writeFile(tempFile, buffer);

    let detectedLanguage = null;
    let languageConfidence = null;
    let languageDistribution = null;
    const languageDetector = settings.languageDetector || {};
    if (languageDetector.enabled && languageDetector.modelPath) {
      const langScript =
        languageDetector.scriptPath ||
        path.join(__dirname, "..", "..", "deep", "infer_multitask.py");
      const langDevice = languageDetector.device || device;
      const langOutput = await runPython(
        pythonPath,
        langScript,
        ["--audio", tempFile, "--model", languageDetector.modelPath, "--device", langDevice],
        timeoutMs
      );
      const langParsed = JSON.parse(langOutput);
      if (langParsed && typeof langParsed === "object" && langParsed.languageDistribution) {
        const entries = Object.entries(langParsed.languageDistribution).filter(([, v]) =>
          Number.isFinite(v)
        );
        if (entries.length) {
          entries.sort((a, b) => b[1] - a[1]);
          [detectedLanguage, languageConfidence] = entries[0];
          languageDistribution = langParsed.languageDistribution;
        }
      }
    }

    const output = await runPython(
      pythonPath,
      scriptPath,
      ["--audio", tempFile, "--model", modelPath, "--device", device],
      timeoutMs
    );

    const parsed = JSON.parse(output);
    if (!parsed || typeof parsed !== "object") {
      return { ok: false, error: new Error("Deep model output invalid.") };
    }

    if (Number.isFinite(parsed.deepScore)) {
      return {
        ok: true,
        score: parsed.deepScore,
        detectedLanguage,
        languageConfidence,
        languageDistribution,
        multiSpeakerScore: null,
      };
    }

    if (!Number.isFinite(parsed.aiScore)) {
      return { ok: false, error: new Error("Deep model output invalid.") };
    }

    if (!languageDistribution && parsed.languageDistribution) {
      const entries = Object.entries(parsed.languageDistribution).filter(([, v]) =>
        Number.isFinite(v)
      );
      if (entries.length) {
        entries.sort((a, b) => b[1] - a[1]);
        [detectedLanguage, languageConfidence] = entries[0];
        languageDistribution = parsed.languageDistribution;
      }
    }

    return {
      ok: true,
      score: parsed.aiScore,
      detectedLanguage,
      languageConfidence,
      languageDistribution,
      multiSpeakerScore: Number.isFinite(parsed.multiSpeakerScore)
        ? parsed.multiSpeakerScore
        : null,
    };
  } catch (err) {
    return { ok: false, error: err };
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
};

module.exports = {
  inferDeepScore,
};
