const crypto = require("crypto");
const fs = require("fs/promises");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");
const { getServer, startServer } = require("./persistent_server");

const runPython = (pythonPath, scriptPath, args, timeoutMs = 30000) =>
  new Promise((resolve, reject) => {
    let proc;
    try {
      proc = spawn(pythonPath, [scriptPath, ...args], {
        stdio: ["ignore", "pipe", "pipe"],
        detached: false,
      });
    } catch (spawnErr) {
      return reject(new Error(`Failed to spawn Python process: ${spawnErr.message}`));
    }

    const stdout = [];
    const stderr = [];
    let finished = false;

    const cleanup = () => {
      if (proc && !proc.killed) {
        try {
          proc.kill("SIGKILL");
        } catch (killErr) {
          console.error('Error killing Python process:', killErr);
        }
      }
    };

    const timer = setTimeout(() => {
      if (finished) return;
      finished = true;
      cleanup();
      reject(new Error("Deep model inference timed out."));
    }, timeoutMs);

    proc.stdout.on("data", (chunk) => stdout.push(chunk));
    proc.stderr.on("data", (chunk) => stderr.push(chunk));
    
    proc.on("error", (err) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      cleanup();
      reject(new Error(`Python process error: ${err.message}`));
    });
    
    proc.on("close", (code, signal) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      
      if (signal) {
        reject(new Error(`Python process killed by signal ${signal}`));
        return;
      }
      
      if (code !== 0) {
        const errMsg = Buffer.concat(stderr).toString("utf8").trim();
        reject(new Error(errMsg || `Deep model inference failed with code ${code}`));
        return;
      }
      
      try {
        const output = Buffer.concat(stdout).toString("utf8").trim();
        if (!output) {
          reject(new Error('Python process returned empty output'));
          return;
        }
        resolve(output);
      } catch (bufferErr) {
        reject(new Error(`Failed to read Python output: ${bufferErr.message}`));
      }
    });
  });

// Initialize persistent server on first use
let serverInitialized = false;
let usePersistentServer = process.env.USE_PERSISTENT_SERVER !== 'false'; // Default true

const inferDeepScore = async (audioBase64, config, selectedLanguage = null, audioFormat = null) => {
  let tempDir = null;
  let tempFile = null;

  try {
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
    
    // Initialize persistent server on first request
    if (usePersistentServer && !serverInitialized) {
      try {
        const pythonPath = settings.pythonPath || "python3";
        const device = settings.device || "cpu";
        await startServer({ pythonPath, device });
        serverInitialized = true;
        console.log('[DeepModel] Persistent server initialized');
      } catch (err) {
        console.error('[DeepModel] Failed to start persistent server, falling back to spawn mode:', err.message);
        usePersistentServer = false;
      }
    }

    // Validate inputs
    if (!audioBase64 || typeof audioBase64 !== 'string') {
      return {
        ok: false,
        error: { code: 'INVALID_AUDIO', message: 'Invalid audio data provided.' },
      };
    }

    const pythonPath = settings.pythonPath || "python3";
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
          error: { code: 'LANGUAGE_REQUIRED', message: 'Language is required when using per-language models.' },
        };
      }
      const mapped = settings.modelByLanguage[selectedLanguage];
      if (!mapped) {
        return {
          ok: false,
          error: { code: 'UNSUPPORTED_LANGUAGE', message: `No model configured for language "${selectedLanguage}".` },
        };
      }
      modelPath = mapped;
    }
    
    const device = settings.device || "cpu";
    const timeoutMs = settings.timeoutMs || 30000;

    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "voice-deep-"));
    const safeFormat =
      typeof audioFormat === "string" ? audioFormat.trim().toLowerCase() : null;
    const ext =
      safeFormat && safeFormat !== "auto" && /^[a-z0-9]+$/.test(safeFormat)
        ? `.${safeFormat}`
        : ".mp3";
    tempFile = path.join(tempDir, `audio-${crypto.randomBytes(8).toString("hex")}${ext}`);

    // Validate and write audio file
    let buffer;
    try {
      buffer = Buffer.from(audioBase64, "base64");
      if (buffer.length === 0) {
        throw new Error('Empty audio buffer');
      }
    } catch (decodeErr) {
      return {
        ok: false,
        error: { code: 'INVALID_BASE64', message: 'Failed to decode audio data.' },
      };
    }

    await fs.writeFile(tempFile, buffer);

    // Try persistent server first for faster inference
    if (usePersistentServer && serverInitialized) {
      try {
        const server = getServer();
        if (server.isReady()) {
          const result = await server.infer(tempFile, modelPath, timeoutMs);
          
          if (result.error) {
            throw new Error(result.error);
          }
          
          // Extract language info from result
          let detectedLanguage = null;
          let languageConfidence = null;
          let languageDistribution = result.languageDistribution || null;
          
          if (languageDistribution) {
            const entries = Object.entries(languageDistribution).filter(([, v]) => Number.isFinite(v));
            if (entries.length) {
              entries.sort((a, b) => b[1] - a[1]);
              [detectedLanguage, languageConfidence] = entries[0];
            }
          }
          
          return {
            ok: true,
            score: result.aiScore,
            detectedLanguage,
            languageConfidence,
            languageDistribution,
            multiSpeakerScore: result.multiSpeakerScore || null,
          };
        }
      } catch (serverErr) {
        console.error('[DeepModel] Persistent server failed, falling back to spawn:', serverErr.message);
        // Fall through to spawn mode
      }
    }

    // Fallback to spawn mode (original implementation)
    let detectedLanguage = null;
    let languageConfidence = null;
    let languageDistribution = null;
    const languageDetector = settings.languageDetector || {};
    if (languageDetector.enabled && languageDetector.modelPath) {
      try {
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
      } catch (langErr) {
        console.error('Language detection failed:', langErr.message);
        // Continue without language detection
      }
    }

    const output = await runPython(
      pythonPath,
      scriptPath,
      ["--audio", tempFile, "--model", modelPath, "--device", device],
      timeoutMs
    );

    let parsed;
    try {
      parsed = JSON.parse(output);
    } catch (parseErr) {
      console.error('Failed to parse Python output:', output);
      return { 
        ok: false, 
        error: { code: 'INVALID_OUTPUT', message: 'Deep model output invalid.' } 
      };
    }

    if (!parsed || typeof parsed !== "object") {
      return { 
        ok: false, 
        error: { code: 'INVALID_OUTPUT', message: 'Deep model output invalid.' } 
      };
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
      return { 
        ok: false, 
        error: { code: 'INVALID_SCORE', message: 'Deep model output invalid.' } 
      };
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
    console.error('Deep model error:', err.message);
    return { 
      ok: false, 
      error: { code: 'DEEP_MODEL_ERROR', message: err.message || 'Deep model failed.' } 
    };
  } finally {
    // Cleanup temp files
    if (tempDir) {
      try {
        await fs.rm(tempDir, { recursive: true, force: true });
      } catch (cleanupErr) {
        console.error('Failed to cleanup temp directory:', cleanupErr);
      }
    }
  }
};

module.exports = {
  inferDeepScore,
};
