const fs = require("fs");
const path = require("path");

const LOG_PATH = process.env.VOICE_DETECT_LOG_PATH || path.join(__dirname, "..", "logs", "voice-ai-detection.log");

const writeLine = (payload) => {
  const line = `${JSON.stringify(payload)}\n`;
  fs.appendFile(LOG_PATH, line, (err) => {
    if (err) {
      // Best-effort logging; avoid crashing on log failures.
      console.error("Failed to write log", err.message || err);
    }
  });
};

const logDetection = (entry) => {
  const payload = {
    ts: new Date().toISOString(),
    event: "voice_detection",
    ...entry,
  };
  writeLine(payload);
};

module.exports = {
  logDetection,
};
