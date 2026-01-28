const http = require("http");
const config = require("./config");
const { handleVoiceDetection } = require("./api/voice_detection");

const PORT = process.env.PORT || 3000;

const allowedOrigins = new Set([
  "http://localhost:5173",
  "http://127.0.0.1:5173",
]);

const applyCors = (req, res) => {
  const origin = req.headers.origin;
  if (origin && allowedOrigins.has(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
  }
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, x-api-key");
  res.setHeader("Access-Control-Max-Age", "600");
};

const sendNotFound = (res) => {
  res.statusCode = 404;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ status: "error", message: "Invalid request or unsupported audio." }));
};

const server = http.createServer((req, res) => {
  applyCors(req, res);

  if (req.method === "OPTIONS") {
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ status: "ok" }));
    return;
  }

  if (req.method !== "POST" || req.url !== "/api/voice-detection") {
    sendNotFound(res);
    return;
  }

  let bodySize = 0;
  const chunks = [];
  req.on("data", (chunk) => {
    chunks.push(chunk);
    bodySize += chunk.length;
    if (bodySize > config.limits.maxBodyBytes) {
      res.statusCode = 413;
      res.setHeader("Content-Type", "application/json");
      res.end(
        JSON.stringify({ status: "error", message: "Invalid request or unsupported audio." })
      );
      req.destroy();
    }
  });

  req.on("end", () => {
    let payload = null;
    try {
      const body = Buffer.concat(chunks).toString("utf8");
      payload = body ? JSON.parse(body) : null;
    } catch (err) {
      res.statusCode = 400;
      res.setHeader("Content-Type", "application/json");
      res.end(
        JSON.stringify({ status: "error", message: "Invalid request or unsupported audio." })
      );
      return;
    }

    handleVoiceDetection(req, res, payload, config).catch(() => {
      res.statusCode = 500;
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ status: "error", message: "Processing failed. Please retry." }));
    });
  });
});

server.listen(PORT, () => {
  console.log(`Voice detection API listening on port ${PORT}`);
});
