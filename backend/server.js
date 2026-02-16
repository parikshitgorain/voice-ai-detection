const http = require("http");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const config = require("./config");
const { handleVoiceDetection } = require("./api/voice_detection");
const { QueueFullError, createRequestQueue } = require("./utils/request_queue");
const { createRateLimiter } = require("./utils/rate_limiter");
const { isValidApiKey } = require("./utils/authentication");
const { getClientIp } = require("./utils/client_ip");
const { logDetection } = require("./utils/logger");
const { adminRouter } = require("./api/admin");

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || "127.0.0.1";
const STRICT_ERROR_MESSAGE = "Invalid API key or malformed request";
const queue = createRequestQueue(config.queue || {});
const rateLimiter = createRateLimiter(config.rateLimit || {});

const buildAllowedOrigins = () => {
  const envValue = process.env.CORS_ORIGINS;
  if (envValue) {
    return new Set(
      envValue
        .split(",")
        .map((origin) => origin.trim())
        .filter(Boolean)
    );
  }
  if (process.env.NODE_ENV !== "production") {
    return new Set(["http://localhost:5173", "http://127.0.0.1:5173"]);
  }
  return new Set();
};

const allowedOrigins = buildAllowedOrigins();

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

const sendJson = (res, statusCode, payload) => {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload));
};

const sendNotFound = (res) => {
  sendJson(res, 404, { status: "error", message: "Not Found" });
};

// Serve static frontend files
const serveFrontend = (req, res) => {
  const frontendDir = path.join(__dirname, "..", "frontend");
  let filePath = path.join(frontendDir, req.url === "/" ? "index.html" : req.url);
  
  // Security: prevent directory traversal
  if (!filePath.startsWith(frontendDir)) {
    sendNotFound(res);
    return false;
  }
  
  // Check if file exists
  if (!fs.existsSync(filePath)) {
    return false;
  }
  
  // Determine content type
  const ext = path.extname(filePath);
  const contentTypes = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml"
  };
  
  const contentType = contentTypes[ext] || "text/plain";
  
  fs.readFile(filePath, (err, data) => {
    if (err) {
      sendNotFound(res);
      return;
    }
    res.writeHead(200, { "Content-Type": contentType });
    res.end(data);
  });
  
  return true;
};

const server = http.createServer((req, res) => {
  if (adminRouter(req, res)) { return; }
  applyCors(req, res);

  if (req.method === "OPTIONS") {
    sendJson(res, 200, { status: "ok" });
    return;
  }

  if (req.method === "GET" && req.url === "/health") {
    sendJson(res, 200, { status: "ok" });
    return;
  }
  
  // Serve frontend files for GET requests
  if (req.method === "GET" && !req.url.startsWith("/api/")) {
    if (serveFrontend(req, res)) {
      return;
    }
  }

  if (req.method === "GET" && req.url === "/api/queue") {
    const authOk = isValidApiKey(req.headers, config);
    const rateKey = `${getClientIp(req)}:${authOk ? "auth" : "anon"}`;
    if (!rateLimiter.allow(rateKey)) {
      if (!authOk) {
        sendNotFound(res);
        return;
      }
      sendJson(res, 429, { status: "error", message: "Too many requests. Please wait." });
      return;
    }
    if (!authOk) {
      sendNotFound(res);
      return;
    }
    sendJson(res, 200, { status: "ok", ...queue.getStats() });
    return;
  }

  if (req.method !== "POST" || req.url !== "/api/voice-detection") {
    sendNotFound(res);
    return;
  }

  const requestId = crypto.randomUUID();
  const clientIp = getClientIp(req);
  req.requestId = requestId;
  req.clientIp = clientIp;
  req.requestReceivedAt = Date.now();

  const authOk = isValidApiKey(req.headers, config);
  const rateKey = `${clientIp}:${authOk ? "auth" : "anon"}`;
  if (!rateLimiter.allow(rateKey)) {
    if (!authOk) {
      sendNotFound(res);
    } else {
      sendJson(res, 429, { status: "error", message: "Too many requests. Please wait." });
    }
    logDetection({
      requestId,
      ip: clientIp,
      status: "error",
      reason: "RATE_LIMIT",
    });
    return;
  }

  if (!authOk) {
    sendNotFound(res);
    logDetection({
      requestId,
      ip: clientIp,
      status: "error",
      reason: "INVALID_API_KEY",
    });
    return;
  }

  let bodySize = 0;
  const chunks = [];
  req.on("data", (chunk) => {
    chunks.push(chunk);
    bodySize += chunk.length;
    if (bodySize > config.limits.maxBodyBytes) {
      sendJson(res, 413, { status: "error", message: STRICT_ERROR_MESSAGE });
      logDetection({
        requestId,
        ip: clientIp,
        status: "error",
        reason: "BODY_TOO_LARGE",
      });
      req.destroy();
    }
  });

  req.on("end", () => {
    let payload = null;
    try {
      const body = Buffer.concat(chunks).toString("utf8");
      payload = body ? JSON.parse(body) : null;
    } catch (err) {
      sendJson(res, 400, { status: "error", message: STRICT_ERROR_MESSAGE });
      logDetection({
        requestId,
        ip: clientIp,
        status: "error",
        reason: "INVALID_JSON",
      });
      return;
    }

    const task = () =>
      handleVoiceDetection(req, res, payload, config).catch((err) => {
        if (!res.headersSent) {
          sendJson(res, 500, { status: "error", message: "Processing failed. Please retry." });
        }
        logDetection({
          requestId,
          ip: clientIp,
          status: "error",
          reason: "UNHANDLED_ERROR",
          detail: err && err.message ? err.message : "unknown",
        });
      });

    try {
      queue.enqueue({ id: requestId, task, req, onError: (err) => {
        if (!res.headersSent) {
          sendJson(res, 500, { status: "error", message: "Processing failed. Please retry." });
        }
        logDetection({
          requestId,
          ip: clientIp,
          status: "error",
          reason: "QUEUE_TASK_ERROR",
          detail: err && err.message ? err.message : "unknown",
        });
      }});
    } catch (err) {
      if (err instanceof QueueFullError || err.code === "QUEUE_FULL") {
        const queueStats = queue.getStats();
        sendJson(res, 503, {
          status: "error",
          message: `Queue is full. Please wait. Users in queue: ${queueStats.queued}.`,
        });
        logDetection({
          requestId,
          ip: clientIp,
          status: "error",
          reason: "QUEUE_FULL",
          queued: queueStats.queued,
          active: queueStats.active,
        });
        return;
      }

      sendJson(res, 500, { status: "error", message: "Processing failed. Please retry." });
      logDetection({
        requestId,
        ip: clientIp,
        status: "error",
        reason: "QUEUE_INIT_ERROR",
        detail: err && err.message ? err.message : "unknown",
      });
    }
  });
});


server.listen(PORT, HOST, () => {
  console.log(`✅ Voice detection API listening on ${HOST}:${PORT}`);
  console.log(`✅ Process ID: ${process.pid}`);
  console.log(`✅ Node version: ${process.version}`);
  console.log(`✅ Platform: ${process.platform}`);
  console.log(`✅ Memory usage: ${Math.round(process.memoryUsage().heapUsed / 1024 / 1024)}MB`);
  // scheduleDailyReset(); // DISABLED - function not defined
}).on('error', (err) => {
  console.error('❌ Failed to start server:', err);
  if (err.code === 'EADDRINUSE') {
    console.error(`❌ Port ${PORT} is already in use`);
  }
  process.exit(1);
});

setInterval(() => {
  const mem = process.memoryUsage();
  console.log(`💓 Health: Memory=${Math.round(mem.heapUsed/1024/1024)}MB, Uptime=${Math.round(process.uptime())}s`);
}, 300000);
