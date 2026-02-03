// Admin API Routes
const url = require("url");
const fs = require("fs");
const path = require("path");
const adminModule = require("../utils/admin");

const ADMIN_DIR = path.join(__dirname, "..", "..", "admin");

// Serve static admin files
const serveStaticFile = (req, res, filePath) => {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not Found");
      return;
    }
    
    const ext = path.extname(filePath);
    const contentType = {
      ".html": "text/html",
      ".js": "application/javascript",
      ".css": "text/css"
    }[ext] || "text/plain";
    
    res.writeHead(200, { "Content-Type": contentType });
    res.end(data);
  });
};

const parseBody = (req) => {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", chunk => { body += chunk.toString(); });
    req.on("end", () => {
      try { resolve(JSON.parse(body)); }
      catch (err) { reject(err); }
    });
  });
};

const handleLogin = async (req, res) => {
  try {
    const { username, password } = await parseBody(req);
    if (!username || !password) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing credentials" }));
      return;
    }
    const valid = await adminModule.verifyAdmin(username, password);
    if (!valid) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Invalid credentials" }));
      return;
    }
    const token = adminModule.createToken(username);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ token, username }));
  } catch (err) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Internal server error" }));
  }
};

const handleSession = (req, res) => {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ username: req.admin.username }));
};

const handleStats = (req, res) => {
  try {
    const stats = adminModule.getDashboardStats();
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(stats));
  } catch (err) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Internal server error" }));
  }
};

const handleGetKeys = (req, res) => {
  try {
    const keys = adminModule.getApiKeys();
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ keys }));
  } catch (err) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Internal server error" }));
  }
};

const handleCreateKey = async (req, res) => {
  try {
    const result = adminModule.createApiKey();
    res.writeHead(201, { "Content-Type": "application/json" });
    res.end(JSON.stringify(result));
  } catch (err) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Internal server error" }));
  }
};

const handleUpdateKey = async (req, res, keyId) => {
  try {
    const { status } = await parseBody(req);
    if (!status || !["active", "inactive"].includes(status)) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Invalid status" }));
      return;
    }
    const success = adminModule.updateApiKeyStatus(keyId, status);
    if (!success) {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Key not found" }));
      return;
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ success: true }));
  } catch (err) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Internal server error" }));
  }
};

const handleDeleteKey = (req, res, keyId) => {
  try {
    const success = adminModule.deleteApiKey(keyId);
    if (!success) {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Key not found" }));
      return;
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ success: true }));
  } catch (err) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Internal server error" }));
  }
};

const adminRouter = (req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;
  const method = req.method;
  
  // Serve static admin files (HTML, JS, CSS)
  if (pathname.startsWith("/admin/") && method === "GET") {
    const staticFiles = [".html", ".js", ".css"];
    const ext = path.extname(pathname);
    if (staticFiles.includes(ext)) {
      const fileName = path.basename(pathname);
      const filePath = path.join(ADMIN_DIR, fileName);
      serveStaticFile(req, res, filePath);
      return true;
    }
  }
  
  // API: Login (no auth)
  if (pathname === "/admin/login" && method === "POST") {
    handleLogin(req, res);
    return true;
  }
  
  // API: Protected routes
  if (pathname.startsWith("/admin/")) {
    adminModule.requireAdmin(req, res, () => {
      if (pathname === "/admin/session" && method === "GET") {
        handleSession(req, res);
      }
      else if (pathname === "/admin/stats" && method === "GET") {
        handleStats(req, res);
      }
      else if (pathname === "/admin/api-keys" && method === "GET") {
        handleGetKeys(req, res);
      }
      else if (pathname === "/admin/api-keys" && method === "POST") {
        handleCreateKey(req, res);
      }
      else if (pathname.match(/^\/admin\/api-keys\/[^/]+$/) && method === "PATCH") {
        const keyId = pathname.split("/").pop();
        handleUpdateKey(req, res, keyId);
      }
      else if (pathname.match(/^\/admin\/api-keys\/[^/]+$/) && method === "DELETE") {
        const keyId = pathname.split("/").pop();
        handleDeleteKey(req, res, keyId);
      }
      else {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Not found" }));
      }
    });
    return true;
  }
  
  return false;
};

module.exports = { adminRouter };
