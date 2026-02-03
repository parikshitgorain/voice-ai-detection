// Admin authentication and management module
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

// Paths
const DATA_DIR = path.join(__dirname, "..", "..", "data");
const ADMIN_FILE = path.join(DATA_DIR, "admin.json");
const API_KEYS_FILE = path.join(DATA_DIR, "api_keys.json");
const USAGE_FILE = path.join(DATA_DIR, "usage.json");

// JWT secret (in production, use environment variable)
const JWT_SECRET = process.env.JWT_SECRET || crypto.randomBytes(32).toString("hex");

// Read JSON file safely
const readJSON = (filePath) => {
  try {
    const data = fs.readFileSync(filePath, "utf8");
    return JSON.parse(data);
  } catch (err) {
    return null;
  }
};

// Write JSON file safely
const writeJSON = (filePath, data) => {
  try {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf8");
    return true;
  } catch (err) {
    console.error("Error writing JSON:", err);
    return false;
  }
};

// Hash API key for storage
const hashApiKey = (key) => {
  return crypto.createHash("sha256").update(key).digest("hex");
};

// Generate random API key
const generateApiKey = () => {
  return "sk_" + crypto.randomBytes(24).toString("hex");
};

// Verify admin credentials
const verifyAdmin = async (username, password) => {
  const admin = readJSON(ADMIN_FILE);
  if (!admin || admin.username !== username) {
    return false;
  }
  
  try {
    return await bcrypt.compare(password, admin.password_hash);
  } catch (err) {
    return false;
  }
};

// Create JWT token
const createToken = (username) => {
  return jwt.sign({ username, role: "admin" }, JWT_SECRET, { expiresIn: "24h" });
};

// Verify JWT token
const verifyToken = (token) => {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch (err) {
    return null;
  }
};

// Middleware: Require admin authentication
const requireAdmin = (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    res.writeHead(401, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Unauthorized" }));
    return;
  }
  
  const token = authHeader.substring(7);
  const decoded = verifyToken(token);
  
  if (!decoded || decoded.role !== "admin") {
    res.writeHead(401, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Unauthorized" }));
    return;
  }
  
  req.admin = decoded;
  next();
};

// Get all API keys (masked)
const getApiKeys = () => {
  const data = readJSON(API_KEYS_FILE);
  if (!data || !data.keys) return [];
  
  const usage = readJSON(USAGE_FILE) || {};
  
  return data.keys.map(key => ({
    id: key.id,
    preview: key.preview,
    status: key.status,
    created_at: key.created_at,
    usage: usage[key.id] || { total_requests: 0, today_requests: 0, last_used: null }
  }));
};

// Create new API key with limits
// FIX: Added limit fields (type, daily_limit, per_minute_limit, total_limit)
const createApiKey = (limits = {}) => {
  const rawKey = generateApiKey();
  const keyHash = hashApiKey(rawKey);
  const keyId = "key_" + crypto.randomBytes(8).toString("hex");
  const preview = rawKey.slice(-8);
  
  const data = readJSON(API_KEYS_FILE) || { keys: [] };
  
  data.keys.push({
    id: keyId,
    hash: keyHash,
    preview: preview,
    status: "active",
    created_at: new Date().toISOString(),
    limits: {
      type: limits.type || "unlimited", // "unlimited" or "limited"
      daily_limit: limits.daily_limit || null,
      per_minute_limit: limits.per_minute_limit || null,
      total_limit: limits.total_limit || null
    }
  });
  
  writeJSON(API_KEYS_FILE, data);
  
  // Initialize usage tracking with minute tracking
  const usage = readJSON(USAGE_FILE) || {};
  usage[keyId] = {
    total_requests: 0,
    today_requests: 0,
    minute_requests: 0,
    last_used: null,
    last_minute_reset: new Date().toISOString()
  };
  writeJSON(USAGE_FILE, usage);
  
  // Return raw key only once - SECURITY: This is the only time raw key is exposed
  return {
    id: keyId,
    key: rawKey,
    preview: preview
  };
};

// Update API key status
const updateApiKeyStatus = (keyId, status) => {
  const data = readJSON(API_KEYS_FILE);
  if (!data || !data.keys) return false;
  
  const key = data.keys.find(k => k.id === keyId);
  if (!key) return false;
  
  key.status = status;
  return writeJSON(API_KEYS_FILE, data);
};

// Update API key limits
// FIX: Allow updating limits for existing API keys
const updateApiKeyLimits = (keyId, limits) => {
  const data = readJSON(API_KEYS_FILE);
  if (!data || !data.keys) return false;
  
  const key = data.keys.find(k => k.id === keyId);
  if (!key) return false;
  
  // Update limits
  key.limits = {
    type: limits.type || key.limits?.type || "unlimited",
    daily_limit: limits.daily_limit !== undefined ? limits.daily_limit : (key.limits?.daily_limit || null),
    per_minute_limit: limits.per_minute_limit !== undefined ? limits.per_minute_limit : (key.limits?.per_minute_limit || null),
    total_limit: limits.total_limit !== undefined ? limits.total_limit : (key.limits?.total_limit || null)
  };
  
  return writeJSON(API_KEYS_FILE, data);
};

// Delete API key
const deleteApiKey = (keyId) => {
  const data = readJSON(API_KEYS_FILE);
  if (!data || !data.keys) return false;
  
  data.keys = data.keys.filter(k => k.id !== keyId);
  const success = writeJSON(API_KEYS_FILE, data);
  
  // Clean up usage data
  if (success) {
    const usage = readJSON(USAGE_FILE) || {};
    delete usage[keyId];
    writeJSON(USAGE_FILE, usage);
  }
  
  return success;
};

// Get dashboard stats
const getDashboardStats = () => {
  const keys = readJSON(API_KEYS_FILE) || { keys: [] };
  const usage = readJSON(USAGE_FILE) || {};
  
  let totalCalls = 0;
  let todayCalls = 0;
  
  Object.values(usage).forEach(u => {
    totalCalls += u.total_requests || 0;
    todayCalls += u.today_requests || 0;
  });
  
  return {
    total_keys: keys.keys.length,
    active_keys: keys.keys.filter(k => k.status === "active").length,
    total_calls: totalCalls,
    today_calls: todayCalls
  };
};

// Validate and track API key usage with limit enforcement
// FIX: Added per-minute tracking and limit enforcement
const validateAndTrackApiKey = (apiKey) => {
  if (!apiKey) return { valid: false, error: "No API key provided" };
  
  const keyHash = hashApiKey(apiKey);
  const data = readJSON(API_KEYS_FILE);
  
  if (!data || !data.keys) return { valid: false, error: "Invalid API key" };
  
  const key = data.keys.find(k => k.hash === keyHash && k.status === "active");
  if (!key) return { valid: false, error: "Invalid or inactive API key" };
  
  // Load usage data
  const usage = readJSON(USAGE_FILE) || {};
  if (!usage[key.id]) {
    usage[key.id] = { 
      total_requests: 0, 
      today_requests: 0, 
      minute_requests: 0,
      last_used: null,
      last_minute_reset: new Date().toISOString()
    };
  }
  
  const now = new Date();
  const lastMinuteReset = new Date(usage[key.id].last_minute_reset || now);
  
  // Reset minute counter if 60 seconds passed
  if (now - lastMinuteReset >= 60000) {
    usage[key.id].minute_requests = 0;
    usage[key.id].last_minute_reset = now.toISOString();
  }
  
  // ENFORCE LIMITS - Only for "limited" type keys
  if (key.limits && key.limits.type === "limited") {
    // Check total limit
    if (key.limits.total_limit !== null && usage[key.id].total_requests >= key.limits.total_limit) {
      return { valid: false, error: "Total request limit exceeded", code: 429 };
    }
    
    // Check daily limit
    if (key.limits.daily_limit !== null && usage[key.id].today_requests >= key.limits.daily_limit) {
      return { valid: false, error: "Daily request limit exceeded", code: 429 };
    }
    
    // Check per-minute limit
    if (key.limits.per_minute_limit !== null && usage[key.id].minute_requests >= key.limits.per_minute_limit) {
      return { valid: false, error: "Rate limit exceeded - too many requests per minute", code: 429 };
    }
  }
  
  // Update usage counters
  usage[key.id].total_requests += 1;
  usage[key.id].today_requests += 1;
  usage[key.id].minute_requests += 1;
  usage[key.id].last_used = now.toISOString();
  
  writeJSON(USAGE_FILE, usage);
  
  return { valid: true, keyId: key.id };
};

// Reset daily counters (call this daily via cron)
const resetDailyCounters = () => {
  const usage = readJSON(USAGE_FILE) || {};
  
  Object.keys(usage).forEach(keyId => {
    usage[keyId].today_requests = 0;
  });
  
  writeJSON(USAGE_FILE, usage);
};

// Change admin password
// FIX: Added password change functionality with bcrypt
const changeAdminPassword = async (currentPassword, newPassword) => {
  const admin = readJSON(ADMIN_FILE);
  if (!admin) {
    return { success: false, error: "Admin data not found" };
  }
  
  // Verify current password
  try {
    const isValid = await bcrypt.compare(currentPassword, admin.password_hash);
    if (!isValid) {
      return { success: false, error: "Current password is incorrect" };
    }
  } catch (err) {
    return { success: false, error: "Password verification failed" };
  }
  
  // Hash new password
  try {
    const newHash = await bcrypt.hash(newPassword, 12);
    admin.password_hash = newHash;
    
    // Write safely
    const success = writeJSON(ADMIN_FILE, admin);
    if (!success) {
      return { success: false, error: "Failed to save new password" };
    }
    
    return { success: true };
  } catch (err) {
    return { success: false, error: "Failed to hash new password" };
  }
};

module.exports = {
  verifyAdmin,
  createToken,
  verifyToken,
  requireAdmin,
  getApiKeys,
  createApiKey,
  updateApiKeyStatus,
  updateApiKeyLimits,
  deleteApiKey,
  getDashboardStats,
  validateAndTrackApiKey,
  resetDailyCounters,
  changeAdminPassword
};
