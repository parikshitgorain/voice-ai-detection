/**
 * GPU Helper: Auto-detect available device (GPU/CPU) with fallback
 */
const { execSync } = require('child_process');
const path = require('path');

/**
 * Detect optimal device for PyTorch inference
 * @param {string} pythonPath - Path to Python executable
 * @returns {string} - 'cuda' or 'cpu'
 */
function detectDevice(pythonPath = 'python3') {
  try {
    const scriptPath = path.join(__dirname, '..', 'deep', 'detect_device.py');
    const result = execSync(`"${pythonPath}" "${scriptPath}"`, {
      encoding: 'utf8',
      timeout: 5000,
      stdio: ['pipe', 'pipe', 'pipe']
    }).trim();
    
    const device = result.split('\n')[0].trim();
    return device === 'cuda' ? 'cuda' : 'cpu';
  } catch (error) {
    // If detection fails, safely fall back to CPU
    console.warn('[GPU Helper] Device detection failed, using CPU:', error.message);
    return 'cpu';
  }
}

/**
 * Get device with environment variable override
 * @param {string} pythonPath - Path to Python executable
 * @returns {string} - Final device to use
 */
function getDevice(pythonPath = 'python3') {
  // If explicitly set in environment, use that
  const envDevice = process.env.DEEP_MODEL_DEVICE;
  if (envDevice) {
    const device = envDevice.toLowerCase();
    if (device === 'cuda' || device === 'gpu') {
      return 'cuda';
    }
    if (device === 'cpu') {
      return 'cpu';
    }
    if (device === 'auto') {
      return detectDevice(pythonPath);
    }
  }
  
  // Auto-detect by default
  return detectDevice(pythonPath);
}

module.exports = {
  detectDevice,
  getDevice
};
