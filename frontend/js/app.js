// DOM Elements
const form = document.getElementById("detect-form");
const languageSelect = document.getElementById("language");
const apiKeyInput = document.getElementById("api-key");
const fileInput = document.getElementById("audio-file");
const base64Input = document.getElementById("base64");
const detectBtn = document.getElementById("detect-btn");
const loading = document.getElementById("loading");
const loadingText = document.getElementById("loading-text");
const errorEl = document.getElementById("error");
const languageWarningEl = document.getElementById("language-warning");
const classificationEl = document.getElementById("classification");
const confidenceTextEl = document.getElementById("confidence-text");
const confidenceBarEl = document.getElementById("confidence-bar");
const explanationEl = document.getElementById("explanation");
const rawResponseEl = document.getElementById("raw-response");

// Configuration
const CONFIG = window.VOICE_AI_CONFIG || {};
const DEFAULT_API_KEY = CONFIG.apiKey || "";
const EFFECTIVE_DEFAULT_KEY = DEFAULT_API_KEY && DEFAULT_API_KEY !== "change-me" ? DEFAULT_API_KEY : "";
const API_BASE_URL = CONFIG.apiBaseUrl || "";
const MAX_FILE_BYTES = 50 * 1024 * 1024;
const MIN_DURATION = 2;
const MAX_DURATION = 300;
const STRICT_ERROR_MESSAGE = "Invalid API key or malformed request";

// AI Loading Messages
const AI_LOADING_STEPS = [
  "Extracting audio features",
  "Analyzing voice patterns",
  "Evaluating neural signatures",
  "Computing authenticity score"
];

let loadingStepIndex = 0;
let loadingInterval = null;

// Loading State Management
const cycleLoadingMessage = () => {
  loadingStepIndex = (loadingStepIndex + 1) % AI_LOADING_STEPS.length;
  if (loadingText) {
    loadingText.textContent = AI_LOADING_STEPS[loadingStepIndex];
  }
};

const startLoadingCycle = () => {
  loadingStepIndex = 0;
  if (loadingText) {
    loadingText.textContent = AI_LOADING_STEPS[0];
  }
  loadingInterval = setInterval(cycleLoadingMessage, 1200);
};

const stopLoadingCycle = () => {
  if (loadingInterval) {
    clearInterval(loadingInterval);
    loadingInterval = null;
  }
  loadingStepIndex = 0;
};

const setLoading = (isLoading) => {
  loading.classList.toggle("is-active", isLoading);
  detectBtn.disabled = isLoading;
  
  if (isLoading) {
    startLoadingCycle();
  } else {
    stopLoadingCycle();
  }
};

// Output Management
const resetOutput = () => {
  classificationEl.textContent = "—";
  classificationEl.classList.remove("status-ai", "status-human");
  classificationEl.classList.add("result-value-placeholder");
  
  confidenceTextEl.textContent = "—";
  confidenceBarEl.style.width = "0%";
  
  explanationEl.textContent = "No analysis available";
  
  rawResponseEl.textContent = "No response yet";
  errorEl.textContent = "";
  languageWarningEl.classList.remove("is-active");
};

const updateOutput = (payload) => {
  const isAi = payload.classification === "AI_GENERATED";
  
  // Classification
  classificationEl.textContent = payload.classification.replace("_", " ");
  classificationEl.classList.remove("result-value-placeholder");
  classificationEl.classList.toggle("status-ai", isAi);
  classificationEl.classList.toggle("status-human", !isAi);
  
  // Confidence
  const confidencePct = Math.round(payload.confidenceScore * 100);
  confidenceTextEl.textContent = `${confidencePct}%`;
  
  // Animate confidence bar
  setTimeout(() => {
    confidenceBarEl.style.width = `${confidencePct}%`;
  }, 50);
  
  // Explanation
  explanationEl.textContent = payload.explanation || "Analysis completed successfully";
  
  // Raw response
  rawResponseEl.textContent = JSON.stringify(payload, null, 2);
  
  // Language warning
  if (payload.languageWarning) {
    languageWarningEl.classList.add("is-active");
  } else {
    languageWarningEl.classList.remove("is-active");
  }
};

// Validation
const validateForm = () => {
  if (!languageSelect.value) {
    return "Please select a language";
  }
  
  const hasBase64 = base64Input.value.trim().length > 0;
  const hasFile = fileInput.files && fileInput.files.length === 1;
  const apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
  
  if (!apiKey && !EFFECTIVE_DEFAULT_KEY) {
    return "API key is required";
  }
  
  if (!hasBase64 && !hasFile) {
    return "Please provide audio file or base64 input";
  }
  
  if (fileInput.files && fileInput.files.length > 1) {
    return "Please attach only one audio file";
  }
  
  if (hasFile) {
    const file = fileInput.files[0];
    const name = file.name.toLowerCase();
    
    if (!name.endsWith(".mp3")) {
      return "Only MP3 format is supported";
    }
    
    if (file.size > MAX_FILE_BYTES) {
      return "File size exceeds 50 MB limit";
    }
  }
  
  if (hasBase64 && base64Input.value.trim().length < 20) {
    return "Base64 payload appears incomplete";
  }
  
  return null;
};

// Audio Processing
const getAudioDuration = (file) => {
  return new Promise((resolve, reject) => {
    const audio = new Audio();
    const url = URL.createObjectURL(file);
    audio.preload = "metadata";
    
    audio.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(audio.duration);
    };
    
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Unable to read audio metadata"));
    };
    
    audio.src = url;
  });
};

const readFileAsBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("Base64 conversion failed"));
        return;
      }
      
      const parts = result.split(",");
      resolve(parts.length > 1 ? parts[1] : result);
    };
    
    reader.onerror = () => reject(new Error("File read failed"));
    reader.readAsDataURL(file);
  });
};

// Input Handlers
fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files.length) {
    base64Input.value = "";
  }
});

base64Input.addEventListener("input", () => {
  if (base64Input.value.trim().length) {
    fileInput.value = "";
  }
});

// Form Submission
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  
  const error = validateForm();
  if (error) {
    errorEl.textContent = error;
    return;
  }
  
  setLoading(true);
  resetOutput();
  
  try {
    const file = fileInput.files && fileInput.files.length === 1 ? fileInput.files[0] : null;
    
    // Validate duration
    if (file) {
      const duration = await getAudioDuration(file);
      
      if (Number.isFinite(duration) && duration < MIN_DURATION) {
        throw new Error(`Audio must be at least ${MIN_DURATION} seconds long`);
      }
      
      if (Number.isFinite(duration) && duration > MAX_DURATION) {
        throw new Error("Audio duration must be under 5 minutes");
      }
    }
    
    // Get base64
    const audioBase64 = base64Input.value.trim()
      ? base64Input.value.trim()
      : await readFileAsBase64(file);
    
    // Prepare request
    const apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
    const finalApiKey = apiKey || EFFECTIVE_DEFAULT_KEY;
    
    if (!finalApiKey) {
      throw new Error("API key is required");
    }
    
    // Make API call
    const response = await fetch(`${API_BASE_URL}/api/voice-detection`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": finalApiKey
      },
      body: JSON.stringify({
        language: languageSelect.value,
        audioFormat: "mp3",
        audioBase64
      })
    });
    
    const text = await response.text();
    let payload = null;
    
    if (text && text.trim()) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }
    
    if (!response.ok) {
      const serverMessage = payload && typeof payload.message === "string" 
        ? payload.message 
        : STRICT_ERROR_MESSAGE;
      errorEl.textContent = serverMessage;
      return;
    }
    
    if (!payload || typeof payload !== "object") {
      errorEl.textContent = "Processing failed. Please try again";
      return;
    }
    
    updateOutput(payload);
    
  } catch (err) {
    errorEl.textContent = err.message || STRICT_ERROR_MESSAGE;
  } finally {
    // Clear sensitive inputs
    base64Input.value = "";
    
    setLoading(false);
  }
});

// Initialize
resetOutput();

// Don't prefill API keys for security
if (apiKeyInput) {
  apiKeyInput.value = "";
}
