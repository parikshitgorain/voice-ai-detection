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

const CONFIG = window.VOICE_AI_CONFIG || {};
const DEFAULT_API_KEY = CONFIG.apiKey || "";
const EFFECTIVE_DEFAULT_KEY = DEFAULT_API_KEY && DEFAULT_API_KEY !== "change-me" ? DEFAULT_API_KEY : "";
const API_KEY_STORAGE = "voiceAiApiKey";
const API_BASE_URL = CONFIG.apiBaseUrl || "";
const MAX_FILE_BYTES = 50 * 1024 * 1024;
const MIN_DURATION = 2;
const MAX_DURATION = 300;

const classificationEl = document.getElementById("classification");
const confidenceTextEl = document.getElementById("confidence-text");
const confidenceBarEl = document.getElementById("confidence-bar");
const explanationEl = document.getElementById("explanation");
const rawResponseEl = document.getElementById("raw-response");
const STRICT_ERROR_MESSAGE = "Invalid API key or malformed request";
const allowedMessages = [STRICT_ERROR_MESSAGE];

const loadingMessages = ["Uploading audio", "Analyzing voice patterns", "Generating decision"];
let loadingTimers = [];

const clearLoadingTimers = () => {
  loadingTimers.forEach((timer) => clearTimeout(timer));
  loadingTimers = [];
};

const loadApiKey = () => {
  if (!apiKeyInput) return;
  const saved = localStorage.getItem(API_KEY_STORAGE);
  if (saved) {
    apiKeyInput.value = saved;
  } else if (EFFECTIVE_DEFAULT_KEY) {
    apiKeyInput.value = EFFECTIVE_DEFAULT_KEY;
  }
};

const persistApiKey = () => {
  if (!apiKeyInput) return;
  const value = apiKeyInput.value.trim();
  if (value) {
    localStorage.setItem(API_KEY_STORAGE, value);
  } else {
    localStorage.removeItem(API_KEY_STORAGE);
  }
};

const startLoadingCycle = () => {
  clearLoadingTimers();
  if (loadingText) loadingText.textContent = loadingMessages[0];
  loadingTimers.push(
    setTimeout(() => {
      if (loadingText) loadingText.textContent = loadingMessages[1];
    }, 400)
  );
  loadingTimers.push(
    setTimeout(() => {
      if (loadingText) loadingText.textContent = loadingMessages[2];
    }, 1400)
  );
};

const setLoading = (isLoading) => {
  loading.classList.toggle("is-active", isLoading);
  detectBtn.disabled = isLoading;
  if (isLoading) {
    startLoadingCycle();
  } else {
    clearLoadingTimers();
    if (loadingText) loadingText.textContent = "Analyzing voice patterns";
  }
};

const resetOutput = () => {
  classificationEl.textContent = "-";
  classificationEl.classList.remove("status-ai", "status-human");
  classificationEl.classList.add("muted");
  confidenceTextEl.textContent = "-";
  confidenceBarEl.style.width = "0%";
  explanationEl.textContent = "-";
  explanationEl.classList.add("muted");
  rawResponseEl.textContent = "No response yet.";
  errorEl.textContent = "";
  languageWarningEl.classList.remove("is-active");
};

const clearInputCache = () => {
  base64Input.value = "";
};

const updateOutput = (payload) => {
  const isAi = payload.classification === "AI_GENERATED";
  classificationEl.textContent = payload.classification;
  classificationEl.classList.remove("muted");
  classificationEl.classList.toggle("status-ai", isAi);
  classificationEl.classList.toggle("status-human", !isAi);

  const confidencePct = Math.round(payload.confidenceScore * 100);
  confidenceTextEl.textContent = `${confidencePct}%`;
  confidenceBarEl.style.width = `${confidencePct}%`;
  explanationEl.textContent = payload.explanation;
  explanationEl.classList.remove("muted");

  rawResponseEl.textContent = JSON.stringify(payload, null, 2);
  if (payload.languageWarning) {
    languageWarningEl.classList.add("is-active");
  } else {
    languageWarningEl.classList.remove("is-active");
  }
};

const mapUserMessage = (message) => {
  if (allowedMessages.includes(message)) return message;
  return STRICT_ERROR_MESSAGE;
};

const validateForm = () => {
  if (!languageSelect.value) return "Select a language.";
  const hasBase64 = base64Input.value.trim().length > 0;
  const hasFile = fileInput.files && fileInput.files.length === 1;
  const apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
  if (!apiKey && !EFFECTIVE_DEFAULT_KEY) {
    return "API key is required. Please enter it.";
  }
  if (!hasBase64 && !hasFile) {
    return "Provide base64 audio or attach one MP3 file.";
  }
  if (fileInput.files && fileInput.files.length > 1) {
    return "Attach exactly one MP3 file.";
  }
  if (hasFile) {
    const file = fileInput.files[0];
    const name = file.name.toLowerCase();
    const allowed = [".mp3"];
    if (!allowed.some((ext) => name.endsWith(ext))) return "Unsupported audio format.";
    if (file.size > MAX_FILE_BYTES) return "File exceeds 50 MB limit.";
  }
  if (hasBase64 && base64Input.value.trim().length < 20) {
    return "Base64 payload looks incomplete.";
  }
  return null;
};

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
      reject(new Error("Unable to read audio metadata."));
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
        reject(new Error("Base64 conversion failed."));
        return;
      }
      const parts = result.split(",");
      resolve(parts.length > 1 ? parts[1] : result);
    };
    reader.onerror = () => reject(new Error("File read failed."));
    reader.readAsDataURL(file);
  });
};

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

if (apiKeyInput) {
  apiKeyInput.addEventListener("input", () => {
    persistApiKey();
  });
}

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
    if (file) {
      const duration = await getAudioDuration(file);
      if (Number.isFinite(duration) && duration < MIN_DURATION) {
        throw new Error(`Audio duration must be at least ${MIN_DURATION} seconds.`);
      }
      if (Number.isFinite(duration) && duration > MAX_DURATION) {
        throw new Error("Audio duration must be under 5 minutes.");
      }
    }

    const audioBase64 = base64Input.value.trim()
      ? base64Input.value.trim()
      : await readFileAsBase64(file);

    const detectAudioFormat = () => "mp3";

    const headers = {
      "Content-Type": "application/json",
    };
    const apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
    const finalApiKey = apiKey || EFFECTIVE_DEFAULT_KEY;
    if (!finalApiKey) {
      errorEl.textContent = "API key is required. Please enter it.";
      return;
    }
    headers["x-api-key"] = finalApiKey;
    const response = await fetch(`${API_BASE_URL}/api/voice-detection`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        language: languageSelect.value,
        audioFormat: detectAudioFormat(file),
        audioBase64,
      }),
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
      const serverMessage = payload && typeof payload.message === "string" ? payload.message : "";
      errorEl.textContent = mapUserMessage(serverMessage);
      return;
    }

    if (!payload || typeof payload !== "object") {
      errorEl.textContent = "Processing failed. Please retry.";
      return;
    }

    updateOutput(payload);
  } catch (err) {
    errorEl.textContent = STRICT_ERROR_MESSAGE;
  } finally {
    clearInputCache();
    setLoading(false);
  }
});

resetOutput();
loadApiKey();
