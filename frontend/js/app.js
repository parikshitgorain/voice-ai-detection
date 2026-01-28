const form = document.getElementById("detect-form");
const languageSelect = document.getElementById("language");
const fileInput = document.getElementById("audio-file");
const base64Input = document.getElementById("base64");
const detectBtn = document.getElementById("detect-btn");
const loading = document.getElementById("loading");
const loadingText = document.getElementById("loading-text");
const errorEl = document.getElementById("error");
const languageWarningEl = document.getElementById("language-warning");

const API_KEY = "change-me";
const API_BASE_URL = "http://localhost:3000";
const MAX_FILE_BYTES = 50 * 1024 * 1024;
const MIN_DURATION = 10;
const MAX_DURATION = 300;

const classificationEl = document.getElementById("classification");
const confidenceTextEl = document.getElementById("confidence-text");
const confidenceBarEl = document.getElementById("confidence-bar");
const explanationEl = document.getElementById("explanation");
const rawResponseEl = document.getElementById("raw-response");
const allowedMessages = [
  "Audio could not be processed. Please try another file.",
  "Invalid request or unsupported audio.",
  "Processing failed. Please retry.",
];

const loadingMessages = ["Uploading audio", "Analyzing voice patterns", "Generating decision"];
let loadingTimers = [];

const clearLoadingTimers = () => {
  loadingTimers.forEach((timer) => clearTimeout(timer));
  loadingTimers = [];
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
  if (!message) return "Processing failed. Please retry.";
  const normalized = message.toLowerCase();
  if (normalized.includes("decode") || normalized.includes("unsupported format")) {
    return "Audio could not be processed. Please try another file.";
  }
  if (normalized.includes("invalid request") || normalized.includes("unsupported audio")) {
    return "Invalid request or unsupported audio.";
  }
  return "Processing failed. Please retry.";
};

const validateForm = () => {
  if (!languageSelect.value) return "Select a language.";
  if (!fileInput.files || fileInput.files.length !== 1) return "Attach exactly one MP3 file.";
  const file = fileInput.files[0];
  if (!file.name.toLowerCase().endsWith(".mp3")) return "File must be an MP3.";
  if (file.size > MAX_FILE_BYTES) return "File exceeds 50 MB limit.";
  if (base64Input.value && base64Input.value.length < 20) return "Base64 payload looks incomplete.";
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
    const file = fileInput.files[0];
    const duration = await getAudioDuration(file);
    if (Number.isFinite(duration) && duration < MIN_DURATION) {
      throw new Error("Audio duration must be at least 10 seconds.");
    }
    if (Number.isFinite(duration) && duration > MAX_DURATION) {
      throw new Error("Audio duration must be under 5 minutes.");
    }

    const audioBase64 = base64Input.value.trim()
      ? base64Input.value.trim()
      : await readFileAsBase64(file);

    const response = await fetch(`${API_BASE_URL}/api/voice-detection`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
      },
      body: JSON.stringify({
        language: languageSelect.value,
        audioFormat: "mp3",
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
    errorEl.textContent = "Processing failed. Please retry.";
  } finally {
    clearInputCache();
    setLoading(false);
  }
});

resetOutput();
