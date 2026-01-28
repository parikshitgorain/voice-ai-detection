const sanitizeBase64 = (value) => value.trim().replace(/\s+/g, "");

const isValidBase64 = (value) => {
  if (!value || value.length % 4 !== 0) return false;
  return /^[A-Za-z0-9+/=]+$/.test(value);
};

const decodeBase64ToBuffer = (audioBase64, maxBytes) => {
  if (!audioBase64 || typeof audioBase64 !== "string") {
    return { ok: false, error: { code: "INVALID_BASE64", message: "audioBase64 is required." } };
  }

  const sanitized = sanitizeBase64(audioBase64);
  if (!isValidBase64(sanitized)) {
    return { ok: false, error: { code: "INVALID_BASE64", message: "Malformed base64 payload." } };
  }

  try {
    const buffer = Buffer.from(sanitized, "base64");
    if (!buffer.length) {
      return { ok: false, error: { code: "EMPTY_AUDIO", message: "Decoded audio is empty." } };
    }
    if (maxBytes && buffer.length > maxBytes) {
      return { ok: false, error: { code: "FILE_TOO_LARGE", message: "File size exceeds 50 MB limit." } };
    }
    return { ok: true, buffer };
  } catch (err) {
    return { ok: false, error: { code: "BASE64_DECODE_FAILED", message: "Base64 decode failed." } };
  }
};

const decodeMp3ToPcm = async (mp3Buffer, decoder) => {
  if (!decoder) {
    return {
      ok: false,
      error: { code: "DECODER_MISSING", message: "MP3 decoder is not configured." },
    };
  }

  try {
    const decoded = await decoder(mp3Buffer);
    if (!decoded || !decoded.pcm || !decoded.sampleRate) {
      return {
        ok: false,
        error: { code: "DECODE_FAILED", message: "MP3 decode returned invalid data." },
      };
    }
    return { ok: true, pcm: decoded.pcm, sampleRate: decoded.sampleRate };
  } catch (err) {
    return { ok: false, error: { code: "DECODE_FAILED", message: "MP3 decode failed." } };
  }
};

const computeDurationSeconds = (pcmLength, sampleRate) => {
  if (!sampleRate || sampleRate <= 0) return 0;
  return pcmLength / sampleRate;
};

const loadAudioFromBase64 = async (audioBase64, decoder, maxBytes) => {
  const base64Result = decodeBase64ToBuffer(audioBase64, maxBytes);
  if (!base64Result.ok) return base64Result;

  const decodeResult = await decodeMp3ToPcm(base64Result.buffer, decoder);
  if (!decodeResult.ok) return decodeResult;

  const duration = computeDurationSeconds(decodeResult.pcm.length, decodeResult.sampleRate);
  return {
    ok: true,
    pcm: decodeResult.pcm,
    sampleRate: decodeResult.sampleRate,
    duration,
  };
};

module.exports = {
  decodeBase64ToBuffer,
  decodeMp3ToPcm,
  computeDurationSeconds,
  loadAudioFromBase64,
};
