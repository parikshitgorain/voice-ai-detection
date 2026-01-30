const crypto = require("crypto");
const fs = require("fs/promises");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

const runCommand = (command, args, inputBuffer) =>
  new Promise((resolve, reject) => {
    const proc = spawn(command, args);
    const chunks = [];
    const errors = [];

    if (inputBuffer) {
      proc.stdin.write(inputBuffer);
      proc.stdin.end();
    }

    proc.stdout.on("data", (chunk) => chunks.push(chunk));
    proc.stderr.on("data", (chunk) => errors.push(chunk));
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(Buffer.concat(errors).toString("utf8")));
        return;
      }
      resolve(Buffer.concat(chunks));
    });
  });

const parseRational = (value) => {
  if (!value) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const parts = String(value).split("/");
  if (parts.length === 1) {
    const parsed = Number(parts[0]);
    return Number.isFinite(parsed) ? parsed : null;
  }
  const numerator = Number(parts[0]);
  const denominator = Number(parts[1]);
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) return null;
  return numerator / denominator;
};

const collectTags = (formatTags, streamTags) => {
  const tags = {};
  const addTags = (source) => {
    if (!source) return;
    for (const [key, value] of Object.entries(source)) {
      if (value === undefined || value === null) continue;
      tags[key.toLowerCase()] = String(value);
    }
  };
  addTags(formatTags);
  addTags(streamTags);
  return tags;
};

const probeMp3 = async (filePath) => {
  const args = [
    "-v",
    "error",
    "-select_streams",
    "a:0",
    "-show_entries",
    "stream=sample_rate,channels,codec_name,codec_tag_string,bit_rate,avg_frame_rate,profile:format=bit_rate,format_name,format_long_name,size,duration,probe_score,tags",
    "-of",
    "json",
    filePath,
  ];
  const output = await runCommand("ffprobe", args);
  const parsed = JSON.parse(output.toString("utf8"));
  const stream = parsed.streams && parsed.streams[0] ? parsed.streams[0] : {};
  const format = parsed.format || {};
  const tags = collectTags(format.tags, stream.tags);

  const sampleRate = stream.sample_rate ? Number(stream.sample_rate) : null;
  const channels = stream.channels ? Number(stream.channels) : null;
  const bitRate = stream.bit_rate
    ? Number(stream.bit_rate)
    : format.bit_rate
    ? Number(format.bit_rate)
    : null;

  return {
    sampleRate: Number.isFinite(sampleRate) ? sampleRate : null,
    channels: Number.isFinite(channels) ? channels : null,
    bitRate: Number.isFinite(bitRate) ? bitRate : null,
    duration: format.duration ? Number(format.duration) : null,
    size: format.size ? Number(format.size) : null,
    formatName: format.format_name ?? null,
    formatLongName: format.format_long_name ?? null,
    codecName: stream.codec_name ?? null,
    codecTag: stream.codec_tag_string ?? null,
    avgFrameRate: parseRational(stream.avg_frame_rate),
    tags,
    tagCount: Object.keys(tags).length,
    encoder: tags.encoder ?? null,
  };
};

const decodeMp3 = async (buffer, formatHint = null) => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "voice-detect-"));
  const safeHint =
    typeof formatHint === "string" ? formatHint.trim().toLowerCase() : null;
  const ext =
    safeHint && safeHint !== "auto" && /^[a-z0-9]+$/.test(safeHint)
      ? `.${safeHint}`
      : ".mp3";
  const tempFile = path.join(
    tempDir,
    `audio-${crypto.randomBytes(16).toString("hex")}${ext}`
  );

  try {
    await fs.writeFile(tempFile, buffer);
    const meta = await probeMp3(tempFile);
    if (!meta.sampleRate || !meta.channels) {
      throw new Error("Missing sample rate metadata.");
    }

    const args = [
      "-v",
      "error",
      "-i",
      tempFile,
      "-f",
      "f32le",
      "-acodec",
      "pcm_f32le",
      "-",
    ];

    const pcmBuffer = await runCommand("ffmpeg", args);
    const floatArray = new Float32Array(
      pcmBuffer.buffer,
      pcmBuffer.byteOffset,
      Math.floor(pcmBuffer.length / 4)
    );

    const mono = toMono(floatArray, meta.channels);
    return { pcm: mono, sampleRate: meta.sampleRate, metadata: meta };
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
};

const encodePcmToMp3 = async (pcm, sampleRate, bitrateKbps = 96) => {
  if (!pcm || !pcm.length) {
    throw new Error("PCM buffer is empty.");
  }
  const rate = Number.isFinite(sampleRate) && sampleRate > 0 ? sampleRate : 16000;
  const args = [
    "-v",
    "error",
    "-f",
    "f32le",
    "-ar",
    String(rate),
    "-ac",
    "1",
    "-i",
    "-",
    "-codec:a",
    "libmp3lame",
    "-b:a",
    `${bitrateKbps}k`,
    "-f",
    "mp3",
    "-",
  ];
  const pcmBuffer = Buffer.from(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  return runCommand("ffmpeg", args, pcmBuffer);
};

const toMono = (interleaved, channels) => {
  if (channels === 1) return interleaved;
  const length = Math.floor(interleaved.length / channels);
  const mono = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    let sum = 0;
    const base = i * channels;
    for (let c = 0; c < channels; c += 1) {
      sum += interleaved[base + c];
    }
    mono[i] = sum / channels;
  }
  return mono;
};

module.exports = {
  decodeMp3,
  encodePcmToMp3,
};
