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

const probeMp3 = async (filePath) => {
  const args = [
    "-v",
    "error",
    "-select_streams",
    "a:0",
    "-show_entries",
    "stream=sample_rate,channels",
    "-of",
    "json",
    filePath,
  ];
  const output = await runCommand("ffprobe", args);
  const parsed = JSON.parse(output.toString("utf8"));
  const stream = parsed.streams && parsed.streams[0];
  return {
    sampleRate: stream ? Number(stream.sample_rate) : null,
    channels: stream ? Number(stream.channels) : null,
  };
};

const decodeMp3 = async (buffer) => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "voice-detect-"));
  const tempFile = path.join(
    tempDir,
    `audio-${crypto.randomBytes(16).toString("hex")}.mp3`
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
    return { pcm: mono, sampleRate: meta.sampleRate };
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
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
};
