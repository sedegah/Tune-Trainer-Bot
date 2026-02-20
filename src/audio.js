let OggVorbisDecoderClass;
let OggOpusDecoderClass;
let MPEGDecoderClass;

function ensureWorkerGlobal() {
  if (typeof globalThis.Worker !== "undefined") return;

  globalThis.Worker = class WorkerUnavailable {
    constructor() {
      throw new Error("Worker threads are unavailable in this runtime");
    }
  };
}

function withPromiseTimeout(promise, timeoutMs, label) {
  let timeoutHandle;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutHandle = setTimeout(() => reject(new Error(`${label}_timeout`)), timeoutMs);
  });

  return Promise.race([promise, timeoutPromise]).finally(() => clearTimeout(timeoutHandle));
}

async function decodeWithDecoder({
  DecoderClass,
  bytes,
  timeoutMs,
  label,
  decodeMethod = "decodeFile",
}) {
  const decoder = new DecoderClass();
  try {
    await withPromiseTimeout(decoder.ready, timeoutMs, `${label}_init`);
    const decoded = await withPromiseTimeout(decoder[decodeMethod](bytes), timeoutMs, label);
    return decoded;
  } finally {
    try {
      const cleanupResult = decoder.free?.();
      if (cleanupResult && typeof cleanupResult.then === "function") {
        cleanupResult.catch(() => null);
      }
    } catch {
    }
  }
}

async function getOggVorbisDecoderClass() {
  ensureWorkerGlobal();
  if (!OggVorbisDecoderClass) {
    const mod = await import("@wasm-audio-decoders/ogg-vorbis");
    OggVorbisDecoderClass = mod.OggVorbisDecoder;
  }
  return OggVorbisDecoderClass;
}

async function getOggOpusDecoderClass() {
  ensureWorkerGlobal();
  if (!OggOpusDecoderClass) {
    const mod = await import("@aldlss/ogg-opus-decoder");
    OggOpusDecoderClass = mod.OggOpusDecoder;
  }
  return OggOpusDecoderClass;
}

async function getMpegDecoderClass() {
  ensureWorkerGlobal();
  if (!MPEGDecoderClass) {
    const mod = await import("mpg123-decoder");
    MPEGDecoderClass = mod.MPEGDecoder;
  }
  return MPEGDecoderClass;
}

function extensionOf(path = "") {
  const idx = path.lastIndexOf(".");
  if (idx === -1) return "";
  return path.slice(idx + 1).toLowerCase();
}

function mergeChannels(channelData) {
  if (!channelData?.length) {
    return new Float32Array(0);
  }

  if (channelData.length === 1) {
    return channelData[0] instanceof Float32Array ? channelData[0] : new Float32Array(channelData[0]);
  }

  const length = channelData[0].length;
  const out = new Float32Array(length);

  for (let i = 0; i < length; i += 1) {
    let sum = 0;
    for (let c = 0; c < channelData.length; c += 1) {
      sum += channelData[c][i] ?? 0;
    }
    out[i] = sum / channelData.length;
  }

  return out;
}

function decodeWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  if (riff !== "RIFF" || wave !== "WAVE") {
    throw new Error("Invalid WAV header");
  }

  let offset = 12;
  let audioFormat = 1;
  let numChannels = 1;
  let sampleRate = 44100;
  let bitsPerSample = 16;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= view.byteLength) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
    const chunkSize = view.getUint32(offset + 4, true);

    if (chunkId === "fmt ") {
      audioFormat = view.getUint16(offset + 8, true);
      numChannels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bitsPerSample = view.getUint16(offset + 22, true);
    } else if (chunkId === "data") {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }

    offset += 8 + chunkSize + (chunkSize % 2);
  }

  if (dataOffset === -1) {
    throw new Error("WAV data chunk not found");
  }

  const availableDataSize = Math.max(0, Math.min(dataSize, view.byteLength - dataOffset));

  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = Math.floor(availableDataSize / bytesPerSample / numChannels);
  const channels = Array.from({ length: numChannels }, () => new Float32Array(totalSamples));

  for (let i = 0; i < totalSamples; i += 1) {
    for (let channel = 0; channel < numChannels; channel += 1) {
      const sampleIndex = dataOffset + (i * numChannels + channel) * bytesPerSample;
      let value = 0;

      if (audioFormat === 3 && bitsPerSample === 32) {
        value = view.getFloat32(sampleIndex, true);
      } else if (bitsPerSample === 16) {
        value = view.getInt16(sampleIndex, true) / 32768;
      } else if (bitsPerSample === 24) {
        const b0 = view.getUint8(sampleIndex);
        const b1 = view.getUint8(sampleIndex + 1);
        const b2 = view.getUint8(sampleIndex + 2);
        let int = (b2 << 16) | (b1 << 8) | b0;
        if (int & 0x800000) int |= 0xff000000;
        value = int / 8388608;
      } else if (bitsPerSample === 32) {
        value = view.getInt32(sampleIndex, true) / 2147483648;
      } else if (bitsPerSample === 8) {
        value = (view.getUint8(sampleIndex) - 128) / 128;
      } else {
        throw new Error(`Unsupported WAV bits per sample: ${bitsPerSample}`);
      }

      channels[channel][i] = value;
    }
  }

  return { samples: mergeChannels(channels), sampleRate };
}

function maybeResample(samples, sampleRate, targetRate = 22050) {
  if (!samples?.length || sampleRate === targetRate) {
    return { samples, sampleRate };
  }

  const ratio = targetRate / sampleRate;
  const outLength = Math.max(1, Math.floor(samples.length * ratio));
  const out = new Float32Array(outLength);

  for (let i = 0; i < outLength; i += 1) {
    const srcPos = i / ratio;
    const left = Math.floor(srcPos);
    const right = Math.min(samples.length - 1, left + 1);
    const frac = srcPos - left;
    out[i] = samples[left] * (1 - frac) + samples[right] * frac;
  }

  return { samples: out, sampleRate: targetRate };
}

function clipDuration(samples, sampleRate, maxSeconds) {
  const maxSamples = Math.floor(sampleRate * maxSeconds);
  if (samples.length <= maxSamples) return samples;
  return samples.subarray(0, maxSamples);
}

async function decodeOgg(arrayBuffer, options = {}) {
  const bytes = new Uint8Array(arrayBuffer);
  const decodeTimeoutMs = Math.max(3000, Number(options.perCodecTimeoutMs ?? 9000));
  const preferOpusOnly = options.preferOpusOnly === true;
  let opusErrorMessage = "unknown";
  let vorbisErrorMessage = "unknown";

  try {
    const OggOpusDecoder = await getOggOpusDecoderClass();
    const decoded = await decodeWithDecoder({
      DecoderClass: OggOpusDecoder,
      bytes,
      timeoutMs: decodeTimeoutMs,
      label: "ogg_opus_decode",
    });

    if (decoded?.channelData?.length) {
      return { samples: mergeChannels(decoded.channelData), sampleRate: decoded.sampleRate };
    }
    opusErrorMessage = "decoded_empty";
  } catch (error) {
    opusErrorMessage = String(error?.message ?? "opus_decode_failed");
  }

  if (preferOpusOnly) {
    throw new Error(`Failed to decode OGG Opus (opus=${opusErrorMessage})`);
  }

  try {
    const OggVorbisDecoder = await getOggVorbisDecoderClass();
    const decoded = await decodeWithDecoder({
      DecoderClass: OggVorbisDecoder,
      bytes,
      timeoutMs: decodeTimeoutMs,
      label: "ogg_vorbis_decode",
    });
    if (!decoded?.channelData?.length) {
      vorbisErrorMessage = "decoded_empty";
    } else {
      return { samples: mergeChannels(decoded.channelData), sampleRate: decoded.sampleRate };
    }
  } catch (error) {
    vorbisErrorMessage = String(error?.message ?? "vorbis_decode_failed");
  }

  throw new Error(`Failed to decode OGG (opus=${opusErrorMessage}; vorbis=${vorbisErrorMessage})`);
}

async function decodeMp3(arrayBuffer) {
  const MPEGDecoder = await getMpegDecoderClass();
  const decoded = await decodeWithDecoder({
    DecoderClass: MPEGDecoder,
    bytes: new Uint8Array(arrayBuffer),
    timeoutMs: 22000,
    label: "mp3_decode",
    decodeMethod: "decode",
  });

  if (!decoded?.channelData?.length) {
    throw new Error("Failed to decode MP3");
  }

  return { samples: mergeChannels(decoded.channelData), sampleRate: decoded.sampleRate };
}

export async function decodeAudioFile(arrayBuffer, mimeType = "", filePath = "", maxSeconds = 30, options = {}) {
  const ext = extensionOf(filePath);
  const mime = mimeType.toLowerCase();
  const decoderTimeoutMs = Math.max(5000, Number(options.decoderTimeoutMs ?? 22000));
  const perCodecTimeoutMs = Math.max(3000, Number(options.perCodecTimeoutMs ?? Math.floor(decoderTimeoutMs / 2)));
  const preferOpusOnly = options.preferOpusOnly === true;

  let decoded;

  if (mime.includes("wav") || ext === "wav") {
    decoded = decodeWav(arrayBuffer);
  } else if (mime.includes("mpeg") || mime.includes("mp3") || ext === "mp3") {
    decoded = await decodeMp3(arrayBuffer);
  } else if (mime.includes("ogg") || mime.includes("opus") || ext === "ogg" || ext === "opus") {
    decoded = await decodeOgg(arrayBuffer, { perCodecTimeoutMs, preferOpusOnly });
  } else if (ext === "wav") {
    decoded = decodeWav(arrayBuffer);
  } else {
    try {
      decoded = await decodeOgg(arrayBuffer, { perCodecTimeoutMs, preferOpusOnly });
    } catch {
      try {
        decoded = await decodeMp3(arrayBuffer);
      } catch {
        decoded = decodeWav(arrayBuffer);
      }
    }
  }

  const mono = decoded.samples instanceof Float32Array ? decoded.samples : new Float32Array(decoded.samples);
  const resampled = maybeResample(mono, decoded.sampleRate, 22050);
  const clipped = clipDuration(resampled.samples, resampled.sampleRate, maxSeconds);

  return {
    samples: clipped,
    sampleRate: resampled.sampleRate,
  };
}
