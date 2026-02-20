const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const MAJOR_TEMPLATE = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1];
const MINOR_TEMPLATE = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0];

function normalize(samples) {
  let max = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = Math.abs(samples[i]);
    if (v > max) max = v;
  }
  if (max <= 0) return samples;
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    out[i] = samples[i] / max;
  }
  return out;
}

function rms(samples) {
  if (!samples.length) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / samples.length);
}

export function detectPitch(samples, sampleRate) {
  if (!samples?.length || sampleRate <= 0) return null;
  const y = normalize(samples);
  if (rms(y) < 0.01) return null;

  const minFreq = 65.41;
  const maxFreq = 2093.0;
  const minLag = Math.floor(sampleRate / maxFreq);
  const maxLag = Math.floor(sampleRate / minFreq);

  let bestLag = -1;
  let bestCorr = 0;

  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let corr = 0;
    let energy1 = 0;
    let energy2 = 0;
    const limit = y.length - lag;
    for (let i = 0; i < limit; i += 1) {
      const a = y[i];
      const b = y[i + lag];
      corr += a * b;
      energy1 += a * a;
      energy2 += b * b;
    }

    const denom = Math.sqrt(energy1 * energy2);
    if (denom === 0) continue;
    const score = corr / denom;

    if (score > bestCorr) {
      bestCorr = score;
      bestLag = lag;
    }
  }

  if (bestLag <= 0 || bestCorr < 0.35) return null;
  return sampleRate / bestLag;
}

export function hzToNote(freq) {
  if (!freq || freq <= 0) {
    return { noteName: null, centsOff: null, targetFreq: null };
  }

  const midiNum = 69 + 12 * Math.log2(freq / 440.0);
  const nearestMidi = Math.round(midiNum);
  const centsOff = (midiNum - nearestMidi) * 100;
  const noteName = `${NOTE_NAMES[((nearestMidi % 12) + 12) % 12]}${Math.floor(nearestMidi / 12) - 1}`;
  const targetFreq = 440.0 * (2 ** ((nearestMidi - 69) / 12));

  return { noteName, centsOff, targetFreq };
}

function pitchClassEnergy(samples, sampleRate, freq) {
  const omega = (2 * Math.PI * freq) / sampleRate;
  let sPrev = 0;
  let sPrev2 = 0;
  const coeff = 2 * Math.cos(omega);

  for (let i = 0; i < samples.length; i += 1) {
    const s = samples[i] + coeff * sPrev - sPrev2;
    sPrev2 = sPrev;
    sPrev = s;
  }

  return sPrev2 * sPrev2 + sPrev * sPrev - coeff * sPrev * sPrev2;
}

function estimateChroma(samples, sampleRate) {
  const chroma = new Array(12).fill(0);
  const frameSize = Math.min(4096, samples.length);
  const hop = Math.max(1024, Math.floor(frameSize / 2));
  if (frameSize < 1024) return chroma;

  for (let start = 0; start + frameSize <= samples.length; start += hop) {
    const frame = samples.subarray(start, start + frameSize);

    for (let pitchClass = 0; pitchClass < 12; pitchClass += 1) {
      let acc = 0;
      for (let octave = 2; octave <= 6; octave += 1) {
        const midi = 12 * (octave + 1) + pitchClass;
        const freq = 440 * (2 ** ((midi - 69) / 12));
        if (freq < 40 || freq > sampleRate / 2 - 100) continue;
        acc += pitchClassEnergy(frame, sampleRate, freq);
      }
      chroma[pitchClass] += acc;
    }
  }

  const norm = Math.hypot(...chroma);
  if (norm > 0) {
    for (let i = 0; i < chroma.length; i += 1) {
      chroma[i] /= norm;
    }
  }

  return chroma;
}

export function detectKey(samples, sampleRate) {
  if (!samples?.length || sampleRate <= 0) {
    return { keyName: null, confidence: 0 };
  }

  const durationLimited = samples.subarray(0, Math.min(samples.length, sampleRate * 30));
  const chroma = estimateChroma(durationLimited, sampleRate);

  let bestScore = -Infinity;
  let bestKey = null;

  for (let i = 0; i < NOTE_NAMES.length; i += 1) {
    for (const [template, suffix] of [[MAJOR_TEMPLATE, "Major"], [MINOR_TEMPLATE, "Minor"]]) {
      const rolled = new Array(12).fill(0);
      for (let j = 0; j < 12; j += 1) {
        rolled[j] = template[(j - i + 12) % 12];
      }

      const norm = Math.hypot(...rolled);
      const normalized = rolled.map((value) => value / norm);
      let score = 0;
      for (let j = 0; j < 12; j += 1) {
        score += chroma[j] * normalized[j];
      }

      if (score > bestScore) {
        bestScore = score;
        bestKey = `${NOTE_NAMES[i]} ${suffix}`;
      }
    }
  }

  const confidence = Math.max(0, Math.min(1, bestScore));
  return { keyName: bestKey, confidence };
}

export function suggestChords(keyName) {
  if (!keyName) return [];
  const [baseNote, scaleType] = keyName.split(" ");
  const baseIdx = NOTE_NAMES.indexOf(baseNote);
  if (baseIdx < 0) return [];

  if (scaleType === "Major") {
    const intervals = [0, 2, 4, 5, 7, 9, 11];
    const suffixes = ["", "m", "m", "", "", "m", "dim"];
    return intervals.map((interval, index) => `${NOTE_NAMES[(baseIdx + interval) % 12]}${suffixes[index]}`);
  }

  const intervals = [0, 2, 3, 5, 7, 8, 10];
  const suffixes = ["m", "dim", "", "m", "m", "", ""];
  return intervals.map((interval, index) => `${NOTE_NAMES[(baseIdx + interval) % 12]}${suffixes[index]}`);
}

export function buildWaveformSvg(samples, sampleRate, noteName, keyName) {
  const width = 1000;
  const height = 320;
  const midY = height / 2;
  const maxSeconds = 5;
  const count = Math.min(samples.length, Math.floor(sampleRate * maxSeconds));
  const points = [];

  if (count > 0) {
    const step = Math.max(1, Math.floor(count / width));
    for (let i = 0; i < count; i += step) {
      const x = (i / count) * width;
      const y = midY - samples[i] * (height * 0.42);
      points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
    }
  }

  const title = `Waveform - Note: ${noteName ?? "N/A"} | Key: ${keyName ?? "N/A"}`;
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">\n  <rect width="100%" height="100%" fill="#ffffff"/>\n  <text x="20" y="28" font-size="20" fill="#111827" font-family="Arial, sans-serif">${title}</text>\n  <line x1="0" y1="${midY}" x2="${width}" y2="${midY}" stroke="#d1d5db" stroke-width="1"/>\n  <polyline fill="none" stroke="#2563eb" stroke-width="1.25" points="${points.join(" ")}"/>\n</svg>`;
}
