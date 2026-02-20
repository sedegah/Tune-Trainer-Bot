import { decodeAudioFile } from "./audio.js";
import { buildWaveformSvg, detectKey, detectPitch, hzToNote, suggestChords } from "./music.js";

const recentUpdateIds = new Set();

function markUpdateSeen(updateId) {
  if (typeof updateId !== "number") return;
  recentUpdateIds.add(updateId);
  setTimeout(() => recentUpdateIds.delete(updateId), 5 * 60 * 1000);
}

function isDuplicateUpdate(updateId) {
  return typeof updateId === "number" && recentUpdateIds.has(updateId);
}

function getRequiredEnv(env, key) {
  const value = env[key];
  if (!value) throw new Error(`${key} is required`);
  return value;
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function htmlResponse(html, status = 200) {
  return new Response(html, {
    status,
    headers: { "content-type": "text/html; charset=utf-8" },
  });
}

function homePageHtml() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TuneTrainerBot: Pitch & Key Analysis on Telegram</title>
  <style>
    :root {
      --bg: #07090d;
      --panel: rgba(255, 255, 255, 0.04);
      --line: rgba(255, 255, 255, 0.14);
      --text: #f5f7ff;
      --muted: #a6adbb;
      --accent: #77f7d1;
      --accent-2: #86a9ff;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      color: var(--text);
      background:
        radial-gradient(60rem 30rem at 10% -10%, rgba(134, 169, 255, 0.20), transparent 55%),
        radial-gradient(50rem 30rem at 95% -5%, rgba(119, 247, 209, 0.20), transparent 50%),
        var(--bg);
      padding: 32px 18px;
    }

    .shell {
      max-width: 1040px;
      margin: 0 auto;
    }

    .hero {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
    }

    .badge {
      width: fit-content;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      background: rgba(119, 247, 209, 0.10);
      border: 1px solid rgba(119, 247, 209, 0.25);
      border-radius: 999px;
      padding: 6px 10px;
      margin-bottom: 14px;
    }

    h1 {
      margin: 0 0 10px;
      font-size: clamp(1.7rem, 4vw, 3rem);
      line-height: 1.08;
      letter-spacing: -0.02em;
      max-width: 18ch;
    }

    .lead {
      margin: 0;
      color: var(--muted);
      font-size: clamp(0.98rem, 1.6vw, 1.15rem);
      max-width: 58ch;
    }

    .actions {
      margin-top: 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .btn {
      text-decoration: none;
      color: #041318;
      background: linear-gradient(90deg, var(--accent), #b9ffe6);
      padding: 11px 16px;
      border-radius: 12px;
      font-weight: 700;
      font-size: 14px;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .btn-secondary {
      text-decoration: none;
      color: var(--text);
      background: transparent;
      border: 1px solid var(--line);
      padding: 11px 16px;
      border-radius: 12px;
      font-weight: 600;
      font-size: 14px;
    }

    .grid {
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
    }

    .card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: var(--panel);
    }

    .card h3 {
      margin: 0 0 8px;
      font-size: 14px;
      color: #d8deea;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }

    .meta {
      margin-top: 16px;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
      color: #8d95a6;
      font-size: 13px;
    }

    .meta a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }

    .meta a:hover {
      text-decoration: underline;
    }

    .commands {
      margin-top: 14px;
      border: 1px dashed rgba(255, 255, 255, 0.18);
      border-radius: 12px;
      padding: 10px 12px;
      color: #b8c0d0;
      font-size: 13px;
      background: rgba(255, 255, 255, 0.02);
    }

    code {
      color: var(--accent-2);
      font-weight: 600;
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="badge">Live on Cloudflare Workers</div>
      <h1>TuneTrainerBot: Pitch, Key, and Chord Intelligence</h1>
      <p class="lead">Send an audio sample on Telegram and instantly get detected note, frequency, tuning offset, estimated key, chord suggestions, and waveform output.</p>
      <div class="actions">
        <a href="https://t.me/TuneTrainerBot" class="btn">Start on Telegram</a>
        <a href="/health" class="btn-secondary">Check System Health</a>
      </div>

      <div class="commands">
        Commands: <code>/start</code> · <code>/help</code>
      </div>

      <div class="grid">
        <article class="card">
          <h3>Pitch Detection</h3>
          <p>Detects the dominant note and frequency from incoming audio clips.</p>
        </article>
        <article class="card">
          <h3>Tuning Feedback</h3>
          <p>Reports sharp/flat offset in cents for fast intonation correction.</p>
        </article>
        <article class="card">
          <h3>Key + Chords</h3>
          <p>Estimates major/minor key and suggests matching chord sets.</p>
        </article>
        <article class="card">
          <h3>Waveform Output</h3>
          <p>Returns a waveform graphic so users can inspect the signal shape.</p>
        </article>
      </div>

      <div class="meta">
        <span>Bot Status: ✅ Live</span>
        <span>Developed by Kimathi Sedegah • <a href="https://www.kimathi.tech/" target="_blank" rel="noopener noreferrer">Portfolio</a></span>
      </div>
    </section>
  </main>
</body>
</html>`;
}

async function telegramJson(env, method, payload) {
  const token = getRequiredEnv(env, "BOT_TOKEN");
  const response = await fetch(`https://api.telegram.org/bot${token}/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(`Telegram ${method} failed: ${JSON.stringify(data)}`);
  }
  return data.result;
}

async function telegramMultipart(env, method, formData) {
  const token = getRequiredEnv(env, "BOT_TOKEN");
  const response = await fetch(`https://api.telegram.org/bot${token}/${method}`, {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(`Telegram ${method} failed: ${JSON.stringify(data)}`);
  }
  return data.result;
}

function startMessage() {
  return "🎺 Welcome to TuneTrainerBot!\n\nSend a voice note or audio file and I'll tell you:\n• 🎵 The musical note\n• 📏 How sharp/flat you are\n• 🎼 Frequency\n• 🎹 Key\n• 🎸 Suggested chords\n\nBest formats: OGG/Opus, MP3, WAV\nCommands:\n/start /help";
}

function helpMessage() {
  return "📌 Send a clear, sustained note.\n⏱ Audio files under 30 seconds work best.\n💾 Supported formats: OGG/Opus, OGG/Vorbis, MP3, WAV";
}

function parseAudioMessage(message) {
  if (message.voice) {
    return {
      fileId: message.voice.file_id,
      fileSize: message.voice.file_size ?? 0,
      mimeType: "audio/ogg",
      source: "voice",
    };
  }

  if (message.audio) {
    return {
      fileId: message.audio.file_id,
      fileSize: message.audio.file_size ?? 0,
      mimeType: message.audio.mime_type ?? "",
      fileName: message.audio.file_name ?? "",
      source: "audio",
    };
  }

  if (message.document) {
    const mimeType = (message.document.mime_type ?? "").toLowerCase();
    const fileName = message.document.file_name ?? "";
    const lowerName = fileName.toLowerCase();
    const looksAudio =
      mimeType.startsWith("audio/") ||
      lowerName.endsWith(".mp3") ||
      lowerName.endsWith(".wav") ||
      lowerName.endsWith(".ogg") ||
      lowerName.endsWith(".opus");

    if (looksAudio) {
      return {
        fileId: message.document.file_id,
        fileSize: message.document.file_size ?? 0,
        mimeType,
        fileName,
        source: "document",
      };
    }
  }

  return null;
}

async function getFileBytes(env, fileId) {
  const file = await telegramJson(env, "getFile", { file_id: fileId });
  if (!file?.file_path) throw new Error("Telegram did not return file_path");

  const token = getRequiredEnv(env, "BOT_TOKEN");
  const downloadUrl = `https://api.telegram.org/file/bot${token}/${file.file_path}`;
  const response = await fetch(downloadUrl);
  if (!response.ok) {
    throw new Error(`File download failed with status ${response.status}`);
  }

  return {
    bytes: await response.arrayBuffer(),
    filePath: file.file_path,
  };
}

function formatAnalysis(noteName, freq, centsOff, keyName, confidence, chords) {
  const centsAbs = Math.abs(centsOff ?? 0);
  const tuningText = centsOff == null
    ? "Could not determine tuning"
    : centsAbs < 5
      ? "🎯 Perfectly in tune!"
      : `⚖️ ${centsAbs.toFixed(1)} cents ${centsOff > 0 ? "sharp" : "flat"}`;

  return [
    `🎵 Detected Note: ${noteName ?? "Unknown"}`,
    `🎼 Frequency: ${freq?.toFixed(2) ?? "N/A"} Hz`,
    `📏 Tuning Status: ${tuningText}`,
    `🎹 Estimated Key: ${keyName ?? "Unknown"} (${Math.round((confidence ?? 0) * 100)}% confidence)`,
    `🎸 Suggested Chords: ${chords.length ? chords.join(", ") : "N/A"}`,
  ].join("\n");
}

async function withTimeout(taskPromise, timeoutMs, label) {
  let timeoutHandle;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutHandle = setTimeout(() => reject(new Error(`${label}_timeout`)), timeoutMs);
  });

  try {
    return await Promise.race([taskPromise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutHandle);
  }
}

async function processAudioUpdate(env, message) {
  const chatId = message.chat.id;
  const processingMessage = await telegramJson(env, "sendMessage", {
    chat_id: chatId,
    text: "🔄 Processing your audio...",
    reply_to_message_id: message.message_id,
  });

  const maxBytes = Number(env.MAX_AUDIO_BYTES ?? "10485760");
  const maxSeconds = Number(env.MAX_AUDIO_SECONDS ?? "30");
  const configuredDecodeTimeout = Number(env.DECODE_TIMEOUT_MS ?? "22000");
  const decodeTimeoutMs = Math.max(8000, Math.min(configuredDecodeTimeout, 28000));

  try {
    const audio = parseAudioMessage(message);
    if (!audio) {
      await telegramJson(env, "editMessageText", {
        chat_id: chatId,
        message_id: processingMessage.message_id,
        text: "❌ Send a voice note or an audio file (MP3/WAV/OGG/OPUS).",
      });
      return;
    }

    if (audio.fileSize > maxBytes) {
      await telegramJson(env, "editMessageText", {
        chat_id: chatId,
        message_id: processingMessage.message_id,
        text: "⚠️ File too large! Please keep audio below 10MB.",
      });
      return;
    }

    const isLikelySupported =
      audio.source === "voice" ||
      (audio.mimeType ?? "").includes("ogg") ||
      (audio.mimeType ?? "").includes("opus") ||
      (audio.mimeType ?? "").includes("mpeg") ||
      (audio.mimeType ?? "").includes("mp3") ||
      (audio.mimeType ?? "").includes("wav") ||
      (audio.fileName ?? "").toLowerCase().endsWith(".mp3") ||
      (audio.fileName ?? "").toLowerCase().endsWith(".wav") ||
      (audio.fileName ?? "").toLowerCase().endsWith(".ogg") ||
      (audio.fileName ?? "").toLowerCase().endsWith(".opus");

    if (!isLikelySupported) {
      await telegramJson(env, "editMessageText", {
        chat_id: chatId,
        message_id: processingMessage.message_id,
        text: "⚠️ Unsupported audio format. Please send OGG/Opus, MP3, or WAV.",
      });
      return;
    }

    const downloaded = await withTimeout(getFileBytes(env, audio.fileId), 15000, "download");

    let decoded;
    try {
      decoded = await withTimeout(
        decodeAudioFile(downloaded.bytes, audio.mimeType, downloaded.filePath, maxSeconds, {
          decoderTimeoutMs: Math.max(5000, decodeTimeoutMs - 2000),
          perCodecTimeoutMs: 9000,
          preferOpusOnly: false,
        }),
        decodeTimeoutMs,
        "decode",
      );
    } catch (decodeError) {
      const rawReason = String(decodeError?.message ?? "decode_error").replace(/\s+/g, " ").trim();
      const reason = rawReason.slice(0, 140);
      console.error("Audio decode failed", {
        reason: rawReason,
        source: audio.source,
        mimeType: audio.mimeType ?? "",
        fileName: audio.fileName ?? "",
        filePath: downloaded?.filePath ?? "",
        fileSize: audio.fileSize ?? 0,
        maxSeconds,
        decodeTimeoutMs,
      });
      await telegramJson(env, "editMessageText", {
        chat_id: chatId,
        message_id: processingMessage.message_id,
        text: `⚠️ Audio decode failed (${reason}). Please send a clear OGG/Opus, MP3, or WAV file under 10MB.`,
      }).catch(() => null);
      return;
    }

    const frequency = await withTimeout(
      Promise.resolve(detectPitch(decoded.samples, decoded.sampleRate)),
      3000,
      "pitch",
    );
    if (!frequency) {
      await telegramJson(env, "editMessageText", {
        chat_id: chatId,
        message_id: processingMessage.message_id,
        text: "😕 I couldn’t detect a clear pitch. Try a sustained single note.",
      });
      return;
    }

    const { noteName, centsOff } = hzToNote(frequency);
    const { keyName, confidence } = await withTimeout(
      Promise.resolve(detectKey(decoded.samples, decoded.sampleRate)),
      5000,
      "key",
    );
    const chords = suggestChords(keyName);

    await telegramJson(env, "editMessageText", {
      chat_id: chatId,
      message_id: processingMessage.message_id,
      text: formatAnalysis(noteName, frequency, centsOff, keyName, confidence, chords),
    });

    const svg = buildWaveformSvg(decoded.samples, decoded.sampleRate, noteName, keyName);
    const form = new FormData();
    form.set("chat_id", String(chatId));
    form.set("caption", `🎵 Waveform\nNote: ${noteName ?? "Unknown"} | Key: ${keyName ?? "Unknown"}`);
    form.set("document", new Blob([svg], { type: "image/svg+xml" }), "waveform.svg");

    await telegramMultipart(env, "sendDocument", form);
  } catch (error) {
    const rawReason = String(error?.message ?? "processing_error").replace(/\s+/g, " ").trim();
    const reason = rawReason.slice(0, 140);
    await telegramJson(env, "editMessageText", {
      chat_id: chatId,
      message_id: processingMessage.message_id,
      text: `❌ Processing failed (${reason}). Please try another clear audio sample.`,
    }).catch(() => null);
    console.error("Audio processing error", error);
    return;
  }
}

async function handleTelegramUpdate(env, update) {
  const message = update?.message;
  if (!message) return;

  const text = message.text?.trim() ?? "";
  if (text.startsWith("/start")) {
    await telegramJson(env, "sendMessage", { chat_id: message.chat.id, text: startMessage() });
    return;
  }

  if (text.startsWith("/help")) {
    await telegramJson(env, "sendMessage", { chat_id: message.chat.id, text: helpMessage() });
    return;
  }

  if (message.voice || message.audio || message.document) {
    await processAudioUpdate(env, message);
    return;
  }
}

async function setWebhook(env, requestOrigin = "") {
  const baseUrl = env.PUBLIC_BASE_URL || requestOrigin;
  if (!baseUrl) {
    throw new Error("PUBLIC_BASE_URL is required when request origin is unavailable");
  }
  const secret = getRequiredEnv(env, "WEBHOOK_SECRET");
  const webhookUrl = `${baseUrl.replace(/\/$/, "")}/webhook`;

  return telegramJson(env, "setWebhook", {
    url: webhookUrl,
    secret_token: secret,
    allowed_updates: ["message"],
  });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/") {
      return htmlResponse(homePageHtml());
    }

    if (request.method === "GET" && url.pathname === "/health") {
      return jsonResponse({ status: "ok", service: "TuneTrainerBot Worker" });
    }

    if (request.method === "POST" && url.pathname === "/webhook") {
      const expectedSecret = env.WEBHOOK_SECRET;
      if (expectedSecret) {
        const received = request.headers.get("x-telegram-bot-api-secret-token") ?? "";
        if (received !== expectedSecret) {
          return jsonResponse({ error: "Forbidden" }, 403);
        }
      }

      try {
        const update = await request.json();
        const updateId = update?.update_id;

        if (isDuplicateUpdate(updateId)) {
          return jsonResponse({ status: "duplicate_ignored" });
        }

        markUpdateSeen(updateId);
        const processingTask = handleTelegramUpdate(env, update).catch((error) => {
          console.error("Webhook background processing error", error);
        });

        if (ctx?.waitUntil) {
          ctx.waitUntil(processingTask);
        } else {
          processingTask.catch(() => null);
        }
      } catch (error) {
        console.error("Webhook processing error", error);
      }

      return jsonResponse({ status: "ok" });
    }

    if (request.method === "POST" && url.pathname === "/setup-webhook") {
      const setupSecret = url.searchParams.get("secret") ?? "";
      if (!env.WEBHOOK_SECRET || setupSecret !== env.WEBHOOK_SECRET) {
        return jsonResponse({ error: "Forbidden" }, 403);
      }

      try {
        const result = await setWebhook(env, url.origin);
        return jsonResponse({ status: "ok", result });
      } catch (error) {
        return jsonResponse({ error: `setWebhook failed: ${error.message}` }, 500);
      }
    }

    return jsonResponse({ error: "Not found" }, 404);
  },
};
