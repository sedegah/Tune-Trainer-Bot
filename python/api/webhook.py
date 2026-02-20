import os
import io
import math
import logging
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request
import uvicorn
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    AIORateLimiter,
)

# ------------------ Logging ------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("TuneTrainerBot")

# ------------------ Environment ------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN environment variable not set!")

# ------------------ Pitch Detection ------------------
def detect_pitch(audio: np.ndarray, sr: int):
    if audio is None or len(audio) == 0:
        return None
    y = audio.astype(np.float32)
    f0_series = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_series = f0_series[~np.isnan(f0_series)]
    if len(f0_series) == 0:
        return None
    return float(np.median(f0_series))

def hz_to_note(freq: float):
    if not freq or freq <= 0:
        return None, None, None
    midi_num = 69 + 12 * math.log2(freq / 440.0)
    nearest_midi = int(round(midi_num))
    cents_off = (midi_num - nearest_midi) * 100
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_name = note_names[nearest_midi % 12] + str((nearest_midi // 12) - 1)
    target_freq = 440.0 * (2 ** ((nearest_midi - 69) / 12))
    return note_name, cents_off, target_freq

# ------------------ Telegram Handlers ------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎺 *Welcome to TuneTrainerBot!* 🎶\n\n"
        "Send me a *voice note* or *audio file* (MP3/WAV/OGG) and I’ll tell you:\n"
        "• The musical note 🎵\n"
        "• How sharp/flat you are 📈\n"
        "• Fundamental frequency (Hz)\n\n"
        "Try it now!",
        parse_mode=constants.ParseMode.MARKDOWN
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Processing your audio... one moment! 🎼")

    file = None
    if update.message.voice:
        file = await update.message.voice.get_file()
    elif update.message.audio:
        file = await update.message.audio.get_file()
    else:
        await update.message.reply_text("Please send a *voice note* or *audio file!* 🎧",
                                        parse_mode=constants.ParseMode.MARKDOWN)
        return

    try:
        buf = io.BytesIO()
        await file.download_to_memory(out=buf)
        buf.seek(0)

        # More robust decoding
        try:
            y, sr = sf.read(buf, always_2d=False)
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = np.mean(y, axis=1)
        except Exception:
            buf.seek(0)
            y, sr = librosa.load(buf, sr=None, mono=True)
    except Exception as e:
        logger.error(f"❌ Error reading audio: {e}")
        await update.message.reply_text("⚠️ I couldn’t read that audio file. Try again with MP3 or WAV.")
        return

    freq = detect_pitch(y, sr)
    if freq is None:
        await update.message.reply_text("😕 I couldn’t detect a clear pitch. Try a sustained tone or single note.")
        return

    note_name, cents_off, target_freq = hz_to_note(freq)
    cents_abs = abs(cents_off)
    tuning_indicator = "✨" if cents_abs < 5 else ("📈" if cents_off > 0 else "📉")
    tuning_text = "Perfectly in tune!" if cents_abs < 5 else (
        f"{tuning_indicator} {cents_abs:.1f} cents {'sharp' if cents_off > 0 else 'flat'}"
    )

    msg = (
        f"*Detected Note:* {note_name}\n"
        f"*Frequency:* {freq:.2f} Hz\n"
        f"*Target Note Frequency:* {target_freq:.2f} Hz\n"
        f"*Tuning Status:* {tuning_text}"
    )
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8, 3))
        duration_to_plot = min(len(y) / sr, 5)
        ax.plot(y[: int(duration_to_plot * sr)])
        ax.set_title(f"Waveform (First {duration_to_plot:.1f}s) – {note_name}")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        plt.close(fig)
        img_buf.seek(0)
        await update.message.reply_photo(img_buf, caption="📈 Audio Waveform Visualization")
    except Exception as e:
        logger.warning(f"Waveform generation failed: {e}")

# ------------------ FastAPI + Telegram Setup ------------------
app = FastAPI(title="TuneTrainerBot Webhook")
telegram_app = Application.builder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()
telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

@app.get("/")
async def home():
    return {"status": "🎺 TuneTrainerBot is live and ready!"}

@app.post(f"/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
    except Exception as e:
        logger.exception("Webhook error: %s", e)
        raise HTTPException(status_code=500, detail="Error processing update")
    return {"status": "ok"}

@app.on_event("startup")
async def on_startup():
    await telegram_app.initialize()
    if WEBHOOK_URL:
        webhook_url = f"{WEBHOOK_URL}/{BOT_TOKEN}"
        await telegram_app.bot.set_webhook(url=webhook_url)
        logger.info(f"✅ Webhook set to: {webhook_url}")
    else:
        logger.warning("⚠️ WEBHOOK_URL not set; bot will not receive updates via webhook.")

# ------------------ Entry ------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    logger.info(f"🚀 Starting TuneTrainerBot on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
