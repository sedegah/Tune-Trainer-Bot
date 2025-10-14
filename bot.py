import os
import io
import math
import tempfile
import logging
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    AIORateLimiter,
)

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Suppress unnecessary warnings ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# --- FastAPI setup ---
app_fastapi = FastAPI()

@app_fastapi.get("/")
async def home():
    return {"status": "🎺 TuneTrainerBot is live and webhook active!"}

# --- Pitch detection helpers ---
def detect_pitch(audio: np.ndarray, sr: int):
    if audio is None or len(audio) == 0:
        return None
    y = audio.astype(np.float32)
    f0_series = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
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

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎺 *Welcome to TuneTrainerBot!* 🎶\n\n"
        "Send me a *voice note* or *audio file* (MP3/WAV/OGG) and I'll tell you:\n"
        "• The musical note 🎵\n"
        "• How sharp/flat you are 📈\n"
        "• Fundamental frequency (Hz)\n\n"
        "Try it now!",
        parse_mode=constants.ParseMode.MARKDOWN,
    )

# --- Hardened Audio Handler ---
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Processing your audio... one moment! 🎼")

    try:
        if update.message.voice:
            file = await update.message.voice.get_file()
            suffix = ".ogg"
        elif update.message.audio:
            file = await update.message.audio.get_file()
            suffix = ".mp3"
        else:
            await update.message.reply_text(
                "Please send a *voice note* or *audio file!* 🎧",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
            return

        # 🔹 Save to temporary file instead of in-memory buffer
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            await file.download_to_drive(custom_path=tmp.name)
            tmp_path = tmp.name

        # 🔹 Load safely
        try:
            y, sr = sf.read(tmp_path, always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        except Exception:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)

        if y is None or len(y) == 0:
            raise ValueError("Empty audio buffer")

    except Exception as e:
        logger.error(f"❌ Error loading audio: {e}")
        await update.message.reply_text("❌ I couldn’t read that audio file. Try again with MP3 or OGG.")
        return

    # --- Pitch detection ---
    freq = detect_pitch(y, sr)
    if freq is None:
        await update.message.reply_text("😕 I couldn’t detect a clear pitch. Try a sustained tone.")
        return

    note_name, cents_off, target_freq = hz_to_note(freq)
    cents_abs = abs(cents_off)
    tuning_indicator = "✨" if cents_abs < 5 else ("📈" if cents_off > 0 else "📉")
    tuning_text = (
        "Perfectly in tune!" if cents_abs < 5
        else f"{tuning_indicator} {cents_abs:.1f} cents {'sharp' if cents_off > 0 else 'flat'}"
    )

    msg = (
        f"🎵 *Detected Note:* {note_name}\n"
        f"🎼 *Frequency:* {freq:.2f} Hz\n"
        f"🎯 *Target Note Frequency:* {target_freq:.2f} Hz\n"
        f"📊 *Tuning Status:* {tuning_text}"
    )
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

    # --- Waveform Visualization ---
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8, 3))
        duration_to_plot = min(len(y) / sr, 5)
        samples_to_plot = int(duration_to_plot * sr)
        ax.plot(y[:samples_to_plot])
        ax.set_title(f"Waveform (First {duration_to_plot:.1f}s) – {note_name}", fontsize=14)
        ax.set_xlabel("Time (Samples)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        await update.message.reply_photo(buf, caption="📈 Audio Waveform Visualization")
    except Exception as e:
        logger.error(f"Waveform error: {e}")
        await update.message.reply_text("⚠️ Could not generate waveform visualization.")

# --- Telegram App Setup ---
def build_app():
    TOKEN = os.getenv("BOT_TOKEN")
    if not TOKEN:
        raise RuntimeError("❌ BOT_TOKEN not set in environment variables.")
    return Application.builder().token(TOKEN).rate_limiter(AIORateLimiter()).build()

telegram_app = build_app()
telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

# --- Webhook route ---
@app_fastapi.post("/{token}")
async def webhook(request: Request, token: str):
    if token != os.getenv("BOT_TOKEN"):
        return {"status": "unauthorized"}
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"status": "ok"}

# --- Startup event: set webhook ---
@app_fastapi.on_event("startup")
async def startup():
    TOKEN = os.getenv("BOT_TOKEN")
    WEBHOOK_URL = "https://tune-trainer-bot.onrender.com"
    if TOKEN:
        full_url = f"{WEBHOOK_URL}/{TOKEN}"
        await telegram_app.initialize()
        await telegram_app.bot.set_webhook(url=full_url)
        logger.info(f"✅ Webhook set successfully at {full_url}")

# --- Local run (Render entrypoint) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_fastapi, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
