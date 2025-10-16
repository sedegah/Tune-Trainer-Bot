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
)
import warnings

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# --- FastAPI setup ---
app_fastapi = FastAPI()


@app_fastapi.get("/")
async def home():
    return {"status": "🎺 TuneTrainerBot is live and webhook active!"}


# --- Pitch detection helpers ---
def detect_pitch(audio: np.ndarray, sr: int):
    try:
        if audio is None or len(audio) == 0:
            return None
        y = audio.astype(np.float32)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        f0_series = librosa.yin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_series = f0_series[~np.isnan(f0_series)]
        f0_series = f0_series[f0_series > 0]
        if len(f0_series) == 0:
            return None
        return float(np.median(f0_series))
    except Exception as e:
        logger.error(f"Pitch detection error: {e}")
        return None


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


# --- Key detection helpers ---
MAJOR_TEMPLATE = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
MINOR_TEMPLATE = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def get_chroma(y: np.ndarray, sr: int):
    S = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    return chroma_mean

def detect_key(y: np.ndarray, sr: int):
    chroma = get_chroma(y, sr)
    best_score = -1
    best_key = None
    for i, note in enumerate(NOTE_NAMES):
        major_rot = np.roll(MAJOR_TEMPLATE, i)
        minor_rot = np.roll(MINOR_TEMPLATE, i)
        major_score = np.dot(chroma, major_rot)
        minor_score = np.dot(chroma, minor_rot)
        if major_score > best_score:
            best_score = major_score
            best_key = note + " Major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = note + " Minor"
    return best_key


# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎺 *Welcome to TuneTrainerBot!* 🎶\n\n"
        "Send me a *voice note* or *audio file* (MP3/WAV/OGG) and I'll tell you:\n"
        "• The musical note 🎵\n"
        "• How sharp/flat you are 📈\n"
        "• Fundamental frequency (Hz)\n"
        "• Detected musical key 🎹\n\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/help - Get help\n\n"
        "Try it now!",
        parse_mode=constants.ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎼 *How to use TuneTrainerBot:*\n\n"
        "1️⃣ Record a voice note or send an audio file\n"
        "2️⃣ Play a single sustained note (works best)\n"
        "3️⃣ Wait for analysis results\n\n"
        "💡 *Tips:*\n"
        "• Use a clear, sustained tone\n"
        "• Avoid background noise\n"
        "• Audio files under 30 seconds work best\n"
        "• Supported formats: MP3, WAV, OGG\n\n"
        "📧 Issues? Contact @your_username",
        parse_mode=constants.ParseMode.MARKDOWN,
    )


# --- Audio Handler ---
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🎵 Processing your audio... one moment!")
    tmp_path = None
    try:
        if update.message.voice:
            file = await update.message.voice.get_file()
            suffix = ".ogg"
        elif update.message.audio:
            file = await update.message.audio.get_file()
            suffix = ".mp3"
        else:
            await msg.edit_text("⚠️ Please send a *voice note* or *audio file!* 🎧", parse_mode=constants.ParseMode.MARKDOWN)
            return

        if file.file_size > 10 * 1024 * 1024:
            await msg.edit_text("⚠️ File too large! Please send audio under 10MB.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(custom_path=tmp_path)

        y = None
        sr = None
        try:
            y, sr = sf.read(tmp_path, always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        except Exception as e:
            logger.warning(f"Soundfile failed: {e}, trying librosa...")
        if y is None:
            y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=30)
        if len(y) > 30 * sr:
            y = y[:30 * sr]

    except Exception as e:
        logger.error(f"❌ Error loading audio: {e}", exc_info=True)
        await msg.edit_text("❌ I couldn't read that audio file.")
        return
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

    try:
        await msg.edit_text("🎵 Analyzing pitch and key...")
        freq = detect_pitch(y, sr)
        note_name, cents_off, target_freq = hz_to_note(freq)
        key_name = detect_key(y, sr)

        tuning_text = ""
        if cents_off is not None:
            cents_abs = abs(cents_off)
            tuning_indicator = "✨" if cents_abs < 5 else ("📈" if cents_off > 0 else "📉")
            tuning_text = "✨ Perfectly in tune!" if cents_abs < 5 else f"{tuning_indicator} *{cents_abs:.1f} cents {'sharp' if cents_off > 0 else 'flat'}*"

        response_msg = (
            f"🎵 *Detected Note:* {note_name}\n"
            f"🎼 *Frequency:* {freq:.2f} Hz\n"
            f"📊 *Tuning Status:* {tuning_text}\n"
            f"🎹 *Estimated Key:* {key_name}"
        )
        await msg.edit_text(response_msg, parse_mode=constants.ParseMode.MARKDOWN)

        # --- Waveform visualization ---
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(10, 4))
            duration_to_plot = min(len(y) / sr, 5)
            samples_to_plot = int(duration_to_plot * sr)
            time_axis = np.linspace(0, duration_to_plot, samples_to_plot)
            ax.plot(time_axis, y[:samples_to_plot], linewidth=0.5)
            ax.set_title(f"Waveform – Detected Note: {note_name} | Key: {key_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, duration_to_plot)

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            buf.seek(0)
            await update.message.reply_photo(buf, caption=f"📊 Waveform visualization\nNote: {note_name} | Key: {key_name}")
        except Exception as e:
            logger.error(f"Waveform error: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        await msg.edit_text("❌ An error occurred during processing. Please try again.")


# --- Telegram Application ---
def build_app():
    TOKEN = os.getenv("BOT_TOKEN")
    if not TOKEN:
        raise RuntimeError("❌ BOT_TOKEN not set in environment variables.")
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler((filters.VOICE | filters.AUDIO) & ~filters.COMMAND, handle_audio)
    )
    return application

telegram_app = build_app()


# --- Webhook route ---
@app_fastapi.post("/{token}")
async def webhook(request: Request, token: str):
    if token != os.getenv("BOT_TOKEN"):
        logger.warning("Unauthorized webhook attempt")
        return {"status": "unauthorized"}
    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# --- Startup/Shutdown events ---
@app_fastapi.on_event("startup")
async def startup():
    TOKEN = os.getenv("BOT_TOKEN")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://tune-trainer-bot.onrender.com")
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN not set")
    try:
        full_url = f"{WEBHOOK_URL}/{TOKEN}"
        await telegram_app.initialize()
        await telegram_app.bot.set_webhook(url=full_url)
        logger.info(f"✅ Webhook set successfully at {full_url}")
        logger.info("🚀 TuneTrainerBot is ready!")
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise


@app_fastapi.on_event("shutdown")
async def shutdown():
    try:
        await telegram_app.shutdown()
        logger.info("👋 Bot shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# --- Local run ---
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(app_fastapi, host="0.0.0.0", port=PORT, log_level="info")
