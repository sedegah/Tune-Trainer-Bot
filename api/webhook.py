import math
import numpy as np
import librosa
import matplotlib.pyplot as plt
from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, AIORateLimiter
import io
import os
import logging

from fastapi import FastAPI, Request, HTTPException
import uvicorn
from telegram.error import TelegramError

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN environment variable not set!")
    raise ValueError("BOT_TOKEN environment variable not set!")

if not WEBHOOK_URL:
    logger.warning("⚠️ WEBHOOK_URL environment variable not set. Startup may fail.")


def detect_pitch(audio: np.ndarray, sr: int):
    if audio is None:
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎺 Welcome to *TuneTrainerBot!* 🎶\n\n"
        "Send me a *voice note* or *audio file* (MP3/WAV) and I’ll tell you:\n"
        "• The musical note 🎵\n"
        "• How sharp/flat you are 📈\n"
        "• Fundamental frequency (Hz)\n\n"
        "Try it now!",
        parse_mode=constants.ParseMode.MARKDOWN
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Processing your audio... one moment! 🎼")

    file = None
    file_type = "unknown audio"

    if update.message.voice:
        file = await update.message.voice.get_file()
        file_type = "voice note (OGG/Opus)"
    elif update.message.audio:
        file = await update.message.audio.get_file()
        file_type = f"audio file ({update.message.audio.mime_type or 'unknown MIME'})"
    else:
        await update.message.reply_text("Please send a *voice note* or *audio file!* 🎧", parse_mode=constants.ParseMode.MARKDOWN)
        return

    try:
        with io.BytesIO() as buf:
            await file.download_to_memory(out=buf)
            buf.seek(0)
            y, sr = librosa.load(buf, sr=None, mono=True)
    except Exception as e:
        logger.error(f"Error loading {file_type} from user {update.effective_user.id}: {e}")
        error_msg = f"❌ Sorry, I had trouble reading that audio file ({file_type}). "
        if "voice note" in file_type:
            error_msg += "If this is a voice note, try sending an explicit MP3 or WAV file instead."
        else:
            error_msg += "Please ensure it's a standard MP3 or WAV file and try again."
        await update.message.reply_text(error_msg)
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
        f" *Detected Note:* {note_name}\n"
        f" *Frequency:* {freq:.2f} Hz\n"
        f" *Target Note Frequency:* {target_freq:.2f} Hz\n"
        f" *Tuning Status:* {tuning_text}"
    )
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

    try:
        plt.style.use('seaborn-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 3))
        duration_to_plot = min(len(y) / sr, 5)
        samples_to_plot = int(duration_to_plot * sr)

        ax.plot(y[:samples_to_plot])
        ax.set_title(f"Waveform (First {duration_to_plot:.1f}s) – {note_name}", fontsize=14)
        ax.set_xlabel("Time (Samples)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches='tight')
        plt.close(fig)
        img_buf.seek(0)

        await update.message.reply_photo(img_buf, caption="📈 Audio Waveform Visualization")
    except Exception as e:
        logger.error(f"Error generating or sending photo: {e}")
        await update.message.reply_text("⚠️ Could not generate the waveform visualization.")


app = FastAPI(title="TuneTrainerBot Webhook")

tg_app = Application.builder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()

tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))


@app.post(f"/{BOT_TOKEN}")
async def telegram_webhook(req: Request):
    try:
        data = await req.json()
    except Exception as e:
        logger.error("Failed to parse request body: %s", e)
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        update = Update.de_json(data, tg_app.bot)
        await tg_app.process_update(update)
    except TelegramError as e:
        logger.error(f"Telegram error while processing update: {e}")
    except Exception as e:
        logger.exception("Exception while processing update: %s", e)

    return {"status": "ok"}


@app.get("/")
async def root():
    return {"status": "TuneTrainerBot is alive and ready for webhooks!"}


@app.on_event("startup")
async def on_startup():
    await tg_app.initialize()
    await tg_app.start()

    if WEBHOOK_URL:
        full_webhook_url = f"{WEBHOOK_URL}/{BOT_TOKEN}"
        try:
            await tg_app.bot.set_webhook(full_webhook_url)
            logger.info("Webhook successfully set to: %s", full_webhook_url)
        except Exception as e:
            logger.exception("Failed to set webhook: %s", e)
            raise e
    else:
        logger.warning("WEBHOOK_URL not set. Webhook was not registered with Telegram.")


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    logger.info("Starting uvicorn server on port %s", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
