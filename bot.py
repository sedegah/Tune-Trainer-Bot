import math
import numpy as np
import librosa
import matplotlib.pyplot as plt
from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, AIORateLimiter
import io
import os
import logging

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# --- Pitch detection helpers ---
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


# --- Telegram bot handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎺 Welcome to *TuneTrainerBot!* 🎶\n\n"
        "Send me a *voice note* or *audio file* (MP3/WAV/OGG) and I'll tell you:\n"
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
        await update.message.reply_text(f"❌ Sorry, I couldn't read that file ({file_type}). Try again with MP3/WAV.")
        return

    freq = detect_pitch(y, sr)
    if freq is None:
        await update.message.reply_text("😕 I couldn't detect a clear pitch. Try a sustained note.")
        return

    note_name, cents_off, target_freq = hz_to_note(freq)
    cents_abs = abs(cents_off)
    tuning_indicator = "✨" if cents_abs < 5 else ("📈" if cents_off > 0 else "📉")
    tuning_text = "Perfectly in tune!" if cents_abs < 5 else (
        f"{tuning_indicator} {cents_abs:.1f} cents {'sharp' if cents_off > 0 else 'flat'}"
    )

    msg = (
        f"🎵 *Detected Note:* {note_name}\n"
        f"🎼 *Frequency:* {freq:.2f} Hz\n"
        f"🎯 *Target Note Frequency:* {target_freq:.2f} Hz\n"
        f"📊 *Tuning Status:* {tuning_text}"
    )
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')

    try:
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
        logger.error(f"Error generating waveform plot: {e}")
        await update.message.reply_text("⚠️ Could not generate the waveform visualization.")


# --- Main startup ---
def main():
    TOKEN = os.getenv("BOT_TOKEN")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    PORT = int(os.environ.get("PORT", "5000"))

    if not TOKEN:
        logger.error("❌ BOT_TOKEN environment variable not set.")
        return

    app = (
        Application.builder()
        .token(TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    if WEBHOOK_URL:
        full_webhook = f"{WEBHOOK_URL}/{TOKEN}"
        logger.info(f"🌐 Starting TuneTrainerBot in Webhook mode: {full_webhook}")
        try:
            app.run_webhook(
                listen="0.0.0.0",
                port=PORT,
                url_path=TOKEN,
                webhook_url=full_webhook,
            )
        except Exception as e:
            logger.error(f"⚠️ Failed to start webhook: {e}")
            logger.info("Switching to polling mode instead...")
            app.run_polling()
    else:
        logger.info("🤖 WEBHOOK_URL not set — starting in Polling mode.")
        app.run_polling()


if __name__ == "__main__":
    main()
