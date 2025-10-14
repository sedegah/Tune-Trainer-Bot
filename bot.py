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
    """Detect pitch using YIN algorithm with better error handling"""
    try:
        if audio is None or len(audio) == 0:
            return None
        
        # Ensure audio is float32
        y = audio.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Detect pitch
        f0_series = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        
        # Filter out NaN and zero values
        f0_series = f0_series[~np.isnan(f0_series)]
        f0_series = f0_series[f0_series > 0]
        
        if len(f0_series) == 0:
            return None
            
        return float(np.median(f0_series))
    except Exception as e:
        logger.error(f"Pitch detection error: {e}")
        return None


def hz_to_note(freq: float):
    """Convert frequency to note name with cents deviation"""
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
    """Handle /start command"""
    await update.message.reply_text(
        "🎺 *Welcome to TuneTrainerBot!* 🎶\n\n"
        "Send me a *voice note* or *audio file* (MP3/WAV/OGG) and I'll tell you:\n"
        "• The musical note 🎵\n"
        "• How sharp/flat you are 📈\n"
        "• Fundamental frequency (Hz)\n\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/help - Get help\n\n"
        "Try it now!",
        parse_mode=constants.ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
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
    """Process audio files and voice notes"""
    msg = await update.message.reply_text("🎵 Processing your audio... one moment!")

    tmp_path = None
    try:
        # Get file
        if update.message.voice:
            file = await update.message.voice.get_file()
            suffix = ".ogg"
            logger.info(f"Processing voice note from user {update.effective_user.id}")
        elif update.message.audio:
            file = await update.message.audio.get_file()
            suffix = ".mp3"
            logger.info(f"Processing audio file from user {update.effective_user.id}")
        else:
            await msg.edit_text(
                "⚠️ Please send a *voice note* or *audio file!* 🎧",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
            return

        # Check file size (limit to 10MB)
        if file.file_size > 10 * 1024 * 1024:
            await msg.edit_text("⚠️ File too large! Please send audio under 10MB.")
            return

        # Download to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(custom_path=tmp_path)
        
        logger.info(f"Downloaded file to {tmp_path}")

        # Load audio with multiple fallback methods
        y = None
        sr = None
        
        # Try soundfile first (faster)
        try:
            y, sr = sf.read(tmp_path, always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            logger.info(f"Loaded with soundfile: sr={sr}, samples={len(y)}")
        except Exception as e:
            logger.warning(f"Soundfile failed: {e}, trying librosa...")
            
        # Fallback to librosa
        if y is None:
            try:
                y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=30)
                logger.info(f"Loaded with librosa: sr={sr}, samples={len(y)}")
            except Exception as e:
                logger.error(f"Librosa also failed: {e}")
                raise ValueError(f"Could not load audio: {e}")

        # Validate audio
        if y is None or len(y) == 0:
            raise ValueError("Empty audio buffer")
        
        if sr is None or sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}")

        # Limit audio length (max 30 seconds)
        max_samples = 30 * sr
        if len(y) > max_samples:
            y = y[:max_samples]
            logger.info(f"Truncated audio to 30 seconds")

    except Exception as e:
        logger.error(f"❌ Error loading audio: {e}", exc_info=True)
        await msg.edit_text(
            "❌ I couldn't read that audio file.\n\n"
            "Please try:\n"
            "• Sending a voice note instead\n"
            "• Using MP3/WAV/OGG format\n"
            "• Ensuring file is under 10MB"
        )
        return
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

    # --- Pitch detection ---
    try:
        await msg.edit_text("🎵 Analyzing pitch...")
        freq = detect_pitch(y, sr)
        
        if freq is None or freq <= 0:
            await msg.edit_text(
                "😕 I couldn't detect a clear pitch.\n\n"
                "💡 *Tips:*\n"
                "• Play a single sustained note\n"
                "• Reduce background noise\n"
                "• Sing or play louder\n"
                "• Avoid chords or multiple notes",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
            return

        note_name, cents_off, target_freq = hz_to_note(freq)
        
        if note_name is None:
            await msg.edit_text("😕 Detected a frequency but couldn't identify the note.")
            return

        cents_abs = abs(cents_off)
        tuning_indicator = "✨" if cents_abs < 5 else ("📈" if cents_off > 0 else "📉")
        tuning_text = (
            "✨ Perfectly in tune!" if cents_abs < 5
            else f"{tuning_indicator} *{cents_abs:.1f} cents {'sharp' if cents_off > 0 else 'flat'}*"
        )

        response_msg = (
            f"🎵 *Detected Note:* {note_name}\n"
            f"🎼 *Frequency:* {freq:.2f} Hz\n"
            f"🎯 *Target Frequency:* {target_freq:.2f} Hz\n"
            f"📊 *Tuning Status:* {tuning_text}"
        )
        
        await msg.edit_text(response_msg, parse_mode=constants.ParseMode.MARKDOWN)

        # --- Waveform Visualization ---
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Plot first 5 seconds
            duration_to_plot = min(len(y) / sr, 5)
            samples_to_plot = int(duration_to_plot * sr)
            time_axis = np.linspace(0, duration_to_plot, samples_to_plot)
            
            ax.plot(time_axis, y[:samples_to_plot], linewidth=0.5)
            ax.set_title(f"Waveform – Detected Note: {note_name} ({freq:.1f} Hz)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, duration_to_plot)

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            buf.seek(0)
            
            await update.message.reply_photo(
                buf, 
                caption=f"📊 Waveform visualization\nNote: {note_name} | Frequency: {freq:.1f} Hz"
            )
            
        except Exception as e:
            logger.error(f"Waveform error: {e}", exc_info=True)
            # Don't fail the whole request if visualization fails
            
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        await msg.edit_text(
            "❌ An error occurred during processing. Please try again."
        )


# --- Telegram Application ---
def build_app():
    """Build telegram application"""
    TOKEN = os.getenv("BOT_TOKEN")
    if not TOKEN:
        raise RuntimeError("❌ BOT_TOKEN not set in environment variables.")
    
    application = Application.builder().token(TOKEN).build()
    
    # Add handlers
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
    """Handle incoming webhook updates"""
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
    """Initialize bot on startup"""
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
    """Cleanup on shutdown"""
    try:
        await telegram_app.shutdown()
        logger.info("👋 Bot shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# --- Local run (Render entrypoint) ---
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app_fastapi, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info"
    )
