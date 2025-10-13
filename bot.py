import math
import numpy as np
import librosa
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import io
import os
import logging
# Ensure this script runs as a web server to satisfy Render deployment requirements.

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------- Pitch Detection Core -------------------

def detect_pitch(audio: np.ndarray, sr: int):
    """Estimate fundamental frequency using librosa.yin."""
    if audio is None:
        return None
    
    # Ensure audio is float32 for librosa processing
    y = audio.astype(np.float32)
    
    # Use a realistic pitch range (C2 to C7) for musical instruments/voice
    f0_series = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    
    # Clean up NaN values and find the median fundamental frequency (F0)
    f0_series = f0_series[~np.isnan(f0_series)]
    
    if len(f0_series) == 0:
        return None
        
    return float(np.median(f0_series))


def hz_to_note(freq: float):
    """Map frequency in Hz to nearest note & cent deviation."""
    if not freq or freq <= 0:
        return None, None, None

    # Calculate MIDI number relative to A4 (440 Hz = MIDI 69)
    midi_num = 69 + 12 * math.log2(freq / 440.0)
    nearest_midi = int(round(midi_num))
    
    # Cent deviation (100 cents per semitone)
    cents_off = (midi_num - nearest_midi) * 100

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    # Calculate octave and note name
    note_name = note_names[nearest_midi % 12] + str((nearest_midi // 12) - 1)
    
    # Calculate the exact frequency of the nearest target note
    target_freq = 440.0 * (2 ** ((nearest_midi - 69) / 12))
    
    return note_name, cents_off, target_freq

# ------------------- Telegram Bot Logic -------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    await update.message.reply_text(
        "🎺 Welcome to *TuneTrainerBot!* 🎶\n\n"
        "Send me a *voice note* or *audio file* (MP3/WAV) and I’ll tell you:\n"
        "• The musical note 🎵\n"
        "• How sharp/flat you are 📈\n"
        "• Fundamental frequency (Hz)\n\n"
        "Try it now!",
        parse_mode="Markdown"
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processes incoming voice notes or audio files."""
    await update.message.reply_text("Processing your audio... one moment! 🎼")
    
    file = None
    if update.message.voice:
        file = await update.message.voice.get_file()
    elif update.message.audio:
        file = await update.message.audio.get_file()
    else:
        # This branch is technically unreachable due to the MessageHandler filter, 
        # but kept for robustness.
        await update.message.reply_text("Please send a *voice note* or *audio file!* 🎧", parse_mode="Markdown")
        return

    try:
        # Download file into an in-memory buffer
        with io.BytesIO() as buf:
            await file.download_to_memory(out=buf)
            buf.seek(0)
            
            # Load the audio data using librosa
            y, sr = librosa.load(buf, sr=None, mono=True)

    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        await update.message.reply_text("❌ Sorry, I had trouble reading that audio file. Is it a standard format?")
        return

    # Detect pitch
    freq = detect_pitch(y, sr)
    if freq is None:
        await update.message.reply_text("😕 I couldn’t detect a clear pitch. Try a sustained tone or single note.")
        return

    # Convert frequency to note/cent deviation
    note_name, cents_off, target_freq = hz_to_note(freq)
    cents_abs = abs(cents_off)
    
    # Generate tuning feedback message
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
    await update.message.reply_text(msg, parse_mode="Markdown")

    # Generate and send waveform plot
    try:
        fig, ax = plt.subplots(figsize=(8, 3))
        # Plot only the first 5 seconds to avoid huge images
        duration_to_plot = min(len(y) / sr, 5) 
        samples_to_plot = int(duration_to_plot * sr)
        
        ax.plot(y[:samples_to_plot])
        ax.set_title(f"Waveform (First {duration_to_plot:.1f}s) – {note_name}", fontsize=14)
        ax.set_xlabel("Time (Samples)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        
        # Save plot to an in-memory buffer
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches='tight')
        plt.close(fig) # Close figure to free up memory
        img_buf.seek(0)
        
        await update.message.reply_photo(img_buf, caption="📈 Audio Waveform Visualization")
        
    except Exception as e:
        logger.error(f"Error generating or sending photo: {e}")
        await update.message.reply_text("⚠️ Could not generate the waveform visualization.")


# ------------------- Run the Bot (Webhook Setup) -------------------

def main():
    """Start the bot using Webhooks for Render deployment."""
    # --- Environment Variables ---
    TOKEN = os.getenv("BOT_TOKEN")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL") # e.g., https://tune-trainer-bot.onrender.com
    PORT = int(os.environ.get("PORT", "5000")) # Render sets this
    
    if not TOKEN:
        logger.error("❌ BOT_TOKEN environment variable not set.")
        return
    if not WEBHOOK_URL:
        logger.warning("⚠️ WEBHOOK_URL environment variable not set. Falling back to Polling.")
        # If WEBHOOK_URL is missing, we revert to polling as a fallback.
        app = ApplicationBuilder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
        logger.info("🤖 TuneTrainerBot starting in Polling mode...")
        app.run_polling()
        return

    # --- Webhook Setup ---
    logger.info(f"Using Webhook URL: {WEBHOOK_URL} on port: {PORT}")
    
    # 1. Build the application instance
    app = ApplicationBuilder().token(TOKEN).build()
    
    # 2. Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    
    # 3. Start the webhook server
    try:
        app.run_webhook(
            listen="0.0.0.0",               # Listen on all interfaces (required by Render)
            port=PORT,                       # Use the port specified by the environment
            url_path=TOKEN,                  # Use the token as a secret path (improves security)
            webhook_url=f"{WEBHOOK_URL}/{TOKEN}" # Telegram needs the full URL
        )
        logger.info(" TuneTrainerBot started successfully with Webhook!")
    except Exception as e:
        logger.error(f"Failed to start webhook: {e}")


if __name__ == "__main__":
    main()
