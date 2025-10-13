import math
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import io
import os

# ------------------- Pitch Detection Core -------------------

def detect_pitch(audio: np.ndarray, sr: int):
    """Estimate fundamental frequency using librosa.yin."""
    if audio is None:
        return None
    y = audio.astype(np.float32)
    f0_series = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_series = f0_series[~np.isnan(f0_series)]
    if len(f0_series) == 0:
        return None
    return float(np.median(f0_series))


def hz_to_note(freq: float):
    """Map frequency in Hz to nearest note & cent deviation."""
    if not freq or freq <= 0:
        return None, None, None

    midi_num = 69 + 12 * math.log2(freq / 440.0)
    nearest_midi = int(round(midi_num))
    cents_off = (midi_num - nearest_midi) * 100

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_name = note_names[nearest_midi % 12] + str((nearest_midi // 12) - 1)
    target_freq = 440.0 * (2 ** ((nearest_midi - 69) / 12))
    return note_name, cents_off, target_freq

# ------------------- Telegram Bot Logic -------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    file = None
    if update.message.voice:
        file = await update.message.voice.get_file()
    elif update.message.audio:
        file = await update.message.audio.get_file()
    else:
        await update.message.reply_text("Please send a *voice note* or *audio file!* 🎧", parse_mode="Markdown")
        return

    with io.BytesIO() as buf:
        await file.download_to_memory(out=buf)
        buf.seek(0)
        y, sr = librosa.load(buf, sr=None, mono=True)

    freq = detect_pitch(y, sr)
    if freq is None:
        await update.message.reply_text("😕 I couldn’t detect a clear pitch. Try a sustained tone or single note.")
        return

    note_name, cents_off, target_freq = hz_to_note(freq)
    cents_abs = abs(cents_off)
    tuning = " Perfectly in tune!" if cents_abs < 5 else (
        f"🔺 {cents_abs:.1f} cents sharp" if cents_off > 0 else f"🔻 {cents_abs:.1f} cents flat"
    )

    msg = (
        f" *Detected Note:* {note_name}\n"
        f" *Frequency:* {freq:.2f} Hz\n"
        f" *Target:* {target_freq:.2f} Hz\n"
        f"{tuning}"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

    # Send waveform
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title(f"Waveform – {note_name}")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)
    await update.message.reply_photo(img_buf, caption="📈 Waveform")

# ------------------- Run the Bot -------------------

def main():
    TOKEN = os.getenv("BOT_TOKEN")  # safer than hardcoding
    if not TOKEN:
        print(" Set BOT_TOKEN as an environment variable first.")
        return

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    print("🤖 TuneTrainerBot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
