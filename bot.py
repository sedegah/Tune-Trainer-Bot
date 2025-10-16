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
import asyncio

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

app_fastapi = FastAPI()
processing_messages = set()

MAJOR_TEMPLATE = np.array([1,0,1,0,1,1,0,1,0,1,0,1])
MINOR_TEMPLATE = np.array([1,0,1,1,0,1,0,1,1,0,1,0])
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def detect_pitch(audio: np.ndarray, sr: int):
    try:
        y = audio.astype(np.float32)
        if np.max(np.abs(y)) > 0:
            y /= np.max(np.abs(y))
        f0_series = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
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
    midi_num = 69 + 12*math.log2(freq/440.0)
    nearest_midi = int(round(midi_num))
    cents_off = (midi_num - nearest_midi) * 100
    note_name = NOTE_NAMES[nearest_midi % 12] + str((nearest_midi//12)-1)
    target_freq = 440.0 * 2**((nearest_midi-69)/12)
    return note_name, cents_off, target_freq

def get_chroma(y: np.ndarray, sr: int):
    S = np.abs(librosa.stft(y, n_fft=4096))
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    return np.mean(chroma, axis=1)

def detect_key(y: np.ndarray, sr: int):
    y = y[:sr*10]  # first 10 seconds only
    chroma = get_chroma(y, sr)
    best_score = -1
    best_key = None
    for i, note in enumerate(NOTE_NAMES):
        for template, suffix in [(MAJOR_TEMPLATE, "Major"), (MINOR_TEMPLATE, "Minor")]:
            score = np.dot(chroma, np.roll(template, i)) / (np.linalg.norm(chroma) * np.linalg.norm(template))
            if score > best_score:
                best_score = score
                best_key = f"{note} {suffix}"
    return best_key, best_score

def suggest_chords(key_name):
    if key_name is None:
        return []
    base_note, scale_type = key_name.split()
    base_idx = NOTE_NAMES.index(base_note)
    if scale_type == "Major":
        chords = [NOTE_NAMES[(base_idx+i)%12]+suffix for i,suffix in zip([0,2,4,5,7,9,11], ["","m","m","","","m","dim"])]
    else:
        chords = [NOTE_NAMES[(base_idx+i)%12]+suffix for i,suffix in zip([0,2,3,5,7,8,10], ["m","dim","","m","m","",""])]
    return chords

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎺 *Welcome to TuneTrainerBot!* 🎶\n\n"
        "Send me a voice note or audio file and I'll tell you:\n"
        "• The musical note\n• How sharp/flat you are\n"
        "• Fundamental frequency (Hz)\n• Detected musical key\n• Suggested chords\n\n"
        "Commands:\n/start /help",
        parse_mode=constants.ParseMode.MARKDOWN
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎼 Send a clear, sustained note.\n"
        "Audio files under 30 seconds work best.\n"
        "Supported formats: MP3, WAV, OGG",
        parse_mode=constants.ParseMode.MARKDOWN
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_id = update.message.message_id
    if message_id in processing_messages:
        return
    processing_messages.add(message_id)

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
            await msg.edit_text("⚠️ Send a voice note or audio file!", parse_mode=constants.ParseMode.MARKDOWN)
            return

        if file.file_size > 10*1024*1024:
            await msg.edit_text("⚠️ File too large!")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(custom_path=tmp_path)

        try:
            y, sr = sf.read(tmp_path, always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        except:
            y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=30)

        if sr > 22050:
            y = librosa.resample(y, sr, 22050)
            sr = 22050
        if len(y) > 30*sr:
            y = y[:30*sr]

        freq, key_info = await asyncio.to_thread(lambda: (detect_pitch(y,sr), detect_key(y,sr)))
        note_name, cents_off, target_freq = hz_to_note(freq)
        key_name, key_conf = key_info
        chords = suggest_chords(key_name)

        tuning_text = ""
        if cents_off is not None:
            cents_abs = abs(cents_off)
            tuning_indicator = "✨" if cents_abs < 5 else ("📈" if cents_off>0 else "📉")
            tuning_text = "✨ Perfectly in tune!" if cents_abs<5 else f"{tuning_indicator} *{cents_abs:.1f} cents {'sharp' if cents_off>0 else 'flat'}*"

        response_msg = (
            f"🎵 *Detected Note:* {note_name}\n"
            f"🎼 *Frequency:* {freq:.2f} Hz\n"
            f"📊 *Tuning Status:* {tuning_text}\n"
            f"🎹 *Estimated Key:* {key_name} ({key_conf*100:.0f}% confidence)\n"
            f"🎸 *Suggested Chords:* {', '.join(chords)}"
        )
        await msg.edit_text(response_msg, parse_mode=constants.ParseMode.MARKDOWN)

        try:
            await asyncio.to_thread(plot_waveform, y, sr, note_name, key_name, update)
        except Exception as e:
            logger.error(f"Waveform error: {e}")

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        await msg.edit_text("❌ Error during processing. Please try again.")
    finally:
        processing_messages.discard(message_id)
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

def plot_waveform(y, sr, note_name, key_name, update):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10,4))
    duration_to_plot = min(len(y)/sr,5)
    samples_to_plot = int(duration_to_plot*sr)
    time_axis = np.linspace(0,duration_to_plot,samples_to_plot)
    ax.plot(time_axis, y[:samples_to_plot], linewidth=0.5)
    ax.set_title(f"Waveform – Note: {note_name} | Key: {key_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0,duration_to_plot)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    asyncio.run(update.message.reply_photo(buf, caption=f"📊 Waveform\nNote: {note_name} | Key: {key_name}"))
