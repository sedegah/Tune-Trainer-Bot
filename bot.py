import os
import io
import logging
import tempfile
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, HTTPException
from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, AIORateLimiter
import warnings
import asyncio

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("TuneTrainerBot")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set!")

processing_messages = set()

# Templates for key detection
MAJOR_TEMPLATE = np.array([1,0,1,0,1,1,0,1,0,1,0,1])
MINOR_TEMPLATE = np.array([1,0,1,1,0,1,0,1,1,0,1,0])
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def get_chroma(y: np.ndarray, sr: int):
    S = np.abs(librosa.stft(y, n_fft=4096))
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    return np.mean(chroma, axis=1)

def detect_key(y: np.ndarray, sr: int):
    y = y[:sr*30]  # Use first 30s for analysis
    chroma = get_chroma(y, sr)
    chroma /= np.linalg.norm(chroma)
    best_score = -1
    best_key = None
    for i, note in enumerate(NOTE_NAMES):
        for template, suffix in [(MAJOR_TEMPLATE, "Major"), (MINOR_TEMPLATE, "Minor")]:
            rolled_template = np.roll(template, i)
            rolled_template = rolled_template / np.linalg.norm(rolled_template)
            score = np.dot(chroma, rolled_template)
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

def plot_waveform(y, sr, note_name, key_name):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10,4))
    duration_to_plot = min(len(y)/sr,5)
    samples_to_plot = int(duration_to_plot*sr)
    time_axis = np.linspace(0,duration_to_plot,samples_to_plot)
    ax.plot(time_axis, y[:samples_to_plot], linewidth=0.5)
    ax.set_title(f"Waveform – Key: {key_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0,duration_to_plot)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

# Telegram bot commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to TuneTrainerBot!\n\nSend a voice note or audio file and I'll tell you:\n"
        "• The song's key (Major/Minor)\n• Confidence\n• Suggested chords\n\n"
        "Commands:\n/start /help",
        parse_mode=constants.ParseMode.MARKDOWN
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send a clear audio clip (up to 30s) of a song.\nSupported formats: MP3, WAV, OGG",
        parse_mode=constants.ParseMode.MARKDOWN
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_id = update.message.message_id
    if message_id in processing_messages:
        return
    processing_messages.add(message_id)
    msg = await update.message.reply_text("Processing your audio to detect the key...")
    tmp_path = None
    try:
        if update.message.voice:
            file = await update.message.voice.get_file()
            suffix = ".ogg"
        elif update.message.audio:
            file = await update.message.audio.get_file()
            suffix = ".mp3"
        else:
            await msg.edit_text("Send a voice note or audio file!")
            return

        if file.file_size > 10*1024*1024:
            await msg.edit_text("File too large!")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(custom_path=tmp_path)

        y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=30)
        if len(y) > 30*sr:
            y = y[:30*sr]

        key_name, key_conf = await asyncio.to_thread(lambda: detect_key(y, sr))
        chords = suggest_chords(key_name)

        response_msg = (
            f"Estimated Key: {key_name}\n"
            f"Confidence: {key_conf*100:.0f}%\n"
            f"Suggested Chords: {', '.join(chords)}"
        )

        await msg.edit_text(response_msg, parse_mode=constants.ParseMode.MARKDOWN)

        buf = await asyncio.to_thread(plot_waveform, y, sr, note_name="N/A", key_name=key_name)
        await update.message.reply_photo(buf, caption=f"Waveform | Key: {key_name}")

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        await msg.edit_text("Error during processing. Please try again.")
    finally:
        processing_messages.discard(message_id)
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

# FastAPI & Telegram setup
app = FastAPI(title="TuneTrainerBot Webhook")
telegram_app = Application.builder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()
telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("help", help_command))
telegram_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

@app.get("/")
async def home():
    return {"status": "TuneTrainerBot is live!"}

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
        logger.info(f"Webhook set: {webhook_url}")
    else:
        logger.warning("WEBHOOK_URL not set; bot won't receive updates via webhook")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    logger.info(f"Starting TuneTrainerBot on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
