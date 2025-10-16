# TuneTrainerBot

TuneTrainerBot is a Telegram bot that analyzes audio files to detect musical notes, assess tuning accuracy, identify the key, and suggest appropriate chords. It utilizes FastAPI for the backend and the python-telegram-bot library for Telegram integration.

## Features

* Detects the musical note from audio input.
* Measures tuning accuracy in cents (sharp or flat).
* Identifies the key of the audio.
* Suggests chords based on the detected key.
* Provides a waveform visualization of the audio.
* Supports audio formats: MP3, WAV, OGG.
* Handles audio files up to 10MB in size.

## Requirements

* Python 3.8+
* Libraries:

  * `fastapi`
  * `python-telegram-bot`
  * `librosa`
  * `soundfile`
  * `matplotlib`
  * `numpy`
  * `uvicorn`
* Telegram Bot Token
* Webhook URL (optional)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sedegah/Tune-Trainer-Bot.git
   cd Tune-Trainer-Bot
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:

   ```bash
   export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
   export WEBHOOK_URL="your-webhook-url"  # Optional
   ```

## Usage

Run the bot with:

```bash
python bot.py
```

The bot will start and listen for incoming audio messages. When a user sends an audio file, the bot will process it and respond with:

* Detected musical note.
* Tuning accuracy (in cents).
* Frequency of the note.
* Estimated key of the audio.
* Suggested chords based on the key.
* A waveform visualization of the audio.

## Contributing

Feel free to fork the repository, submit issues, and send pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License.

