# TuneTrainerBot (Cloudflare Worker + JavaScript)

TuneTrainerBot is now rebuilt in JavaScript for Cloudflare Workers.

It receives Telegram webhook updates, downloads incoming audio/voice files, detects pitch and note, estimates key, suggests chords, and sends a waveform visualization as an SVG document.

## Features

- Detects note from audio input
- Measures tuning offset in cents (sharp/flat)
- Estimates musical key (major/minor)
- Suggests diatonic chords from detected key
- Returns waveform visualization (SVG)
- Supports Telegram audio (OGG/Vorbis, MP3, WAV)
- Enforces file size and duration limits

## Project Layout

- `src/index.js` — Cloudflare Worker webhook + Telegram integration
- `src/audio.js` — audio decoding/resampling pipeline
- `src/music.js` — pitch, note, key, chords, waveform SVG generation
- `scripts/set-webhook.mjs` — webhook setup helper
- `python/` — legacy Python implementation preserved as requested

## Requirements

- Node.js 20+
- Cloudflare account + Workers API token
- Telegram bot token

## Local Setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Create local env file:

   ```bash
   cp .dev.vars.example .dev.vars
   ```

3. Fill `.dev.vars` values:

   - `BOT_TOKEN`
   - `WEBHOOK_SECRET`
   - `PUBLIC_BASE_URL` (your deployed worker URL, required for `npm run set-webhook`)
   - `CLOUDFLARE_API_TOKEN`
   - `DECODE_TIMEOUT_MS` (optional, default `35000`)

4. Run locally:

   ```bash
   npm run dev
   ```

## Deploy to Cloudflare

```bash
npm run deploy
```

After deploy, configure Telegram webhook:

```bash
npm run set-webhook
```

Or call the Worker endpoint directly:

```bash
curl -X POST "https://<your-worker-url>/setup-webhook?secret=<WEBHOOK_SECRET>"
```

## Webhook Endpoints

- `GET /` health check
- `POST /webhook` Telegram webhook receiver (validates `X-Telegram-Bot-Api-Secret-Token`)
- `POST /setup-webhook?secret=...` helper endpoint to set Telegram webhook (auto-uses request origin)

## Notes

- Keep `.dev.vars` private (already git-ignored)
- Rotate tokens if shared accidentally
- The Worker sends waveform as SVG document instead of PNG image

