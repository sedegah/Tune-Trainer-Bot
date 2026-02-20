import fs from "node:fs";

function parseEnvFile(path) {
  const out = {};
  if (!fs.existsSync(path)) return out;
  const lines = fs.readFileSync(path, "utf-8").split(/\r?\n/);
  for (const line of lines) {
    if (!line || line.trim().startsWith("#")) continue;
    const idx = line.indexOf("=");
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    if (key) out[key] = value;
  }
  return out;
}

const local = parseEnvFile(".dev.vars");
const token = process.env.BOT_TOKEN || local.BOT_TOKEN;
const baseUrl = process.env.PUBLIC_BASE_URL || local.PUBLIC_BASE_URL;
const secret = process.env.WEBHOOK_SECRET || local.WEBHOOK_SECRET;

if (!token || !baseUrl || !secret) {
  console.error("Missing BOT_TOKEN, PUBLIC_BASE_URL, or WEBHOOK_SECRET");
  process.exit(1);
}

const webhookUrl = `${baseUrl.replace(/\/$/, "")}/webhook`;

const response = await fetch(`https://api.telegram.org/bot${token}/setWebhook`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    url: webhookUrl,
    secret_token: secret,
    allowed_updates: ["message"],
  }),
});

const data = await response.json();
if (!response.ok || !data.ok) {
  console.error("Failed to set webhook:", JSON.stringify(data));
  process.exit(1);
}

console.log("Webhook configured:", webhookUrl);
