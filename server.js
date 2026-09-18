import express from 'express';
import cors from 'cors';
import { Readable } from 'stream';

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// ============================================================
// SINGLE SOURCE OF TRUTH — edit only this block when models change.
//
// THINKING_STYLES: the different chat_template_kwargs shapes seen
// across model families. Add a new style only if a new vendor uses
// a shape you don't already have. These are DEFAULTS — if the
// incoming request already sets chat_template_kwargs, those values
// win (thinking is not forced).
//
// MODEL_CONFIG: key = short name your client sends,
//               value = [realName, style]
// style must match a key in THINKING_STYLES, or "none" for no kwargs.
// ============================================================
const THINKING_STYLES = {
  glm:      { enable_thinking: true },
  deepseek: { thinking: true },
  kimi:     { thinking_mode: "enabled" },
  none:     null
};

const MODEL_CONFIG = {
  glm:          ["z-ai/glm-5.3",                "glm"],
  glmflash:     ["z-ai/glm-5.3-flash",          "glm"],
  deepseek:     ["deepseek-ai/deepseek-v4-pro", "deepseek"],
  minimax:      ["minimaxai/minimax-m3",        "none"],
  kimi:         ["moonshotai/kimi-k3",          "kimi"],
};
// ============================================================

const HOP_BY_HOP_OR_UNSAFE_HEADERS = [
  'content-encoding',
  'content-length',
  'transfer-encoding',
  'connection'
];

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const incomingBody = req.body;
    const requestedKey = incomingBody.model?.toLowerCase();
    const entry = MODEL_CONFIG[requestedKey];

    // Fall back to passing the raw model string through untouched
    // if it's not one of our known short names.
    const [realModelName, style] = entry || [incomingBody.model, null];
    const defaultKwargs = style ? THINKING_STYLES[style] : null;

    if (!realModelName) {
      return res.status(400).json({ error: "No model specified in request body." });
    }

    const proxyBody = {
      ...incomingBody,
      model: realModelName
    };

    if (defaultKwargs) {
      proxyBody.chat_template_kwargs = {
        ...defaultKwargs,
        ...(incomingBody.chat_template_kwargs || {})
      };
    }

    const fetchResponse = await fetch("https://integrate.api.nvidia.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.NVIDIA_API_KEY}`,
        "User-Agent": "curl/8.5.0",
        "Accept": "*/*"
      },
      body: JSON.stringify(proxyBody)
    });

    fetchResponse.headers.forEach((value, name) => {
      if (!HOP_BY_HOP_OR_UNSAFE_HEADERS.includes(name.toLowerCase())) {
        res.setHeader(name, value);
      }
    });
    res.status(fetchResponse.status);

    if (fetchResponse.body) {
      Readable.fromWeb(fetchResponse.body).pipe(res);
    } else {
      res.end();
    }
  } catch (error) {
    console.error("Proxy Error:", error);
    if (!res.headersSent) {
      res.status(500).json({ error: error.message });
    } else {
      res.end();
    }
  }
});

if (!process.env.NVIDIA_API_KEY) {
  console.warn("WARNING: NVIDIA_API_KEY is not set. All upstream requests will fail with 401.");
}

app.listen(port, () => {
  console.log(`Render Proxy listening on port ${port}`);
});
