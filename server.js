import express from 'express';
import cors from 'cors';
import { Readable } from 'stream';

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// ============================================================
// SINGLE SOURCE OF TRUTH — edit only this block when models change.
// key      = the short name your client sends (e.g. "glm")
// realName = the exact upstream model string NVIDIA expects
// kwargs   = the chat_template_kwargs to inject for that model
//            (set to null if a model needs no special kwargs)
// ============================================================
const MODEL_CONFIG = {
  glm: {
    realName: "z-ai/glm-5.3",
    kwargs: { enable_thinking: true, clear_thinking: false }
  },
  deepseek: {
    realName: "deepseek-ai/deepseek-v4-pro",
    kwargs: { thinking: true }
  },
  minimax: {
    realName: "minimaxai/minimax-m3",
    kwargs: null
  },
  kimi: {
    realName: "moonshotai/kimi-k3",
    kwargs: { thinking_mode: "enabled" }
  }
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
    const config = MODEL_CONFIG[requestedKey];

    // Fall back to passing the raw model string through untouched
    // if it's not one of our known short names.
    const realModelName = config ? config.realName : incomingBody.model;

    if (!realModelName) {
      return res.status(400).json({ error: "No model specified in request body." });
    }

    const proxyBody = {
      ...incomingBody,
      model: realModelName
    };

    if (config?.kwargs) {
      proxyBody.chat_template_kwargs = config.kwargs;
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
