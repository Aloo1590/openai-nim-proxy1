import express from 'express';
import OpenAI from 'openai';

const app = express();
app.use(express.json({ limit: '15mb' }));

// --- Config ---
const PORT = process.env.PORT || 3000;
const NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1';
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'z-ai/glm-5.2';
// Fallback key if you don't want to pass a key from the client (e.g. from Janitor AI's "API key" field)
const SERVER_NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;

// Simple CORS so browser-based clients don't choke
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

function getClient(req) {
  // Janitor AI sends whatever "API key" you configure in its proxy settings
  // as a standard `Authorization: Bearer <key>` header. We use that as the
  // NVIDIA NIM key, so you can just paste your NVIDIA API key into Janitor AI.
  const authHeader = req.headers['authorization'];
  const clientKey = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;
  const apiKey = clientKey || SERVER_NVIDIA_API_KEY;

  if (!apiKey) {
    const err = new Error(
      'No API key provided. Either send "Authorization: Bearer <NVIDIA_API_KEY>" or set NVIDIA_API_KEY on the server.'
    );
    err.status = 401;
    throw err;
  }

  return new OpenAI({ apiKey, baseURL: NVIDIA_BASE_URL });
}

// --- OpenAI-compatible: GET /v1/models ---
// Janitor AI (and most OpenAI-compatible clients) call this to populate the model dropdown.
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: [
      {
        id: DEFAULT_MODEL,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'nvidia-nim',
      },
    ],
  });
});

// --- OpenAI-compatible: POST /v1/chat/completions ---
app.post('/v1/chat/completions', async (req, res) => {
  let client;
  try {
    client = getClient(req);
  } catch (err) {
    return res.status(err.status || 401).json({
      error: { message: err.message, type: 'invalid_request_error' },
    });
  }

  const {
    messages,
    temperature = 1,
    top_p = 1,
    max_tokens = 16384,
    stream = false,
    seed,
    stop,
    presence_penalty,
    frequency_penalty,
    // model is intentionally ignored/overridden below so the proxy always
    // targets GLM 5.2 on NIM, regardless of what the client thinks it picked
  } = req.body || {};

  if (!Array.isArray(messages)) {
    return res.status(400).json({
      error: { message: '`messages` array is required', type: 'invalid_request_error' },
    });
  }

  const payload = {
    model: DEFAULT_MODEL,
    messages,
    temperature,
    top_p,
    max_tokens,
    stream,
  };
  if (seed !== undefined) payload.seed = seed;
  if (stop !== undefined) payload.stop = stop;
  if (presence_penalty !== undefined) payload.presence_penalty = presence_penalty;
  if (frequency_penalty !== undefined) payload.frequency_penalty = frequency_penalty;

  try {
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders();

      const completion = await client.chat.completions.create(payload);

      for await (const chunk of completion) {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      const completion = await client.chat.completions.create(payload);
      res.json(completion);
    }
  } catch (err) {
    console.error('NVIDIA NIM upstream error:', err?.message || err);
    if (!res.headersSent) {
      res.status(err.status || 500).json({
        error: {
          message: err.message || 'Unknown error from NVIDIA NIM',
          type: 'upstream_error',
        },
      });
    } else {
      // Streaming already started; just end the connection cleanly.
      res.end();
    }
  }
});

// Health check
app.get('/', (req, res) => {
  res.send(`NIM -> GLM 5.2 OpenAI-compatible proxy is running. Model: ${DEFAULT_MODEL}`);
});

app.listen(PORT, () => {
  console.log(`OpenAI-compatible NVIDIA NIM proxy listening on port ${PORT}`);
  console.log(`Forwarding to ${NVIDIA_BASE_URL} using model "${DEFAULT_MODEL}"`);
});
