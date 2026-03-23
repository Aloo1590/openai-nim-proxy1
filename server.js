// server.js — Minimal Fast OpenAI → NVIDIA NIM Proxy

const express = require("express");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "20mb" }));

const NIM_API_KEY = process.env.NIM_API_KEY;
const NIM_API_BASE = "https://integrate.api.nvidia.com/v1";

if (!NIM_API_KEY) {
  console.warn("⚠️ NIM_API_KEY not set");
}

// --- MODEL MAPPING (kept as requested) ---
const MODEL_MAPPING = {
  "deepseek-v3.1-terminus": "deepseek-ai/deepseek-v3.1-terminus",
  "deepseek-v3.2": "deepseek-ai/deepseek-v3.2",
  "mistral": "mistralai/mistral-large-3-675b-instruct-2512",
  "deepseek-v3.1": "deepseek-ai/deepseek-v3.1",

  "minimax": "minimaxai/minimax-m2.1",
  "stepfun": "stepfun-ai/step-3.5-flash",
  "kimi": "moonshotai/kimi-k2.5",

  "glm4.7": "z-ai/glm4.7",
  "glm5": "z-ai/glm5",

  "meta/llama-3.1-8b-instruct": "meta/llama-3.1-8b-instruct",
  "meta/llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
  "meta/llama-3.1-405b-instruct": "meta/llama-3.1-405b-instruct",
  "deepseek-ai/deepseek-v3.1": "deepseek-ai/deepseek-v3.1",
  "qwen/qwen3-coder-480b-a35b-instruct": "qwen/qwen3-coder-480b-a35b-instruct",
  "nvidia/llama-3.1-nemotron-ultra-253b-v1":
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
};

// --- model selector (unchanged logic, just simplified) ---
function selectModel(model) {
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];
  if (model.includes("/")) return model;

  const m = model.toLowerCase();

  if (m.includes("405b") || m.includes("ultra"))
    return "meta/llama-3.1-405b-instruct";

  if (m.includes("70b") || m.includes("gpt-4"))
    return "meta/llama-3.1-70b-instruct";

  return "meta/llama-3.1-8b-instruct";
}

// --- health ---
app.get("/health", (req, res) => {
  res.json({ ok: true });
});

// --- models ---
app.get("/v1/models", (req, res) => {
  res.json({
    object: "list",
    data: Object.keys(MODEL_MAPPING).map((id) => ({
      id,
      object: "model",
    })),
  });
});

// --- MAIN ENDPOINT ---
app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(401).json({ error: "Missing NIM_API_KEY" });
    }

    const body = req.body;

    // remap model only
    const nimBody = {
      ...body,
      model: selectModel(body.model),
    };

    const response = await fetch(
      `${NIM_API_BASE}/chat/completions`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(nimBody),
      }
    );

    // --- STREAMING ---
    if (body.stream) {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });

      res.flushHeaders?.();

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(decoder.decode(value));
      }

      res.end();
      return;
    }

    // --- NON-STREAM ---
    const data = await response.json();
    res.json(data);

  } catch (err) {
    console.error("Error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// --- fallback ---
app.all("*", (req, res) => {
  res.status(404).json({ error: "Not found" });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Running on port ${PORT}`);
});