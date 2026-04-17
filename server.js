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

/* ------------------ MODELS ------------------ */

const REASONING_MODELS = new Set([
  "minimaxai/minimax-m2.7",
  "z-ai/glm4.7",
  "google/gemma-4-31b-it",
  "deepseek-ai/deepseek-v3.2",
  "moonshotai/kimi-k2.5",
]);

const MODEL_MAP = {
  "minimax": "minimaxai/minimax-m2.7",
  "glm4.7": "z-ai/glm4.7",
  "gemma4": "google/gemma-4-31b-it",
  "deepseek-v3.2": "deepseek-ai/deepseek-v3.2",
  "kimi2.5": "moonshotai/kimi-k2.5",
};

/* ------------------ HELPERS ------------------ */

function resolveModel(model = "") {
  if (MODEL_MAP[model]) return MODEL_MAP[model];
  if (model.includes("/")) return model;
  return "meta/llama-3.1-8b-instruct";
}

function buildBody(body, model) {
  const { enable_reasoning, clear_thinking, ...rest } = body;

  const final = {
    ...rest,
    model,
  };

  if (enable_reasoning && REASONING_MODELS.has(model)) {
    final.chat_template_kwargs = {
      ...(rest.chat_template_kwargs || {}),
      enable_thinking: true,
      clear_thinking: clear_thinking !== false,
    };
  }

  return final;
}

/* ------------------ STREAM FIX ------------------ */

function rewriteChunk(raw) {
  return raw
    .split("\n")
    .map((line) => {
      if (!line.startsWith("data: ")) return line;

      const jsonStr = line.slice(6);
      if (jsonStr === "[DONE]") return line;

      try {
        const parsed = JSON.parse(jsonStr);

        if (!parsed.choices) return line;

        parsed.choices = parsed.choices.map((choice) => {
          const delta = choice.delta || {};

          const reasoning = delta.reasoning_content;
          const content = delta.content || "";

          if (reasoning) {
            delta.content = `<think>${reasoning}</think>` + content;
          }

          return { ...choice, delta };
        });

        return `data: ${JSON.stringify(parsed)}`;
      } catch {
        return line;
      }
    })
    .join("\n");
}

/* ------------------ ROUTES ------------------ */

app.get("/health", (_, res) => {
  res.json({ ok: true });
});

app.get("/v1/models", (_, res) => {
  res.json({
    object: "list",
    data: Object.keys(MODEL_MAP).map((id) => ({
      id,
      object: "model",
      reasoning_capable: REASONING_MODELS.has(MODEL_MAP[id]),
    })),
  });
});

/* ------------------ MAIN ------------------ */

app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(401).json({ error: "Missing NIM_API_KEY" });
    }

    const body = req.body;

    if (!body.messages || !Array.isArray(body.messages)) {
      return res.status(400).json({ error: "messages required" });
    }

    const model = resolveModel(body.model);
    const nimBody = buildBody(body, model);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000);

    const response = await fetch(`${NIM_API_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(nimBody),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    /* -------- STREAM -------- */

    if (body.stream) {
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        return res.status(response.status).json(err);
      }

      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        let chunk = decoder.decode(value, { stream: true });

        if (body.enable_reasoning) {
          chunk = rewriteChunk(chunk);
        }

        res.write(chunk);
      }

      res.end();
      return;
    }

    /* -------- NON STREAM -------- */

    const data = await response.json();

    if (body.enable_reasoning && data.choices) {
      data.choices = data.choices.map((choice) => {
        const reasoning = choice.message?.reasoning_content;
        const content = choice.message?.content || "";

        if (reasoning) {
          choice.message.content =
            `<think>\n${reasoning}\n</think>\n\n` + content;
        }

        return choice;
      });
    }

    res.status(response.status).json(data);

  } catch (err) {
    if (err.name === "AbortError") {
      return res.status(504).json({ error: "timeout" });
    }

    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

/* ------------------ FALLBACK ------------------ */

app.all("*", (_, res) => {
  res.status(404).json({ error: "Not found" });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`running on ${PORT}`);
});
