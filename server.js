const express = require("express");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "20mb" }));

const NIM_API_KEY = process.env.NIM_API_KEY;
const NIM_API_BASE = "https://integrate.api.nvidia.com/v1";
const REQUEST_TIMEOUT_MS = 120_000;

if (!NIM_API_KEY) {
  console.warn("⚠️ NIM_API_KEY not set");
}

/* ------------------ MODELS ------------------ */

const DEFAULT_MODEL = "meta/llama-3.1-8b-instruct";

/* ------------------ HELPERS ------------------ */

function resolveModel(model) {
  if (typeof model !== "string" || model.trim().length === 0) return DEFAULT_MODEL;
  return model.trim();
}

function buildBody(body, model) {
  const { enable_reasoning, clear_thinking, ...rest } = body;

  const final = {
    ...rest,
    model,
  };

  if (enable_reasoning) {
    final.chat_template_kwargs = {
      ...(rest.chat_template_kwargs || {}),
      enable_thinking: true,
      clear_thinking: clear_thinking !== false,
    };
  }

  return final;
}

/* ------------------ STREAM FIX ------------------ */
/**
 * Creates a stateful transformer for one streaming request.
 *
 * Fixes vs. the original implementation:
 *  1. Buffers partial lines across chunk boundaries. `reader.read()` chunks
 *     do not align with SSE "line" boundaries, so a JSON payload could
 *     previously be split across two reads, fail JSON.parse, and pass
 *     through unmodified (silently dropping the reasoning wrap for that
 *     event, or corrupting the message the client renders).
 *  2. Emits exactly one opening `<think>` and one closing `</think>` per
 *     reasoning span, instead of wrapping every individual delta chunk in
 *     its own `<think>...</think>` pair (which produced many broken/nested
 *     tags client-side instead of one continuous reasoning block).
 *  3. Strips `reasoning_content` from the outgoing delta once it's been
 *     folded into `content`, since most downstream clients (Janitor AI
 *     included) only render `content`.
 */
function createReasoningTransformer() {
  let buffer = "";
  let thinkOpen = false;

  function transformLine(line) {
    if (!line.startsWith("data: ")) return line;

    const jsonStr = line.slice(6).trim();
    if (!jsonStr || jsonStr === "[DONE]") return line;

    let parsed;
    try {
      parsed = JSON.parse(jsonStr);
    } catch {
      return line;
    }

    if (!parsed.choices) return line;

    parsed.choices = parsed.choices.map((choice) => {
      const delta = { ...(choice.delta || {}) };
      const reasoning = delta.reasoning_content;
      let content = delta.content || "";

      if (reasoning) {
        content = (thinkOpen ? "" : "<think>") + reasoning + content;
        thinkOpen = true;
      } else if (thinkOpen) {
        content = "</think>" + content;
        thinkOpen = false;
      }

      delta.content = content;
      delete delta.reasoning_content;

      return { ...choice, delta };
    });

    return `data: ${JSON.stringify(parsed)}`;
  }

  return {
    push(raw) {
      buffer += raw;
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep the possibly-incomplete final line
      return lines.map(transformLine).join("\n") + (lines.length ? "\n" : "");
    },
    // Call once the upstream stream ends, to flush any trailing partial
    // line and close an unterminated <think> block.
    flush() {
      let out = "";
      if (buffer) {
        out += transformLine(buffer);
        buffer = "";
      }
      if (thinkOpen) {
        out += (out ? "\n" : "") + `data: ${JSON.stringify({
          choices: [{ index: 0, delta: { content: "</think>" } }],
        })}`;
        thinkOpen = false;
      }
      return out;
    },
  };
}

/* ------------------ ROUTES ------------------ */

app.get("/health", (_, res) => {
  res.json({ ok: true });
});

app.get("/v1/models", async (_, res) => {
  if (!NIM_API_KEY) {
    return res.status(401).json({ error: "Missing NIM_API_KEY" });
  }

  try {
    const upstream = await fetch(`${NIM_API_BASE}/models`, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}` },
    });
    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err) {
    console.error("models fetch failed:", err);
    res.status(502).json({ error: "failed to fetch model list from NIM" });
  }
});

/* ------------------ MAIN ------------------ */

app.post("/v1/chat/completions", async (req, res) => {
  const body = req.body;

  if (!NIM_API_KEY) {
    return res.status(401).json({ error: "Missing NIM_API_KEY" });
  }

  if (!body || !Array.isArray(body.messages)) {
    return res.status(400).json({ error: "messages required" });
  }

  const model = resolveModel(body.model);
  const nimBody = buildBody(body, model);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  // If the client disconnects, stop the upstream request/stream too.
  res.on("close", () => controller.abort());

  let upstream;
  try {
    upstream = await fetch(`${NIM_API_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(nimBody),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === "AbortError") {
      return res.headersSent ? res.end() : res.status(504).json({ error: "timeout" });
    }
    console.error(err);
    return res.status(502).json({ error: "upstream request failed" });
  }

  /* -------- STREAM -------- */
  if (body.stream) {
    if (!upstream.ok) {
      clearTimeout(timeout);
      const err = await upstream.json().catch(() => ({}));
      return res.status(upstream.status).json(err);
    }

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    const reader = upstream.body.getReader();
    const decoder = new TextDecoder();
    const transformer = body.enable_reasoning ? createReasoningTransformer() : null;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        res.write(transformer ? transformer.push(chunk) : chunk);
      }

      if (transformer) {
        const tail = transformer.flush();
        if (tail) res.write(tail);
      }
    } catch (err) {
      if (err.name !== "AbortError") console.error("stream error:", err);
    } finally {
      clearTimeout(timeout);
      reader.cancel().catch(() => {});
      res.end();
    }
    return;
  }

  /* -------- NON STREAM -------- */
  try {
    const data = await upstream.json();

    if (body.enable_reasoning && Array.isArray(data.choices)) {
      data.choices = data.choices.map((choice) => {
        const reasoning = choice.message?.reasoning_content;
        const content = choice.message?.content || "";

        if (reasoning) {
          choice.message.content = `<think>\n${reasoning}\n</think>\n\n${content}`;
          delete choice.message.reasoning_content;
        }

        return choice;
      });
    }

    res.status(upstream.status).json(data);
  } catch (err) {
    console.error(err);
    res.status(502).json({ error: "invalid upstream response" });
  } finally {
    clearTimeout(timeout);
  }
});

/* ------------------ FALLBACK ------------------ */

app.all("*", (_, res) => {
  res.status(404).json({ error: "Not found" });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`running on ${PORT}`);
});
