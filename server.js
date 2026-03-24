const express = require("express");
const cors = require("cors");
const app = express();
const PORT = process.env.PORT || 3000;
app.use(cors());
app.use(express.json({ limit: "20mb" }));

const NIM_API_KEY = process.env.NIM_API_KEY;
const NIM_API_BASE = "https://integrate.api.nvidia.com/v1";

if (!NIM_API_KEY) {
  console.warn("⚠️  NIM_API_KEY not set");
}

// Models that support reasoning/thinking
const REASONING_CAPABLE_MODELS = new Set([
  "z-ai/glm5",
  "z-ai/glm4.7",
  "glm5",
  "glm4.7",
  "deepseek-ai/deepseek-v3.1",
  "deepseek-ai/deepseek-v3.1-terminus",
  "deepseek-v3.1",
  "deepseek-v3.1-terminus",
  "moonshotai/kimi-k2.5",
  "kimi",
]);

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
  "nvidia/llama-3.1-nemotron-ultra-253b-v1": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
};

function selectModel(model) {
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];
  if (model.includes("/")) return model;
  const m = model.toLowerCase();
  if (m.includes("405b") || m.includes("ultra")) return "meta/llama-3.1-405b-instruct";
  if (m.includes("70b") || m.includes("gpt-4")) return "meta/llama-3.1-70b-instruct";
  return "meta/llama-3.1-8b-instruct";
}

// Build the NIM request body, injecting reasoning params if requested
function buildNimBody(body, resolvedModel) {
  // Pull out our custom flag — don't forward it raw to NIM
  const { enable_reasoning, clear_thinking, ...rest } = body;

  const nimBody = {
    ...rest,
    model: resolvedModel,
  };

  // Check if reasoning was requested AND the model supports it
  const reasoningRequested = enable_reasoning === true;
  const modelSupportsReasoning = REASONING_CAPABLE_MODELS.has(resolvedModel);

  if (reasoningRequested && modelSupportsReasoning) {
    nimBody.chat_template_kwargs = {
      enable_thinking: true,
      clear_thinking: clear_thinking !== false, // default true (clears <think> tags from final output)
      ...(rest.chat_template_kwargs || {}),     // allow manual override
    };
  }

  return nimBody;
}

// --- health ---
app.get("/health", (req, res) => res.json({ ok: true }));

// --- models ---
app.get("/v1/models", (req, res) => {
  res.json({
    object: "list",
    data: Object.keys(MODEL_MAPPING).map((id) => ({
      id,
      object: "model",
      reasoning_capable: REASONING_CAPABLE_MODELS.has(MODEL_MAPPING[id]),
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
    if (!body.messages || !Array.isArray(body.messages)) {
      return res.status(400).json({ error: "messages array is required" });
    }

    const resolvedModel = selectModel(body.model);
    const nimBody = buildNimBody(body, resolvedModel);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120_000);

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

    // --- STREAMING ---
    if (body.stream) {
      if (!response.ok) {
        const errData = await response.json().catch(() => ({ error: "Unknown NIM error" }));
        return res.status(response.status).json(errData);
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

        const chunk = decoder.decode(value, { stream: true });

        // If reasoning is on, prepend reasoning_content into the streamed
        // content so clients that don't know about reasoning_content still see it.
        // Clients that DO understand reasoning_content get it untouched.
        if (body.enable_reasoning) {
          const rewritten = rewriteReasoningChunks(chunk);
          res.write(rewritten);
        } else {
          res.write(chunk);
        }
      }

      res.end();
      return;
    }

    // --- NON-STREAM ---
    const data = await response.json();

    // For non-streaming, merge reasoning_content into content if present
    if (body.enable_reasoning && data.choices) {
      data.choices = data.choices.map((choice) => {
        const reasoning = choice.message?.reasoning_content;
        if (reasoning) {
          choice.message.content = `<think>\n${reasoning}\n</think>\n\n${choice.message.content || ""}`;
        }
        return choice;
      });
    }

    res.status(response.status).json(data);

  } catch (err) {
    if (err.name === "AbortError") {
      return res.status(504).json({ error: "Request to NIM timed out" });
    }
    console.error("Error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * Rewrites SSE chunks to prepend reasoning_content into content
 * so Janitor AI (which only reads `content`) still sees the thinking.
 *
 * Each SSE chunk looks like:
 *   data: {"choices":[{"delta":{"content":"...","reasoning_content":"..."}}]}\n\n
 */
function rewriteReasoningChunks(raw) {
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
          // If there's reasoning content but no regular content yet,
          // surface it wrapped in <think> tags inside content
          if (reasoning && !delta.content) {
            delta.content = reasoning;  // surfaces in clients unaware of reasoning_content
          }
          return { ...choice, delta };
        });

        return `data: ${JSON.stringify(parsed)}`;
      } catch {
        return line; // unparseable line, pass through untouched
      }
    })
    .join("\n");
}

// --- fallback ---
app.all("*", (req, res) => {
  res.status(404).json({ error: "Not found" });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Running on port ${PORT}`);
});
