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

app.get("/health", (_, res) => res.json({ ok: true }));

app.get("/v1/models", async (_, res) => {
  const upstream = await fetch(`${NIM_API_BASE}/models`, {
    headers: { Authorization: `Bearer ${NIM_API_KEY}` },
  });
  const data = await upstream.json();
  res.status(upstream.status).json(data);
});

app.post("/v1/chat/completions", async (req, res) => {
  const body = req.body;

  if (!body || !Array.isArray(body.messages)) {
    return res.status(400).json({ error: "messages required" });
  }

  // Pure passthrough. Whatever the client sends is what NIM gets.
  // No forced chat_template_kwargs, no injected params.
  console.log(`[proxy] -> ${body.model} | stream=${!!body.stream}`);

  const upstream = await fetch(`${NIM_API_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${NIM_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (body.stream) {
    if (!upstream.ok) {
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
    let buffer = "";
    let thinkOpen = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) {
            res.write(line + "\n");
            continue;
          }
          const jsonStr = line.slice(6).trim();
          if (!jsonStr || jsonStr === "[DONE]") {
            res.write(line + "\n");
            continue;
          }

          let parsed;
          try {
            parsed = JSON.parse(jsonStr);
          } catch {
            res.write(line + "\n");
            continue;
          }

          if (parsed.choices) {
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
          }

          res.write(`data: ${JSON.stringify(parsed)}\n`);
        }
      }

      if (thinkOpen) {
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "</think>" } }] })}\n`);
      }
    } catch (err) {
      console.error("stream error:", err);
    } finally {
      reader.cancel().catch(() => {});
      res.end();
    }
    return;
  }

  // Non-streaming
  const data = await upstream.json();

  if (Array.isArray(data.choices)) {
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
});

app.all("*", (_, res) => res.status(404).json({ error: "Not found" }));

app.listen(PORT, "0.0.0.0", () => {
  console.log(`running on ${PORT}`);
});
