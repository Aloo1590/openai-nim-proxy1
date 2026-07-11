import express from 'express';
import cors from 'cors';
import { Readable } from 'stream';

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
// 50mb limit permanently fixes the 413 Payload Too Large error for long chats
app.use(express.json({ limit: '50mb' }));

const modelMap = {
  "glm": "z-ai/glm-5.2",
  "deepseek": "deepseek-ai/deepseek-v4-pro", 
  "minimax": "minimaxai/minimax-m3",
  "stepfun": "stepfun-ai/step-3.7-flash"
};

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const incomingBody = req.body;
    
    const requestedModel = incomingBody.model?.toLowerCase();
    const realModelName = modelMap[requestedModel] || incomingBody.model;

    // Pass the payload exactly as your Janitor AI UI generated it.
    // Zero interference with your max_tokens, temperature, or top_p.
    const proxyBody = {
      ...incomingBody,
      model: realModelName
    };

    // INJECT REASONING KWARGS BASED ON THE MODEL
    if (realModelName === "z-ai/glm-5.2") {
      proxyBody.chat_template_kwargs = {
        "enable_thinking": true,
        "clear_thinking": false
      };
    } else if (realModelName === "deepseek-ai/deepseek-v4-pro") {
      proxyBody.chat_template_kwargs = {
        "thinking": true
      };
    } else if (realModelName === "minimaxai/minimax-m3") {
      proxyBody.chat_template_kwargs = {
        "thinking_mode": "enabled"
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
      res.setHeader(name, value);
    });
    res.status(fetchResponse.status);

    if (fetchResponse.body) {
      Readable.fromWeb(fetchResponse.body).pipe(res);
    } else {
      res.end();
    }

  } catch (error) {
    console.error("Proxy Error:", error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Render Proxy listening on port ${port}`);
});
