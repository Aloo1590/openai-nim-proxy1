import express from 'express';
import cors from 'cors';
import { Readable } from 'stream';

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const modelMap = {
  "glm": "z-ai/glm-5.2",
  "deepseek": "deepseek-ai/deepseek-v4-pro", 
  "minimax": "minimaxai/minimax-m3",
  "stepfun": "stepfun-ai/stepfun-flash"
};

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const incomingBody = req.body;
    
    const requestedModel = incomingBody.model?.toLowerCase();
    const realModelName = modelMap[requestedModel] || incomingBody.model;

    const proxyBody = {
      ...incomingBody,
      model: realModelName,
      max_tokens: incomingBody.max_tokens > 0 ? incomingBody.max_tokens : 4096
    };

    // THE ADDITION: Force the thinking parameters on for GLM 5.2
    if (realModelName === "z-ai/glm-5.2") {
      proxyBody.chat_template_kwargs = {
        "enable_thinking": true,
        "clear_thinking": false
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
