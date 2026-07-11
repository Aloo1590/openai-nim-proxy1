import express from 'express';
import cors from 'cors';
import { Readable } from 'stream';

const app = express();
const port = process.env.PORT || 3000;

// Enable CORS so Janitor AI can connect
app.use(cors());
app.use(express.json());

// THE TRANSLATOR DICTIONARY
// You can add or edit any model mappings here.
const modelMap = {
  "glm": "z-ai/glm-5.2",
  "deepseek": "deepseek-ai/deepseek-v4-pro", 
  "minimax": "minimaxai/minimax-m3",
  "stepfun": "stepfun-ai/stepfun-flash"
};

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const incomingBody = req.body;
    
    // Check if the user typed a simple name in Janitor. 
    // If it's not in the dictionary, it just passes whatever they typed directly.
    const requestedModel = incomingBody.model?.toLowerCase();
    const realModelName = modelMap[requestedModel] || incomingBody.model;

    const proxyBody = {
      ...incomingBody,
      model: realModelName,
      // Fallback token limit just in case Janitor omits it
      max_tokens: incomingBody.max_tokens > 0 ? incomingBody.max_tokens : 4096
    };

    // Forward the translated payload to NVIDIA
    const fetchResponse = await fetch("https://integrate.api.nvidia.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.NVIDIA_API_KEY}`,
        // Spoofing the User-Agent to avoid any leftover WAF blocks
        "User-Agent": "curl/8.5.0",
        "Accept": "*/*"
      },
      body: JSON.stringify(proxyBody)
    });

    // Copy NVIDIA's response headers back to Janitor
    fetchResponse.headers.forEach((value, name) => {
      res.setHeader(name, value);
    });
    res.status(fetchResponse.status);

    // Stream the data back flawlessly
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
