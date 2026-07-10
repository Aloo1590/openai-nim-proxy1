const express = require('express');
const cors = require('cors');
const https = require('https'); // Built-in Node module, zero SDKs required

const app = express();
app.use(cors());
app.use(express.json());

app.post('/v1/chat/completions', (req, res) => {
  // 1. Force a safe max_tokens if missing. Unbounded requests on GLM's 1M context cause internal stalls.
  if (!req.body.max_tokens) {
    req.body.max_tokens = 4096;
  }
  
  const payload = JSON.stringify(req.body);

  const options = {
    hostname: 'integrate.api.nvidia.com',
    port: 443,
    path: '/v1/chat/completions',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.NVIDIA_API_KEY}`,
      'Content-Length': Buffer.byteLength(payload),
      // 🔥 THE MAGIC BULLET: Spoof curl to bypass the WAF blocking Node.js/OpenAI SDKs
      'User-Agent': 'curl/7.68.0',
      'Accept': '*/*'
    }
  };

  // 2. Open a raw TCP-like connection directly to NVIDIA
  const proxyReq = https.request(options, (proxyRes) => {
    // Pass NVIDIA's response headers back to ReqBin/Janitor
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    // Pipe the raw byte stream. This natively handles both JSON and SSE Streaming without manual chunk parsing.
    proxyRes.pipe(res);
  });

  proxyReq.on('error', (error) => {
    console.error('NVIDIA Connection Error:', error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: error.message });
    }
  });

  // 3. Fire the payload
  proxyReq.write(payload);
  proxyReq.end();
});

app.listen(process.env.PORT || 3000, () => {
  console.log('Stealth cURL Proxy running on port', process.env.PORT || 3000);
});
