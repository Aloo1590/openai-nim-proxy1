const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const payload = JSON.stringify(req.body);

    const response = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.NVIDIA_API_KEY}`,
        // 🔥 THE FIX: Tell NVIDIA to close the socket immediately so Node doesn't pool it
        'Connection': 'close' 
      },
      body: payload,
      // 🔥 THE FIX: Explicitly disable Node.js's native socket pooling
      keepalive: false 
    });

    // Pass the exact status back
    res.status(response.status);

    // Stream the raw response directly back to ReqBin / Janitor
    if (response.body) {
      // Set stream headers if the frontend requested a stream
      if (req.body.stream) {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'close');
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(decoder.decode(value, { stream: true }));
      }
      res.end();
    } else {
      res.end();
    }

  } catch (error) {
    console.error('Fetch error:', error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: error.message });
    }
  }
});

app.listen(process.env.PORT || 3000, () => {
  console.log('Dead-Socket Bypass Proxy running on port', process.env.PORT || 3000);
});
