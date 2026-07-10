const express = require('express');
const cors = require('cors');
const OpenAI = require('openai'); // Requires: npm install openai cors express

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;

// Initialize the OpenAI SDK pointing directly to NVIDIA NIM
const openai = new OpenAI({
  apiKey: process.env.NVIDIA_API_KEY, 
  baseURL: 'https://integrate.api.nvidia.com/v1',
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    // Feeds the EXACT raw payload from your frontend directly into the SDK.
    // ZERO hardcoded config overrides.
    const completion = await openai.chat.completions.create(req.body);

    // Handle streaming smoothly via SDK async iterator
    if (req.body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      for await (const chunk of completion) {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
      
      res.write('data: [DONE]\n\n');
      return res.end();
    } 
    
    // Handle standard JSON response
    res.json(completion);

  } catch (error) {
    console.error('API Error:', error.message);
    res.status(error.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error'
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`Raw Passthrough Proxy running on port ${PORT}`);
});
