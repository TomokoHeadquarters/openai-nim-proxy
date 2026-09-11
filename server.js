// server.js - OpenAI to NVIDIA NIM Proxy (optimizado velocidad + streaming)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Deja estos en false para máxima velocidad
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'nvidia/nemotron-3-nano-30b-a3b',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-sonnet': '"google/gemma-4-31b-it',
  'gemini-pro': 'nvidia/nemotron-3-nano-30b-a3b'
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'NIM Proxy', streaming: true });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    let nimModel = MODEL_MAPPING[model] || model;

    // Construir request limpio
    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.8,
      max_tokens: max_tokens || 2048,
      stream: !!stream          // respeta lo que pida Janitor
    };

    // Clamp temperature (algunos modelos de NIM no aceptan > 1.0)
    if (nimRequest.temperature > 1.0) nimRequest.temperature = 1.0;
    if (nimRequest.temperature < 0) nimRequest.temperature = 0;

    // Limpieza de parámetros que suelen romper
    delete nimRequest.presence_penalty;
    delete nimRequest.frequency_penalty;
    delete nimRequest.stop;

    if (nimRequest.max_tokens > 4096) nimRequest.max_tokens = 4096;

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          'Accept': stream ? 'text/event-stream' : 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 180000 // 3 minutos
      }
    );

    // ========== STREAMING (rápido) ==========
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // importante en algunos hosts

      response.data.pipe(res); // reenvío directo, mínimo overhead

      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        if (!res.headersSent) res.status(500).end();
        else res.end();
      });

      return;
    }

    // ========== NO-STREAMING ==========
    const data = response.data;
    const choice = data.choices?.[0];

    let content = choice?.message?.content || '';

    // Solo si SHOW_REASONING está activo
    if (SHOW_REASONING && choice?.message?.reasoning_content) {
      content = `<think>\n\( {choice.message.reasoning_content}\n</think>\n\n \){content}`;
    }

    res.json({
      id: data.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: data.created || Math.floor(Date.now() / 1000),
      model: model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content
        },
        finish_reason: choice?.finish_reason || 'stop'
      }],
      usage: data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    });

  } catch (error) {
    const status = error.response?.status || 500;
    const detail = error.response?.data || { message: error.message };

    console.error('NIM Error:', status, JSON.stringify(detail).slice(0, 500));

    res.status(status).json({
      error: {
        message: detail.message || detail.error?.message || 'Request failed',
        type: 'invalid_request_error',
        code: status,
        details: detail
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`NIM Proxy running on port ${PORT}`);
});
