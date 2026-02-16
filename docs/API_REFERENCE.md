# 📡 API Reference

Complete API documentation for Voice AI Detection.

## Base URL

```
http://YOUR_SERVER_IP:3000
```

---

## Authentication

All API requests require an API key in the header:

```http
X-API-Key: your_api_key_here
```

Get API keys from the admin panel at `/admin`.

---

## Endpoints

### 1. Voice Detection

Classify audio as HUMAN or AI_GENERATED.

**Endpoint:** `POST /api/voice-detection`

**Request Headers:**
```http
Content-Type: application/json
X-API-Key: your_api_key_here
```

**Request Body:**
```json
{
  "audioBase64": "base64_encoded_audio_data",
  "language": "English",
  "audioFormat": "mp3"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audioBase64 | string | Yes | Base64-encoded audio file |
| language | string | Yes | Audio language: `English`, `Hindi`, `Tamil`, `Malayalam`, `Telugu` |
| audioFormat | string | Yes | Audio format: `mp3`, `wav`, `ogg`, `flac`, `m4a` |

**Response (Success):**
```json
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.97
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| status | string | Always `"success"` for successful requests |
| classification | string | `"HUMAN"` or `"AI_GENERATED"` |
| confidenceScore | float | Confidence score (0.0 to 1.0) |

**Response (Error):**
```json
{
  "status": "error",
  "message": "Error description"
}
```

**Example (cURL):**
```bash
# Encode audio file
AUDIO_BASE64=$(base64 -w0 audio.mp3)

# Make request
curl -X POST http://localhost:3000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_your_api_key_here" \
  -d "{
    \"audioBase64\": \"$AUDIO_BASE64\",
    \"language\": \"English\",
    \"audioFormat\": \"mp3\"
  }"
```

**Example (JavaScript):**
```javascript
// Read file as base64
const fs = require('fs');
const audioBase64 = fs.readFileSync('audio.mp3', 'base64');

// Make request
const response = await fetch('http://localhost:3000/api/voice-detection', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'sk_your_api_key_here'
  },
  body: JSON.stringify({
    audioBase64: audioBase64,
    language: 'English',
    audioFormat: 'mp3'
  })
});

const result = await response.json();
console.log(result);
// { status: 'success', classification: 'AI_GENERATED', confidenceScore: 0.97 }
```

**Example (Python):**
```python
import requests
import base64

# Read and encode audio
with open('audio.mp3', 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# Make request
response = requests.post(
    'http://localhost:3000/api/voice-detection',
    headers={
        'Content-Type': 'application/json',
        'X-API-Key': 'sk_your_api_key_here'
    },
    json={
        'audioBase64': audio_base64,
        'language': 'English',
        'audioFormat': 'mp3'
    }
)

result = response.json()
print(result)
# {'status': 'success', 'classification': 'AI_GENERATED', 'confidenceScore': 0.97}
```

---

### 2. Health Check

Check if the API server is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

**Example:**
```bash
curl http://localhost:3000/health
```

---

## Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_REQUEST | Missing or invalid request parameters |
| 401 | UNAUTHORIZED | Missing or invalid API key |
| 413 | PAYLOAD_TOO_LARGE | Audio file too large (>10MB) |
| 422 | VALIDATION_ERROR | Invalid audio format or language |
| 429 | RATE_LIMIT_EXCEEDED | Too many requests |
| 500 | INTERNAL_ERROR | Server error |

**Error Response Format:**
```json
{
  "status": "error",
  "message": "Detailed error message"
}
```

---

## Rate Limiting

**Default Limits:**
- 60 requests per minute per API key
- 1000 requests per day per API key

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1234567890
```

When rate limit is exceeded:
```json
{
  "status": "error",
  "message": "Rate limit exceeded. Try again in 30 seconds."
}
```

---

## Supported Languages

| Language | Code | Model |
|----------|------|-------|
| English | `English` | multitask_English.pt |
| Hindi | `Hindi` | multitask_Hindi.pt |
| Tamil | `Tamil` | multitask_Tamil.pt |
| Malayalam | `Malayalam` | multitask_Malayalam.pt |
| Telugu | `Telugu` | multitask_Telugu.pt |

**Note:** Language parameter is case-sensitive.

---

## Supported Audio Formats

| Format | Extension | MIME Type |
|--------|-----------|-----------|
| MP3 | .mp3 | audio/mpeg |
| WAV | .wav | audio/wav |
| OGG | .ogg | audio/ogg |
| FLAC | .flac | audio/flac |
| M4A | .m4a | audio/mp4 |

**Audio Requirements:**
- Max file size: 10MB
- Duration: 1-60 seconds (optimal: 3-10 seconds)
- Sample rate: Any (will be resampled to 16kHz)
- Channels: Mono or Stereo (will be converted to mono)

---

## Response Time

**Typical Response Times:**
- With GPU: 0.2-0.4 seconds
- With CPU: 2-5 seconds

**Timeout:** 30 seconds

---

## Best Practices

### 1. Audio Quality
- Use clear speech with minimal background noise
- Avoid music or multiple speakers
- Optimal duration: 3-10 seconds
- Higher quality audio = better accuracy

### 2. Error Handling
```javascript
try {
  const response = await fetch(url, options);
  const result = await response.json();
  
  if (result.status === 'error') {
    console.error('API Error:', result.message);
    // Handle error
  } else {
    // Process result
    console.log('Classification:', result.classification);
    console.log('Confidence:', result.confidenceScore);
  }
} catch (error) {
  console.error('Network Error:', error);
  // Handle network error
}
```

### 3. Rate Limiting
```javascript
// Implement exponential backoff
async function makeRequestWithRetry(url, options, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, options);
      if (response.status === 429) {
        // Rate limited, wait and retry
        const waitTime = Math.pow(2, i) * 1000; // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, waitTime));
        continue;
      }
      return await response.json();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
    }
  }
}
```

### 4. Caching
```javascript
// Cache results to avoid duplicate requests
const cache = new Map();

function getCacheKey(audioBase64, language) {
  return `${language}:${audioBase64.substring(0, 100)}`;
}

async function classifyWithCache(audioBase64, language) {
  const key = getCacheKey(audioBase64, language);
  
  if (cache.has(key)) {
    return cache.get(key);
  }
  
  const result = await classify(audioBase64, language);
  cache.set(key, result);
  
  return result;
}
```

---

## Admin API

### Get API Keys

**Endpoint:** `GET /api/admin/keys`

**Authentication:** Admin session required

**Response:**
```json
{
  "keys": [
    {
      "id": "key_abc123",
      "name": "Production Key",
      "preview": "sk_...ea7",
      "status": "active",
      "created_at": "2026-02-16T08:00:00.000Z",
      "type": "limited",
      "daily_limit": 1000
    }
  ]
}
```

### Create API Key

**Endpoint:** `POST /api/admin/keys`

**Request Body:**
```json
{
  "name": "My API Key",
  "type": "limited",
  "daily_limit": 1000
}
```

**Response:**
```json
{
  "key": "sk_089fbb8ceb046bc2c3c1496d9c56be510818c164548e8ea7",
  "id": "key_abc123",
  "name": "My API Key"
}
```

**Note:** Save the key immediately - it won't be shown again!

---

## Webhooks (Coming Soon)

Receive notifications when classification is complete.

**Webhook Payload:**
```json
{
  "event": "classification.completed",
  "timestamp": "2026-02-16T08:00:00.000Z",
  "data": {
    "classification": "AI_GENERATED",
    "confidenceScore": 0.97,
    "language": "English"
  }
}
```

---

## SDKs (Coming Soon)

Official SDKs for popular languages:
- JavaScript/TypeScript
- Python
- PHP
- Ruby
- Go

---

## Support

- Documentation: https://github.com/parikshitgorain/voice-ai-detection
- Issues: https://github.com/parikshitgorain/voice-ai-detection/issues
- Email: parikshitgorain@yahoo.com
