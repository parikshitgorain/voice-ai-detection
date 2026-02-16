# Quick Start Guide - Hackathon Compliance Testing

## Your API Configuration

**API Key:** `sk_089fbb8ceb046bc2c3c1496d9c56be510818c164548e8ea7`
**Key Name:** testing
**Key Type:** limited (1000 requests/day)
**Status:** active

## Quick Test Commands

### 1. Test API with curl (Manual Test)

```bash
curl -X POST http://localhost:3000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_089fbb8ceb046bc2c3c1496d9c56be510818c164548e8ea7" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "your_base64_audio_here"
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "classification": "HUMAN",
  "confidenceScore": 0.85
}
```

### 2. Run Self-Evaluation (Automated Test)

**Prerequisites:**
- Add real MP3 audio files to `test_data/` directory
- Update `test_data/test_manifest.json` with file paths
- Ensure API server is running

**Command:**
```bash
python3 scripts/self_evaluation.py \
  --endpoint http://localhost:3000/api/voice-detection \
  --api-key sk_089fbb8ceb046bc2c3c1496d9c56be510818c164548e8ea7 \
  --manifest test_data/test_manifest.json
```

**Output:**
- Console: Real-time test results
- File: `evaluation_results.json` with detailed scoring

### 3. Check API Server Status

```bash
curl http://localhost:3000/health
```

**Expected Response:**
```json
{
  "status": "ok"
}
```

## Available API Keys

You have 3 active API keys configured:

1. **GPU-Test-Key** (unlimited)
   - Preview: `d3027d44`
   - Type: unlimited
   
2. **Native-Server-Key** (unlimited)
   - Preview: `8685f1ad`
   - Type: unlimited
   
3. **testing** (limited) ← **Currently Using**
   - Preview: `548e8ea7`
   - Type: limited (1000/day)
   - Full Key: `sk_089fbb8ceb046bc2c3c1496d9c56be510818c164548e8ea7`

## Next Steps

### Step 1: Add Test Audio Files

Create or download MP3 audio files:
- At least 1 HUMAN voice sample
- At least 1 AI_GENERATED voice sample
- 5-10 seconds duration recommended

Place them in `test_data/` directory:
```
test_data/
├── sample_human_english.mp3
├── sample_ai_english.mp3
└── test_manifest.json
```

### Step 2: Update Test Manifest

Edit `test_data/test_manifest.json`:
```json
{
  "test_cases": [
    {
      "file_path": "test_data/sample_human_english.mp3",
      "language": "English",
      "expected_classification": "HUMAN",
      "duration_seconds": 5.0,
      "source": "Recorded from native speaker",
      "quality": "clear"
    },
    {
      "file_path": "test_data/sample_ai_english.mp3",
      "language": "English",
      "expected_classification": "AI_GENERATED",
      "duration_seconds": 5.0,
      "source": "Generated with ElevenLabs",
      "quality": "high"
    }
  ]
}
```

### Step 3: Start API Server

```bash
cd backend
node server.js
```

Server will start on `http://localhost:3000`

### Step 4: Run Self-Evaluation

```bash
python3 scripts/self_evaluation.py \
  --endpoint http://localhost:3000/api/voice-detection \
  --api-key sk_089fbb8ceb046bc2c3c1496d9c56be510818c164548e8ea7 \
  --manifest test_data/test_manifest.json
```

### Step 5: Review Results

Check `evaluation_results.json`:
```bash
cat evaluation_results.json
```

Look for:
- ✅ All format checks passing
- ✅ Response times < 30 seconds
- ✅ Correct classifications
- ✅ High confidence scores (≥ 0.8)

## Troubleshooting

### API Key Not Working
- Verify key is active in `backend/data/api_keys.json`
- Check server logs for authentication errors
- Ensure `x-api-key` header is set correctly

### Self-Evaluation Fails
- Ensure API server is running
- Check test audio files exist at specified paths
- Verify test_manifest.json is valid JSON
- Check Python dependencies: `pip install requests`

### Response Format Issues
- API should return exactly 3 fields: status, classification, confidenceScore
- Check `backend/api/voice_detection.js` for response formatting
- Verify no extra fields are being added

## Compliance Checklist

Before hackathon submission:
- [ ] Response format: Exactly 3 fields
- [ ] HTTP status: 200 for success
- [ ] Classification: "HUMAN" or "AI_GENERATED" (exact case)
- [ ] Confidence: Float 0.0-1.0
- [ ] Response time: < 30 seconds
- [ ] Self-evaluation: All tests passing
- [ ] Documentation: README updated
- [ ] Deployment: API accessible via HTTPS
- [ ] GitHub: Repository public and complete

## Support Files

- `HACKATHON_COMPLIANCE.md` - Full compliance documentation
- `UPGRADE_SUMMARY.md` - Implementation summary
- `test_data/README.md` - Test data setup guide
- `test_data/INSTRUCTIONS.txt` - Quick instructions
- `README.md` - Main API documentation

## Contact

Maintainer: Parikshit Gorain
Email: parikshitgorain@yahoo.com

---

**Ready to test!** 🚀
