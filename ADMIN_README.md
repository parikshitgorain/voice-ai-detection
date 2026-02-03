# Admin Panel

## Access

**URL**: `https://voiceai.parikshit.dev/admin/login.html`

**Default Credentials**:
- Username: `admin`
- Password: `admin123`

## Features

### Dashboard
- View total and active API keys
- Monitor total API calls
- Track today's API usage

### API Key Management
- Create new API keys (format: `sk_...`)
- Enable/disable keys
- Delete keys
- View usage statistics per key
- Copy keys securely (shown only once)

## Security

- JWT-based authentication (24h expiration)
- Bcrypt password hashing (12 rounds)
- API keys stored as SHA-256 hashes
- Unlimited API usage with tracking

## Storage

All data stored in JSON files:
- `/var/www/voice-ai-detection/data/admin.json` - Admin credentials
- `/var/www/voice-ai-detection/data/api_keys.json` - API key hashes
- `/var/www/voice-ai-detection/data/usage.json` - Usage tracking

## API Endpoints

### Public (No Auth)
- `POST /admin/login` - Authenticate and get JWT token

### Protected (Requires Bearer Token)
- `GET /admin/session` - Verify token
- `GET /admin/stats` - Get dashboard statistics
- `GET /admin/api-keys` - List all API keys
- `POST /admin/api-keys` - Create new API key
- `PATCH /admin/api-keys/:id` - Update key status (active/inactive)
- `DELETE /admin/api-keys/:id` - Delete API key

## Changing Admin Password

```bash
cd /var/www/voice-ai-detection/backend
node -e "const bcrypt = require('bcrypt'); bcrypt.hash('YOUR_NEW_PASSWORD', 12, (err, hash) => { console.log(hash); });"
# Copy the hash and update /var/www/voice-ai-detection/data/admin.json
```

## Notes

- Daily usage counters can be reset via cron job calling `resetDailyCounters()` from admin module
- API keys are validated and tracked automatically on each API request
- Old config-based API key (RAPIDAPI_KEY) still works as fallback
