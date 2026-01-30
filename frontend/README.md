# Frontend

Static single-page UI for the voice detection API.

## Local run
```bash
python3 -m http.server 5173 --directory frontend
```

## Runtime config
Edit `frontend/config.js`:
- `apiBaseUrl`: optional base URL for the API (empty = same origin)
- `apiKey`: leave empty (UI never stores or pre-fills keys)

## Notes
- API key is entered by the user on each session.
- No browser storage is used for the API key.
