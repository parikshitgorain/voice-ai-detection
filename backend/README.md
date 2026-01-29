# Backend

Node.js API server and audio analysis pipeline.

Key files:
- `server.js` API entrypoint
- `config.js` runtime configuration
- `services/` processing pipeline and classifiers
- `deep/` optional Python deep model bridge

Run (local):
```
cd backend
npm install
node server.js
```
