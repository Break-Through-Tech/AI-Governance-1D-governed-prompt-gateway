# Gateway frontend

A plain HTML/CSS/JS page for the backend in `src/backend`. No build step, no npm.

## Run it

Start the backend from the repo root. It serves this folder at `/`:

```bash
uvicorn src.backend.main:app --port 8000
```

Then open http://localhost:8000.

To serve the page from somewhere else (or open `index.html` straight from disk),
point it at the API with `?api=`:

```
index.html?api=http://localhost:8000
```

`?tab=check|live|batch|log` opens a specific view.

## Views

- **Check a prompt**: run one prompt (and optionally the LLM's reply) through both
  guard models and see the decision, each model's vote and the reasons.
- **Watch a reply**: streams a reply word by word over the `/v1/moderate/stream`
  WebSocket. The colored strip shows the risk level after every piece; if the
  reply turns unsafe, the gateway cuts it off and the rest is shown struck out.
  This is the demo for why we use Qwen3Guard-Stream.
- **Batch test**: paste up to 64 prompts, one per line, and compare results.
- **Decision log**: the latest entries from the audit log (`GET /v1/audit`).

The "Block when" menu on each view overrides the server's `GUARD_POLICY` for that request.
The status pills at the top show whether each model loaded; hover one for details.

## Files

- `index.html`: markup for the four views
- `styles.css`: colors and type are defined as variables at the top, with a dark theme
- `app.js`: API calls, the live-stream client and rendering
