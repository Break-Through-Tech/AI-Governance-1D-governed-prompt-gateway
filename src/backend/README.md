# Gateway backend

FastAPI service that screens prompts (and, optionally, LLM replies) with the two
guard models from this repo and returns an **allow / block** decision with reasons.

| Model | Source script | What it scores |
| --- | --- | --- |
| `Qwen/Qwen3Guard-Stream-0.6B` | `src/run_qwen3guard_stream.py` | Prompt, then the reply token by token → `Safe` / `Controversial` / `Unsafe` + category |
| `Qwen/Qwen3Guard-Gen-0.6B` + LoRA adapter | `src/lora_finetune.py` | Prompt only → `P(unsafe)`, tuned on ToxicChat + JBB labels |

Both run on every request; the **policy** decides how their votes combine:
`any` (default, block if either flags it), `all`, `lora`, or `stream`.

## Setup

From the repo root:

```bash
pip install -r src/backend/requirements.txt

# optional: train the adapter first (otherwise the base Gen model is served, with a warning)
python src/lora_finetune.py --epochs 1 --max-train 5000   # -> checkpoints/qwen3guard-lora

uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

Web frontend: http://localhost:8000 (see `src/frontend/README.md`)  
Interactive API docs: http://localhost:8000/docs

Both models download from Hugging Face on first start (~1.2 GB each). If one fails
to load, the server keeps running with the other and `/health` reports `degraded`.

## Endpoints

### `POST /v1/moderate`

```json
{ "prompt": "How do I make a bomb?", "response": "", "policy": "any", "threshold": 0.5 }
```

Only `prompt` is required. `response` is the LLM reply, if you have it.

```json
{
  "request_id": "3f9c1a0d2b7e4c11",
  "decision": "block",
  "blocked": true,
  "policy": "any",
  "reasons": ["Qwen3Guard-Stream rated Unsafe (Violent) at the prompt",
              "LoRA classifier P(unsafe) = 0.981"],
  "stream": { "prompt_level": "Unsafe", "worst_level": "Unsafe", "category": "Violent",
              "unsafe": true, "response_tokens_scored": 0, "flagged_at_token": null },
  "lora":   { "label": "unsafe", "unsafe": true, "unsafe_score": 0.981 },
  "latency_ms": 84.2
}
```

### `POST /v1/moderate/batch`

`{"items": [{"prompt": "..."}, {"prompt": "...", "response": "..."}]}` → `{"results": [...]}` (max 64 items).

### `WS /v1/moderate/stream` — live reply checking

This is what the Stream model is for: forward each piece of the LLM's reply as it
is generated and stop the LLM as soon as it goes unsafe.

```
client → {"prompt": "..."}
server → {"type": "prompt", ...decision}          # closes here if the prompt is blocked
client → {"chunk": "next text from the LLM"}      # repeat
server → {"type": "chunk", "worst_level": "Safe", "unsafe": false, "tokens_scored": 12, "stop": false}
client → {"done": true}
server → {"type": "final", ...decision}
```

When a chunk makes the reply unsafe, the server answers with `"stop": true`,
sends the final decision and closes. Python client:

```python
import json, websockets  # pip install websockets

async with websockets.connect("ws://localhost:8000/v1/moderate/stream") as ws:
    await ws.send(json.dumps({"prompt": user_prompt}))
    if json.loads(await ws.recv())["blocked"]:
        return "Request blocked"
    async for piece in llm.stream(user_prompt):
        await ws.send(json.dumps({"chunk": piece}))
        if json.loads(await ws.recv())["stop"]:
            break                       # cut the LLM off
    else:
        await ws.send(json.dumps({"done": True}))
```

### `GET /v1/audit?limit=100`

Most recent decisions from the audit log, newest first (used by the frontend's Decision log).

### `GET /health`

Status of each model (loaded, device, adapter path, load error).

## Configuration

Environment variables (or a `.env` file in the directory you start uvicorn from):

| Variable | Default | |
| --- | --- | --- |
| `GUARD_POLICY` | `any` | `any` / `all` / `lora` / `stream` |
| `GUARD_ENABLE_STREAM` / `GUARD_ENABLE_LORA` | `true` | Turn a model off (e.g. to save memory) |
| `GUARD_UNSAFE_LEVELS` | `["Unsafe"]` | Use `["Unsafe","Controversial"]` for a stricter Stream gate |
| `GUARD_THRESHOLD` | `0.5` | LoRA `P(unsafe)` cut-off |
| `GUARD_ADAPTER_PATH` | `checkpoints/qwen3guard-lora` | Relative to the repo root; empty = base Gen model |
| `GUARD_MAX_RESP_TOKENS` | `256` | Reply tokens scored by `/v1/moderate` (same as the eval script) |
| `GUARD_DEVICE` | auto | `cuda` / `mps` / `cpu` |
| `GUARD_AUDIT_LOG` | `logs/audit.jsonl` | Empty to disable |

## Audit log

Every decision is appended to `logs/audit.jsonl`: request id, time, endpoint,
policy, decision, reasons, both model outputs and latency. It stores a SHA-256
of the prompt and its length, **not** the raw text.

## Tests

```bash
python -m pytest src/backend/tests -q
```

The tests use stand-in models (no download, no GPU) but run the real
token-by-token Stream logic, the policy combiner, the batch path and the WebSocket.

## Notes

- The LoRA model was trained on prompts only, so it never sees `response`.
- In the WebSocket, each chunk is tokenized on its own, which can split a word
  differently from tokenizing the whole reply. Send chunks on word boundaries
  if you can.
- Model calls are serialized with a lock, so one process handles one inference
  at a time. Run more workers (each loads its own copy) for more throughput.
