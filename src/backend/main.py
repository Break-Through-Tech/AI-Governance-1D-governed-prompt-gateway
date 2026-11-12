"""Governed prompt gateway API: Qwen3Guard-Stream + LoRA-tuned Qwen3Guard-Gen.

Run from the repo root:
    uvicorn src.backend.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health                   model status
    POST /v1/moderate              one prompt (+ optional LLM response) -> allow/block
    POST /v1/moderate/batch        many prompts at once
    WS   /v1/moderate/stream       check an LLM reply live, chunk by chunk, while it streams
    GET  /v1/audit                 most recent decisions from the audit log
    GET  /                         the web frontend (src/frontend)
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import REPO_ROOT, Settings, get_settings
from .gateway import AuditLog, audit_record, decide, new_request_id
from .schema import (
    BatchModerateRequest, BatchModerateResponse, Decision, HealthResponse, LoraResult,
    ModelStatus, ModerateRequest, StreamResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")


@dataclass
class Guards:
    stream: object | None = None   # StreamGuard
    lora: object | None = None     # GuardClassifier
    errors: dict[str, str] = field(default_factory=dict)
    loading: bool = False


async def _load(name: str, factory, settings: Settings, guards: Guards) -> None:
    try:
        setattr(guards, name, await asyncio.to_thread(factory, settings))
    except Exception as exc:  # noqa: BLE001 - keep serving with the other model
        log.exception("failed to load %s model", name)
        guards.errors[name] = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = app.state.settings
    # Tests (or a notebook) can pre-set app.state.guards to skip model loading.
    if getattr(app.state, "guards", None) is None:
        guards = app.state.guards = Guards(loading=True)
        jobs = []
        if s.enable_stream:
            from .stream_guard import StreamGuard
            jobs.append(_load("stream", StreamGuard, s, guards))
        if s.enable_lora:
            from .classifier import GuardClassifier
            jobs.append(_load("lora", GuardClassifier, s, guards))
        await asyncio.gather(*jobs)
        guards.loading = False
    app.state.audit = AuditLog(s)
    yield
    app.state.guards = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Governed Prompt Gateway", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["*"], allow_headers=["*"])

    # ---------------------------------------------------------------- helpers
    def guards_of(request_app: FastAPI) -> Guards:
        g = getattr(request_app.state, "guards", None)
        if g is None or g.loading:
            raise HTTPException(503, "Models are still loading")
        if g.stream is None and g.lora is None:
            raise HTTPException(503, f"No guard model is available: {g.errors or 'all disabled'}")
        return g

    def to_decision(rid, policy, blocked, reasons, sv, lv, started) -> Decision:
        return Decision(
            request_id=rid, decision="block" if blocked else "allow", blocked=blocked,
            policy=policy, reasons=reasons,
            stream=StreamResult(**asdict(sv)) if sv else None,
            lora=LoraResult(label=lv.label, unsafe=lv.label == "unsafe",
                            unsafe_score=lv.unsafe_score) if lv else None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def run_one(g: Guards, prompt: str, response: str, threshold):
        sv = g.stream.moderate(prompt, response) if g.stream else None
        lv = g.lora.classify([prompt], threshold)[0] if g.lora else None
        return sv, lv

    def finish(request_app, endpoint, policy, prompt, response, sv, lv, started, rid=None) -> Decision:
        blocked, reasons = decide(policy, sv, lv)
        rid = rid or new_request_id()
        request_app.state.audit.write(audit_record(
            rid, endpoint, policy, prompt, response, blocked, reasons, sv, lv, started))
        return to_decision(rid, policy, blocked, reasons, sv, lv, started)

    # ---------------------------------------------------------------- routes
    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        g: Guards | None = getattr(request.app.state, "guards", None)
        s = request.app.state.settings
        stream = g.stream if g else None
        lora = g.lora if g else None
        errors = g.errors if g else {}
        wanted = [n for n, on in (("stream", s.enable_stream), ("lora", s.enable_lora)) if on]
        loaded = [n for n in wanted if (stream if n == "stream" else lora) is not None]
        status = "loading" if (g is None or g.loading) else ("ok" if len(loaded) == len(wanted) else "degraded")
        return HealthResponse(
            status=status, policy=s.policy,
            stream=ModelStatus(enabled=s.enable_stream, loaded=stream is not None, model=s.stream_model,
                               device=getattr(stream, "device", None), error=errors.get("stream")),
            lora=ModelStatus(enabled=s.enable_lora, loaded=lora is not None, model=s.base_model,
                             adapter=getattr(lora, "adapter", None),
                             device=getattr(lora, "device", None), error=errors.get("lora")),
        )

    @app.post("/v1/moderate", response_model=Decision)
    async def moderate(req: ModerateRequest, request: Request):
        g = guards_of(request.app)
        started = time.perf_counter()
        policy = req.policy or request.app.state.settings.policy
        try:
            sv, lv = await asyncio.to_thread(run_one, g, req.prompt, req.response, req.threshold)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return finish(request.app, "/v1/moderate", policy, req.prompt, req.response, sv, lv, started)

    @app.post("/v1/moderate/batch", response_model=BatchModerateResponse)
    async def moderate_batch(req: BatchModerateRequest, request: Request):
        g = guards_of(request.app)
        s = request.app.state.settings
        if len(req.items) > s.max_batch_items:
            raise HTTPException(413, f"At most {s.max_batch_items} items per request")
        if any(not it.prompt.strip() for it in req.items):
            raise HTTPException(422, "Prompts must be non-empty")
        policy = req.policy or s.policy
        started = time.perf_counter()

        def work():
            prompts = [it.prompt for it in req.items]
            lora_vs = g.lora.classify(prompts, req.threshold) if g.lora else [None] * len(prompts)
            stream_vs = [g.stream.moderate(it.prompt, it.response) if g.stream else None for it in req.items]
            return stream_vs, lora_vs

        stream_vs, lora_vs = await asyncio.to_thread(work)
        return BatchModerateResponse(results=[
            finish(request.app, "/v1/moderate/batch", policy, it.prompt, it.response, sv, lv, started)
            for it, sv, lv in zip(req.items, stream_vs, lora_vs)
        ])

    @app.websocket("/v1/moderate/stream")
    async def moderate_stream(ws: WebSocket):
        """Live check of an LLM reply while it is generated.

        client -> {"prompt": "...", "policy"?: "...", "threshold"?: 0.5}
        server -> {"type": "prompt", ...Decision}            (closes if blocked)
        client -> {"chunk": "next piece of the reply"}       (repeat)
        server -> {"type": "chunk", "worst_level", "category", "unsafe", "tokens_scored", "stop"}
        client -> {"done": true}
        server -> {"type": "final", ...Decision}             then closes
        If a chunk makes the reply unsafe, the server sends "stop": true, then the
        final decision, and closes - the caller should cut off generation.
        """
        await ws.accept()
        s = ws.app.state.settings
        try:
            g = guards_of(ws.app)
        except HTTPException as exc:
            await ws.send_json({"type": "error", "detail": exc.detail})
            await ws.close(code=1013)
            return

        sess = None
        try:
            try:
                first = ModerateRequest(**(await ws.receive_json()))
            except (ValidationError, TypeError, ValueError) as exc:
                await ws.send_json({"type": "error", "detail": f"first message must be {{'prompt': ...}}: {exc}"})
                await ws.close(code=1003)
                return
            policy = first.policy or s.policy
            started = time.perf_counter()
            rid = new_request_id()
            response_parts: list[str] = []

            lv = await asyncio.to_thread(g.lora.classify, [first.prompt], first.threshold) if g.lora else None
            lv = lv[0] if lv else None
            if g.stream:
                sess = await asyncio.to_thread(g.stream.open_session, first.prompt)
            sv = sess.verdict() if sess else None
            blocked, reasons = decide(policy, sv, lv)
            await ws.send_json({"type": "prompt", **to_decision(
                rid, policy, blocked, reasons, sv, lv, started).model_dump()})

            while not blocked:
                msg = await ws.receive_json()
                if msg.get("done"):
                    break
                chunk = msg.get("chunk")
                if not isinstance(chunk, str):
                    await ws.send_json({"type": "error", "detail": "expected {'chunk': str} or {'done': true}"})
                    continue
                response_parts.append(chunk)
                if sess is None:
                    await ws.send_json({"type": "chunk", "scored": False, "stop": False,
                                        "detail": "stream model not loaded; reply is not scored"})
                    continue
                sv = await asyncio.to_thread(sess.feed, chunk)
                blocked, reasons = decide(policy, sv, lv)
                await ws.send_json({"type": "chunk", "scored": True, "worst_level": sv.worst_level,
                                    "category": sv.category, "unsafe": sv.unsafe,
                                    "tokens_scored": sv.response_tokens_scored, "stop": blocked})

            sv = sess.verdict() if sess else None
            final = finish(ws.app, "/v1/moderate/stream", policy, first.prompt,
                           "".join(response_parts), sv, lv, started, rid)
            await ws.send_json({"type": "final", **final.model_dump()})
            await ws.close()
        except WebSocketDisconnect:
            pass
        finally:
            if sess is not None:
                await asyncio.to_thread(sess.close)

    @app.get("/v1/audit")
    def audit(request: Request, limit: int = 100):
        """Most recent decisions, newest first."""
        return {"enabled": request.app.state.audit.path is not None,
                "records": request.app.state.audit.tail(max(1, min(limit, 1000)))}

    # Serve the frontend last so it never shadows an API route.
    frontend = REPO_ROOT / "src" / "frontend"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


app = create_app()
