"""Qwen3Guard-Stream (from src/run_qwen3guard_stream.py) as a live moderator.

The Stream model scores token by token: the user prompt is prefilled once, then
each assistant token is fed in with the running `stream_state`. We keep the
worst risk level seen, exactly like `moderate()` in run_qwen3guard_stream.py.

Two ways to use it:
  * `moderate(prompt, response)` - one-shot, for a finished prompt/response pair.
  * `open_session(prompt)` + `session.feed(chunk)` - incremental, for checking an
    LLM's reply while it is being generated (used by the WebSocket endpoint).
"""
import logging
import threading
from dataclasses import dataclass, field

from .classifier import pick_device
from .config import Settings

log = logging.getLogger(__name__)

SEVERITY = {"Safe": 0, "Controversial": 1, "Unsafe": 2}


def _parse(result: dict) -> tuple[str, str | None]:
    """(risk_level, category) from one stream_moderate_from_ids call.

    Model revisions return either scalars or per-token lists, so accept both
    (same handling as app.py)."""
    level = result["risk_level"]
    cat = result.get("category")
    if isinstance(level, (list, tuple)):
        level = level[-1]
    if isinstance(cat, (list, tuple)):
        cat = cat[-1] if cat else None
    return str(level), (str(cat) if cat else None)


@dataclass
class StreamVerdict:
    prompt_level: str
    worst_level: str
    category: str | None
    unsafe: bool
    response_tokens_scored: int = 0
    flagged_at_token: int | None = None  # index of the response token that hit an unsafe level


@dataclass
class StreamSession:
    """Running moderation state for one prompt + an assistant reply arriving in chunks."""
    guard: "StreamGuard"
    state: object
    prompt_level: str
    worst_level: str
    category: str | None
    tokens_scored: int = 0
    flagged_at_token: int | None = None
    closed: bool = field(default=False, repr=False)

    @property
    def unsafe(self) -> bool:
        return self.worst_level in self.guard.unsafe_levels

    def verdict(self) -> StreamVerdict:
        return StreamVerdict(self.prompt_level, self.worst_level, self.category, self.unsafe,
                             self.tokens_scored, self.flagged_at_token)

    def feed(self, text: str) -> StreamVerdict:
        """Score the next chunk of the assistant reply."""
        if self.closed:
            raise RuntimeError("session is closed")
        self.guard.feed(self, text)
        return self.verdict()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.guard.close(self)


class StreamGuard:
    name = "stream"

    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.s = settings
        self.unsafe_levels = set(settings.unsafe_levels)
        self.device = pick_device(settings.device)
        if self.device == "cpu":
            dtype = torch.float32
        elif self.device == "cuda" and not torch.cuda.is_bf16_supported():
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        self.tok = AutoTokenizer.from_pretrained(settings.stream_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            settings.stream_model, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device).eval()

        self._lock = threading.Lock()
        log.info("Stream guard ready on %s", self.device)

    # -- low level -----------------------------------------------------------------
    def _step(self, ids: list[int], role: str, state):
        import torch

        with torch.no_grad():
            result, state = self.model.stream_moderate_from_ids(
                torch.tensor(ids, device=self.device), role=role, stream_state=state
            )
        return _parse(result), state

    def open_session(self, prompt: str) -> StreamSession:
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError("prompt is empty after tokenization")
        with self._lock:
            (level, cat), state = self._step(ids, "user", None)
        return StreamSession(self, state, prompt_level=level, worst_level=level, category=cat)

    def feed(self, sess: StreamSession, text: str, budget: int | None = None) -> None:
        ids = self.tok.encode(text, add_special_tokens=False)
        if budget is not None:
            ids = ids[: max(0, budget - sess.tokens_scored)]
        with self._lock:
            for tid in ids:
                (level, cat), sess.state = self._step([tid], "assistant", sess.state)
                sess.tokens_scored += 1
                if SEVERITY.get(level, 0) > SEVERITY.get(sess.worst_level, 0):
                    sess.worst_level, sess.category = level, cat
                    if sess.flagged_at_token is None and level in self.unsafe_levels:
                        sess.flagged_at_token = sess.tokens_scored - 1
                if sess.worst_level == "Unsafe":
                    break  # can't get any worse; stop spending compute

    def close(self, sess: StreamSession) -> None:
        close = getattr(self.model, "close_stream", None)
        if callable(close):
            try:
                with self._lock:
                    close(sess.state)
            except Exception:  # noqa: BLE001 - cleanup must never fail a request
                log.debug("close_stream failed", exc_info=True)

    # -- one shot ------------------------------------------------------------------
    def moderate(self, prompt: str, response: str = "") -> StreamVerdict:
        sess = self.open_session(prompt)
        try:
            if response and sess.worst_level != "Unsafe":
                self.feed(sess, response, budget=self.s.max_resp_tokens)
            return sess.verdict()
        finally:
            sess.close()
