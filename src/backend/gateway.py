"""Combine the Stream and LoRA verdicts into one allow/block decision + audit record."""
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone

from .classifier import LoraVerdict
from .config import Settings
from .stream_guard import StreamVerdict

log = logging.getLogger(__name__)


def decide(policy: str, stream: StreamVerdict | None, lora: LoraVerdict | None) -> tuple[bool, list[str]]:
    """Return (blocked, reasons) for the given policy."""
    votes: dict[str, bool] = {}
    reasons: list[str] = []
    if stream is not None:
        votes["stream"] = stream.unsafe
        if stream.unsafe:
            where = "the prompt" if stream.flagged_at_token is None else f"reply token {stream.flagged_at_token + 1}"
            cat = f" ({stream.category})" if stream.category else ""
            reasons.append(f"Qwen3Guard-Stream rated {stream.worst_level}{cat} at {where}")
    if lora is not None:
        votes["lora"] = lora.label == "unsafe"
        if votes["lora"]:
            reasons.append(f"LoRA classifier P(unsafe) = {lora.unsafe_score:.3f}")

    if not votes:
        raise RuntimeError("no guard model is enabled")
    if policy in ("stream", "lora") and policy in votes:
        blocked = votes[policy]
        reasons = [r for r in reasons if r.startswith("Qwen3Guard-Stream" if policy == "stream" else "LoRA")]
    elif policy == "all":
        blocked = all(votes.values())
    else:  # "any", or a single-model policy whose model is disabled
        blocked = any(votes.values())
    return blocked, (reasons if blocked else [])


class AuditLog:
    """Append-only JSONL log of every decision. Stores a hash of the text, not the text."""

    def __init__(self, settings: Settings):
        self.path = settings.local_path(settings.audit_log)
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def tail(self, n: int) -> list[dict]:
        """Last n records, newest first."""
        if not self.path or not self.path.exists():
            return []
        with self._lock, self.path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=n)
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def write(self, record: dict) -> None:
        if not self.path:
            return
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("could not write audit log")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def audit_record(request_id: str, endpoint: str, policy: str, prompt: str, response: str,
                 blocked: bool, reasons: list[str], stream: StreamVerdict | None,
                 lora: LoraVerdict | None, started: float) -> dict:
    return {
        "request_id": request_id,
        "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "endpoint": endpoint,
        "policy": policy,
        "decision": "block" if blocked else "allow",
        "reasons": reasons,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_chars": len(prompt),
        "response_chars": len(response),
        "stream": asdict(stream) if stream else None,
        "lora": asdict(lora) if lora else None,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
