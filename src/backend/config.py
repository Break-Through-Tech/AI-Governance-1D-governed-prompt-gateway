"""Runtime settings, read from environment variables (prefix GUARD_) or a .env file."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (…/AI-Governance-1D-governed-prompt-gateway). lora_finetune.py is run from
# here, so relative paths like checkpoints/qwen3guard-lora resolve against it.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GUARD_", env_file=".env", extra="ignore")

    # ---- Qwen3Guard-Stream (same defaults as src/run_qwen3guard_stream.py) ----
    enable_stream: bool = True
    stream_model: str = "Qwen/Qwen3Guard-Stream-0.6B"
    # A risk level in this list counts as unsafe. Add "Controversial" for a stricter gate.
    unsafe_levels: list[str] = ["Unsafe"]
    max_resp_tokens: int = 256        # cap on response tokens scored per request

    # ---- Qwen3Guard-Gen + LoRA adapter (same defaults as src/lora_finetune.py) ----
    enable_lora: bool = True
    base_model: str = "Qwen/Qwen3Guard-Gen-0.6B"
    adapter_path: str | None = "checkpoints/qwen3guard-lora"  # empty -> serve the base model
    merge_adapter: bool = True        # fold LoRA weights into the base for faster inference
    max_len: int = 768                # token budget, matches MAX_LEN in training
    max_chars: int = 3000             # prompts were clipped to 3000 chars in training
    batch_size: int = 8
    threshold: float = 0.5            # P(unsafe) at or above this -> "unsafe"

    # ---- Gateway policy ----
    # any:    block if either model flags the prompt (safest)
    # all:    block only if every enabled model flags it
    # lora:   decide with the LoRA model only (Stream still reported)
    # stream: decide with the Stream model only (LoRA still reported)
    policy: Literal["any", "all", "lora", "stream"] = "any"

    # ---- Service ----
    device: str | None = None         # cuda / mps / cpu; auto-detect when unset
    max_batch_items: int = 64
    cors_origins: list[str] = ["*"]
    audit_log: str | None = "logs/audit.jsonl"  # empty -> no audit file

    @staticmethod
    def local_path(path: str | None) -> Path | None:
        """Resolve a local path against the repo root so the server works from any cwd."""
        if not path:
            return None
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
