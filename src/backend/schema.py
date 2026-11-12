from typing import Literal

from pydantic import BaseModel, Field

Policy = Literal["any", "all", "lora", "stream"]


class ModerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20_000, description="User prompt to check")
    response: str = Field("", max_length=50_000,
                          description="Optional LLM reply; scored token by token by Qwen3Guard-Stream")
    threshold: float | None = Field(None, ge=0, le=1, description="Override the LoRA P(unsafe) threshold")
    policy: Policy | None = Field(None, description="Override the server's combine policy")


class BatchItem(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20_000)
    response: str = Field("", max_length=50_000)


class BatchModerateRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1)
    threshold: float | None = Field(None, ge=0, le=1)
    policy: Policy | None = None


class StreamResult(BaseModel):
    prompt_level: str = Field(..., examples=["Safe"])
    worst_level: str = Field(..., examples=["Unsafe"])
    category: str | None = Field(None, examples=["Violent"])
    unsafe: bool
    response_tokens_scored: int = 0
    flagged_at_token: int | None = None


class LoraResult(BaseModel):
    label: str = Field(..., examples=["unsafe"])
    unsafe: bool
    unsafe_score: float = Field(..., description="P(unsafe) from the Safe/Unsafe logits")


class Decision(BaseModel):
    request_id: str
    decision: Literal["allow", "block"]
    blocked: bool
    policy: Policy
    reasons: list[str]
    stream: StreamResult | None = None
    lora: LoraResult | None = None
    latency_ms: float


class BatchModerateResponse(BaseModel):
    results: list[Decision]


class ModelStatus(BaseModel):
    enabled: bool
    loaded: bool
    model: str
    adapter: str | None = None
    device: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "degraded"]
    policy: Policy
    stream: ModelStatus
    lora: ModelStatus
