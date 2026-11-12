"""Qwen3Guard-Gen + LoRA adapter (from src/lora_finetune.py) as a prompt classifier.

Inference mirrors `evaluate()` in lora_finetune.py: build the chat prompt, append
the "Safety:" cue, and compare the next-token logits for " Safe" vs " Unsafe".
Softmax over those two logits gives P(unsafe).
"""
import logging
import threading
from dataclasses import dataclass

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class LoraVerdict:
    label: str            # "safe" | "unsafe"
    unsafe_score: float   # P(unsafe) in [0, 1]


def pick_device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GuardClassifier:
    name = "lora"

    def __init__(self, settings: Settings):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.s = settings
        self.device = pick_device(settings.device)
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        adapter = settings.local_path(settings.adapter_path)
        if adapter and not adapter.exists():
            log.warning(
                "LoRA adapter not found at %s - serving the base %s. "
                "Train one with: python src/lora_finetune.py", adapter, settings.base_model,
            )
            adapter = None
        self.adapter = str(adapter) if adapter else None

        # lora_finetune.py saves the tokenizer next to the adapter; fall back to the base.
        try:
            self.tok = AutoTokenizer.from_pretrained(self.adapter or settings.base_model)
        except OSError:
            self.tok = AutoTokenizer.from_pretrained(settings.base_model)
        self.tok.padding_side = "left"
        # Keep the END of long prompts so the "Safety:" cue is never cut off
        # (training also kept the last tokens: p[-(MAX_LEN - len(a)):]).
        self.tok.truncation_side = "left"

        model = AutoModelForCausalLM.from_pretrained(settings.base_model, torch_dtype=dtype)
        if self.adapter:
            log.info("Loading LoRA adapter from %s", self.adapter)
            model = PeftModel.from_pretrained(model, self.adapter)
            if settings.merge_adapter:
                model = model.merge_and_unload()
        self.model = model.to(self.device).eval()

        self.safe_id = self.tok(" Safe", add_special_tokens=False)["input_ids"][0]
        self.unsafe_id = self.tok(" Unsafe", add_special_tokens=False)["input_ids"][0]

        self._lock = threading.Lock()  # torch modules aren't safe to call concurrently
        log.info("LoRA classifier ready on %s", self.device)

    def _build(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt[: self.s.max_chars]}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + "Safety:"

    def classify(self, prompts: list[str], threshold: float | None = None) -> list[LoraVerdict]:
        import torch

        thr = self.s.threshold if threshold is None else threshold
        out: list[LoraVerdict] = []
        with self._lock, torch.no_grad():
            for i in range(0, len(prompts), self.s.batch_size):
                texts = [self._build(p) for p in prompts[i : i + self.s.batch_size]]
                enc = self.tok(
                    texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.s.max_len, add_special_tokens=False,
                ).to(self.device)
                logits = self.model(**enc).logits[:, -1].float()
                pair = torch.stack([logits[:, self.safe_id], logits[:, self.unsafe_id]], dim=-1)
                p_unsafe = pair.softmax(-1)[:, 1].tolist()
                out += [LoraVerdict("unsafe" if p >= thr else "safe", round(p, 6)) for p in p_unsafe]
        return out
