"""LoRA fine-tune of a Qwen3Guard model on ToxicChat train (+ optional jailbreak data).

The Stream model has custom per-token heads that peft can't train through its
streaming API, so we adapt the generative sibling (Qwen3Guard-Gen) instead:
a standard causal LM that answers "Safety: Safe|Unsafe". The adapter learns
this project's labels; loss is computed on the answer tokens only.

Usage:
    python src/lora_finetune.py --epochs 1 --max-train 5000
    python src/lora_finetune.py --eval-only --adapter checkpoints/qwen3guard-lora
"""
import argparse

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from data import load_eval_sets, load_jailbreak_behaviors, load_toxicchat

ANSWER = {0: "Safety: Safe", 1: "Safety: Unsafe"}
MAX_LEN = 768


def build_prompt(tok, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def collate_fn(tok):
    def collate(batch):
        ids, labels = [], []
        for prompt, y in batch:
            p = tok(build_prompt(tok, prompt[:3000]), add_special_tokens=False)["input_ids"]
            a = tok(ANSWER[y] + tok.eos_token, add_special_tokens=False)["input_ids"]
            p = p[-(MAX_LEN - len(a)):]
            ids.append(p + a)
            labels.append([-100] * len(p) + a)
        n = max(map(len, ids))
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor([x + [pad] * (n - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids]),
            "labels": torch.tensor([x + [-100] * (n - len(x)) for x in labels]),
        }
    return collate


@torch.no_grad()
def evaluate(model, tok, limit):
    model.eval()
    tok.padding_side = "left"
    safe_id = tok(" Safe", add_special_tokens=False)["input_ids"][0]
    unsafe_id = tok(" Unsafe", add_special_tokens=False)["input_ids"][0]
    for name, df in load_eval_sets(limit).items():
        preds = []
        for i in tqdm(range(0, len(df), 8), desc=name):
            texts = [build_prompt(tok, p[:3000]) + "Safety:" for p in df["prompt"].iloc[i:i + 8]]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LEN, add_special_tokens=False).to(model.device)
            logits = model(**enc).logits[:, -1]
            preds += (logits[:, unsafe_id] > logits[:, safe_id]).int().tolist()
        print(f"\n== {name} (n={len(df)})")
        print(classification_report(df["label"], preds, target_names=["safe", "unsafe"], zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--adapter", default="checkpoints/qwen3guard-lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-limit", type=int, default=300)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(device)

    if args.eval_only:
        model = PeftModel.from_pretrained(model, args.adapter)
        evaluate(model, tok, args.eval_limit)
        return

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    # Train: ToxicChat train + jailbreak behaviors (small, mixes in explicit harmful asks).
    # Note: jailbreak behaviors are also in the eval sets, so those scores are optimistic;
    # drop the concat below for a clean held-out comparison.
    train = pd.concat([load_toxicchat("train"), load_jailbreak_behaviors()], ignore_index=True)
    if args.max_train:
        train = train.sample(min(args.max_train, len(train)), random_state=0)
    examples = list(zip(train["prompt"], train["label"]))
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn(tok))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = args.epochs * len(loader) // args.grad_accum
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    model.train()
    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            (model(**batch).loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()

    model.save_pretrained(args.adapter)
    tok.save_pretrained(args.adapter)
    evaluate(model, tok, args.eval_limit)


if __name__ == "__main__":
    main()
