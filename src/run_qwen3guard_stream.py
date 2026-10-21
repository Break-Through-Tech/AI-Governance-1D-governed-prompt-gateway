"""Run Qwen3Guard-Stream from Hugging Face over the jailbreak and ToxicChat data.

Qwen3Guard-Stream classifies token by token. We prefill the user prompt, then
stream the assistant response (if any) one token at a time and keep the worst
risk level seen. A sample is predicted unsafe if that level is in
--unsafe-levels (default: Unsafe only; add Controversial for a stricter gate).

NOTE: written against the model card API (`stream_moderate_from_ids`); the
exact shape of the returned dict may differ by model revision - check
`_parse` if results look empty.

Usage:
    python src/run_qwen3guard_stream.py --limit 200
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data import load_eval_sets

SEVERITY = {"Safe": 0, "Controversial": 1, "Unsafe": 2}


def _parse(result: dict) -> tuple[str, str | None]:
    """Return (risk_level, category) from one stream_moderate_from_ids call."""
    level = result["risk_level"][-1]
    cat = result.get("category", [None])[-1]
    return level, cat


@torch.no_grad()
def moderate(model, tok, prompt: str, response: str, max_resp_tokens: int):
    device = model.device
    user_ids = tok.encode(prompt, add_special_tokens=False)
    result, state = model.stream_moderate_from_ids(
        torch.tensor(user_ids, device=device), role="user", stream_state=None
    )
    worst, worst_cat = _parse(result)
    prompt_level = worst

    if response:
        for tid in tok.encode(response, add_special_tokens=False)[:max_resp_tokens]:
            result, state = model.stream_moderate_from_ids(
                torch.tensor([tid], device=device), role="assistant", stream_state=state
            )
            level, cat = _parse(result)
            if SEVERITY[level] > SEVERITY[worst]:
                worst, worst_cat = level, cat
    return {"prompt_level": prompt_level, "worst_level": worst, "category": worst_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3Guard-Stream-0.6B")
    ap.add_argument("--limit", type=int, default=None, help="samples per dataset")
    ap.add_argument("--max-resp-tokens", type=int, default=256)
    ap.add_argument("--unsafe-levels", nargs="+", default=["Unsafe"])
    ap.add_argument("--out", default="results/qwen3guard_stream")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in load_eval_sets(args.limit).items():
        rows = [
            moderate(model, tok, r.prompt, r.response, args.max_resp_tokens)
            for r in tqdm(df.itertuples(), total=len(df), desc=name)
        ]
        df = df.reset_index(drop=True).join(__import__("pandas").DataFrame(rows))
        df["pred"] = df["worst_level"].isin(args.unsafe_levels).astype(int)
        df.to_csv(out_dir / f"{name}.csv", index=False)

        report = classification_report(
            df["label"], df["pred"], target_names=["safe", "unsafe"],
            output_dict=True, zero_division=0,
        )
        (out_dir / f"{name}_report.json").write_text(json.dumps(report, indent=2))
        print(f"\n== {name} (n={len(df)})")
        print(classification_report(df["label"], df["pred"],
                                    target_names=["safe", "unsafe"], zero_division=0))


if __name__ == "__main__":
    main()
