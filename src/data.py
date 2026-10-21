"""Load the two zipped datasets into a common (prompt, response, label) frame.

label: 1 = unsafe / harmful, 0 = safe.
"""
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read(zip_name: str, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(DATA_DIR / zip_name) as z, z.open(member) as f:
        return pd.read_csv(f)


def load_toxicchat(split: str = "test") -> pd.DataFrame:
    df = _read("lmsys_toxicchat.zip", f"lmsys_toxicchat/{split}.csv")
    return pd.DataFrame({
        "source": "toxicchat",
        "prompt": df["user_input"],
        "response": df["model_output"],
        "label": df["toxicity"].astype(int),
    })


def load_jailbreak_behaviors() -> pd.DataFrame:
    harmful = _read("jailbreak.zip", "jailbreak/behaviors_harmful.csv")
    benign = _read("jailbreak.zip", "jailbreak/behaviors_benign.csv")
    return pd.concat([
        pd.DataFrame({"source": "jbb_harmful", "prompt": harmful["Goal"], "response": "", "label": 1}),
        pd.DataFrame({"source": "jbb_benign", "prompt": benign["Goal"], "response": "", "label": 0}),
    ], ignore_index=True)


def load_judge_comparison() -> pd.DataFrame:
    """Jailbreak prompts + target model responses; label = human majority vote."""
    df = _read("jailbreak.zip", "jailbreak/judge_comparison_test.csv")
    return pd.DataFrame({
        "source": "judge_comparison",
        "prompt": df["prompt"],
        "response": df["target_response"].fillna(""),
        "label": df["human_majority"].astype(int),
    })


def load_eval_sets(limit: int | None = None) -> dict[str, pd.DataFrame]:
    sets = {
        "toxicchat_test": load_toxicchat("test"),
        "jbb_behaviors": load_jailbreak_behaviors(),
        "judge_comparison": load_judge_comparison(),
    }
    if limit:
        sets = {k: v.sample(min(limit, len(v)), random_state=0) for k, v in sets.items()}
    return sets
