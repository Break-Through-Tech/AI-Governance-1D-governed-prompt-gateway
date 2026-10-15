import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_toxic_chat():
    train_path = os.path.join(OUTPUT_DIR, "toxic_chat_train.csv")
    test_path = os.path.join(OUTPUT_DIR, "toxic_chat_test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("[1/3] LMSYS Toxic-Chat already downloaded. Skipping.")
        return

    print("[1/3] Downloading LMSYS Toxic-Chat...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    ds["train"].to_pandas().to_csv(train_path, index=False)
    ds["test"].to_pandas().to_csv(test_path, index=False)
    print(f" -> Saved: {train_path} and {test_path}")

def fetch_jailbreakbench():
    out_path = os.path.join(OUTPUT_DIR, "jailbreakbench_behaviors.csv")
    if os.path.exists(out_path):
        print("[2/3] JailbreakBench already downloaded. Skipping.")
        return

    print("\n[2/3] Downloading JailbreakBench...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    harmful_df = ds["harmful"].to_pandas()
    benign_df = ds["benign"].to_pandas()
    combined_df = pd.concat([harmful_df, benign_df], ignore_index=True)
    combined_df.to_csv(out_path, index=False)
    print(f" -> Saved: {out_path} ({len(combined_df)} records)")

def fetch_beavertails():
    print("\n[3/3] Downloading BeaverTails (30k subset)...")
    train_path = os.path.join(OUTPUT_DIR, "beavertails_30k_train.csv")
    test_path = os.path.join(OUTPUT_DIR, "beavertails_30k_test.csv")

    # Load using split='30k_train' and split='30k_test'
    ds_train = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    ds_test = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")

    ds_train.to_pandas().to_csv(train_path, index=False)
    ds_test.to_pandas().to_csv(test_path, index=False)
    print(f" -> Saved: {train_path} and {test_path}")

if __name__ == "__main__":
    fetch_toxic_chat()
    fetch_jailbreakbench()
    fetch_beavertails()
    print("\nAll datasets successfully prepared in data/datasets/!")
