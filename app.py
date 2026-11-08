"""Streamlit dashboard for classifying a prompt with Qwen3Guard-Stream."""

import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3Guard-Stream-0.6B"

st.set_page_config(page_title="Prompt Safety Check", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="Loading Qwen3Guard-Stream…")
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModel.from_pretrained(
            model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=True
        )
    else:
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
    return tokenizer, model.eval()


def classify_prompt(tokenizer, model, prompt: str) -> tuple[str, str | None]:
    """Return the model's risk level and optional category for a user prompt."""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        raise ValueError("Enter some text to classify.")
    device = next(model.parameters()).device
    result, _ = model.stream_moderate_from_ids(
        torch.tensor(input_ids, device=device), role="user", stream_state=None
    )
    level = result["risk_level"]
    category = result.get("category")
    if isinstance(level, (list, tuple)):
        level = level[-1]
    if isinstance(category, (list, tuple)):
        category = category[-1] if category else None
    return str(level), str(category) if category else None


st.title("Prompt Safety Check")
st.write("Submit a prompt to check whether Qwen3Guard-Stream considers it negative or unsafe.")

with st.form("prompt_check"):
    prompt = st.text_area(
        "Prompt to classify",
        height=180,
        placeholder="Paste or write a prompt here…",
        help="Your prompt is sent to the model running in this app for classification.",
    )
    threshold = st.radio(
        "Count as negative when the risk is",
        options=["Unsafe only", "Controversial or Unsafe"],
        horizontal=True,
        help="The stricter option treats controversial content as negative too.",
    )
    submitted = st.form_submit_button("Check prompt", type="primary")

if submitted:
    if not prompt.strip():
        st.warning("Enter a prompt before submitting.")
    else:
        try:
            with st.spinner("Classifying prompt…"):
                tokenizer, model = load_model(DEFAULT_MODEL)
                risk_level, category = classify_prompt(tokenizer, model, prompt.strip())
            negative_levels = {"Unsafe"}
            if threshold == "Controversial or Unsafe":
                negative_levels.add("Controversial")
            is_negative = risk_level in negative_levels
            if is_negative:
                st.error("Negative: the prompt meets the selected risk threshold.")
            else:
                st.success("Not negative: the prompt is below the selected risk threshold.")
            col1, col2 = st.columns(2)
            col1.metric("Model risk level", risk_level)
            col2.metric("Category", category or "Not provided")
            st.caption(f"Model: `{DEFAULT_MODEL}` · Prompt-only classification")
        except Exception as exc:
            st.error(f"Could not classify this prompt: {exc}")
            st.info("Check that dependencies are installed and that the model can be downloaded.")

st.caption("This is an automated safety signal. Review important decisions with a person.")
