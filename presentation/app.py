import streamlit as st
import torch
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from neural_networks.rnn import RNNClassifier
from neural_networks.lstm import LSTMClassifier
from nlp.preprocessor import NLPPreprocessor

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
label_emojis = {"Negative": "🔴", "Neutral": "🟡", "Positive": "🟢"}

if "model_dict" not in st.session_state:

    def _load_model(ckpt_path: str, net_cls):
        ckpt = torch.load(ckpt_path, map_location=device)
        config = ckpt["config"]
        model = net_cls(**config).to(device)
        state_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
        model.load_state_dict(ckpt[state_key])
        model.eval()
        return model

    model_dict = {}
    model_dict["RNN"] = _load_model("models/best_rnn.pt", RNNClassifier)
    model_dict["LSTM"] = _load_model("models/best_lstm.pt", LSTMClassifier)
    st.session_state.model_dict = model_dict

st.title("💬 ChatGPDiss - ChatGPT Tweet Sentiment Analysis 🧠")
st.write(
    "This is a simple app to analyze the sentiment of tweets about ChatGPT using RNN and LSTM models."
)

tweet_text = st.text_area("Enter a tweet:", "", max_chars=280)

if tweet_text:
    preprocessor = NLPPreprocessor(vocab_path="nlp/vocab.json")
    x = preprocessor.preprocess_text(tweet_text).to(device)

    predictions = {}
    with torch.no_grad():
        for name, model in st.session_state.model_dict.items():
            logits = model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
            label = label_map.get(pred_idx, "unknown")
            emoji = label_emojis.get(label, "")
            predictions[name] = f"{emoji} {label}"

    st.subheader("🎯 Predicted sentiment")
    for model, result in predictions.items():
        st.markdown(f"**{model}**: {result}")

    with st.expander("🔍 Preprocessing Steps", expanded=False):
        st.markdown("📝 Raw text")
        st.code(preprocessor.raw_text)

        st.markdown("✂️ Stripped text")
        st.code(preprocessor.stripped_text)

        st.markdown("🔤 Normalized text")
        st.code(preprocessor.normalized_text)

        st.markdown("🔢 Indexed text")
        st.code(preprocessor.indexed_text)

        st.markdown("🧠 Tensor")
        st.code(preprocessor.tensor)
