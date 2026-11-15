
import streamlit as st
import joblib
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# Configure webpage
st.set_page_config(
    page_title="Toxicity & Fairness Explorer",
    page_icon="🔍",
    layout="wide"
)

# Load logistic regression model from GitHub repository
@st.cache_resource
def load_logistic_model():
    path = Path(__file__).parent / "models" / "logistic_regression.joblib"
    return joblib.load(path)

# Load fine-tuned DistilBERT from Hugging Face
# Cache the models to avoid reloading them for every prediction
@st.cache_resource
def load_distilbert():
    repo_id = "will55555/cs72-toxicity-distilbert"

    # Download model and tokenizer from Hugging Face
    # use the original tokenizer from the pretrained DistilBERT model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()

    # Download the classification threshold and tokenizer length settings
    settings_path = hf_hub_download(
        repo_id=repo_id,
        filename="inference_settings.json"
    )

    with open(settings_path) as f:
        settings = json.load(f)

    return model, tokenizer, settings["threshold"], settings["max_length"]

# Load both classifiers
lr_model, lr_vectorizer, lr_threshold = load_logistic_model()
bert_model, bert_tokenizer, bert_threshold, bert_max_length = load_distilbert()

# Display page title
st.title("Toxicity & Fairness Explorer")
st.write(
    "Compare NLP models for toxicity classification "
    "and explore potential demographic biases in their predictions."
)

# Create three application tabs
tab1, tab2, tab3 = st.tabs([
    "Classify Comments",
    "Explore Fairness",
    "Counterfactual Analysis"
])

# Tab 1: Compare classifier predictions on user-entered comments
with tab1:
    st.header("Classify Comments")

    comment = st.text_area(
        "Enter an online comment",
        placeholder="Type a comment here..."
    )

    if st.button("Classify"):
        if comment.strip():

            # Logistic regression: convert text into TF-IDF features
            lr_features = lr_vectorizer.transform([comment])
            lr_prob = float(lr_model.predict_proba(lr_features)[0, 1])
            lr_pred = lr_prob >= lr_threshold

            # DistilBERT: tokenize text using saved training settings
            inputs = bert_tokenizer(
                [comment],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=bert_max_length
            )

            # Run DistilBERT inference without calculating gradients
            with torch.inference_mode():
                logits = bert_model(**inputs).logits
                bert_prob = float(torch.softmax(logits, dim=1)[0, 1])

            bert_pred = bert_prob >= bert_threshold

            # Display the two models side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Logistic Regression")
                st.metric("Toxicity Score", f"{lr_prob:.1%}")

                if lr_pred:
                    st.error("Toxic")
                else:
                    st.success("Non-toxic")

                st.caption(f"Threshold: {lr_threshold:.3f}")

            with col2:
                st.subheader("DistilBERT")
                st.metric("Toxicity Score", f"{bert_prob:.1%}")

                if bert_pred:
                    st.error("Toxic")
                else:
                    st.success("Non-toxic")

                st.caption(f"Threshold: {bert_threshold:.3f}")

            st.caption(
                "Scores are model estimates, not verified measures "
                "of toxicity or guaranteed calibrated probabilities."
            )

        else:
            st.warning("Please enter a comment")

# Tab 2: Fairness dashboard (next step)
with tab2:
    st.header("Explore Fairness")
    st.write("Compare model error rates across demographic groups")

# Tab 3: Counterfactual comparison (to be added)
with tab3:
    st.header("Counterfactual Analysis")
    st.write("Compare predictions when identity terms change")
