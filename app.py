"""Streamlit demo for CS72 Toxicity Classification and Fairness project.

Run locally: python -m streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths and model identifiers used by the team's existing notebooks
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LR_FILE = ROOT / "models" / "logistic_regression.joblib"
BERT_REPO = "will55555/cs72-toxicity-distilbert"

st.set_page_config(page_title="Toxicity & Fairness Explorer", page_icon="🔍", layout="wide")


# Cache models because Streamlit re-runs this script after user interactions
@st.cache_resource(show_spinner="Loading logistic regression model...")
def load_logistic_model():
    model, vectorizer, threshold = joblib.load(LR_FILE)
    return model, vectorizer, float(threshold)


@st.cache_resource(show_spinner="Loading fine-tuned DistilBERT...")
def load_distilbert():
    # Load our team's fine-tuned weights from Hugging Face Hub
    model = AutoModelForSequenceClassification.from_pretrained(BERT_REPO)

    # Notebook 03 used the standard uncased DistilBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Use the threshold and token limit recorded alongside the trained weights
    settings_file = hf_hub_download(repo_id=BERT_REPO, filename="inference_settings.json")
    with open(settings_file, encoding="utf-8") as file:
        settings = json.load(file)

    model.eval()
    return model, tokenizer, float(settings["threshold"]), int(settings["max_length"])


@st.cache_data(show_spinner=False)
def load_results():
    # These are the held-out test results exported by Notebook 04
    fairness = pd.read_csv(RESULTS / "fairness_results.csv")
    overall = pd.read_csv(RESULTS / "model_comparison.csv")
    return fairness, overall


def predict_comments(comments):
    """Return both model scores and labels for a list of comments."""
    lr_model, vectorizer, lr_threshold = load_logistic_model()
    bert_model, tokenizer, bert_threshold, max_length = load_distilbert()

    # TF-IDF representation and logistic regression predictions
    tfidf = vectorizer.transform(comments)
    lr_scores = lr_model.predict_proba(tfidf)[:, 1]

    # Transformer tokenization; batch inference avoids repeated overhead
    tokens = tokenizer(
        comments,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = bert_model(**tokens).logits
        bert_scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    return pd.DataFrame({
        "Comment": comments,
        "Logistic Regression": lr_scores,
        "DistilBERT": bert_scores,
        "LR Label": ["Toxic" if p >= lr_threshold else "Non-toxic" for p in lr_scores],
        "BERT Label": ["Toxic" if p >= bert_threshold else "Non-toxic" for p in bert_scores],
    }), lr_threshold, bert_threshold


# A model-loading failure should be explained rather than displaying a raw traceback
# These inputs are synthetic examples, not held-out test examples
EXAMPLES = {
    "Neutral statement": "I really enjoyed reading this article.",
    "Direct insult": "You are an idiot and nobody wants to hear your opinion.",
    "Harmless religious identity": "I am Muslim and enjoy discussing my faith.",
    "Harmless sexual identity": "I am gay and proud of my identity.",
}
IDENTITIES = [
    "Muslim", "Christian", "Jewish", "Hindu", "Buddhist",
    "Black", "White", "Asian", "Hispanic",
    "gay", "lesbian", "bisexual", "transgender",
]
TEMPLATES = {
    "Positive self-description": "I am {} and proud of my identity.",
    "Positive statement about friends": "I have many {} friends who are wonderful people.",
    "Hostile statement": "I hate {} people and think they are disgusting.",
}


st.title("Toxicity & Fairness Explorer")
st.write(
    "Compare our TF-IDF + logistic regression and fine-tuned DistilBERT toxicity "
    "classifiers, and explore potential disparities across identity-related comments."
)
st.caption(
    "Course project using the Civil Comments dataset (2015–2017). "
    "Predictions are model estimates, not verified judgments about users or content."
)

tab1, tab2, tab3 = st.tabs([
    "Classify Comments", "Explore Fairness", "Counterfactual Comparison"
])

# ---------------------------------------------------------------
# Tab 1: Apply both trained classifiers to a user-provided comment
# ---------------------------------------------------------------
with tab1:
    st.subheader("Classify an online comment")

    # Example selector makes normal and challenging inputs easy to demonstrate
    def use_selected_example():
        st.session_state["comment_text"] = EXAMPLES[st.session_state["example_choice"]]

    if "comment_text" not in st.session_state:
        st.session_state["comment_text"] = EXAMPLES["Neutral statement"]

    st.selectbox(
        "Try an example (or enter your own text below)",
        options=list(EXAMPLES),
        key="example_choice",
        on_change=use_selected_example,
    )
    st.text_area("Comment", key="comment_text", height=110, max_chars=2000)

    if st.button("Classify comment", type="primary"):
        comment = st.session_state["comment_text"].strip()
        if not comment:
            st.warning("Please enter a comment first")
        else:
            try:
                predictions, lr_threshold, bert_threshold = predict_comments([comment])
                row = predictions.iloc[0]
                left, right = st.columns(2)
                for container, name, score_col, label_col, threshold in [
                    (left, "Logistic Regression", "Logistic Regression", "LR Label", lr_threshold),
                    (right, "DistilBERT", "DistilBERT", "BERT Label", bert_threshold),
                ]:
                    with container:
                        st.markdown(f"**{name}**")
                        st.metric("Estimated toxicity score", f"{row[score_col]:.1%}")
                        if row[label_col] == "Toxic":
                            st.error("Predicted toxic")
                        else:
                            st.success("Predicted non-toxic")
                        st.caption(f"Decision threshold: {threshold:.3f}")
                st.caption(
                    "Each model uses the decision threshold selected on validation data. "
                    "Scores should not be assumed perfectly calibrated."
                )
            except Exception as exc:
                st.error(f"Unable to run the classifiers: {exc}")

# ---------------------------------------------------------------
# Tab 2: Explore the saved test-set group-level error statistics
# ---------------------------------------------------------------
with tab2:
    st.subheader("Fairness across identity mentions")
    st.write(
        "These are error rates measured on the same held-out test set. "
        "Select a group to compare each model with its overall error rate."
    )

    try:
        fairness_df, overall_df = load_results()
        group = st.selectbox("Demographic identity mentioned", sorted(fairness_df["Group"].unique()))
        selected = fairness_df[fairness_df["Group"] == group].set_index("Model")
        overall = overall_df.set_index("Model")

        # Sample counts are shared across models because they use the same test rows
        counts = selected.iloc[0]
        cols = st.columns(3)
        cols[0].metric("Comments mentioning group", f"{int(counts['Comments']):,}")
        cols[1].metric("Non-toxic comments", f"{int(counts['Non-toxic']):,}")
        cols[2].metric("Toxic comments", f"{int(counts['Toxic']):,}")

        # Tables combine group-specific results with each model's overall test FPR/FNR
        for metric, heading, description in [
            ("FPR", "False-positive rate", "Share of labeled non-toxic comments incorrectly flagged as toxic"),
            ("FNR", "False-negative rate", "Share of labeled toxic comments incorrectly classified as non-toxic"),
        ]:
            st.markdown(f"#### {heading}")
            st.caption(description)
            comparison = pd.DataFrame({
                "Selected identity group": selected[metric] * 100,
                "Overall test set": overall[metric] * 100,
            }).reindex(["Logistic Regression", "DistilBERT"])
            # display error rates for both models
            st.bar_chart(comparison)
            st.dataframe(comparison.round(2).rename_axis("Model"), use_container_width=True)

        # Highlight smaller denominators instead of over-interpreting noisy percentages
        if counts["Non-toxic"] < 100 or counts["Toxic"] < 30:
            st.warning("Small subgroup sample: these error rates may be unstable")

        with st.expander("View all demographic groups"):
            metric_choice = st.radio("Metric", ["FPR", "FNR"], horizontal=True)
            all_groups = fairness_df.pivot(index="Group", columns="Model", values=metric_choice) * 100
            st.dataframe(all_groups.round(2), use_container_width=True)

        st.caption(
            "Groups were detected using keyword/regex matches, may overlap, and may include "
            "non-demographic word uses. Differences do not alone prove discriminatory bias."
        )
    except (OSError, KeyError, ValueError) as exc:
        st.error(f"Cannot display fairness results: {exc}")
        st.info("Commit results/fairness_results.csv and results/model_comparison.csv to GitHub")

# ---------------------------------------------------------------
# Tab 3: Run live identity-substitution tests on fixed sentence templates
# ---------------------------------------------------------------
with tab3:
    st.subheader("Counterfactual identity substitutions")
    st.write(
        "Keep the surrounding sentence fixed while changing the identity word. "
        "Compare how each trained model responds to the same phrasing."
    )

    template_name = st.selectbox("Sentence template", list(TEMPLATES))
    chosen_identities = st.multiselect(
        "Identity terms to compare",
        options=IDENTITIES,
        default=["Muslim", "Christian", "Black", "gay", "lesbian", "transgender"],
    )

    if st.button("Compare identity terms", type="primary"):
        if len(chosen_identities) < 2:
            st.warning("Select at least two identity terms")
        else:
            # Generate new predictions rather than reusing the saved CSV
            comments = [TEMPLATES[template_name].format(x) for x in chosen_identities]
            try:
                comparison, lr_threshold, bert_threshold = predict_comments(comments)
                comparison.insert(0, "Identity", chosen_identities)

                st.markdown("#### Predicted toxicity scores")
                chart_data = comparison.set_index("Identity")[["Logistic Regression", "DistilBERT"]] * 100
                # Display predicted toxicity scores for each identity
                st.caption("Predicted toxicity (%)")
                # Display the counterfactual predictions
                # Plot only the identity terms and predicted toxicity scores
                chart_df = comparison.set_index("Identity")[
                    ["Logistic Regression", "DistilBERT"]
                ]

                # Display the model predictions side by side
                st.bar_chart(chart_df)

                # Show binary outcomes as well as continuous model scores
                display_df = comparison[[
                    "Identity", "Comment", "Logistic Regression", "LR Label", "DistilBERT", "BERT Label"
                ]].copy()
                display_df["Logistic Regression"] = display_df["Logistic Regression"].map(lambda x: f"{x:.1%}")
                display_df["DistilBERT"] = display_df["DistilBERT"].map(lambda x: f"{x:.1%}")
                st.dataframe(display_df, hide_index=True, use_container_width=True)
                st.caption(
                    f"Thresholds: LR {lr_threshold:.3f}, DistilBERT {bert_threshold:.3f}. "
                    "Changing one word can change grammar or meaning; these synthetic tests "
                    "are diagnostics, not a complete fairness assessment."
                )
            except Exception as exc:
                st.error(f"Unable to compare identity terms: {exc}")

st.divider()
st.caption(
    "Research demonstration only. Models were trained on labeled Civil Comments data; "
    "neither model should be used as an automatic moderation decision-maker."
)
