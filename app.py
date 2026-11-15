
import streamlit as st
import joblib
from pathlib import Path

# Configure the webpage
st.set_page_config(
    page_title="Toxicity & Fairness Explorer",
    page_icon="🔍",
    layout="wide"
)

# Load the saved logistic regression model and vectorizer
# Cache the model so it only loads once rather than on every interaction
@st.cache_resource
def load_logistic_model():
    model_path = Path(__file__).parent / "models" / "logistic_regression.joblib"
    model, vectorizer, threshold = joblib.load(model_path)
    return model, vectorizer, threshold

# Load the model components
lr_model, lr_vectorizer, lr_threshold = load_logistic_model()

# Website title and description
st.title("Toxicity & Fairness Explorer")
st.write(
    "Compare NLP models for toxicity classification "
    "and explore potential demographic biases in their predictions."
)

# Create the three main tabs
tab1, tab2, tab3 = st.tabs([
    "Classify Comments",
    "Explore Fairness",
    "Counterfactual Analysis"
])

# Tab 1: Classify user-entered comments
with tab1:
    st.header("Classify Comments")

    # Allow users to enter their own text
    comment = st.text_area(
        "Enter an online comment",
        placeholder="Type a comment here..."
    )

    if st.button("Classify"):
        if comment.strip():

            # Convert the comment into TF-IDF features
            features = lr_vectorizer.transform([comment])

            # Calculate predicted toxicity probability
            probability = lr_model.predict_proba(features)[0, 1]

            # Apply the classification threshold selected in Notebook 02
            prediction = probability >= lr_threshold

            # Display the result
            st.subheader("Logistic Regression Results")
            st.metric("Toxicity Score", f"{probability:.1%}")

            if prediction:
                st.error("Predicted Classification: Toxic")
            else:
                st.success("Predicted Classification: Non-toxic")

            st.caption(f"Classification threshold: {lr_threshold:.3f}")

        else:
            st.warning("Please enter a comment")

# Tab 2: Fairness analysis (to be implemented)
with tab2:
    st.header("Explore Fairness")
    st.write("Compare model error rates across demographic groups")

# Tab 3: Counterfactual analysis (to be implemented)
with tab3:
    st.header("Counterfactual Analysis")
    st.write("Compare predictions when identity terms change")
