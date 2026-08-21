import joblib
import streamlit as st
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_char_vectorizer.joblib"


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    return vectorizer, model


vectorizer, model = load_model()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📱 SMS Spam Classifier")

st.write(
    "Enter an SMS message below to determine whether "
    "it is likely to be spam or a legitimate message."
)


# --------------------------------------------------
# Message input
# --------------------------------------------------

message = st.text_area(
    "Enter your SMS message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict", use_container_width=True):

    if not message.strip():

        st.warning("Please enter an SMS message.")

    else:

        features = vectorizer.transform([message])

        prediction = model.predict(features)[0]

        st.divider()

        if prediction == 1:

            st.error("🚨 SPAM DETECTED")

            st.write(
                "This message has been classified as **spam**."
            )

        else:

            st.success("✅ HAM — NOT SPAM")

            st.write(
                "This message has been classified as **legitimate**."
            )


# --------------------------------------------------
# Model information
# --------------------------------------------------

with st.expander("Model Information"):

    st.write("**Model:** Linear SVM")

    st.write("**Features:** Character-level TF-IDF")

    st.write("**Accuracy:** 98.74%")

    st.write("**Precision:** 99.17%")

    st.write("**Recall:** 90.84%")

    st.write("**F1 Score:** 94.82%")