from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_char_vectorizer.joblib"


def test_model_exists():
    """Check that the trained classifier exists."""
    assert MODEL_PATH.exists()


def test_vectorizer_exists():
    """Check that the trained TF-IDF vectorizer exists."""
    assert VECTORIZER_PATH.exists()


def test_model_can_be_loaded():
    """Check that the saved classifier can be loaded."""
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_vectorizer_can_be_loaded():
    """Check that the saved vectorizer can be loaded."""
    vectorizer = joblib.load(VECTORIZER_PATH)

    assert vectorizer is not None


def test_spam_prediction():
    """Check that the model can classify a spam message."""

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    message = [
        "Congratulations! You have won a free prize. Call now!"
    ]

    features = vectorizer.transform(message)

    prediction = model.predict(features)[0]

    assert prediction == 1


def test_ham_prediction():
    """Check that the model can classify a legitimate message."""

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    message = [
        "Hey, are you coming home for dinner?"
    ]

    features = vectorizer.transform(message)

    prediction = model.predict(features)[0]

    assert prediction == 0