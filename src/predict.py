import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_char_vectorizer.joblib"


vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)


def predict_spam(message):
    """
    Predict whether an SMS message is spam or ham.

    Returns:
        str: 'spam' or 'ham'
    """

    features = vectorizer.transform([message])

    prediction = model.predict(features)[0]

    if prediction == 1:
        return "spam"

    return "ham"


if __name__ == "__main__":

    message = input("Enter an SMS message: ")

    result = predict_spam(message)

    print(f"\nPrediction: {result.upper()}")