import joblib
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "models"

VECTORIZER_PATH = MODEL_DIR / "tfidf_char_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_data():
    """Load and prepare the SMS dataset."""

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    X = df["message"]

    y = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    return X, y


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate_model():

    print("=" * 60)
    print("SMS SPAM CLASSIFIER - EVALUATION")
    print("=" * 60)

    # Load data
    X, y = load_data()

    # IMPORTANT:
    # Use the same split as train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load trained artifacts
    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(f"\nTest samples: {len(X_test)}")

    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Predict
    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    print("\nModel Performance")
    print("-" * 30)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Classification report
    print("\nClassification Report")
    print("-" * 30)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"]
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix")
    print("-" * 30)
    print(cm)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()