import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


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
    """Load the UCI SMS Spam Collection dataset."""

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    return df


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

def prepare_data(df):
    """Clean data and prepare features and target."""

    # Remove duplicate messages
    df = df.drop_duplicates().reset_index(drop=True)

    # Features
    X = df["message"]

    # Target
    y = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    return X, y


# --------------------------------------------------
# Train model
# --------------------------------------------------

def train_model(X_train, y_train):
    """Train character-level TF-IDF + Linear SVM."""

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LinearSVC(
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    return vectorizer, model


# --------------------------------------------------
# Save model
# --------------------------------------------------

def save_model(vectorizer, model):
    """Save vectorizer and trained model."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Model saved to: {MODEL_PATH}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("SMS SPAM CLASSIFIER - TRAINING")
    print("=" * 60)

    # Load
    df = load_data()

    print(f"\nOriginal dataset: {len(df)} messages")

    # Prepare
    X, y = prepare_data(df)

    print(f"After removing duplicates: {len(X)} messages")

    print("\nClass distribution:")
    print(y.value_counts())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train
    print("\nTraining Character TF-IDF + Linear SVM...")

    vectorizer, model = train_model(
        X_train,
        y_train
    )

    print("Training completed.")

    # Save
    save_model(
        vectorizer,
        model
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()