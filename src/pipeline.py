"""End-to-end NLP text classification pipeline.

This module provides tools for preprocessing text, building datasets,
training classifiers, and making predictions on new data.

Run as: python -m src.pipeline
"""
from typing import List, Tuple, Dict
import re
import logging

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """Clean and normalize text data.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text with lowercase and punctuation removed
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_demo_dataset() -> pd.DataFrame:
    """Create a demo sentiment classification dataset.
    
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    data = {
        "text": [
            "I love this product, it is great!",
            "Terrible experience, will not buy again.",
            "This is an excellent purchase.",
            "Waste of money, very disappointed.",
            "Amazing quality and fast delivery!",
            "Absolutely hate it, return immediately.",
            "Good value for money.",
            "Disappointing, low quality.",
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    df["text"] = df["text"].map(preprocess_text)
    logger.info(f"Created demo dataset with {len(df)} samples")
    return df


def train_and_evaluate(df: pd.DataFrame) -> Dict[str, float]:
    """Train a text classification model and evaluate performance.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        
    Returns:
        Dictionary with model metrics
    """
    X = df["text"].values
    y = df["label"].values
    
    # Lazy-import scikit-learn to avoid import errors in environments
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    logger.info("Building TF-IDF vectorizer...")
    vect = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    Xv = vect.fit_transform(X)
    
    logger.info("Splitting data into train/test sets...")
    Xtr, Xte, ytr, yte = train_test_split(Xv, y, test_size=0.25, random_state=42)
    
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(Xtr, ytr)
    
    logger.info("Evaluating model...")
    preds = model.predict(Xte)
    accuracy = accuracy_score(yte, preds)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(yte, preds, target_names=["Negative", "Positive"]))
    
    return {
        "accuracy": accuracy,
        "model": model,
        "vectorizer": vect,
    }


def predict(text: str, model, vectorizer) -> Dict[str, any]:
    """Make a prediction on new text.
    
    Args:
        text: Input text
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        Dictionary with prediction and confidence
    """
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]
    
    return {
        "text": text,
        "prediction": "Positive" if prediction == 1 else "Negative",
        "confidence": float(confidence[prediction]),
    }


def main():
    """Main entry point for the pipeline."""
    df = build_demo_dataset()
    results = train_and_evaluate(df)
    
    logger.info("\nTesting predictions on new data:")
    test_texts = [
        "This is amazing!",
        "I hate this product.",
        "It's okay.",
    ]
    
    for text in test_texts:
        pred = predict(text, results["model"], results["vectorizer"])
        logger.info(f"Text: {pred['text']}")
        logger.info(f"  Prediction: {pred['prediction']} ({pred['confidence']:.4f})")


if __name__ == "__main__":
    main()
