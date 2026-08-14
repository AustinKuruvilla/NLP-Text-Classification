"""Minimal NLP text classification pipeline demo.

Run as: python -m src.pipeline
"""
from typing import List
import re
import logging

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def build_demo_dataset() -> pd.DataFrame:
    data = {
        "text": [
            "I love this product, it is great!",
            "Terrible experience, will not buy again.",
            "This is an excellent purchase.",
            "Waste of money, very disappointed.",
        ],
        "label": [1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    df["text"] = df["text"].map(preprocess_text)
    return df


def train_and_evaluate(df: pd.DataFrame):
    X = df["text"].values
    y = df["label"].values
    # Lazy-import scikit-learn to avoid import errors in environments
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    vect = TfidfVectorizer()
    Xv = vect.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xv, y, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    print(classification_report(yte, preds))


def main():
    df = build_demo_dataset()
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
