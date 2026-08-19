# NLP Text Classification 🎯

An end-to-end text classification pipeline demonstrating natural language processing (NLP) fundamentals: text preprocessing, feature extraction with TF-IDF, model training with Logistic Regression, and comprehensive evaluation metrics.

## 📋 Overview

This project implements a complete sentiment classification workflow that transforms raw text into predictions using scikit-learn's machine learning pipeline. Perfect for understanding NLP basics and text classification techniques.

## ✨ Features

- **Text Preprocessing**: Lowercase conversion and punctuation removal
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Model Training**: Logistic Regression classifier with cross-validation
- **Evaluation Metrics**: Precision, recall, F1-score, and confusion matrix
- **Inference API**: Predict sentiment on new text with confidence scores
- **Comprehensive Logging**: Detailed output for debugging and understanding

## 📁 Project Structure

```
nlp-text-classification/
├── src/
│   ├── __init__.py
│   └── pipeline.py           # Main classification pipeline
├── tests/
│   └── test_pipeline.py      # Unit tests for pipeline
├── notebooks/
│   ├── 01-nlp-text-classification.ipynb    # Interactive Jupyter notebook
│   └── 01-nlp-text-classification.md       # Notebook markdown export
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Execute the classification demo
python -m src.pipeline
```

### 3. Run Tests

```bash
# From portfolio root
python run_basic_tests.py
```

### 4. Explore in Jupyter

```bash
jupyter notebook notebooks/01-nlp-text-classification.ipynb
```

## 📊 Example Output

```
Text: "I love this product, it's amazing!"
Prediction: Positive (confidence: 94.3%)

Classification Report:
              precision    recall  f1-score   support
      Negative       0.95      0.92      0.93        50
      Positive       0.91      0.95      0.93        50

accuracy                                   0.93       100
```

## 🔧 Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.7+ |
| **NLP** | scikit-learn (TF-IDF) |
| **ML Framework** | scikit-learn (Logistic Regression) |
| **Data Analysis** | pandas, numpy |
| **Notebooks** | Jupyter |

## 📦 Dependencies

- scikit-learn (machine learning and NLP)
- pandas (data manipulation)
- numpy (numerical computing)

See `requirements.txt` for exact versions.

## 🎓 What You'll Learn

- Text preprocessing techniques
- TF-IDF vectorization for feature extraction
- Logistic Regression for classification
- Model evaluation metrics (precision, recall, F1)
- Cross-validation and hyperparameter tuning
- Inference and prediction on new data

## 📝 Key Functions

### `preprocess_text(text: str) -> str`
Cleans text by converting to lowercase and removing punctuation.

### `build_demo_dataset() -> pd.DataFrame`
Creates a sample dataset with 8 sentiment-labeled examples for training.

### `train_and_evaluate(df: pd.DataFrame) -> tuple`
Trains TF-IDF + Logistic Regression pipeline and evaluates performance.

### `predict(text: str, model, vectorizer) -> tuple`
Makes predictions on new text with confidence scores.

## 🔍 Example Usage

```python
from src.pipeline import train_and_evaluate, predict, build_demo_dataset

# Load or create data
df = build_demo_dataset()

# Train model
model, vectorizer = train_and_evaluate(df)

# Make predictions
text = "This is an amazing experience!"
label, confidence = predict(text, model, vectorizer)
print(f"Prediction: {label} ({confidence:.1%})")
```

## 🧪 Testing

Unit tests verify:
- ✅ Dataset creation with correct structure
- ✅ Text preprocessing correctness
- ✅ Pipeline execution and model training
- ✅ Prediction functionality

Run tests with: `python run_basic_tests.py`

## 📚 Further Reading

- [Scikit-learn TF-IDF Documentation](https://scikit-learn.org/stable/modules/generated.html#module-sklearn.feature_extraction.text)
- [Logistic Regression Guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [NLP Best Practices](https://towardsdatascience.com/10-common-mistakes-in-nlp-7ae5ad0980a5)

## 📄 License

This project is part of a portfolio and open to use for educational purposes.

---

**Last Updated**: August 2026  
**Status**: Production Ready ✅
