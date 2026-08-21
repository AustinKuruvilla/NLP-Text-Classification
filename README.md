# SMS Spam Classification using Machine Learning

An end-to-end Natural Language Processing (NLP) project that classifies SMS messages as **spam** or **ham (legitimate)** using character-level TF-IDF features and a Linear Support Vector Machine.

## Project Overview

Spam messages are unwanted messages that may contain advertisements, fraudulent offers, malicious links, or other unwanted content.

The goal of this project is to automatically classify SMS messages as either:

- **Ham**: legitimate message
- **Spam**: unwanted message

The workflow covers data loading, cleaning, exploratory analysis, train/test splitting, TF-IDF feature engineering, model training, evaluation, serialization, and Streamlit deployment.

## Dataset

The project uses the [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), stored at `data/raw/SMSSpamCollection` as tab-separated label/message pairs.

The original dataset contains 5,572 messages. After removing 403 duplicate messages, the working dataset contains 5,169 unique messages: 4,516 ham messages and 653 spam messages.

Spam messages are longer on average than ham messages (138.67 versus 71.48 characters). The dataset is imbalanced, so evaluation focuses on precision, recall, and F1-score rather than accuracy alone.

## Machine Learning Approach

Character-level TF-IDF was selected because SMS messages often contain abbreviations, unusual spelling, URLs, phone numbers, fragmented words, and informal language.

```text
Analyzer: character
N-gram range: 3-5
Minimum document frequency: 2
Maximum features: 30,000
```

The classifier is a `LinearSVC` with `class_weight="balanced"`. The stratified train/test split uses a 20% test size and `random_state=42`.

### Model Performance

| Metric    | Score  |
|-----------|-------:|
| Accuracy  | 98.74% |
| Precision | 99.17% |
| Recall    | 90.84% |
| F1-score  | 94.82% |

## 📁 Project Structure

```text
nlp-text-classification/
├── app.py                              # Streamlit web application
├── data/
│   ├── processed/                      # Processed data outputs
│   └── raw/SMSSpamCollection           # Raw SMS Spam Collection dataset
├── models/
│   ├── spam_classifier.joblib          # Trained LinearSVC model
│   └── tfidf_char_vectorizer.joblib    # Fitted character TF-IDF vectorizer
├── src/
│   ├── __init__.py
│   ├── evaluate.py                     # Metrics and confusion matrix output
│   ├── load_data.py                    # Dataset loading helper
│   ├── predict.py                      # Command-line prediction helper
│   └── train.py                        # Training and model serialization
├── tests/
│   └── test_pipeline.py                # Tests
├── notebooks/
│   └── sms_spam_analysis.ipynb         # Analysis notebook
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Set up the environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.train
```

This writes the trained classifier and vectorizer to `models/`.

### 3. Evaluate the model

```bash
python -m src.evaluate
```

The command prints accuracy, precision, recall, F1-score, a classification report, and a confusion matrix.

### 4. Predict from the command line

```bash
python -m src.predict
```

Enter an SMS message when prompted.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```
[streamlit demo](https://austinkuruvilla-nlp-text-classification-app-1zee8h.streamlit.app/)

The app loads the serialized model artifacts and returns a ham or spam prediction for a message entered in the text area.

## Testing

Run the available tests with:

```bash
pytest
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Data processing | pandas, NumPy |
| Feature engineering | scikit-learn `TfidfVectorizer` |
| Classifier | scikit-learn `LinearSVC` |
| Model serialization | joblib |
| Web deployment | Streamlit |
| Analysis | Jupyter, Matplotlib, Seaborn |

## License

This project is part of a portfolio and is intended for educational purposes.
