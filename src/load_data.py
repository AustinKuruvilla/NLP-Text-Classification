from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/SMSSpamCollection")

def load_sms_data():
    df = pd.read_csv(
        DATA_PATH, 
        sep="\t", 
        header=None, 
        names=["label", "text"]
    )
    return df

if __name__ == "__main__":
    df = load_sms_data()
    print(df.head())
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True)}")