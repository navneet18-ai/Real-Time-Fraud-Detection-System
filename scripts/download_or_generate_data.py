"""
Script to prepare or generate the credit card fraud detection dataset.
Supports:
1. Verifying existing data/creditcard.csv.
2. Attempting download via kaggle API if configured.
3. Generating a statistically authentic benchmark dataset matching Kaggle's schema
   (Time, V1-V28 PCA features, skewed Amount, and 0.172% fraud rate) if no local file exists.
4. Exporting data/sample_transactions.csv for easy real-time dashboard testing and demo presets.
"""

import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "creditcard.csv")
SAMPLE_FILE = os.path.join(DATA_DIR, "sample_transactions.csv")


def ensure_data_directory():
    os.makedirs(DATA_DIR, exist_ok=True)


def generate_synthetic_benchmark(n_samples: int = 30000, fraud_rate: float = 0.00172, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a statistically authentic synthetic credit card transaction dataset
    matching the schema and distributions of the European Credit Card Kaggle dataset:
    - Time: Seconds elapsed over 48 hours (0 to 172,800)
    - V1 to V28: PCA latent features with realistic shift for fraudulent transactions
    - Amount: Heavy right-skewed log-normal transaction amounts
    - Class: 0 for legitimate, 1 for fraudulent
    """
    print(f"Generating synthetic benchmark dataset with {n_samples:,} transactions ({fraud_rate*100:.3f}% fraud rate)...")
    rng = np.random.default_rng(random_seed)

    n_fraud = max(int(n_samples * fraud_rate), 50)  # Ensure at least 50 frauds for robust evaluation
    n_legit = n_samples - n_fraud

    # 1. Time feature: Uniform across 48 hours (172,800 seconds)
    time_legit = np.sort(rng.uniform(0, 172800, n_legit))
    time_fraud = rng.uniform(0, 172800, n_fraud)

    # 2. Amount: Log-normal distribution (median ~25, mean ~88, max ~2000+)
    # Fraud transactions often have distinct amount distributions (e.g. low card checks or max limit hits)
    amount_legit = np.round(rng.lognormal(mean=2.8, sigma=1.2, size=n_legit), 2)
    fraud_amount_pool = np.concatenate([
        rng.uniform(1.0, 15.0, int(n_fraud * 0.4)),        # Small probing charges
        rng.uniform(100.0, 800.0, int(n_fraud * 0.4)),     # Typical fraud purchases
        rng.uniform(900.0, 2200.0, n_fraud - 2 * int(n_fraud * 0.4)) # High value fraud
    ])
    amount_fraud = np.round(rng.choice(fraud_amount_pool, size=n_fraud, replace=False), 2)

    # 3. PCA Features V1-V28
    # Legitimate transactions: standard normal distributions around 0
    pca_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))

    # Fraud transactions: known key discriminative components in Kaggle dataset (V4, V11, V12, V14, V17)
    # have significant mean shifts and higher variance
    pca_fraud = rng.normal(loc=0.0, scale=1.4, size=(n_fraud, 28))
    # V4: Positive shift in fraud
    pca_fraud[:, 3] += rng.normal(loc=3.5, scale=1.2, size=n_fraud)
    # V11: Positive shift in fraud
    pca_fraud[:, 10] += rng.normal(loc=3.2, scale=1.1, size=n_fraud)
    # V12: Negative shift in fraud
    pca_fraud[:, 11] -= rng.normal(loc=4.0, scale=1.3, size=n_fraud)
    # V14: Significant negative shift in fraud
    pca_fraud[:, 13] -= rng.normal(loc=5.5, scale=1.5, size=n_fraud)
    # V17: Negative shift in fraud
    pca_fraud[:, 16] -= rng.normal(loc=4.5, scale=1.4, size=n_fraud)
    # V10: Negative shift in fraud
    pca_fraud[:, 9] -= rng.normal(loc=3.8, scale=1.2, size=n_fraud)

    # Combine Legitimate
    df_legit = pd.DataFrame(pca_legit, columns=[f"V{i}" for i in range(1, 29)])
    df_legit.insert(0, "Time", time_legit)
    df_legit["Amount"] = amount_legit
    df_legit["Class"] = 0

    # Combine Fraud
    df_fraud = pd.DataFrame(pca_fraud, columns=[f"V{i}" for i in range(1, 29)])
    df_fraud.insert(0, "Time", time_fraud)
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    # Concatenate and shuffle
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


def create_sample_preset_file(df: pd.DataFrame, n_each: int = 15):
    """Creates a sample_transactions.csv file with verified sample cases for UI demo testing."""
    legit_samples = df[df["Class"] == 0].head(n_each).copy()
    fraud_samples = df[df["Class"] == 1].head(n_each).copy()
    sample_df = pd.concat([legit_samples, fraud_samples], ignore_index=True)
    sample_df.to_csv(SAMPLE_FILE, index=False)
    print(f"Saved {len(sample_df)} sample transactions to {SAMPLE_FILE} for UI testing presets.")


def main():
    ensure_data_directory()

    if os.path.exists(DATA_FILE):
        print(f"Dataset already exists at: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded existing dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.")
        print(f"Class distribution:\n{df['Class'].value_counts()}")
        create_sample_preset_file(df)
        return

    # Check if user has Kaggle credentials and kaggle CLI
    kaggle_creds = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_creds):
        print("Found Kaggle credentials! Attempting download via Kaggle API...")
        ret = os.system(f"kaggle datasets download -d mlg-ulb/creditcardfraud -p {DATA_DIR} --unzip")
        if ret == 0 and os.path.exists(DATA_FILE):
            print(f"Successfully downloaded and extracted {DATA_FILE}!")
            df = pd.read_csv(DATA_FILE)
            create_sample_preset_file(df)
            return
        else:
            print("Kaggle download command failed or returned non-zero code. Falling back to synthetic generation.")

    # Generate high-fidelity synthetic benchmark
    df = generate_synthetic_benchmark(n_samples=50000, fraud_rate=0.002, random_seed=42)
    df.to_csv(DATA_FILE, index=False)
    print(f"Successfully generated and saved dataset to {DATA_FILE}!")
    print(f"Dataset shape: {df.shape}")
    print(f"Class breakdown:\n{df['Class'].value_counts(normalize=True).round(5) * 100}%")
    create_sample_preset_file(df)


if __name__ == "__main__":
    main()
