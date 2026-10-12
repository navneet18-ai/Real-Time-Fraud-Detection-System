"""
Preprocessing and feature engineering module for credit card fraud detection.
Handles:
- Feature schema validation
- Data sanitization and missing value checks
- Robust scaling for skewed features (Amount, Time)
- Preprocessing pipeline serialization and loading
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Canonical feature list expected by model
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
NUMERICAL_FEATURES = ["Time"] + PCA_FEATURES + ["Amount"]
TARGET_COLUMN = "Class"


class FraudDataPreprocessor:
    """
    Standardized preprocessor for fraud detection transactions.
    Fits RobustScaler on 'Amount' and 'Time' strictly on training data
    to prevent data leakage, preserving PCA features V1-V28.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_names = NUMERICAL_FEATURES.copy()

    def fit(self, X: pd.DataFrame) -> "FraudDataPreprocessor":
        """Fits scaler on Time and Amount columns."""
        self._validate_columns(X, fit_mode=True)
        # We scale Time and Amount because PCA features (V1-V28) are already zero-centered
        self.scaler.fit(X[["Time", "Amount"]])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms features using fitted scaler and returns formatted numpy array."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        
        self._validate_columns(X, fit_mode=False)
        X_df = X[self.feature_names].copy()

        # Fill any unexpected missing values with 0.0 (baseline mean for PCA)
        X_df = X_df.fillna(0.0)

        # Scale Time and Amount
        scaled_time_amount = self.scaler.transform(X_df[["Time", "Amount"]])
        
        # Combine scaled Time, V1-V28, and scaled Amount
        # Order: [scaled_Time, V1, V2, ..., V28, scaled_Amount]
        transformed = np.column_stack([
            scaled_time_amount[:, 0],          # scaled Time
            X_df[PCA_FEATURES].to_numpy(),     # V1 - V28
            scaled_time_amount[:, 1]           # scaled Amount
        ])
        return transformed

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def _validate_columns(self, X: pd.DataFrame, fit_mode: bool = False):
        """Ensures all 30 expected feature columns exist and target is not leaked."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        
        missing = [col for col in self.feature_names if col not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required features: {missing}")

    def save(self, filepath: str):
        """Saves fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FraudDataPreprocessor":
        """Loads preprocessor from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
        return joblib.load(filepath)


def validate_transaction_dict(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Validates a raw dictionary input for a single transaction.
    Raises ValueError with a clear message if values are invalid.
    """
    sanitized: Dict[str, float] = {}

    # Check for target leakage
    if "Class" in data or "class" in data:
        # Silently ignore or drop target if accidentally supplied
        pass

    # Validate Time
    if "Time" not in data or data["Time"] is None:
        sanitized["Time"] = 0.0
    else:
        try:
            time_val = float(data["Time"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid 'Time' value: {data.get('Time')}") from e
        if time_val < 0:
            raise ValueError("Transaction 'Time' cannot be negative.")
        sanitized["Time"] = time_val

    # Validate Amount
    if "Amount" not in data or data["Amount"] is None:
        raise ValueError("Transaction 'Amount' is required.")
    try:
        amount_val = float(data["Amount"])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid 'Amount' value: {data.get('Amount')}") from e
    if amount_val < 0:
        raise ValueError("Transaction 'Amount' cannot be negative.")
    sanitized["Amount"] = amount_val

    # Validate PCA features V1-V28
    for feat in PCA_FEATURES:
        val = data.get(feat, 0.0)
        try:
            sanitized[feat] = float(val) if val is not None else 0.0
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric value for {feat}: {val}") from e

    return sanitized


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates a batch DataFrame for CSV uploads.
    Returns (cleaned_df, error_messages).
    """
    errors: List[str] = []

    if df.empty:
        return df, ["The uploaded CSV file is empty."]

    # If Class column is present, strip it to avoid target leakage during inference
    df_clean = df.copy()
    if TARGET_COLUMN in df_clean.columns:
        df_clean = df_clean.drop(columns=[TARGET_COLUMN])

    # Check missing required columns
    missing_cols = [c for c in NUMERICAL_FEATURES if c not in df_clean.columns]
    if missing_cols:
        errors.append(f"Missing required columns ({len(missing_cols)}): {', '.join(missing_cols[:5])}...")
        return pd.DataFrame(), errors

    # Check for invalid Amount
    invalid_amounts = df_clean[pd.to_numeric(df_clean["Amount"], errors="coerce").isna()]
    if not invalid_amounts.empty:
        errors.append(f"Found {len(invalid_amounts)} rows with non-numeric Amount.")

    negative_amounts = df_clean[pd.to_numeric(df_clean["Amount"], errors="coerce") < 0]
    if not negative_amounts.empty:
        errors.append(f"Found {len(negative_amounts)} rows with negative Amount.")

    # Convert all columns to float
    try:
        df_clean = df_clean[NUMERICAL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    except Exception as e:
        errors.append(f"Data type conversion failed: {str(e)}")

    return df_clean, errors
