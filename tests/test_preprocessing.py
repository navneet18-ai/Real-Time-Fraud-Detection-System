"""
Unit tests for data validation, preprocessing, and leakage prevention.
"""

import pytest
import numpy as np
import pandas as pd
from preprocessing import (
    FraudDataPreprocessor,
    validate_transaction_dict,
    validate_batch_dataframe,
    NUMERICAL_FEATURES,
    PCA_FEATURES,
)


@pytest.fixture
def sample_raw_dataframe():
    """Generates synthetic DataFrame with expected 30 features + Class."""
    n_rows = 50
    data = {f"V{i}": np.random.randn(n_rows) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 10000, n_rows)
    data["Amount"] = np.random.exponential(50, n_rows)
    data["Class"] = np.random.choice([0, 1], n_rows, p=[0.9, 0.1])
    return pd.DataFrame(data)


def test_preprocessor_fit_transform(sample_raw_dataframe):
    preprocessor = FraudDataPreprocessor()
    X = sample_raw_dataframe[NUMERICAL_FEATURES]
    
    transformed = preprocessor.fit_transform(X)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (50, 30)
    assert preprocessor.is_fitted is True


def test_preprocessor_unfitted_raises_error(sample_raw_dataframe):
    preprocessor = FraudDataPreprocessor()
    X = sample_raw_dataframe[NUMERICAL_FEATURES]
    with pytest.raises(RuntimeError, match="must be fitted before transforming"):
        preprocessor.transform(X)


def test_preprocessor_missing_columns_raises_error():
    preprocessor = FraudDataPreprocessor()
    incomplete_df = pd.DataFrame({"Time": [10.0], "Amount": [50.0]})
    with pytest.raises(ValueError, match="missing required features"):
        preprocessor.fit(incomplete_df)


def test_validate_transaction_dict_valid():
    sample_input = {"Time": 500.0, "Amount": 45.50, "V1": 1.2, "V2": -0.8}
    validated = validate_transaction_dict(sample_input)
    assert validated["Time"] == 500.0
    assert validated["Amount"] == 45.50
    assert validated["V1"] == 1.2
    assert validated["V2"] == -0.8
    # Unsupplied PCA features should default to 0.0
    assert validated["V3"] == 0.0
    assert validated["V28"] == 0.0


def test_validate_transaction_dict_negative_amount():
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_transaction_dict({"Time": 100.0, "Amount": -15.0})


def test_validate_transaction_dict_negative_time():
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_transaction_dict({"Time": -5.0, "Amount": 100.0})


def test_validate_transaction_dict_invalid_type():
    with pytest.raises(ValueError, match="Invalid 'Amount' value"):
        validate_transaction_dict({"Amount": "not_a_number"})


def test_validate_batch_dataframe_strips_target_column(sample_raw_dataframe):
    assert "Class" in sample_raw_dataframe.columns
    clean_df, errors = validate_batch_dataframe(sample_raw_dataframe)
    assert len(errors) == 0
    # Crucial leakage test: Class column must be eliminated from inference inputs
    assert "Class" not in clean_df.columns
    assert clean_df.shape[1] == 30


def test_validate_batch_dataframe_empty():
    clean_df, errors = validate_batch_dataframe(pd.DataFrame())
    assert clean_df.empty
    assert "empty" in errors[0]
