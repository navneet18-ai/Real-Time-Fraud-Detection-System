"""
Unit tests for fraud prediction engine, risk categorization, and batch scoring.
"""

import pytest
import pandas as pd
import numpy as np
from predict import FraudPredictor, get_predictor


@pytest.fixture
def predictor():
    pred = get_predictor()
    assert pred.is_ready, "Model artifacts must be present for predictor tests."
    return pred


def test_predictor_is_ready(predictor):
    assert predictor.is_ready is True
    assert predictor.model is not None
    assert predictor.preprocessor is not None


def test_predict_single_legitimate(predictor):
    # Typical legitimate transaction with mild/zero PCA features and modest amount
    legit_input = {"Time": 100.0, "Amount": 25.50}
    for i in range(1, 29):
        legit_input[f"V{i}"] = 0.0

    result = predictor.predict_single(legit_input, save_to_db=False)
    assert "fraud_probability" in result
    assert "risk_level" in result
    assert "prediction" in result
    assert "explanation" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_single_fraud_anomalous(predictor):
    # Anomalous transaction mimicking extreme fraud shifts in V4, V11, V14, V17
    fraud_input = {
        "Time": 500.0,
        "Amount": 1200.0,
        "V4": 4.5,
        "V11": 3.8,
        "V12": -5.0,
        "V14": -6.5,
        "V17": -5.5
    }
    result = predictor.predict_single(fraud_input, save_to_db=False)
    assert result["fraud_probability"] > 0.50
    assert result["risk_level"] in ["MEDIUM", "HIGH"]
    assert result["is_flagged"] is True


def test_risk_threshold_classification(predictor):
    predictor.threshold_low = 0.20
    predictor.threshold_high = 0.65

    assert predictor.classify_risk(0.05) == "LOW"
    assert predictor.classify_risk(0.19) == "LOW"
    assert predictor.classify_risk(0.20) == "MEDIUM"
    assert predictor.classify_risk(0.50) == "MEDIUM"
    assert predictor.classify_risk(0.65) == "HIGH"
    assert predictor.classify_risk(0.95) == "HIGH"


def test_single_and_batch_prediction_consistency(predictor):
    """Verifies that scoring a single row yields the exact same probability as batch scoring."""
    sample_txn = {"Time": 350.0, "Amount": 89.20, "V1": -1.2, "V4": 2.1, "V14": -2.8}
    for i in range(1, 29):
        if f"V{i}" not in sample_txn:
            sample_txn[f"V{i}"] = 0.0

    # Single prediction
    single_res = predictor.predict_single(sample_txn, save_to_db=False)
    single_prob = single_res["fraud_probability"]

    # Batch prediction with identical row
    df_batch = pd.DataFrame([sample_txn])
    scored_df, _ = predictor.predict_batch(df_batch, save_to_db=False)
    batch_prob = float(scored_df.iloc[0]["fraud_probability"])

    assert round(single_prob, 3) == round(batch_prob, 3), (
        f"Inconsistency detected: single={single_prob}, batch={batch_prob}"
    )


def test_predict_batch_summary_stats(predictor):
    rows = []
    for i in range(10):
        row = {"Time": float(i * 10), "Amount": 50.0 + i}
        for v in range(1, 29):
            row[f"V{v}"] = 0.0
        rows.append(row)
    
    df = pd.DataFrame(rows)
    scored_df, summary = predictor.predict_batch(df, save_to_db=False)
    
    assert summary["total_processed"] == 10
    assert len(scored_df) == 10
    assert "fraud_probability" in scored_df.columns
    assert "risk_level" in scored_df.columns
    assert "prediction" in scored_df.columns
