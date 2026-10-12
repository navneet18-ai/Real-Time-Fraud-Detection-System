"""
Reusable prediction engine for Real-Time Fraud Detection.
Loads trained model and preprocessor artifacts, calculates calibrated fraud probabilities,
assigns configurable risk tiers, and triggers XAI explanations and SQLite persistence.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib
import numpy as np
import pandas as pd

import database
from preprocessing import (
    FraudDataPreprocessor,
    NUMERICAL_FEATURES,
    validate_transaction_dict,
    validate_batch_dataframe,
)
from explainability import explain_single_prediction, get_global_feature_importance

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fraud_model.joblib")
DEFAULT_PREPROCESSOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "preprocessor.joblib")


class FraudPredictor:
    """
    Inference service managing model lifecycle, input transformation,
    fraud risk scoring, and audit logging.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
        threshold_low: float = 0.20,
        threshold_high: float = 0.65,
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.threshold_low = float(os.getenv("RISK_THRESHOLD_LOW", threshold_low))
        self.threshold_high = float(os.getenv("RISK_THRESHOLD_HIGH", threshold_high))
        
        self.model = None
        self.preprocessor = None
        self.global_feature_importances = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model and preprocessing transformer into memory."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.global_feature_importances = get_global_feature_importance(self.model)
        else:
            self.model = None

        if os.path.exists(self.preprocessor_path):
            self.preprocessor = FraudDataPreprocessor.load(self.preprocessor_path)
        else:
            self.preprocessor = None

    @property
    def is_ready(self) -> bool:
        """Returns True if both model and preprocessor are loaded in memory."""
        return self.model is not None and self.preprocessor is not None

    def classify_risk(self, probability: float) -> str:
        """Maps probability to operational risk tiers."""
        if probability >= self.threshold_high:
            return "HIGH"
        elif probability >= self.threshold_low:
            return "MEDIUM"
        return "LOW"

    def predict_single(
        self,
        raw_data: Dict[str, Any],
        save_to_db: bool = True,
        source: str = "realtime",
        transaction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes and scores a single transaction payload.
        Returns full prediction result, risk categorization, and XAI breakdown.
        """
        if not self.is_ready:
            raise RuntimeError("Model or Preprocessor artifact is not loaded. Train the model first.")

        # 1. Validate and sanitize input
        sanitized = validate_transaction_dict(raw_data)
        input_df = pd.DataFrame([sanitized])

        # 2. Transform through preprocessor
        X_trans = self.preprocessor.transform(input_df)

        # 3. Model inference
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_trans)[0]
            # Class 1 is Fraud
            fraud_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        else:
            # Fallback if model only outputs discrete prediction
            decision = float(self.model.predict(X_trans)[0])
            fraud_prob = 0.90 if decision == 1 else 0.05

        risk_level = self.classify_risk(fraud_prob)
        prediction = 1 if (fraud_prob >= 0.50 or risk_level == "HIGH") else 0
        is_flagged = bool(risk_level in ["HIGH", "MEDIUM"] or prediction == 1)

        # 4. Generate local feature explanations
        explanation = explain_single_prediction(
            model=self.model,
            raw_features=sanitized,
            transformed_vector=X_trans,
            fraud_probability=fraud_prob,
            top_k=5,
            feature_names=NUMERICAL_FEATURES,
            global_importances=self.global_feature_importances
        )

        # 5. Persist to SQLite
        saved_id = None
        if save_to_db:
            database.init_db()
            saved_id = database.insert_transaction(
                amount=sanitized["Amount"],
                time_elapsed=sanitized["Time"],
                fraud_probability=fraud_prob,
                risk_level=risk_level,
                prediction=prediction,
                top_features=explanation["top_features"],
                source=source,
                features_dict=sanitized,
                transaction_id=transaction_id
            )

        return {
            "transaction_id": saved_id or transaction_id or "TXN-SIMULATED",
            "prediction": prediction,
            "prediction_label": "Fraudulent" if prediction == 1 else "Legitimate",
            "fraud_probability": round(fraud_prob, 4),
            "risk_level": risk_level,
            "is_flagged": is_flagged,
            "amount": sanitized["Amount"],
            "time_elapsed": sanitized["Time"],
            "thresholds_applied": {
                "low": self.threshold_low,
                "high": self.threshold_high
            },
            "explanation": explanation
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        save_to_db: bool = False,
        source: str = "batch_csv"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scores an entire batch DataFrame.
        Appends prediction, probability, risk_level, and is_flagged columns.
        """
        if not self.is_ready:
            raise RuntimeError("Model or Preprocessor artifact is not loaded. Train the model first.")

        clean_df, errors = validate_batch_dataframe(df)
        if errors and clean_df.empty:
            raise ValueError(f"Batch validation failed: {'; '.join(errors)}")

        # Transform entire batch
        X_trans = self.preprocessor.transform(clean_df)

        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_trans)
            fraud_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            preds = self.model.predict(X_trans)
            fraud_probs = np.where(preds == 1, 0.90, 0.05)

        # Vectorized risk categorization
        result_df = clean_df.copy()
        result_df["fraud_probability"] = np.round(fraud_probs, 4)
        
        conditions = [
            result_df["fraud_probability"] >= self.threshold_high,
            result_df["fraud_probability"] >= self.threshold_low
        ]
        choices = ["HIGH", "MEDIUM"]
        result_df["risk_level"] = np.select(conditions, choices, default="LOW")
        result_df["prediction"] = np.where(
            (result_df["fraud_probability"] >= 0.50) | (result_df["risk_level"] == "HIGH"), 1, 0
        )
        result_df["is_flagged"] = result_df["risk_level"].isin(["HIGH", "MEDIUM"]) | (result_df["prediction"] == 1)

        # Aggregate stats
        total_rows = len(result_df)
        flagged_count = int(result_df["is_flagged"].sum())
        high_risk_count = int((result_df["risk_level"] == "HIGH").sum())
        medium_risk_count = int((result_df["risk_level"] == "MEDIUM").sum())
        low_risk_count = int((result_df["risk_level"] == "LOW").sum())

        summary = {
            "total_processed": total_rows,
            "flagged_count": flagged_count,
            "high_risk_count": high_risk_count,
            "medium_risk_count": medium_risk_count,
            "low_risk_count": low_risk_count,
            "fraud_rate_percentage": round(flagged_count / total_rows * 100, 2) if total_rows > 0 else 0.0,
            "total_amount_processed": round(float(result_df["Amount"].sum()), 2),
            "flagged_amount_total": round(float(result_df[result_df["is_flagged"]]["Amount"].sum()), 2)
        }

        # Optional batch database insert
        if save_to_db:
            database.init_db()
            records_to_save = []
            for _, row in result_df.iterrows():
                feat_dict = {col: float(row[col]) for col in NUMERICAL_FEATURES}
                records_to_save.append({
                    "amount": float(row["Amount"]),
                    "time_elapsed": float(row["Time"]),
                    "fraud_probability": float(row["fraud_probability"]),
                    "risk_level": row["risk_level"],
                    "prediction": int(row["prediction"]),
                    "source": source,
                    "features_dict": feat_dict
                })
            database.insert_transactions_batch(records_to_save)

        return result_df, summary


# Global singleton instance
_predictor_instance = None

def get_predictor() -> FraudPredictor:
    """Returns singleton instance of FraudPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FraudPredictor()
    return _predictor_instance
