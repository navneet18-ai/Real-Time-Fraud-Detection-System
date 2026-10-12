"""
Explainable AI (XAI) module for fraud prediction.
Provides:
1. Global model feature importance (coefficients for linear models, Gini importance for tree models).
2. Local transaction risk explanation (feature impact scoring relative to population distributions).
3. Plain-language synthesis for fraud analysts and end users.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

from preprocessing import PCA_FEATURES, NUMERICAL_FEATURES


def get_global_feature_importance(model: Any, feature_names: List[str] = NUMERICAL_FEATURES) -> List[Dict[str, Any]]:
    """
    Extracts global feature importance from the trained model.
    Supports RandomForest, XGBoost, and LogisticRegression.
    """
    importance_scores = []

    # Tree-based models (Random Forest, XGBoost)
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
        for feat, score in zip(feature_names, raw_importances):
            importance_scores.append({
                "feature": feat,
                "importance": float(score),
                "direction": "risk_discriminator"
            })
    # Linear models (Logistic Regression)
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0]
        for feat, score in zip(feature_names, coefs):
            importance_scores.append({
                "feature": feat,
                "importance": float(abs(score)),
                "raw_coefficient": float(score),
                "direction": "increases_risk" if score > 0 else "decreases_risk"
            })
    else:
        # Fallback if model structure is custom
        for feat in feature_names:
            importance_scores.append({"feature": feat, "importance": 0.0, "direction": "neutral"})

    # Sort descending by importance magnitude
    importance_scores.sort(key=lambda x: x["importance"], reverse=True)
    return importance_scores


def explain_single_prediction(
    model: Any,
    raw_features: Dict[str, float],
    transformed_vector: np.ndarray,
    fraud_probability: float,
    top_k: int = 5,
    feature_names: List[str] = NUMERICAL_FEATURES,
    global_importances: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Computes a localized explanation for an individual transaction.
    Measures feature extremity and model weights to highlight the top features
    driving the risk assessment.
    """
    # Build dictionary of global importance lookup
    if not global_importances:
        global_importances = get_global_feature_importance(model, feature_names)
    
    importance_lookup = {item["feature"]: item["importance"] for item in global_importances}

    # Vector is shaped (1, 30): scaled_Time, V1..V28, scaled_Amount
    vec = transformed_vector.flatten()
    
    feature_impacts = []
    
    # Analyze PCA and financial features
    for idx, feat in enumerate(feature_names):
        val = raw_features.get(feat, 0.0)
        transformed_val = float(vec[idx])
        global_weight = importance_lookup.get(feat, 0.01)

        # Deviation from normal baseline (PCA features have baseline mean=0, std=1)
        # Scaled amount/time have baseline median=0 in RobustScaler
        deviation = abs(transformed_val)
        
        # Combined anomaly impact score
        impact_score = deviation * (global_weight ** 0.5)

        # Determine directional tendency
        if feat == "Amount":
            detail = f"Transaction amount of €{val:,.2f}"
            is_anomaly = val > 500 or val < 2.0
        elif feat == "Time":
            detail = f"Timestamp offset: {val:.0f}s"
            is_anomaly = False
        else:
            # PCA features
            detail = f"Latent component value: {val:+.3f}"
            is_anomaly = abs(val) > 2.5

        feature_impacts.append({
            "feature": feat,
            "raw_value": round(float(val), 3),
            "transformed_value": round(transformed_val, 3),
            "impact_score": round(float(impact_score), 4),
            "detail": detail,
            "is_anomaly": is_anomaly
        })

    # Sort features by impact score descending
    feature_impacts.sort(key=lambda x: x["impact_score"], reverse=True)
    top_drivers = feature_impacts[:top_k]

    # Generate plain language summary
    if fraud_probability >= 0.65:
        summary_intro = (
            f"Transaction evaluated as HIGH RISK ({fraud_probability*100:.1f}% estimated probability). "
            f"The primary anomaly drivers are {', '.join([d['feature'] for d in top_drivers[:3]])}."
        )
    elif fraud_probability >= 0.20:
        summary_intro = (
            f"Transaction evaluated as MEDIUM RISK ({fraud_probability*100:.1f}% estimated probability). "
            f"Notable deviations observed in {', '.join([d['feature'] for d in top_drivers[:2]])}."
        )
    else:
        summary_intro = (
            f"Transaction evaluated as LOW RISK ({fraud_probability*100:.1f}% estimated probability). "
            f"Feature values are consistent with legitimate transaction baselines."
        )

    disclaimer = (
        "Educational & Statistical Notice: Model probability and feature explanations reflect "
        "statistical correlation with historical transaction patterns. They do not represent verified "
        "evidence or definitive legal proof of fraudulent intent."
    )

    return {
        "summary": summary_intro,
        "fraud_probability": round(fraud_probability, 4),
        "top_features": top_drivers,
        "all_impacts": feature_impacts,
        "disclaimer": disclaimer
    }
