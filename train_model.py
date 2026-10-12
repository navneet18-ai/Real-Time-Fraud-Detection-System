"""
Machine Learning Training and Model Evaluation Pipeline.
Trains and compares:
1. Baseline: Logistic Regression (class_weight='balanced')
2. Advanced: Random Forest Classifier (class_weight='balanced')
3. Optional: XGBoost Classifier (scale_pos_weight)

Features:
- Stratified 80/20 train/test split (no data leakage)
- Robust scaling fitted strictly on training partition
- Exhaustive evaluation: PR-AUC, Recall, Precision, F1, ROC-AUC, Confusion Matrix
- Automated model selection and joblib artifact persistence
- Exports detailed metrics JSON for Streamlit dashboard visualization
"""

import os
import sys
import json
import time
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Safe optional import of XGBoost for systems without libomp
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as xgb_err:
    HAS_XGBOOST = False
    print(f"[Notice] XGBoost native library unavailable ({xgb_err}). Using Scikit-Learn tree models.")

from preprocessing import FraudDataPreprocessor, NUMERICAL_FEATURES, TARGET_COLUMN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "fraud_model.joblib")
PREPROCESSOR_SAVE_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "model_metrics.json")


def load_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Loads transaction dataset. If missing, runs generation script."""
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Running scripts/download_or_generate_data.py...")
        from scripts.download_or_generate_data import main as prepare_data
        prepare_data()

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset successfully loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


def evaluate_model_performance(
    model_name: str,
    model: Any,
    X_test_trans: np.ndarray,
    y_test: np.ndarray,
    training_time_sec: float
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics with imbalanced-data focus."""
    y_pred = model.predict(X_test_trans)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_trans)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Core imbalanced metrics
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))

    # Curve coordinates for Plotly charts (downsampled for fast dashboard rendering)
    prec_pts, rec_pts, _ = precision_recall_curve(y_test, y_prob)
    fpr_pts, tpr_pts, _ = roc_curve(y_test, y_prob)

    step_pr = max(1, len(prec_pts) // 80)
    step_roc = max(1, len(fpr_pts) // 80)

    return {
        "model_name": model_name,
        "training_time_seconds": round(training_time_sec, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "pr_curve": {
            "precision": [round(float(p), 4) for p in prec_pts[::step_pr]],
            "recall": [round(float(r), 4) for r in rec_pts[::step_pr]],
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr_pts[::step_roc]],
            "tpr": [round(float(y), 4) for y in tpr_pts[::step_roc]],
        }
    }


def train_and_evaluate(
    quick_run: bool = False,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes end-to-end ML training workflow:
    1. Loads dataset
    2. Performs stratified 80/20 train/test split
    3. Fits RobustScaler preprocessor strictly on X_train
    4. Trains Logistic Regression, Random Forest, and XGBoost
    5. Evaluates and compares all models
    6. Selects and serializes best model
    """
    start_time = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_dataset()
    
    # 1. Feature / Target separation (guarantee no leakage)
    X = df[NUMERICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    total_transactions = len(df)
    fraud_transactions = int(y.sum())
    legit_transactions = total_transactions - fraud_transactions
    fraud_ratio = fraud_transactions / total_transactions

    print(f"\n--- Dataset Class Distribution ---")
    print(f"Total Transactions: {total_transactions:,}")
    print(f"Legitimate (Class 0): {legit_transactions:,} ({(1 - fraud_ratio)*100:.3f}%)")
    print(f"Fraudulent (Class 1): {fraud_transactions:,} ({fraud_ratio*100:.3f}%)")

    # 2. Stratified 80/20 train-test split
    print("\nExecuting Stratified Train-Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_seed,
        stratify=y
    )
    print(f"Training set: {len(X_train):,} rows ({int(y_train.sum())} fraud cases)")
    print(f"Testing set:  {len(X_test):,} rows ({int(y_test.sum())} fraud cases)")

    # 3. Fit preprocessor strictly on training fold
    print("\nFitting RobustScaler strictly on X_train to prevent data leakage...")
    preprocessor = FraudDataPreprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Save preprocessor artifact
    preprocessor.save(PREPROCESSOR_SAVE_PATH)
    print(f"Saved fitted preprocessor to: {PREPROCESSOR_SAVE_PATH}")

    models_to_train = {}
    
    # Model 1: Logistic Regression (Baseline)
    models_to_train["Logistic Regression (Baseline)"] = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_seed
    )

    # Model 2: Random Forest
    rf_trees = 50 if quick_run else 100
    models_to_train["Random Forest"] = RandomForestClassifier(
        n_estimators=rf_trees,
        max_depth=12,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1
    )

    # Model 3: XGBoost (or HistGradientBoostingClassifier if OpenMP is not installed)
    if HAS_XGBOOST:
        scale_pos = (legit_transactions / fraud_transactions) if fraud_transactions > 0 else 1.0
        xgb_trees = 50 if quick_run else 100
        models_to_train["XGBoost"] = XGBClassifier(
            n_estimators=xgb_trees,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=min(scale_pos, 100.0),
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1
        )
    else:
        models_to_train["Gradient Boosting (Hist)"] = HistGradientBoostingClassifier(
            max_iter=100,
            class_weight="balanced",
            random_state=random_seed
        )

    results = {}
    trained_model_objects = {}

    print("\n--- Model Training & Comparative Evaluation ---")
    for name, model in models_to_train.items():
        print(f"\nTraining [{name}]...")
        t0 = time.time()
        model.fit(X_train_trans, y_train)
        duration = time.time() - t0
        print(f"Finished training [{name}] in {duration:.2f} seconds.")

        eval_stats = evaluate_model_performance(
            model_name=name,
            model=model,
            X_test_trans=X_test_trans,
            y_test=y_test.to_numpy(),
            training_time_sec=duration
        )
        results[name] = eval_stats
        trained_model_objects[name] = model

        print(f"  Precision: {eval_stats['precision']:.4f}")
        print(f"  Recall:    {eval_stats['recall']:.4f}")
        print(f"  F1 Score:  {eval_stats['f1_score']:.4f}")
        print(f"  PR-AUC:    {eval_stats['pr_auc']:.4f}")
        print(f"  ROC-AUC:   {eval_stats['roc_auc']:.4f}")
        cm = eval_stats["confusion_matrix"]
        print(f"  Confusion Matrix: TP={cm['true_positives']}, FP={cm['false_positives']}, FN={cm['false_positives']}, TN={cm['true_negatives']}")

    # 4. Model Selection Logic (Ranked primarily by PR-AUC and F1-Score)
    best_model_name = max(
        results.keys(),
        key=lambda m: (results[m]["pr_auc"] * 0.6 + results[m]["f1_score"] * 0.4)
    )
    best_model = trained_model_objects[best_model_name]
    print(f"\n=======================================================")
    print(f"Selected Champion Model: [{best_model_name}]")
    print(f"PR-AUC: {results[best_model_name]['pr_auc']} | F1: {results[best_model_name]['f1_score']}")
    print(f"=======================================================")

    # 5. Serialize Champion Model
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"Saved serialized champion model to: {MODEL_SAVE_PATH}")

    # 6. Save comprehensive evaluation metrics JSON
    metrics_export = {
        "dataset_metadata": {
            "total_transactions": total_transactions,
            "fraud_count": fraud_transactions,
            "legitimate_count": legit_transactions,
            "fraud_percentage": round(fraud_ratio * 100, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "selected_model": best_model_name,
        "models": results,
        "features": NUMERICAL_FEATURES,
        "trained_at_epoch": time.time(),
        "total_pipeline_time_seconds": round(time.time() - start_time, 2)
    }

    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"Saved model metrics and evaluation curves to: {METRICS_SAVE_PATH}")

    return metrics_export


if __name__ == "__main__":
    train_and_evaluate()
