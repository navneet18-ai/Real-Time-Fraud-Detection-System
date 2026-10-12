"""
FastAPI REST API for Real-Time Fraud Detection System.
Provides high-throughput prediction endpoints, input validation via Pydantic schemas,
historical transaction retrieval from SQLite, and model performance metrics.
Loads model artifacts once during startup lifecycle.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

import database
from predict import get_predictor, FraudPredictor
from preprocessing import NUMERICAL_FEATURES, PCA_FEATURES


# ---------------------------------------------------------
# Pydantic Schemas for Request Validation and Swagger Docs
# ---------------------------------------------------------

class SingleTransactionInput(BaseModel):
    """Schema for a single financial transaction submission."""
    Time: float = Field(default=0.0, ge=0.0, description="Elapsed seconds since reference transaction")
    Amount: float = Field(..., ge=0.0, description="Transaction amount in EUR (€)")
    # PCA features V1 through V28 default to 0.0 (baseline mean)
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 406.0,
                "Amount": 149.62,
                "V1": -2.31,
                "V2": 1.95,
                "V3": -1.60,
                "V4": 3.99,
                "V14": -4.28,
                "V17": -3.81
            }
        }
    )


class FeatureImpact(BaseModel):
    feature: str
    raw_value: float
    transformed_value: float
    impact_score: float
    detail: str
    is_anomaly: bool


class PredictionExplanation(BaseModel):
    summary: str
    fraud_probability: float
    top_features: List[FeatureImpact]
    disclaimer: str


class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: int
    prediction_label: str
    fraud_probability: float
    risk_level: str
    is_flagged: bool
    amount: float
    time_elapsed: float
    explanation: Optional[PredictionExplanation] = None


class BatchPredictionRequest(BaseModel):
    transactions: List[SingleTransactionInput]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    flagged_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    fraud_rate_percentage: float
    total_amount_processed: float
    flagged_amount_total: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str]
    thresholds: Dict[str, float]


# ---------------------------------------------------------
# Lifespan Management (Load model once at startup)
# ---------------------------------------------------------

predictor_service: Optional[FraudPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database and loads ML model artifacts once on startup."""
    global predictor_service
    print("[API Lifespan] Initializing SQLite database...")
    database.init_db()

    print("[API Lifespan] Preloading Machine Learning artifacts...")
    predictor_service = get_predictor()
    if predictor_service.is_ready:
        print("[API Lifespan] ML Model & Preprocessor successfully loaded into memory!")
    else:
        print("[API Lifespan] WARNING: Model artifacts not found yet. Run train_model.py.")
    yield
    print("[API Lifespan] Shutting down API service.")


# ---------------------------------------------------------
# Application Initialization
# ---------------------------------------------------------

app = FastAPI(
    title="Real-Time Fraud Detection System API",
    description=(
        "Financial transaction fraud detection and risk assessment service powered by Machine Learning. "
        "Provides single and batch transaction scoring, dynamic risk tiers (Low/Medium/High), "
        "Explainable AI feature breakdowns, and SQLite audit logging."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Checks service health and model loading readiness."""
    global predictor_service
    if predictor_service is None:
        predictor_service = get_predictor()

    is_ready = predictor_service.is_ready
    model_type = predictor_service.model.__class__.__name__ if is_ready else None

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_type=model_type,
        thresholds={
            "low_risk": predictor_service.threshold_low,
            "high_risk": predictor_service.threshold_high
        }
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_transaction(
    payload: SingleTransactionInput,
    save_to_db: bool = Query(True, description="Persist prediction in SQLite history")
):
    """
    Evaluates a single financial transaction.
    Returns calibrated fraud probability, operational risk tier, and feature explanations.
    """
    global predictor_service
    if not predictor_service or not predictor_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded. Train the model using train_model.py."
        )

    try:
        data_dict = payload.model_dump()
        result = predictor_service.predict_single(
            raw_data=data_dict,
            save_to_db=save_to_db,
            source="api_realtime"
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_transactions(
    payload: BatchPredictionRequest,
    save_to_db: bool = Query(False, description="Persist all batch records into SQLite")
):
    """
    Evaluates a batch of financial transactions.
    Computes predictions, risk levels, and aggregate summary metrics.
    """
    global predictor_service
    if not predictor_service or not predictor_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded."
        )

    if not payload.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transactions list cannot be empty."
        )

    import pandas as pd
    try:
        raw_list = [t.model_dump() for t in payload.transactions]
        batch_df = pd.DataFrame(raw_list)
        result_df, summary = predictor_service.predict_batch(
            df=batch_df,
            save_to_db=save_to_db,
            source="api_batch"
        )

        predictions_list = []
        for idx, row in result_df.iterrows():
            predictions_list.append(PredictionResponse(
                transaction_id=f"TXN-BATCH-{idx+1:04d}",
                prediction=int(row["prediction"]),
                prediction_label="Fraudulent" if row["prediction"] == 1 else "Legitimate",
                fraud_probability=float(row["fraud_probability"]),
                risk_level=str(row["risk_level"]),
                is_flagged=bool(row["is_flagged"]),
                amount=float(row["Amount"]),
                time_elapsed=float(row["Time"]),
                explanation=None
            ))

        return BatchPredictionResponse(
            total_processed=summary["total_processed"],
            flagged_count=summary["flagged_count"],
            high_risk_count=summary["high_risk_count"],
            medium_risk_count=summary["medium_risk_count"],
            low_risk_count=summary["low_risk_count"],
            fraud_rate_percentage=summary["fraud_rate_percentage"],
            total_amount_processed=summary["total_amount_processed"],
            flagged_amount_total=summary["flagged_amount_total"],
            predictions=predictions_list
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch error: {str(e)}")


@app.get("/transactions", tags=["History & Audit"])
async def list_transactions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    risk_level: Optional[str] = Query(None, description="Filter by LOW, MEDIUM, or HIGH"),
    is_flagged: Optional[bool] = Query(None, description="Filter flagged transactions"),
    search: Optional[str] = Query(None, description="Search by transaction ID or source")
):
    """Retrieves paginated historical transaction predictions and audit logs from SQLite."""
    try:
        records = database.get_transactions(
            limit=limit,
            offset=offset,
            risk_level=risk_level,
            is_flagged=is_flagged,
            search_query=search
        )
        return {
            "count": len(records),
            "limit": limit,
            "offset": offset,
            "transactions": records
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/metrics", tags=["Analytics & KPIs"])
async def get_system_metrics():
    """
    Returns both offline machine learning model benchmark metrics
    and live transactional database KPI metrics.
    """
    import json
    # 1. Live transactional stats from SQLite
    db_stats = database.get_transaction_stats()

    # 2. Offline ML metrics from file if available
    metrics_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_metrics.json")
    ml_metrics = None
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                ml_metrics = json.load(f)
        except Exception:
            pass

    return {
        "live_database_kpis": db_stats,
        "offline_model_metrics": ml_metrics
    }
