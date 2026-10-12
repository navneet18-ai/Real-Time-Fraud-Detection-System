# 🛡️ Real-Time Fraud Detection System

A complete, production-grade, end-to-end Machine Learning and Data Science system for detecting fraudulent financial transactions in real time. Designed for portfolio demonstration, academic evaluation, and practical financial analytics.

---

## 🌟 Key Features

- **⚡ Real-Time Fraud Inference**: Scores incoming transactions in milliseconds with calibrated fraud probabilities and dynamic risk tiering (**Low**, **Medium**, **High**).
- **🧠 Explainable AI (XAI)**: Identifies top feature drivers and anomaly deviations (e.g. latent PCA components, transaction amount anomalies) for each flagged transaction.
- **📊 Interactive Streamlit Dashboard**: 
  - **Executive Overview**: Financial KPI cards, at-risk capital metrics, and risk tier distribution donut charts.
  - **Real-Time Detection**: Interactive submission workbench with one-click authentic sample presets.
  - **Bulk CSV Analysis**: Automated batch scoring, error validation, and one-click export of predictions.
  - **Fraud Analytics**: Plotly interactive visualizations (log-scaled amount distributions, latent anomaly correlation, 48-hour cyclical volume).
  - **Model Performance**: Comparative leaderboards, confusion matrices, Precision-Recall (PR) curves, and ROC curves.
  - **Transaction Audit History**: Searchable SQLite database log with risk filtering and full audit export.
- **🚀 High-Throughput FastAPI REST Backend**:
  - `GET /health` — Service readiness and model status.
  - `POST /predict` — Single transaction evaluation with Pydantic validation.
  - `POST /predict-batch` — High-speed batch scoring for up to thousands of transactions.
  - `GET /transactions` — Paginated history with risk filtering and search.
  - `GET /metrics` — Live operational KPIs and offline model evaluation curves.
  - Built-in interactive Swagger UI documentation at `/docs`.
- **💾 SQLite Transactional Persistence**: Indexed audit trail storing synthetic transaction IDs, amounts, timestamps, probabilities, risk tiers, and feature JSON without storing sensitive PII or raw card numbers.
- **🧪 100% Automated Test Suite**: 26 unit and integration tests covering preprocessing, data leakage protection, model inference, database indexing, and API endpoints.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data & Training Pipeline
        A[Credit Card Dataset creditcard.csv] --> B[Data Preprocessing & Robust Scaling]
        B --> C[Stratified Train/Test Split 80/20]
        C --> D[Model Training: Logistic Regression, Random Forest, XGBoost]
        D --> E[Model Serialization: models/fraud_model.joblib]
    end

    subgraph Backend Engine [FastAPI Port 8000]
        E --> F[Predictor Service]
        G[REST API: /health, /predict, /predict-batch, /transactions] --> F
        F --> H[Dynamic Risk Classifier & XAI Explainer]
        H --> I[(SQLite: fraud_detection.db)]
    end

    subgraph Client Application [Streamlit Port 8501]
        J[Real-Time Scoring Workbench & Sample Presets] --> G
        K[Bulk CSV Batch Processing] --> G
        L[Plotly Interactive Analytics & Model Curves] <-- G
    end
```

---

## 📊 Dataset & Mathematical Methodology

### Dataset
This project is built around the European Credit Card Fraud Detection benchmark (available on Kaggle at `mlg-ulb/creditcardfraud`):
- **284,807 transactions** collected over 48 hours.
- **492 fraudulent transactions** (**0.172% extreme class imbalance**).
- **Features**: `Time`, `Amount` (Euros €), and `V1`–`V28` (orthogonal PCA principal components preserving cardholder privacy).

### Imbalance Strategy & Leakage Prevention
1. **Strict Stratified Splitting**: 80% train / 20% test maintaining identical class proportions in both splits.
2. **Leakage Prevention**: `RobustScaler` (median and IQR) is fitted **strictly on the training fold** and saved as an artifact (`models/preprocessor.joblib`). Test sets and real-time inputs are transformed without refitting.
3. **Cost-Sensitive Learning**: Cost-sensitive weights (`class_weight='balanced'`) penalize missed fraud cases.
4. **Metric Focus**:
   - **PR-AUC (Average Precision)** is the primary optimization metric.
   - Raw accuracy is deceptive (a dummy model predicting all legitimate achieves 99.83% accuracy while catching 0% of fraud).

### Risk Threshold Logic
- **Low Risk** ($P < 0.20$): Automatically cleared.
- **Medium Risk** ($0.20 \le P < 0.65$): Step-up verification (2FA/OTP challenge).
- **High Risk** ($P \ge 0.65$): Transaction temporarily blocked for analyst review.

---

## 📁 Project Structure

```text
Real-Time Fraud Detection System/
│
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git rules excluding datasets, joblib models, virtualenvs
├── requirements.txt              # Pinned Python package dependencies
├── README.md                     # Comprehensive documentation & setup instructions
├── PROJECT_REPORT.md             # Formal Data Science & ML project report
│
├── data/
│   ├── .gitkeep
│   ├── creditcard.csv            # Training dataset (50,000+ rows)
│   └── sample_transactions.csv   # Held-out sample cases for UI presets
│
├── models/
│   ├── .gitkeep
│   ├── fraud_model.joblib        # Serialized champion ML model
│   ├── preprocessor.joblib       # Serialized RobustScaler pipeline
│   └── model_metrics.json        # Evaluation curves, PR-AUC, confusion matrix
│
├── scripts/
│   ├── download_or_generate_data.py  # Dataset verification and synthetic generator
│   └── inspect_data.py           # Exploratory Data Analysis (EDA) auditor
│
├── preprocessing.py              # Input validation, schema sanitization, RobustScaler
├── train_model.py                # Model training, comparison, and evaluation pipeline
├── predict.py                    # Inference engine, probability calibration, risk scoring
├── explainability.py             # Feature importance and localized XAI explanations
├── database.py                   # SQLite database models, connections, and audit logging
├── api.py                        # FastAPI REST service with Pydantic validation
├── app.py                        # Streamlit multi-tab financial analytics dashboard
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py     # Leakage protection & schema validation unit tests
    ├── test_predict.py           # Inference consistency & risk tier tests
    ├── test_database.py          # SQLite persistence and statistics tests
    └── test_api.py               # FastAPI integration tests with TestClient
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone or navigate to the repository directory
cd "Real-Time Fraud Detection System"

# Create a virtual environment using Python 3.12 (or 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Preparation & Exploratory Data Analysis
```bash
# Verify, download, or generate the benchmark dataset
python scripts/download_or_generate_data.py

# Run Exploratory Data Analysis (EDA)
python scripts/inspect_data.py
```

### 3. Train and Compare Models
```bash
# Trains Logistic Regression, Random Forest, and XGBoost
# Evaluates PR-AUC, Recall, Precision, and serializes the champion model
python train_model.py
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 5. Launch the Application
Open two terminal windows (with `.venv` activated):

**Terminal 1 — FastAPI Backend:**
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
*Interactive Swagger Documentation*: `http://127.0.0.1:8000/docs`

**Terminal 2 — Streamlit Dashboard:**
```bash
streamlit run app.py --server.port 8501
```
*Interactive Dashboard*: `http://127.0.0.1:8501`

---

## 📡 REST API Examples

### Single Transaction Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict?save_to_db=true" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 400.0,
    "Amount": 1400.0,
    "V4": 5.2,
    "V11": 4.1,
    "V12": -5.9,
    "V14": -7.5,
    "V17": -6.2
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN-6FE19FAC6E",
  "prediction": 1,
  "prediction_label": "Fraudulent",
  "fraud_probability": 0.9999,
  "risk_level": "HIGH",
  "is_flagged": true,
  "amount": 1400.0,
  "time_elapsed": 400.0,
  "explanation": {
    "summary": "Transaction evaluated as HIGH RISK (100.0% estimated probability). The primary anomaly drivers are Amount, V14, V12.",
    "top_features": [
      {
        "feature": "Amount",
        "raw_value": 1400.0,
        "impact_score": 13.44,
        "is_anomaly": true
      }
    ],
    "disclaimer": "Educational & Statistical Notice: Model probability and feature explanations reflect statistical correlation with historical transaction patterns."
  }
}
```

---

## 🔒 Security & Privacy Notice

- **No Real Banking Credentials**: The system processes anonymized PCA features and transaction amounts. No real card numbers, CVVs, PINs, or cardholder names are accepted or stored.
- **Educational Prototype**: Built as a demonstration system. In production, banking security requires defense-in-depth, device fingerprinting, behavioural biometrics, and multi-factor authentication.
