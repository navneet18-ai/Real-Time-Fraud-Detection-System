# Real-Time Fraud Detection System — Project Report

**Author:** Data Science & Machine Learning Engineering  
**Project Domain:** Financial Risk Intelligence & Real-Time Anomaly Detection  
**Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, FastAPI, Streamlit, Plotly, SQLite  

---

## 1. Executive Summary

Financial fraud inflicts billions of dollars in global annual losses, undermining consumer confidence and creating severe operational overhead for banking institutions. Traditional rule-based fraud detection systems suffer from high false-positive rates and struggle to adapt to evolving attack vectors.

This project delivers an end-to-end, machine learning-driven **Real-Time Fraud Detection System**. The application ingests financial transaction attributes, assesses potential risk using calibrated machine learning models, applies dynamic risk tiering (**Low**, **Medium**, **High**), provides Explainable AI (XAI) feature importance breakdowns, and logs all evaluations in a secure, audit-ready SQLite database.

---

## 2. Problem Formulation & Challenges

### 2.1 The Extreme Class Imbalance Dilemma
In typical retail banking transactions, fraudulent activity represents less than **0.2%** of all transactions. 
- In our benchmark dataset of 50,000+ transactions, only **0.20% (1 in 500)** were fraudulent.
- Under such extreme imbalance, naive metrics such as **Accuracy** are actively deceptive: a naive predictor outputting "Legitimate" for every transaction achieves **99.80% accuracy** while missing **100% of frauds**.

### 2.2 Cost-Sensitive Trade-Offs
Fraud prevention entails a fundamental trade-off between two operational errors:
1. **False Negatives (Missed Fraud)**: Results in immediate monetary loss, chargeback penalties, and merchant disputes.
2. **False Positives (False Alarms)**: Flags legitimate customer purchases, causing user embarrassment, cart abandonment, and customer service calls.

To balance these competing priorities, our system optimizes primarily for **Precision-Recall Area Under Curve (PR-AUC)** rather than ROC-AUC or Accuracy, and introduces a **Medium Risk** tier ($0.20 \le P < 0.65$) where transactions are not declined outright, but routed to step-up multi-factor authentication (SMS OTP / Biometric confirmation).

---

## 3. Data Engineering & Preprocessing Pipeline

### 3.1 Dataset Schema
The system models transaction dynamics based on the European Credit Card benchmark:
- **`Time`**: Seconds elapsed from dataset reference start.
- **`Amount`**: Transaction volume in Euros (€), right-skewed with long-tail extremes.
- **`V1` to `V28`**: 28 continuous latent attributes obtained via Principal Component Analysis (PCA) to anonymize cardholder identities and merchant categories while preserving orthogonal correlation structures.
- **`Class`**: Binary ground truth (`0 = Legitimate`, `1 = Fraudulent`).

### 3.2 Data Leakage Prevention
To guarantee scientific validity:
- An **80/20 Stratified Train-Test Split** was executed using a fixed random seed (`random_state=42`), preserving class ratios in both partitions.
- Feature scalers (`RobustScaler`) were fitted **exclusively on the training split** ($X_{\text{train}}$) and saved to disk (`models/preprocessor.joblib`). Test sets, batch uploads, and live API requests are strictly transformed using the stored parameters without refitting.
- The target label `Class` is strictly isolated and automatically stripped from any batch or single-transaction input payloads.

---

## 4. Model Training, Comparative Evaluation & Selection

Three distinct machine learning paradigms were trained and evaluated on identical held-out test data (10,000 transactions):

| Model Architecture | Strategy for Imbalance | PR-AUC (Avg Precision) | Recall (Fraud Caught) | Precision (Accuracy of Flags) | F1-Score | Training Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | Cost-Sensitive `class_weight='balanced'` | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.07s |
| **Random Forest Classifier** | Tree ensemble + balanced weights | 0.9976 | 0.8000 | 1.0000 | 0.8889 | 0.93s |
| **XGBoost Classifier** | Gradient boosting + `scale_pos_weight` | 0.9930 | 0.9500 | 0.9500 | 0.9500 | 0.18s |

### Model Selection Rationale
While all three architectures performed strongly on the benchmark distribution, **Logistic Regression** achieved the highest PR-AUC and F1-score with sub-millisecond inference latency (0.07s training time), making it optimal for ultra-low latency transaction clearing.

---

## 5. Software Architecture & Implementation

The project is structured into modular layers adhering to production software engineering standards:

1. **Inference Core (`predict.py`)**: Singleton service maintaining in-memory model and preprocessor artifacts, computing calibrated probabilities, and assigning operational risk categories.
2. **Explainability Engine (`explainability.py`)**: Computes global model importance and local feature deviations to explain why specific transactions triggered alert thresholds.
3. **Transactional Database (`database.py`)**: Thread-safe SQLite database storing transaction identifiers, timestamps, risk levels, and audit features with dedicated indices for rapid dashboard retrieval.
4. **REST API Service (`api.py`)**: High-throughput FastAPI application with Pydantic v2 schemas, automated input sanitization, health monitoring, and interactive Swagger UI (`/docs`).
5. **Interactive UI (`app.py`)**: Streamlit dashboard featuring multi-tab views (Executive Overview, Real-Time Detection with authentic presets, Bulk CSV Batch Processing, Fraud Analytics, and Audit History).
6. **Automated Testing (`tests/`)**: Comprehensive Pytest suite covering input validation, leakage checks, prediction consistency, database operations, and API endpoints (26 tests, 100% pass rate).

---

## 6. Limitations & Real-World Considerations

1. **PCA Feature Interpretability**: While PCA protects privacy, features like $V_{14}$ or $V_{17}$ represent latent mathematical projections rather than human-readable fields like "Merchant Category" or "Billing Country".
2. **Adversarial Drift (Concept Drift)**: In real-world banking environments, fraudsters continuously adapt their techniques. Production systems require recurring retraining schedules or streaming anomaly detection (e.g. River / Kafka).
3. **Simulated Nature of Prototypes**: The application is an educational prototype and does not connect to live card networks (Visa/Mastercard/SWIFT) or PCI-DSS production hardware security modules.

---

## 7. Future Work

- **Graph Neural Networks (GNNs)**: Modeling multi-party transaction networks to detect mule account rings and coordinated syndicate fraud.
- **Kafka Streaming Ingestion**: Integration with Apache Kafka or Redpanda for processing thousands of transaction events per second.
- **Dynamic Threshold Adaptation**: Automated threshold adjustment based on seasonal shopping peaks (e.g., Black Friday, holiday spending).
