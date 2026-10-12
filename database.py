"""
Database layer for Real-Time Fraud Detection System.
Provides thread-safe SQLite persistence for transaction predictions, audit logging,
and analytics querying without storing PII or sensitive banking credentials.
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_detection.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH):
    """Initializes the SQLite schema and indices."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                time_elapsed REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                is_flagged INTEGER NOT NULL,
                top_features TEXT,
                source TEXT DEFAULT 'realtime',
                features_json TEXT
            )
        """)
        # Create performance indices for fast dashboard filtering
        conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions (transaction_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON transactions (timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_level ON transactions (risk_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_is_flagged ON transactions (is_flagged)")
    conn.close()


def insert_transaction(
    amount: float,
    time_elapsed: float,
    fraud_probability: float,
    risk_level: str,
    prediction: int,
    top_features: Optional[Dict[str, Any]] = None,
    source: str = "realtime",
    features_dict: Optional[Dict[str, float]] = None,
    transaction_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """
    Inserts a single transaction record into the database.
    Returns the assigned transaction_id.
    """
    if not transaction_id:
        transaction_id = f"TXN-{uuid.uuid4().hex[:10].upper()}"

    timestamp_str = datetime.now(timezone.utc).isoformat()
    is_flagged = 1 if risk_level in ["HIGH", "MEDIUM"] or prediction == 1 else 0
    top_features_json = json.dumps(top_features) if top_features else None
    features_json = json.dumps(features_dict) if features_dict else None

    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO transactions (
                transaction_id, timestamp, amount, time_elapsed,
                fraud_probability, risk_level, prediction, is_flagged,
                top_features, source, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                timestamp_str,
                float(amount),
                float(time_elapsed),
                float(fraud_probability),
                risk_level,
                int(prediction),
                is_flagged,
                top_features_json,
                source,
                features_json,
            ),
        )
    conn.close()
    return transaction_id


def insert_transactions_batch(
    records: List[Dict[str, Any]],
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Efficiently inserts multiple transaction records using executemany.
    Returns the number of rows inserted.
    """
    if not records:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    rows_to_insert = []
    for r in records:
        txn_id = r.get("transaction_id") or f"TXN-{uuid.uuid4().hex[:10].upper()}"
        risk_lvl = r.get("risk_level", "LOW")
        pred = int(r.get("prediction", 0))
        is_flagged = 1 if risk_lvl in ["HIGH", "MEDIUM"] or pred == 1 else 0

        rows_to_insert.append((
            txn_id,
            r.get("timestamp", now_iso),
            float(r.get("amount", 0.0)),
            float(r.get("time_elapsed", 0.0)),
            float(r.get("fraud_probability", 0.0)),
            risk_lvl,
            pred,
            is_flagged,
            json.dumps(r.get("top_features")) if r.get("top_features") else None,
            r.get("source", "batch_csv"),
            json.dumps(r.get("features_dict")) if r.get("features_dict") else None,
        ))

    conn = get_db_connection(db_path)
    with conn:
        conn.executemany(
            """
            INSERT INTO transactions (
                transaction_id, timestamp, amount, time_elapsed,
                fraud_probability, risk_level, prediction, is_flagged,
                top_features, source, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert
        )
    conn.close()
    return len(rows_to_insert)


def get_transactions(
    limit: int = 100,
    offset: int = 0,
    risk_level: Optional[str] = None,
    is_flagged: Optional[bool] = None,
    search_query: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieves paginated transactions with optional filters."""
    query = "SELECT * FROM transactions WHERE 1=1"
    params: List[Any] = []

    if risk_level:
        query += " AND risk_level = ?"
        params.append(risk_level.upper())

    if is_flagged is not None:
        query += " AND is_flagged = ?"
        params.append(1 if is_flagged else 0)

    if search_query:
        query += " AND (transaction_id LIKE ? OR source LIKE ?)"
        like_term = f"%{search_query}%"
        params.extend([like_term, like_term])

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        d = dict(row)
        if d.get("top_features"):
            try:
                d["top_features"] = json.loads(d["top_features"])
            except Exception:
                pass
        results.append(d)
    return results


def get_transaction_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Computes summary KPI statistics across all recorded transactions."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_count,
            SUM(CASE WHEN is_flagged = 1 THEN 1 ELSE 0 END) as flagged_count,
            SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as confirmed_fraud_count,
            SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) as high_risk_count,
            SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) as medium_risk_count,
            SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END) as low_risk_count,
            SUM(amount) as total_volume,
            SUM(CASE WHEN is_flagged = 1 THEN amount ELSE 0 END) as flagged_volume,
            AVG(fraud_probability) as avg_fraud_prob
        FROM transactions
    """)
    row = cursor.fetchone()
    conn.close()

    total = row["total_count"] or 0
    flagged = row["flagged_count"] or 0
    fraud_rate = (flagged / total * 100) if total > 0 else 0.0

    return {
        "total_transactions": total,
        "flagged_transactions": flagged,
        "confirmed_fraud_predictions": row["confirmed_fraud_count"] or 0,
        "high_risk_count": row["high_risk_count"] or 0,
        "medium_risk_count": row["medium_risk_count"] or 0,
        "low_risk_count": row["low_risk_count"] or 0,
        "fraud_rate_percentage": round(fraud_rate, 2),
        "total_volume_amount": round(row["total_volume"] or 0.0, 2),
        "flagged_volume_amount": round(row["flagged_volume"] or 0.0, 2),
        "average_fraud_probability": round(row["avg_fraud_prob"] or 0.0, 4)
    }


def clear_transactions(db_path: str = DEFAULT_DB_PATH):
    """Utility to reset transaction history for fresh demonstrations."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("DELETE FROM transactions")
    conn.close()
