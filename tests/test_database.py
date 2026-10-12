"""
Unit tests for SQLite database operations, indexing, and KPI queries.
"""

import os
import pytest
import database


@pytest.fixture
def temp_db(tmp_path):
    """Provides a fresh isolated test database path for each test."""
    db_file = str(tmp_path / "test_fraud.db")
    database.init_db(db_file)
    return db_file


def test_init_db_creates_table(temp_db):
    conn = database.get_db_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
    table = cursor.fetchone()
    conn.close()
    assert table is not None
    assert table["name"] == "transactions"


def test_insert_and_retrieve_single_transaction(temp_db):
    txn_id = database.insert_transaction(
        amount=250.75,
        time_elapsed=1200.0,
        fraud_probability=0.88,
        risk_level="HIGH",
        prediction=1,
        top_features=[{"feature": "V14", "impact_score": 0.45}],
        source="test_runner",
        db_path=temp_db
    )
    assert txn_id.startswith("TXN-")

    records = database.get_transactions(limit=10, db_path=temp_db)
    assert len(records) == 1
    r = records[0]
    assert r["transaction_id"] == txn_id
    assert r["amount"] == 250.75
    assert r["risk_level"] == "HIGH"
    assert r["prediction"] == 1
    assert r["is_flagged"] == 1
    assert isinstance(r["top_features"], list)


def test_insert_batch_and_stats(temp_db):
    batch = [
        {"amount": 50.0, "time_elapsed": 100.0, "fraud_probability": 0.05, "risk_level": "LOW", "prediction": 0},
        {"amount": 80.0, "time_elapsed": 200.0, "fraud_probability": 0.35, "risk_level": "MEDIUM", "prediction": 0},
        {"amount": 900.0, "time_elapsed": 300.0, "fraud_probability": 0.95, "risk_level": "HIGH", "prediction": 1},
    ]
    inserted = database.insert_transactions_batch(batch, db_path=temp_db)
    assert inserted == 3

    stats = database.get_transaction_stats(temp_db)
    assert stats["total_transactions"] == 3
    assert stats["flagged_transactions"] == 2  # MEDIUM and HIGH are flagged
    assert stats["confirmed_fraud_predictions"] == 1
    assert stats["total_volume_amount"] == 1030.0
    assert stats["flagged_volume_amount"] == 980.0


def test_filtering_and_search(temp_db):
    database.insert_transaction(10.0, 1.0, 0.02, "LOW", 0, transaction_id="TXN-AAA111", db_path=temp_db)
    database.insert_transaction(500.0, 2.0, 0.92, "HIGH", 1, transaction_id="TXN-BBB222", db_path=temp_db)

    # Filter by risk
    high_records = database.get_transactions(risk_level="HIGH", db_path=temp_db)
    assert len(high_records) == 1
    assert high_records[0]["transaction_id"] == "TXN-BBB222"

    # Search by partial ID
    search_res = database.get_transactions(search_query="AAA", db_path=temp_db)
    assert len(search_res) == 1
    assert search_res[0]["transaction_id"] == "TXN-AAA111"
