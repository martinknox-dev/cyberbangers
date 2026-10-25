"""Phase 5 temporal pattern mining tests."""

import polars as pl

from features.temporal import compute_temporal_features


def test_temporal_features_detect_hourly_burst() -> None:
    rows = []
    for i in range(20):
        rows.append({
            "txn_id": f"T{i}", "sender_account": "A", "receiver_account": f"B{i}",
            "amount": 1000.0,
            "timestamp": f"2024-06-01 02:{i%60:02d}:00",
            "payment_mode": "UPI",
        })
    txns = pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_datetime())

    features = compute_temporal_features(txns)
    row = features.filter(pl.col("account_no") == "A").to_dicts()[0]

    assert row["night_txn_ratio"] > 0.8
    assert row["peak_hour"] == 2
    assert row["velocity_txns_per_hour"] > 10


def test_temporal_features_detect_normal_hours() -> None:
    rows = []
    for i in range(20):
        rows.append({
            "txn_id": f"T{i}", "sender_account": "B", "receiver_account": f"C{i}",
            "amount": 1000.0,
            "timestamp": f"2024-06-01 10:{i%60:02d}:00",
            "payment_mode": "UPI",
        })
    txns = pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_datetime())

    features = compute_temporal_features(txns)
    row = features.filter(pl.col("account_no") == "B").to_dicts()[0]

    assert row["night_txn_ratio"] < 0.2
    assert row["peak_hour"] == 10
    assert row["velocity_txns_per_hour"] > 10


def test_temporal_features_detect_weekend_concentration() -> None:
    rows = []
    for i in range(10):
        rows.append({
            "txn_id": f"T{i}", "sender_account": "C", "receiver_account": f"D{i}",
            "amount": 1000.0,
            "timestamp": f"2024-06-01 14:{i*6:02d}:00",
            "payment_mode": "UPI",
        })
    for i in range(10):
        rows.append({
            "txn_id": f"T{i+10}", "sender_account": "C", "receiver_account": f"E{i}",
            "amount": 1000.0,
            "timestamp": f"2024-06-02 14:{i*6:02d}:00",
            "payment_mode": "UPI",
        })
    txns = pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_datetime())

    features = compute_temporal_features(txns)
    row = features.filter(pl.col("account_no") == "C").to_dicts()[0]

    # Two days (Sat/Sun) with 10 txns each -> max freq = 10/20 = 0.5
    assert row["dow_concentration"] == 0.5
    assert row["active_days"] == 2


def test_temporal_features_integration_with_account_features() -> None:
    from core.contracts import RiskConfig
    from ui import state

    rows = []
    for i in range(15):
        rows.append({
            "txn_id": f"T{i}", "sender_account": "X", "receiver_account": f"Y{i}",
            "amount": 5000.0,
            "timestamp": f"2024-06-01 03:{i*4:02d}:00",
            "payment_mode": "UPI",
        })
    txns = pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_datetime())

    bundle = state.run_pipeline(txns, risk_config=RiskConfig())
    features = bundle.features

    assert "night_txn_ratio" in features.columns
    assert "peak_hour" in features.columns
    assert "velocity_txns_per_hour" in features.columns
    assert "dow_concentration" in features.columns
    assert "active_days" in features.columns

    row = features.filter(pl.col("account_no") == "X").to_dicts()[0]
    assert row["night_txn_ratio"] > 0.8
    assert row["peak_hour"] == 3
