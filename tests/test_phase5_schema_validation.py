"""Phase 5 schema validation report tests."""

import polars as pl

from core.contracts import ACCOUNT_METADATA_SCHEMA, COMPLAINT_SCHEMA, TRANSACTION_SCHEMA
from validation.schema_report import SchemaValidator


def test_schema_validator_detects_null_values() -> None:
    """SchemaValidator should detect null values in required columns."""
    df = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "sender_account": ["A", None, "C"],
        "receiver_account": ["B", "B", "C"],
        "amount": [100.0, 200.0, 300.0],
        "timestamp": ["2024-06-01 10:00:00", "2024-06-01 10:01:00", "2024-06-01 10:02:00"],
        "payment_mode": ["UPI", "UPI", "UPI"],
    }).with_columns(pl.col("timestamp").str.to_datetime())

    validator = SchemaValidator(TRANSACTION_SCHEMA)
    report = validator.validate(df)

    assert report.null_counts["sender_account"] == 1
    assert report.null_counts["txn_id"] == 0
    assert not report.passed


def test_schema_validator_detects_type_mismatches() -> None:
    """SchemaValidator should detect type mismatches."""
    # Create a DataFrame with string amount, then try to cast to float
    df = pl.DataFrame({
        "txn_id": ["T1", "T2"],
        "sender_account": ["A", "B"],
        "receiver_account": ["B", "C"],
        "amount": ["not-a-number", "200.0"],  # String that can't be cast to float
        "timestamp": ["2024-06-01 10:00:00", "2024-06-01 10:01:00"],
        "payment_mode": ["UPI", "UPI"],
    }).with_columns(pl.col("timestamp").str.to_datetime())

    validator = SchemaValidator(TRANSACTION_SCHEMA)
    report = validator.validate(df)

    assert report.type_mismatches["amount"] >= 1
    assert not report.passed


def test_schema_validator_detects_outliers() -> None:
    """SchemaValidator should detect statistical outliers using IQR method."""
    # Create data with one extreme outlier
    amounts = [100.0] * 10 + [1000000.0]  # 10 normal + 1 extreme
    df = pl.DataFrame({
        "txn_id": [f"T{i}" for i in range(11)],
        "sender_account": [f"A{i}" for i in range(11)],
        "receiver_account": [f"B{i}" for i in range(11)],
        "amount": amounts,
        "timestamp": ["2024-06-01 10:00:00"] * 11,
        "payment_mode": ["UPI"] * 11,
    }).with_columns(pl.col("timestamp").str.to_datetime())

    validator = SchemaValidator(TRANSACTION_SCHEMA)
    report = validator.validate(df)

    assert report.outliers["amount"] >= 1
    assert report.column_stats["amount"]["max"] > 100000.0


def test_schema_validator_passes_clean_data() -> None:
    """SchemaValidator should pass clean, valid data."""
    df = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "sender_account": ["A", "B", "C"],
        "receiver_account": ["B", "C", "D"],
        "amount": [100.0, 200.0, 300.0],
        "timestamp": ["2024-06-01 10:00:00", "2024-06-01 10:01:00", "2024-06-01 10:02:00"],
        "payment_mode": ["UPI", "IMPS", "NEFT"],
    }).with_columns(pl.col("timestamp").str.to_datetime())

    validator = SchemaValidator(TRANSACTION_SCHEMA)
    report = validator.validate(df)

    assert report.passed
    assert report.total_rows == 3
    assert all(v == 0 for v in report.null_counts.values())
    assert all(v == 0 for v in report.type_mismatches.values())


def test_schema_validator_works_on_accounts() -> None:
    """SchemaValidator should work on account metadata schema."""
    df = pl.DataFrame({
        "account_no": ["A1", "A2"],
        "ifsc": ["SBIN0001", "SBIN0002"],
        "branch_code": ["BR1", "BR2"],
        "mobile_no": ["9876543210", "9876543211"],
        "imei": ["123456789012345", "123456789012346"],
        "kyc_pincode": ["110001", "110002"],
        "account_opening_date": ["2020-01-01", "2021-01-01"],
        "account_type": ["SAVINGS", "CURRENT"],
    }).with_columns(pl.col("account_opening_date").str.to_date())

    validator = SchemaValidator(ACCOUNT_METADATA_SCHEMA)
    report = validator.validate(df)

    assert report.passed
    assert report.total_rows == 2


def test_schema_validator_works_on_complaints() -> None:
    """SchemaValidator should work on complaints schema."""
    df = pl.DataFrame({
        "ack_no": ["ACK1", "ACK2"],
        "victim_account": ["V1", "V2"],
        "suspect_account": ["S1", "S2"],
        "disputed_utr": ["T1", "T2"],
        "fraud_category": ["UPI Fraud", "Card Fraud"],
        "reported_timestamp": ["2024-06-01 10:00:00", "2024-06-01 11:00:00"],
        "disputed_amount": [10000.0, 20000.0],
        "state": ["DL", "MH"],
    }).with_columns(pl.col("reported_timestamp").str.to_datetime())

    validator = SchemaValidator(COMPLAINT_SCHEMA)
    report = validator.validate(df)

    assert report.passed
    assert report.total_rows == 2