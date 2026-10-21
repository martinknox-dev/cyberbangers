"""Phase 1 acceptance tests: contracts, fuzzy ingestion, deterministic simulator."""
import json
from datetime import date, datetime

import polars as pl
import pytest
from pydantic import ValidationError

from core.contracts import (
    ACCOUNT_COLUMNS,
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_COLUMNS,
    COMPLAINT_SCHEMA,
    IFSC_PATTERN,
    MANDATORY_BY_TABLE,
    MANDATORY_COLUMNS,
    TRANSACTION_COLUMNS,
    TRANSACTION_SCHEMA,
    AccountMetadata,
    ColumnMapping,
    Complaint,
    IngestionReport,
    Transaction,
)

VALID_TXN_ROW: dict[str, object] = {
    "txn_id": "UTR2024060100001",
    "sender_account": "2900000001",
    "sender_ifsc": "SBIN0001234",
    "receiver_account": "2900000002",
    "receiver_ifsc": "SBIN0009876",
    "amount": 15000.0,
    "timestamp": datetime(2024, 6, 1, 10, 30, 0),
    "payment_mode": "UPI",
    "device_id": "D000123",
    "ip_address": "10.20.30.40",
}


class TestContracts:
    def test_valid_transaction_row_validates(self) -> None:
        txn = Transaction.model_validate(VALID_TXN_ROW)
        assert txn.amount == 15000.0
        assert txn.payment_mode == "UPI"

    def test_transaction_rejects_negative_amount(self) -> None:
        with pytest.raises(ValidationError):
            Transaction.model_validate({**VALID_TXN_ROW, "amount": -1.0})

    def test_transaction_rejects_bad_payment_mode(self) -> None:
        with pytest.raises(ValidationError):
            Transaction.model_validate({**VALID_TXN_ROW, "payment_mode": "VISA-KEEP"})

    def test_transaction_rejects_empty_txn_id(self) -> None:
        with pytest.raises(ValidationError):
            Transaction.model_validate({**VALID_TXN_ROW, "txn_id": ""})

    def test_account_metadata_constraints(self) -> None:
        ok = AccountMetadata(
            account_no="2900000001",
            ifsc="SBIN0001234",
            account_opening_date=date(2023, 1, 15),
            branch_code="BR001",
            mobile_no="9178451234",
            imei="351104510123456",
            kyc_pincode="800013",
            account_type="PMJDY",
        )
        assert ok.account_type == "PMJDY"
        with pytest.raises(ValidationError):
            AccountMetadata(
                account_no="1", ifsc="BAD", branch_code="B", account_type="JOINT"  # type: ignore[arg-type]
            )

    def test_complaint_roundtrip_json(self) -> None:
        c = Complaint(
            ack_no="ACK20240001",
            victim_account="2900000007",
            suspect_account="5500000001",
            fraud_category="AEPS_CASHOUT",
            reported_timestamp=datetime(2024, 6, 2, 9, 0, 0),
            disputed_amount=32000.0,
            disputed_utr="UTR2024060100001",
            state="Bihar",
        )
        dumped = json.loads(c.model_dump_json())
        assert dumped["disputed_amount"] == 32000.0
        assert dumped["disputed_utr"] == "UTR2024060100001"

    def test_complaint_disputed_utr_optional(self) -> None:
        with_utr = Complaint(
            ack_no="ACK20240002",
            victim_account="2900000007",
            fraud_category="UPI_FRAUD",
            reported_timestamp=datetime(2024, 6, 2, 9, 0, 0),
            disputed_amount=1500.0,
            state="Bihar",
        )
        assert with_utr.disputed_utr is None   # nullable, links only when known

    def test_report_defaults_hold_dataframe(self) -> None:
        report = IngestionReport()
        assert report.rows_total == 0 and report.rows_accepted == 0
        assert report.rejected_rows.is_empty()

    def test_ifsc_pattern_is_strict(self) -> None:
        assert IFSC_PATTERN.fullmatch("SBIN0001234") is not None
        assert IFSC_PATTERN.fullmatch("sbin0001234") is None
        assert IFSC_PATTERN.fullmatch("SBIN00123") is None

    def test_transaction_schema_is_string_first(self) -> None:
        assert list(TRANSACTION_COLUMNS) == [
            "txn_id", "sender_account", "sender_ifsc", "receiver_account",
            "receiver_ifsc", "amount", "timestamp", "payment_mode",
            "device_id", "ip_address",
        ]
        assert TRANSACTION_SCHEMA["amount"] == pl.Float64
        assert TRANSACTION_SCHEMA["timestamp"] == pl.Datetime("us")
        for c in TRANSACTION_SCHEMA:
            if c not in ("amount", "timestamp"):
                assert TRANSACTION_SCHEMA[c] == pl.String, c

    def test_account_and_complaint_schemas(self) -> None:
        assert [ACCOUNT_METADATA_SCHEMA[c] for c in ("account_opening_date", "account_type")] == [
            pl.Date, pl.String,
        ]
        assert list(COMPLAINT_SCHEMA) == [
            "ack_no", "victim_account", "suspect_account", "fraud_category",
            "reported_timestamp", "disputed_amount", "disputed_utr", "state",
        ]
        assert list(COMPLAINT_COLUMNS) == list(COMPLAINT_SCHEMA)
        assert list(ACCOUNT_COLUMNS) == list(ACCOUNT_METADATA_SCHEMA)
        assert [COMPLAINT_SCHEMA[c] for c in ("reported_timestamp", "disputed_amount")] == [
            pl.Datetime("us"), pl.Float64,
        ]

    def test_mandatory_columns_subset_of_schema(self) -> None:
        assert MANDATORY_COLUMNS == (
            "txn_id", "sender_account", "receiver_account",
            "amount", "timestamp", "payment_mode",
        )
        assert set(MANDATORY_COLUMNS) <= set(TRANSACTION_SCHEMA)
        assert MANDATORY_BY_TABLE == {
            "transactions": MANDATORY_COLUMNS,
            "complaints": (
                "ack_no", "victim_account", "fraud_category",
                "reported_timestamp", "disputed_amount", "state",
            ),
            "accounts": ("account_no", "ifsc", "branch_code", "account_type"),
        }
        assert set(MANDATORY_BY_TABLE["complaints"]) <= set(COMPLAINT_SCHEMA)
        assert set(MANDATORY_BY_TABLE["accounts"]) <= set(ACCOUNT_METADATA_SCHEMA)

    def test_column_mapping_confidence_bounds(self) -> None:
        m = ColumnMapping(raw_name="Payee", canonical_name="receiver_account",
                          matched_synonym="payee", confidence=1.0)
        assert m.confidence == 1.0
        with pytest.raises(ValidationError):
            ColumnMapping(raw_name="x", canonical_name=None, confidence=1.5)
