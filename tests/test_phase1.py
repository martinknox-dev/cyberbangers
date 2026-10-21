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


from ingestion.mapper import map_headers, normalize_header

STYLE_A_HEADERS: list[str] = [   # core-banking CSV export (bank style, underscored codes)
    "UTR", "Debit_Acc", "Fr_IFSC", "PAYEE", "To_IFSC", "Amount_INR",
    "Txn_Time", "Mode", "Device", "IP",
]
STYLE_B_HEADERS: list[str] = [   # branch XLSX sheet (spaced title-case)
    "Transaction Ref", "Originator Acct", "Remitter IFSC", "To Account",
    "Beneficiary Branch IFSC", "Txn Amount", "Txn DateTime", "Pay Method",
    "Device No", "Customer IP",
]
STYLE_C_HEADERS: list[str] = [   # complaint-portal dump (mixed snake + CamelCase)
    "txn ref id", "sender account", "Sender IFSC Code", "beneficiary_account",
    "Receiver IFSC Code", "amount", "datetime_of_txn", "channel",
    "imei_device", "IP",
]

STYLE_A_EXPECTED: dict[str, str] = {
    "UTR": "txn_id", "Debit_Acc": "sender_account", "Fr_IFSC": "sender_ifsc",
    "PAYEE": "receiver_account", "To_IFSC": "receiver_ifsc",
    "Amount_INR": "amount", "Txn_Time": "timestamp", "Mode": "payment_mode",
    "Device": "device_id", "IP": "ip_address",
}
STYLE_B_EXPECTED: dict[str, str] = {
    "Transaction Ref": "txn_id", "Originator Acct": "sender_account",
    "Remitter IFSC": "sender_ifsc", "To Account": "receiver_account",
    "Beneficiary Branch IFSC": "receiver_ifsc", "Txn Amount": "amount",
    "Txn DateTime": "timestamp", "Pay Method": "payment_mode",
    "Device No": "device_id", "Customer IP": "ip_address",
}
STYLE_C_EXPECTED: dict[str, str] = {
    "txn ref id": "txn_id", "sender account": "sender_account",
    "Sender IFSC Code": "sender_ifsc", "beneficiary_account": "receiver_account",
    "Receiver IFSC Code": "receiver_ifsc", "amount": "amount",
    "datetime_of_txn": "timestamp", "channel": "payment_mode",
    "imei_device": "device_id", "IP": "ip_address",
}

COMPLAINT_HEADERS: list[str] = [   # NCRP-portal style dump (report-officer wording)
    "Complaint Ack No", "Victim Acc No", "Suspect Acc No", "Fraud Category",
    "Reported On", "Cheated Amount", "Disputed UTR Ref", "State",
]
ACCOUNT_HEADERS: list[str] = [    # core-banking customer master sheet
    "Account Number", "IFSC Code", "Date of Opening", "Home Branch Code",
    "Mobile Number", "Device IMEI", "KYC Pincode", "Acct Type",
]
COMPLAINT_EXPECTED: dict[str, str] = {
    "Complaint Ack No": "ack_no", "Victim Acc No": "victim_account",
    "Suspect Acc No": "suspect_account", "Fraud Category": "fraud_category",
    "Reported On": "reported_timestamp", "Cheated Amount": "disputed_amount",
    "Disputed UTR Ref": "disputed_utr", "State": "state",
}
ACCOUNT_EXPECTED: dict[str, str] = {
    "Account Number": "account_no", "IFSC Code": "ifsc",
    "Date of Opening": "account_opening_date", "Home Branch Code": "branch_code",
    "Mobile Number": "mobile_no", "Device IMEI": "imei",
    "KYC Pincode": "kyc_pincode", "Acct Type": "account_type",
}
MULTI_TABLE_EXACT_CONFS: dict[tuple[str, str], float] = {
    # (table, raw header): probe-computed confidence
    ("complaints", "Complaint Ack No"): 0.8966,
    ("complaints", "Disputed UTR Ref"): 0.8571,
    ("accounts", "Date of Opening"): 0.8889,
    ("accounts", "Home Branch Code"): 0.8148,
}


class TestHeaderMapping:
    def test_normalize_header(self) -> None:
        assert normalize_header("Beneficiary-Acc  No#2") == "beneficiary acc no 2"
        assert normalize_header("  UTR ") == "utr"

    def test_style_a_full_coverage(self) -> None:
        mapping = map_headers(STYLE_A_HEADERS)
        for raw, canonical in STYLE_A_EXPECTED.items():
            assert mapping[raw].canonical_name == canonical, raw
            assert mapping[raw].confidence >= 0.80, (raw, mapping[raw].confidence)

    def test_style_b_full_coverage(self) -> None:
        mapping = map_headers(STYLE_B_HEADERS)
        for raw, canonical in STYLE_B_EXPECTED.items():
            assert mapping[raw].canonical_name == canonical, raw
            assert mapping[raw].confidence >= 0.80, raw

    def test_style_c_full_coverage_with_exact_confidences(self) -> None:
        mapping = map_headers(STYLE_C_HEADERS)
        exact_confs: dict[str, float] = {
            "txn ref id": 0.824, "sender account": 1.0, "Sender IFSC Code": 0.815,
            "beneficiary_account": 1.0, "Receiver IFSC Code": 0.839,
            "amount": 1.0, "datetime_of_txn": 0.889, "channel": 1.0,
            "imei_device": 0.80, "IP": 1.0,
        }
        for raw, canonical in STYLE_C_EXPECTED.items():
            assert mapping[raw].canonical_name == canonical, raw
            assert abs(mapping[raw].confidence - exact_confs[raw]) <= 1e-3, raw

    def test_complaint_and_account_style_coverage_with_exact_confs(self) -> None:
        m_c = map_headers(COMPLAINT_HEADERS, table="complaints")
        for raw, canonical in COMPLAINT_EXPECTED.items():
            assert m_c[raw].canonical_name == canonical, raw
            assert m_c[raw].confidence >= 0.80, raw
        m_a = map_headers(ACCOUNT_HEADERS, table="accounts")
        for raw, canonical in ACCOUNT_EXPECTED.items():
            assert m_a[raw].canonical_name == canonical, raw
            assert m_a[raw].confidence >= 0.80, raw
        # exact-confidence checks (the marginal headers per table):
        for raw in ("Complaint Ack No", "Disputed UTR Ref"):
            assert abs(m_c[raw].confidence - MULTI_TABLE_EXACT_CONFS[("complaints", raw)]) <= 1e-3, raw
        for raw in ("Date of Opening", "Home Branch Code"):
            assert abs(m_a[raw].confidence - MULTI_TABLE_EXACT_CONFS[("accounts", raw)]) <= 1e-3, raw

    def test_default_call_maps_transactions_only(self) -> None:
        # Without `table=`, complaint/account-specific headers must NOT claim
        # transaction canonicals, and transaction headers must not claim
        # complaint canonicals (tables are siloed by the synonym registry).
        tx_style = map_headers(["Cheated Amount", "Victim Acc No"])
        assert tx_style["Cheated Amount"].canonical_name is None
        assert tx_style["Victim Acc No"].canonical_name is None
        cx_style = map_headers(["Amount_INR", "Txn_Time"], table="complaints")
        assert cx_style["Amount_INR"].canonical_name is None
        assert cx_style["Txn_Time"].canonical_name is None

    def test_noise_headers_unmapped(self) -> None:
        mapping = map_headers(
            ["Narration", "Remarks", "Sl No", "Balance", "SenderIFSC", "ReceiverBranchIFSC"]
        )
        for raw, entry in mapping.items():
            assert entry.canonical_name is None, raw
            assert entry.confidence < 0.80, raw

    def test_below_threshold_is_dropped(self) -> None:
        mapping = map_headers(["Narration"], confidence_threshold=0.55)
        assert mapping["Narration"].canonical_name is None  # best score 52.2 < 55

    def test_canonical_collision_exact_tie_keeps_first(self) -> None:
        mapping = map_headers(["To Account", "PAYEE"])  # both hit receiver_account @ 100
        assert mapping["To Account"].canonical_name == "receiver_account"
        assert mapping["PAYEE"].canonical_name is None     # tie -> first-seen wins
        assert mapping["PAYEE"].confidence == 1.0          # loser keeps its own score

    def test_canonical_collision_higher_confidence_wins(self) -> None:
        # "Credit_Acc_No" scores ~0.741 against credit_account; "To Account" 1.0.
        mapping = map_headers(["Credit_Acc_No", "To Account"])
        assert mapping["To Account"].canonical_name == "receiver_account"
        assert mapping["Credit_Acc_No"].canonical_name is None
        assert mapping["Credit_Acc_No"].confidence < 0.80
