"""Phase 1 acceptance tests: contracts, fuzzy ingestion, deterministic simulator."""
import io
import json
from datetime import date, datetime

import openpyxl
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


from ingestion.mapper import DatasetIngester, map_headers, normalize_header

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


def csv_bytes(rows: list[str]) -> bytes:
    header = ",".join(STYLE_A_HEADERS) + "\n"
    body = "\n".join(rows)
    return (header + body + ("\n" if body else "")).encode("utf-8")


def style_a_xlsx(header: list[str], rows: list[list[object]]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    for c, name in enumerate(header, start=1):
        ws.cell(column=c, row=1, value=name)
    for r, row in enumerate(rows, start=2):
        for c, value in enumerate(row, start=1):
            ws.cell(column=c, row=r, value=value)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestIngestHappyPath:
    def test_csv_path_all_three_header_styles_accepted(self, tmp_path) -> None:
        style_rows: dict[str, list[str]] = {
            "a": [
                "UTR_A1,2900000001,SBIN0001234,2900000002,SBIN0009876,15000.50,2024-06-01 10:30:00,UPI,D000123,10.20.30.40",
            ],
            "b": [
                "UTR_B1,2900000001,SBIN0001234,2900000002,SBIN0009876,2499.00,01/06/2024 11:45:30,IMPS,D000124,10.20.30.41",
            ],
            "c": [
                "UTR_C1,2900000001,SBIN0001234,2900000002,SBIN0009876,777.75,2024-06-01T12:00:00,NEFT,D000125,10.20.30.42",
            ],
        }
        headers_by: dict[str, list[str]] = {
            "a": STYLE_A_HEADERS, "b": STYLE_B_HEADERS, "c": STYLE_C_HEADERS,
        }
        for key, rows in style_rows.items():
            path = tmp_path / f"style_{key}.csv"
            path.write_bytes(
                (",".join(headers_by[key]) + "\n" + rows[0] + "\n").encode("utf-8")
            )
            df, report = DatasetIngester().ingest(str(path))
            assert df.height == 1
            assert report.rows_total == 1
            assert report.rows_accepted == 1 and report.rows_rejected == 0
            assert list(df.columns) == list(TRANSACTION_COLUMNS)
            assert df.dtypes == [TRANSACTION_SCHEMA[c] for c in TRANSACTION_COLUMNS]
            assert df["txn_id"].item() == f"UTR_{key.upper()}1"

    def test_dd_mm_precedes_mm_dd(self, tmp_path) -> None:
        rows = [
            "UTR_D1,2900000001,SBIN0001234,2900000002,SBIN0009876,500.00,05/06/2024 09:15:00,UPI,,",
            "UTR_D2,2900000001,SBIN0001234,2900000002,SBIN0009876,500.00,12/31/2024 09:15:00,UPI,,",
            "UTR_D3,2900000001,SBIN0001234,2900000002,SBIN0009876,500.00,05/06/2024 09:15:52 PM,UPI,,",
        ]
        path = tmp_path / "dd_mm.csv"
        path.write_bytes((",".join(STYLE_A_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest(str(path))
        assert report.rows_rejected == 0
        stamps = {r["txn_id"]: r["timestamp"] for r in df.to_dicts()}
        assert stamps["UTR_D1"] == datetime(2024, 6, 5, 9, 15, 0)
        assert stamps["UTR_D2"] == datetime(2024, 12, 31, 9, 15, 0)
        assert stamps["UTR_D3"] == datetime(2024, 6, 5, 21, 15, 52)

    def test_unix_timestamp_seconds_and_ms(self, tmp_path) -> None:
        rows = [
            "UTR_E1,1,SBIN0000001,2,SBIN0000002,10,1717200000,UPI,,",
            "UTR_E2,1,SBIN0000001,2,SBIN0000002,10,1717200002000,IMPS,,",
        ]
        path = tmp_path / "epoch.csv"
        path.write_bytes((",".join(STYLE_A_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest(str(path))
        assert report.rows_rejected == 0
        assert sorted(df["timestamp"].to_list()) == [
            datetime(2024, 6, 1, 0, 0, 0), datetime(2024, 6, 1, 0, 0, 2),
        ]

    def test_xlsx_bytes_with_native_cells(self) -> None:
        payload = style_a_xlsx(
            STYLE_A_HEADERS,
            [[
                "UTR_X1", "2900000001", "sbin0001234", "2900000002",
                "SBIN0009876", 100.5, datetime(2024, 6, 1, 3, 4, 5), "BHIM UPI",
                "D000001", "10.0.0.1",
            ]],
        )
        df, report = DatasetIngester().ingest(payload)   # bytes -> xlsx via PK magic sniff
        assert report.rows_accepted == 1
        assert df["amount"].item() == pytest.approx(100.5)
        assert df["timestamp"].item() == datetime(2024, 6, 1, 3, 4, 5)
        assert df["payment_mode"].item() == "UPI"
        assert df["sender_ifsc"].item() == "SBIN0001234"   # lowercased input upcased + validated

    def test_bytes_csv_via_sniff(self) -> None:
        payload = csv_bytes([
            "UTR_F1,1,SBIN0000001,2,SBIN0000002,25.00,2024-06-01 10:00:00,CASH,dev1,1.2.3.4",
        ])
        df, report = DatasetIngester().ingest(payload)
        assert report.rows_accepted == 1
        assert df["payment_mode"].item() == "CASH"

    def test_explicit_reader_override_on_suffixless_path(self, tmp_path) -> None:
        path = tmp_path / "only"
        path.write_bytes(csv_bytes([
            "UTR_G1,1,SBIN0000001,2,SBIN0000002,25.00,2024-06-01 10:00:00,CASH,dev2,1.2.3.5",
        ]))
        as_csv, report = DatasetIngester().ingest_csv(str(path))
        assert as_csv.height == 1 and report.rows_accepted == 1
        with pytest.raises(openpyxl.utils.exceptions.InvalidFileException):
            DatasetIngester().ingest_xlsx(str(path))

    def test_mapping_report_records_confidence(self, tmp_path) -> None:
        path = tmp_path / "m.csv"
        path.write_bytes(csv_bytes([
            "UTR_H1,1,SBIN0000001,2,SBIN0000002,25.00,2024-06-01 10:00:00,CASH,dev3,1.2.3.6",
        ]))
        _df, report = DatasetIngester().ingest(str(path))
        m = report.column_mapping["Debit_Acc"]
        assert m.canonical_name == "sender_account"
        assert m.matched_synonym == "debit_account"
        assert m.confidence >= 0.80
        assert report.column_mapping["UTR"].canonical_name == "txn_id"


def complaint_csv_bytes(rows: list[str]) -> bytes:
    header = ",".join(COMPLAINT_HEADERS) + "\n"
    return (header + "\n".join(rows) + "\n").encode("utf-8")


def account_xlsx_bytes(rows: list[list[object]]) -> bytes:
    return style_a_xlsx(ACCOUNT_HEADERS, rows)


class TestIngestMultiTableHappy:
    def test_complaints_csv_happy_path(self, tmp_path) -> None:
        path = tmp_path / "ncrp.csv"
        path.write_bytes(complaint_csv_bytes([
            'ACK20240001,2900000007,2990100000,AEPS_CASHOUT,02/06/2024 09:15:00,"32,000.50",UTR2024060100001,Bihar',
        ]))
        df, report = DatasetIngester().ingest_complaints(str(path))
        assert df.height == 1
        assert list(df.columns) == list(COMPLAINT_COLUMNS)
        assert df["ack_no"].item() == "ACK20240001"
        assert df["disputed_utr"].item() == "UTR2024060100001"
        # DD/MM precedence + comma-junk amount survive the shared pipeline:
        assert df["reported_timestamp"].item() == datetime(2024, 6, 2, 9, 15, 0)
        assert df["disputed_amount"].item() == pytest.approx(32000.5)
        assert df["suspect_account"].item() == "2990100000"
        assert report.rows_accepted == 1 and report.rows_rejected == 0

    def test_complaints_headers_do_not_claim_transaction_columns(self, tmp_path) -> None:
        path = tmp_path / "ncrp.csv"
        path.write_bytes(complaint_csv_bytes([
            "ACK20240002,2900000007,2990100001,UPI_FRAUD,02/06/2024 09:15:00,1500,UTR2024060100002,Delhi",
        ]))
        df, report = DatasetIngester().ingest_complaints(str(path))
        assert df.height == 1 and report.rows_accepted == 1

    def test_accounts_xlsx_native_date_cell(self) -> None:
        payload = account_xlsx_bytes([[
            "2900000007", "sbin0001234", date(2023, 1, 15), "BR001",
            9178451234, "351104510123456", "800013", "savings",
        ]])
        df, report = DatasetIngester().ingest_accounts(payload)   # PK magic sniff
        assert df.height == 1
        assert list(df.columns) == list(ACCOUNT_COLUMNS)
        # openpyxl-native datetime cell -> String "2023-01-15 00:00:00.000000" -> Date:
        assert df["account_opening_date"].item() == date(2023, 1, 15)
        assert df["account_type"].item() == "SAVINGS"     # lowercase alias upcased
        assert df["ifsc"].item() == "SBIN0001234"         # lowercase input upcased
        assert df["mobile_no"].item() == "9178451234"     # numeric cell cast to String
        assert df["imei"].item() == "351104510123456"
        assert report.rows_accepted == 1

    def test_payment_mode_alias_expansion(self, tmp_path) -> None:
        rows = [
            "UTR_N1,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:00:00,RTGS,,",
            "UTR_N2,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:05:00,ATM WITHDRAWAL,,",
            "UTR_N3,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:10:00,MICRO ATM,,",
            "UTR_N4,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:15:00,UPI/P2M,,",
            "UTR_N5,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:20:00,UPI/P2P,,",
        ]
        path = tmp_path / "modes.csv"
        path.write_bytes((",".join(STYLE_A_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest(str(path))
        assert report.rows_rejected == 0
        assert df["payment_mode"].to_list() == ["RTGS", "ATM", "AEPS", "UPI", "UPI"]


class TestIngestRejections:
    ROW = (
        "UTR_R1,2900000001,SBIN0001234,2900000002,SBIN0009876,100.00,"
        "2024-06-01 10:30:00,UPI,D000001,10.0.0.9"
    )

    def _ingest_rows(self, tmp_path, rows: list[str]):
        path = tmp_path / "reject.csv"
        path.write_bytes(csv_bytes(rows))
        return DatasetIngester().ingest(str(path))

    def test_negative_amount_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            self.ROW,
            "UTR_R2,2900000001,SBIN0001234,2900000002,SBIN0009876,-5.00,2024-06-01 10:31:00,UPI,,",
        ])
        assert report.rows_total == 2
        assert report.rows_rejected == 1 and df.height == 1
        assert report.failure_counts == {"negative_amount": 1}
        assert report.rejected_rows["UTR"].item() == "UTR_R2"
        assert report.rejected_rows["rejection_reason"].item() == "negative_amount"

    def test_unparseable_timestamp_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R3,1,SBIN0000001,2,SBIN0000002,10.00,NOT-A-DATE,UPI,,",
        ])
        assert report.failure_counts == {"unparseable_timestamp": 1}
        assert df.height == 0

    def test_unparseable_amount_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R4,1,SBIN0000001,2,SBIN0000002,ten,2024-06-01 10:30:00,UPI,,",
        ])
        assert report.failure_counts == {"unparseable_amount": 1}
        assert df.height == 0

    def test_missing_mandatory_fields(self, tmp_path) -> None:
        rows = [
            ",1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:30:00,UPI,,",              # missing:txn_id
            "UTR_M2,,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:30:00,UPI,,",         # missing:sender_account
            "UTR_M3,1,SBIN0000001,,SBIN0000002,10,2024-06-01 10:30:00,UPI,,",         # missing:receiver_account
            "UTR_M4,1,SBIN0000001,2,SBIN0000002,,2024-06-01 10:30:00,UPI,,",          # missing:amount
            "UTR_M5,1,SBIN0000001,2,SBIN0000002,10,,UPI,,",                            # missing:timestamp
            "UTR_M6,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:30:00,,,1.1.1.1",     # missing:payment_mode
        ]
        df, report = self._ingest_rows(tmp_path, rows)
        assert df.height == 0
        assert report.rows_rejected == 6
        assert report.failure_counts == {
            "missing:txn_id": 1, "missing:sender_account": 1,
            "missing:receiver_account": 1, "missing:amount": 1,
            "missing:timestamp": 1, "missing:payment_mode": 1,
        }

    def test_duplicate_utr_keep_first(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            self.ROW,
            self.ROW.replace("100.00", "200.00"),
        ])
        assert report.failure_counts == {"duplicate_utr": 1}
        assert df.height == 1 and df["amount"].item() == 100.0
        assert report.rejected_rows["UTR"].item() == "UTR_R1"

    def test_invalid_ifsc_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R5,1,SBIN1234,2,SBIN0000002,10,2024-06-01 10:30:00,UPI,,",
        ])
        assert report.failure_counts == {"invalid_ifsc": 1}
        assert df.height == 0

    def test_unsupported_payment_mode_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R6,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:30:00,CHEQUE,,",
        ])
        assert report.failure_counts == {"unsupported_payment_mode": 1}

    def test_priority_missing_beats_all(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R7,,SBIN0000001,2,SBIN0000002,-10,NOT-A-DATE,ZZZ,,",
        ])
        assert report.failure_counts == {"missing:sender_account": 1}

    def test_mixed_reasons_aggregate(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R8,1,SBIN0000001,2,SBIN0000002,-1,2024-06-01 10:30:00,UPI,,",          # negative_amount
            "UTR_R9,1,SBIN0000001,2,SBIN0000002,10,NOPE,UPI,,",                          # unparseable_timestamp
        ])
        assert report.failure_counts == {"negative_amount": 1, "unparseable_timestamp": 1}
        assert report.rows_accepted == 0 and df.height == 0
        assert set(report.rejected_rows.columns) == set(STYLE_A_HEADERS) | {"rejection_reason"}

    def test_zero_amount_accepted(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_Z1,1,SBIN0000001,2,SBIN0000002,0,2024-06-01 10:30:00,UPI,,",
        ])
        assert report.rows_rejected == 0 and df.height == 1
        assert df["amount"].item() == 0.0

    def test_header_only_csv_returns_empty_report(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [])
        assert df.height == 0 and df.width == len(TRANSACTION_COLUMNS)
        assert report.rows_total == 0 and report.rows_accepted == 0
        assert report.rows_rejected == 0 and report.failure_counts == {}
        assert report.column_mapping["UTR"].canonical_name == "txn_id"
        assert report.rejected_rows.is_empty()

    def test_ragged_csv_rows_do_not_crash(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            self.ROW + ",EXTRA,CELLS",          # reader truncates extras; row parses
            self.ROW.replace("UTR_R1", "UTR_R10"),
        ])
        assert report.rows_total == 2
        assert report.rows_rejected == 0
        assert df.height == 2

    def test_quoted_timestamp_cell_still_parses(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            'UTR_CX1,1,SBIN0000001,2,SBIN0000002,10,"2024-06-01 10:30:00",UPI,,',
        ])
        assert report.rows_rejected == 0
        assert df["timestamp"].item() == datetime(2024, 6, 1, 10, 30, 0)


class TestIngestMultiTableRejections:
    def test_complaint_missing_ack_duplicate_ack(self, tmp_path) -> None:
        rows = [
            # empty ack -> missing:ack_no (also: suspect + disputed UTR optional, fine)
            ",2900000007,2990100000,AEPS_CASHOUT,02/06/2024 09:15:00,32000,UTR2024060100001,Bihar",
            "ACK20240005,2900000008,2990100001,UPI_FRAUD,02/06/2024 10:15:00,1500,UTR2024060100002,Delhi",
            "ACK20240005,2900000008,2990100001,UPI_FRAUD,02/06/2024 10:20:00,1500,UTR2024060100002,Delhi",  # dup ack
        ]
        path = tmp_path / "ncrp_reject.csv"
        path.write_bytes(complaint_csv_bytes(rows))
        df, report = DatasetIngester().ingest_complaints(str(path))
        assert report.rows_total == 3
        assert df.height == 1
        assert df["ack_no"].item() == "ACK20240005"
        assert report.failure_counts == {"duplicate_ack_no": 1, "missing:ack_no": 1}
        assert report.rejected_rows["rejection_reason"].to_list()[0] == "missing:ack_no"

    def test_complaint_missing_mandatory_field(self, tmp_path) -> None:
        rows = [
            # empty state -> missing:state; amount/timestamp fine
            "ACK20240006,2900000007,2990100002,AEPS_CASHOUT,02/06/2024 09:15:00,9000,UTR2024060100003,",
        ]
        path = tmp_path / "ncrp_state.csv"
        path.write_bytes(complaint_csv_bytes(rows))
        df, report = DatasetIngester().ingest_complaints(str(path))
        assert report.failure_counts == {"missing:state": 1}
        assert df.height == 0

    def test_accounts_rejections(self, tmp_path) -> None:
        rows = [
            # ifsc malformed -> invalid_ifsc
            "2994000000,SBIN00,15/03/2024,BR042,9812000000,351104510900001,800013,CURRENT",
            # unsupported account type -> unsupported_account_type
            "2994000001,SBIN0001234,15/03/2024,BR042,9812000001,351104510900002,800013,JOINT",
            # pincode not digits -> invalid_kyc_pincode
            "2994000002,SBIN0001234,15/03/2024,BR042,9812000002,351104510900003,12AB56,CURRENT",
            # mobile too short -> invalid_mobile_no (5 digits)
            "2994000003,SBIN0001234,15/03/2024,BR042,98123,351104510900004,800013,CURRENT",
            # clean row -> accepted, plus duplicate account_no below
            "2994000004,SBIN0001234,2024-03-15,BR042,9812000004,351104510900005,800013,SAVINGS",
            "2994000004,SBIN0001234,2024-03-20,BR042,9812000005,351104510900006,800013,SAVINGS",  # dup account_no
        ]
        path = tmp_path / "accounts_reject.csv"
        path.write_bytes((",".join(ACCOUNT_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest_accounts(str(path))
        assert report.rows_total == 6
        assert report.rows_accepted == 1 and df.height == 1
        assert df["account_no"].item() == "2994000004"
        assert df["account_opening_date"].item() == date(2024, 3, 15)   # keep-first
        assert report.failure_counts == {
            "duplicate_account_no": 1, "invalid_ifsc": 1, "invalid_kyc_pincode": 1,
            "invalid_mobile_no": 1, "unsupported_account_type": 1,
        }

    def test_accounts_missing_mandatory(self, tmp_path) -> None:
        rows = [
            # empty branch code -> missing:branch_code
            "2994000005,SBIN0001234,16/03/2024,,9812000006,351104510900007,800013,CURRENT",
        ]
        path = tmp_path / "accounts_missing.csv"
        path.write_bytes((",".join(ACCOUNT_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest_accounts(str(path))
        assert report.failure_counts == {"missing:branch_code": 1}
        assert df.height == 0
