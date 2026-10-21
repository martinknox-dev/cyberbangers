"""Phase 1 data contracts: Pydantic models and canonical Polars schemas.

Single source of truth for every entity flowing through PraharGraph.
Identifier-like values (accounts, IFSC, IMEI, pins, mobiles) are strings to
preserve leading zeros; only amount is Float64 and timestamps are
Datetime("us") (naive, ledger-style).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

PaymentMode = Literal["UPI", "IMPS", "NEFT", "RTGS", "AEPS", "ATM", "CASH"]
AccountType = Literal["SAVINGS", "PMJDY", "CURRENT"]
TableKind = Literal["transactions", "complaints", "accounts"]

IFSC_PATTERN = re.compile(r"^[A-Z]{4}0[0-9A-Z]{6}$")

MANDATORY_COLUMNS: tuple[str, ...] = (
    "txn_id",
    "sender_account",
    "receiver_account",
    "amount",
    "timestamp",
    "payment_mode",
)

TRANSACTION_SCHEMA: dict[str, pl.DataType] = {
    "txn_id": pl.String(),
    "sender_account": pl.String(),
    "sender_ifsc": pl.String(),
    "receiver_account": pl.String(),
    "receiver_ifsc": pl.String(),
    "amount": pl.Float64(),
    "timestamp": pl.Datetime("us"),
    "payment_mode": pl.String(),
    "device_id": pl.String(),
    "ip_address": pl.String(),
}

TRANSACTION_COLUMNS: tuple[str, ...] = tuple(TRANSACTION_SCHEMA)

ACCOUNT_METADATA_SCHEMA: dict[str, pl.DataType] = {
    "account_no": pl.String(),
    "ifsc": pl.String(),
    "account_opening_date": pl.Date(),
    "branch_code": pl.String(),
    "mobile_no": pl.String(),
    "imei": pl.String(),
    "kyc_pincode": pl.String(),
    "account_type": pl.String(),
}

COMPLAINT_SCHEMA: dict[str, pl.DataType] = {
    "ack_no": pl.String(),
    "victim_account": pl.String(),
    "suspect_account": pl.String(),
    "fraud_category": pl.String(),
    "reported_timestamp": pl.Datetime("us"),
    "disputed_amount": pl.Float64(),
    "disputed_utr": pl.String(),
    "state": pl.String(),
}

COMPLAINT_COLUMNS: tuple[str, ...] = tuple(COMPLAINT_SCHEMA)
ACCOUNT_COLUMNS: tuple[str, ...] = tuple(ACCOUNT_METADATA_SCHEMA)

MANDATORY_BY_TABLE: dict[TableKind, tuple[str, ...]] = {
    "transactions": MANDATORY_COLUMNS,
    "complaints": (
        "ack_no",
        "victim_account",
        "fraud_category",
        "reported_timestamp",
        "disputed_amount",
        "state",
    ),
    "accounts": ("account_no", "ifsc", "branch_code", "account_type"),
}


class Transaction(BaseModel):
    """One financial transaction (ledger-level ingest record)."""

    txn_id: str = Field(min_length=1)  # unique UTR/reference
    sender_account: str = Field(min_length=1)
    sender_ifsc: str | None = Field(default=None, pattern=IFSC_PATTERN.pattern)
    receiver_account: str = Field(min_length=1)
    receiver_ifsc: str | None = Field(default=None, pattern=IFSC_PATTERN.pattern)
    amount: float = Field(ge=0)
    timestamp: datetime
    payment_mode: PaymentMode
    device_id: str | None = None
    ip_address: str | None = None


class AccountMetadata(BaseModel):
    """Customer/master-level account attributes (KYC-style)."""

    account_no: str = Field(min_length=1)
    ifsc: str = Field(pattern=IFSC_PATTERN.pattern)
    account_opening_date: date | None = None
    branch_code: str = Field(min_length=1)
    mobile_no: str | None = Field(default=None, pattern=r"^\d{10,12}$")
    imei: str | None = Field(default=None, pattern=r"^\d{15}$")
    kyc_pincode: str | None = Field(default=None, pattern=r"^\d{6}$")
    account_type: AccountType


class Complaint(BaseModel):
    """Cyber-crime acknowledgment (NCRP-lite) linking victim to suspect."""

    ack_no: str = Field(min_length=1)
    victim_account: str = Field(min_length=1)
    suspect_account: str | None = None
    fraud_category: str = Field(min_length=1)
    reported_timestamp: datetime
    disputed_amount: float = Field(ge=0)
    disputed_utr: str | None = None   # transaction reference binding the dispute to the ledger
    state: str = Field(min_length=1)


class ColumnMapping(BaseModel):
    """Fuzzy header-to-canonical mapping result for one source column."""

    raw_name: str
    canonical_name: str | None = None
    matched_synonym: str | None = None
    confidence: float = Field(ge=0, le=1)


class IngestionReport(BaseModel):
    """Machine-readable audit of one `DatasetIngester` run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str | None = None
    rows_total: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0
    failure_counts: dict[str, int] = Field(default_factory=dict)
    column_mapping: dict[str, ColumnMapping] = Field(default_factory=dict)
    rejected_rows: pl.DataFrame = pl.DataFrame()
