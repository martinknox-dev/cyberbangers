"""Fuzzy schema ingestion gateway.

Header mapping follows the spec's RapidFuzz token-sort-ratio approach;
readers + normalization (Task 4) and validation + reporting (Task 5)
complete the pipeline in this same module. Everything runs as Polars
expression chains: no row-level Python loops over dataframes, no Pandas.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import polars as pl
from rapidfuzz import fuzz

from core.contracts import (
    ACCOUNT_COLUMNS,
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_COLUMNS,
    COMPLAINT_SCHEMA,
    IFSC_PATTERN,
    TRANSACTION_COLUMNS,
    TRANSACTION_SCHEMA,
    ColumnMapping,
    IngestionReport,
    TableKind,
)

# --- header mapping -------------------------------------------------------

DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.80

CANONICAL_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "txn_id": (
        "txn_id", "utr", "utr_no", "transaction_id", "txn_ref",
        "transaction_reference", "ref_no", "rrn", "utrid",
    ),
    "sender_account": (
        "sender_account", "sender_acc_no", "remitter", "payer", "from_account",
        "debit_account", "debitor", "originator_account", "sender",
        "sender_account_no",
    ),
    "sender_ifsc": (
        "sender_ifsc", "from_ifsc", "fr_ifsc", "sender_branch_ifsc",
        "debit_ifsc", "remitter_ifsc",
    ),
    "receiver_account": (
        "receiver_account", "receiver_acc_no", "beneficiary_acc",
        "beneficiary_account", "payee", "to_account", "credit_account",
        "beneficiary", "payee_account",
    ),
    "receiver_ifsc": (
        "receiver_ifsc", "to_ifsc", "beneficiary_ifsc", "credit_ifsc",
        "beneficiary_branch_ifsc",
    ),
    "amount": (
        "amount", "txn_amount", "transaction_amount", "amount_inr", "amt",
        "value",
    ),
    "timestamp": (
        "timestamp", "txn_timestamp", "txn_time", "txn_datetime",
        "transaction_datetime", "date_time", "txn_date",
    ),
    "payment_mode": (
        "payment_mode", "mode_of_payment", "payment_channel", "txn_mode",
        "payment_method", "channel", "mode",
    ),
    "device_id": ("device_id", "device", "imei", "device_no", "dev_id"),
    "ip_address": ("ip_address", "ip", "ip_addr", "customer_ip", "client_ip"),
}

COMPLAINT_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "ack_no": (
        "ack_no", "acknowledgement_no", "acknowledgment_no", "complaint_ack",
        "complaint_no", "ncrp_ack",
    ),
    "victim_account": (
        "victim_account", "victim_acc_no", "complainant_account",
        "complainant_acc", "defrauded_account",
    ),
    "suspect_account": (
        "suspect_account", "suspect_acc_no", "accused_account",
        "culprit_account", "fraudster_account",
    ),
    "fraud_category": (
        "fraud_category", "category_of_fraud", "complaint_category", "fraud_type",
    ),
    "reported_timestamp": (
        "reported_timestamp", "report_datetime", "complaint_datetime",
        "reported_on", "report_time",
    ),
    "disputed_amount": (
        "disputed_amount", "cheated_amount", "fraud_amount", "loss_amount",
        "complaint_amount",
    ),
    "disputed_utr": (
        "disputed_utr", "disputed_utr_no", "disputed_transaction_id", "utr",
        "transaction_reference",
    ),
    "state": ("state", "states", "jurisdiction", "in_state"),
}

ACCOUNT_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "account_no": (
        "account_no", "account_number", "acct_no", "account", "cust_account_no",
    ),
    "ifsc": ("ifsc", "ifsc_code", "branch_ifsc"),
    "account_opening_date": (
        "account_opening_date", "opening_date", "account_opened_on", "opened_date",
    ),
    "branch_code": ("branch_code", "branch", "home_branch", "sol_id"),
    "mobile_no": (
        "mobile_no", "mobile_number", "phone_no", "contact_number", "cust_mobile",
    ),
    "imei": ("imei", "imei_number", "device_imei"),
    "kyc_pincode": ("kyc_pincode", "pin_code", "pincode", "zip", "postal_code"),
    "account_type": ("account_type", "acct_type", "account_category"),
}

TABLE_SYNONYMS: Final[dict[TableKind, dict[str, tuple[str, ...]]]] = {
    "transactions": CANONICAL_SYNONYMS,
    "complaints": COMPLAINT_SYNONYMS,
    "accounts": ACCOUNT_SYNONYMS,
}

_HEADER_NORM_RE: Final = re.compile(r"[^a-z0-9]+")


def normalize_header(name: str) -> str:
    """Lowercase and collapse every non-alphanumeric run to single spaces."""
    return _HEADER_NORM_RE.sub(" ", name.lower()).strip()


def _best_synonym(
    norm_raw: str, synonyms: dict[str, tuple[str, ...]]
) -> tuple[float, str, str] | None:
    """Highest synonym score across all canonicals, or None when nothing matched."""
    best_score: float = 0.0
    best_canonical: str = ""
    best_synonym: str = ""
    for canonical, candidates in synonyms.items():
        for synonym in candidates:
            score = fuzz.token_sort_ratio(norm_raw, normalize_header(synonym))
            if score > best_score:
                best_score, best_canonical, best_synonym = score, canonical, synonym
    if best_score == 0.0:
        return None
    return (best_score, best_canonical, best_synonym)


def map_headers(
    raw_columns: Sequence[str],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    table: TableKind = "transactions",
) -> dict[str, ColumnMapping]:
    """Map raw header names to canonical columns with fuzzy confidence.

    One raw header maps to at most one canonical; one canonical accepts at
    most one raw header. Highest score wins; exact ties keep the first-seen
    raw column. A losing claimant stays in the result with
    ``canonical_name=None`` and its own confidence score (audit trail).
    The synonym registry is per table (`TABLE_SYNONYMS` keyed on `TableKind`):
    transaction, complaint and account dumps never cross-claim headers.
    """
    cutoff = confidence_threshold - 1e-9
    synonyms = TABLE_SYNONYMS[table]
    claims: dict[str, tuple[float, int]] = {}
    best_per_raw: dict[int, tuple[float, str, str] | None] = {}
    for index, raw in enumerate(raw_columns):
        scored = _best_synonym(normalize_header(raw), synonyms)
        eligible = (
            scored if scored is not None and scored[0] / 100.0 >= cutoff else None
        )
        best_per_raw[index] = eligible
        if eligible is not None:
            score, canonical, _synonym = eligible
            current = claims.get(canonical)
            if current is None or score > current[0]:
                claims[canonical] = (score, index)

    out: dict[str, ColumnMapping] = {}
    for index, raw in enumerate(raw_columns):
        scored = best_per_raw[index]
        if scored is None:
            out[raw] = ColumnMapping(
                raw_name=raw, canonical_name=None,
                matched_synonym=None, confidence=0.0,
            )
            continue
        score, canonical, synonym = scored
        if claims.get(canonical) == (score, index):
            out[raw] = ColumnMapping(
                raw_name=raw, canonical_name=canonical,
                matched_synonym=synonym, confidence=round(score / 100.0, 4),
            )
        else:
            out[raw] = ColumnMapping(
                raw_name=raw, canonical_name=None,
                matched_synonym=None, confidence=round(score / 100.0, 4),
            )
    return out


# --- canonical value expressions -------------------------------------------

TS_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%Y-%m-%d",
)

MODE_CANONICAL: Final[dict[str, str]] = {
    "UPI": "UPI",
    "BHIM UPI": "UPI",
    "UPI P2P": "UPI",
    "UPI/P2P": "UPI",
    "UPI/P2M": "UPI",
    "IMPS": "IMPS",
    "NEFT": "NEFT",
    "RTGS": "RTGS",
    "AEPS": "AEPS",
    "AEPS CASH OUT": "AEPS",
    "MICRO ATM": "AEPS",
    "ATM": "ATM",
    "ATM WITHDRAWAL": "ATM",
    "CASH": "CASH",
    "CASH DEPOSIT": "CASH",
    "CASHOUT": "CASH",
}

# Account-type aliases (indian passbook vocabulary -> canonical AccountType).
ACCOUNT_TYPE_CANONICAL: Final[dict[str, str]] = {
    "SAVINGS": "SAVINGS",
    "SAVINGS ACCOUNT": "SAVINGS",
    "SB": "SAVINGS",
    "SB ACCOUNT": "SAVINGS",
    "PMJDY": "PMJDY",
    "CURRENT": "CURRENT",
    "CURRENT ACCOUNT": "CURRENT",
    "CA": "CURRENT",
}

# Per-table canonical column set + typed schema (single source: contracts).
TABLE_COLUMNS: Final[dict[TableKind, tuple[str, ...]]] = {
    "transactions": TRANSACTION_COLUMNS,
    "complaints": COMPLAINT_COLUMNS,
    "accounts": ACCOUNT_COLUMNS,
}
TABLE_SCHEMA: Final[dict[TableKind, dict[str, pl.DataType]]] = {
    "transactions": TRANSACTION_SCHEMA,
    "complaints": COMPLAINT_SCHEMA,
    "accounts": ACCOUNT_METADATA_SCHEMA,
}

_XLSX_MAGIC: Final[bytes] = b"PK\x03\x04"
_NUMERIC_RE: Final = re.compile(r"^\d+$")
_CURRENCY_JUNK_RE: Final = re.compile(r"[,\s\u20b9]")


def _looks_like_xlsx(source: str | Path | bytes) -> bool:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source[:4]) == _XLSX_MAGIC
    return Path(source).suffix.lower() in {".xlsx", ".xls"}


def _source_label(source: str | Path | bytes) -> str | None:
    if isinstance(source, (bytes, bytearray)):
        return "bytes"
    return str(source)


def _read_csv(source: str | Path | bytes) -> pl.DataFrame:
    """All-String frame: every normalization decision happens downstream."""
    if isinstance(source, (bytes, bytearray)):
        data: str | io.BytesIO = io.BytesIO(bytes(source))
    else:
        data = str(source)
    return pl.read_csv(
        data,
        infer_schema_length=0,
        truncate_ragged_lines=True,
        quote_char='"',
    )


def _read_xlsx(source: str | Path | bytes) -> pl.DataFrame:
    """First worksheet; every cell coerced to String for the unified pipeline."""
    if isinstance(source, (bytes, bytearray)):
        df = pl.read_excel(io.BytesIO(bytes(source)), engine="openpyxl")
    else:
        df = pl.read_excel(str(source), engine="openpyxl")
    return df.select(
        [pl.col(name).cast(pl.String, strict=False).alias(name) for name in df.columns]
    )


def _canonical_raw(mapping: dict[str, ColumnMapping], canonical: str) -> str | None:
    """Original source column claimed by the fuzzy mapper for one canonical."""
    for raw_name, entry in mapping.items():
        if entry.canonical_name == canonical:
            return raw_name
    return None


def _epoch_expr(src: pl.Expr) -> pl.Expr:
    """UNIX seconds/milliseconds detection for purely numeric timestamps."""
    digits = pl.when(src.str.contains(_NUMERIC_RE.pattern)).then(src).otherwise(None)
    as_int = digits.cast(pl.Int64, strict=False)
    return (
        pl.when(as_int >= 10**12)
        .then(pl.from_epoch(as_int, time_unit="ms"))
        .when(as_int >= 10**9)
        .then(pl.from_epoch(as_int, time_unit="s"))
        .otherwise(None)
    )


def _timestamp_expr(src: pl.Expr) -> pl.Expr:
    """Epoch first, then DD/MM-before-MM/DD format fallbacks (task Decision 2)."""
    fallback = pl.coalesce(
        *[
            src.str.strptime(pl.Datetime("us"), fmt, strict=False)
            for fmt in TS_FORMATS
        ]
    )
    return pl.coalesce(_epoch_expr(src), fallback)


def _amount_expr(src: pl.Expr) -> pl.Expr:
    """Strip commas/currency/whitespace, quiet cast; garbage -> null."""
    return src.str.replace_all(_CURRENCY_JUNK_RE.pattern, "").cast(
        pl.Float64, strict=False
    )


def _mode_expr(src: pl.Expr) -> pl.Expr:
    return (
        src.str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(r"\s+", " ")
        .replace_strict(MODE_CANONICAL, default=None)
    )


def _ifsc_expr(src: pl.Expr) -> pl.Expr:
    uppercase = src.str.strip_chars().str.to_uppercase()
    return pl.when(
        uppercase.is_not_null() & uppercase.str.contains(IFSC_PATTERN.pattern)
    ).then(uppercase).otherwise(None)


def _date_expr(src: pl.Expr) -> pl.Expr:
    """Calendar date via the shared timestamp chain (DD/MM precedence inherited),
    then truncated to Date — covers ISO dates and openpyxl-native datetime strings."""
    as_dt = _timestamp_expr(src)
    return as_dt.cast(pl.Date, strict=False)


def _pattern_expr(src: pl.Expr, pattern: str) -> pl.Expr:
    """Keep the cleaned value when it matches `pattern`, else null (audit-grade)."""
    cleaned = src.str.strip_chars()
    return pl.when(
        cleaned.is_not_null() & cleaned.str.contains(pattern)
    ).then(cleaned).otherwise(None)


def _account_type_expr(src: pl.Expr) -> pl.Expr:
    return (
        src.str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(r"\s+", " ")
        .replace_strict(ACCOUNT_TYPE_CANONICAL, default=None)
    )


def _normalize(
    df: pl.DataFrame, mapping: dict[str, ColumnMapping], table: TableKind
) -> pl.DataFrame:
    """Compose raw-text audit columns + typed canonical columns.

    Guarantees: for every canonical in TABLE_COLUMNS[table], both a cleaned
    `_raw_<canonical>` (String, "" -> null) and the canonical column itself
    exist afterwards, exactly once, regardless of the source's raw names.
    """
    columns = TABLE_COLUMNS[table]
    raw_steps: list[pl.Expr] = []
    for canonical in columns:
        source = _canonical_raw(mapping, canonical)
        raw_expr = (
            pl.col(source).cast(pl.String, strict=False).str.strip_chars().replace("", None)
            if source is not None
            else pl.lit(None, dtype=pl.String())
        )
        raw_steps.append(raw_expr.alias(f"_raw_{canonical}"))
    work = df.with_columns(raw_steps)

    canonical_steps: list[pl.Expr] = []
    for canonical in columns:
        src = pl.col(f"_raw_{canonical}")
        expr: pl.Expr
        if table == "transactions":
            if canonical == "amount":
                expr = _amount_expr(src)
            elif canonical == "timestamp":
                expr = _timestamp_expr(src)
            elif canonical == "payment_mode":
                expr = _mode_expr(src)
            elif canonical in ("sender_ifsc", "receiver_ifsc"):
                expr = _ifsc_expr(src)
            else:
                expr = src
        elif table == "complaints":
            if canonical == "disputed_amount":
                expr = _amount_expr(src)
            elif canonical == "reported_timestamp":
                expr = _timestamp_expr(src)
            else:
                expr = src
        else:
            if canonical == "account_opening_date":
                expr = _date_expr(src)
            elif canonical == "ifsc":
                expr = _ifsc_expr(src)
            elif canonical == "account_type":
                expr = _account_type_expr(src)
            elif canonical == "mobile_no":
                expr = _pattern_expr(src, r"^\d{10,12}$")
            elif canonical == "imei":
                expr = _pattern_expr(src, r"^\d{15}$")
            elif canonical == "kyc_pincode":
                expr = _pattern_expr(src, r"^\d{6}$")
            else:
                expr = src
        canonical_steps.append(expr.alias(canonical))
    return work.with_columns(canonical_steps)


class DatasetIngester:
    """CSV/XLSX ingester with fuzzy headers, validation and audit reporting.

    One class, three table kinds: transactions (default), NCRP complaints,
    and account metadata — each with its own synonym registry, canonical
    set and validation rules (Decision 12).
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = confidence_threshold

    def ingest(
        self, source: str | Path | bytes,
        *, table: TableKind = "transactions",
    ) -> tuple[pl.DataFrame, IngestionReport]:
        """Dispatch CSV vs XLSX by PK-zip magic (bytes) or suffix (paths)."""
        if _looks_like_xlsx(source):
            return self.ingest_xlsx(source, table=table)
        return self.ingest_csv(source, table=table)

    def ingest_csv(
        self, source: str | Path | bytes,
        *, table: TableKind = "transactions",
    ) -> tuple[pl.DataFrame, IngestionReport]:
        return self._ingest_frame(_read_csv(source), _source_label(source), table)

    def ingest_xlsx(
        self, source: str | Path | bytes,
        *, table: TableKind = "transactions",
    ) -> tuple[pl.DataFrame, IngestionReport]:
        return self._ingest_frame(_read_xlsx(source), _source_label(source), table)

    def ingest_transactions(self, source: str | Path | bytes) -> tuple[pl.DataFrame, IngestionReport]:
        return self.ingest(source, table="transactions")

    def ingest_complaints(self, source: str | Path | bytes) -> tuple[pl.DataFrame, IngestionReport]:
        return self.ingest(source, table="complaints")

    def ingest_accounts(self, source: str | Path | bytes) -> tuple[pl.DataFrame, IngestionReport]:
        return self.ingest(source, table="accounts")

    def _ingest_frame(
        self, raw: pl.DataFrame, source_label: str | None, table: TableKind
    ) -> tuple[pl.DataFrame, IngestionReport]:
        """Happy-path: normalize + emit every row as accepted (Task 5 wires rejections)."""
        mapping = map_headers(raw.columns, self.confidence_threshold, table=table)
        work = _normalize(raw, mapping, table)
        accepted = work.select(
            [
                pl.col(c).cast(TABLE_SCHEMA[table][c], strict=False)
                for c in TABLE_COLUMNS[table]
            ]
        )
        rejected = pl.DataFrame(
            schema={
                name: pl.String()
                for name in (*[str(c) for c in raw.columns], "rejection_reason")
            }
        )
        report = IngestionReport(
            source=source_label,
            rows_total=raw.height,
            rows_accepted=accepted.height,
            rows_rejected=0,
            failure_counts={},
            column_mapping=mapping,
            rejected_rows=rejected,
        )
        return accepted, report
