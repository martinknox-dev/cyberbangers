"""Fuzzy schema ingestion gateway.

Header mapping follows the spec's RapidFuzz token-sort-ratio approach;
readers + normalization (Task 4) and validation + reporting (Task 5)
complete the pipeline in this same module. Everything runs as Polars
expression chains: no row-level Python loops over dataframes, no Pandas.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from rapidfuzz import fuzz

from core.contracts import ColumnMapping, TableKind

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
