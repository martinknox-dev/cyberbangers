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
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


# --- Phase 2: behavioural features, graph indicators, risk report ------------

ACCOUNT_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "account_no": pl.String(),
    "txn_count": pl.Int64(),
    "inflow_count": pl.Int64(),
    "outflow_count": pl.Int64(),
    "inflow_total": pl.Float64(),
    "outflow_total": pl.Float64(),
    "net_flow": pl.Float64(),
    "unique_senders": pl.Int64(),
    "unique_receivers": pl.Int64(),
    "fan_in_fan_out_asymmetry": pl.Float64(),
    "pass_through_ratio_15m": pl.Float64(),
    "median_retention_minutes": pl.Float64(),
    "smurf_bulk_inflow_count": pl.Int64(),
    "smurf_split_count": pl.Int64(),
    "large_payout_count": pl.Int64(),
    "burst_24h_max_total": pl.Float64(),
    "dormancy_days_before_burst": pl.Float64(),
    "dormancy_burst_score": pl.Float64(),
    "shared_device_account_count": pl.Int64(),
    "branch_cohort_48h_count": pl.Int64(),
    "aeps_cashout_inflow_count": pl.Int64(),
    "direct_complaint_count": pl.Int64(),
    "pass_through_utrs": pl.List(pl.String()),
    "smurf_utrs": pl.List(pl.String()),
    "first_txn_timestamp": pl.Datetime("us"),
    "last_txn_timestamp": pl.Datetime("us"),
}

GRAPH_INDICATOR_SCHEMA: dict[str, pl.DataType] = {
    "account_no": pl.String(),
    "pagerank": pl.Float64(),
    "betweenness_centrality": pl.Float64(),
    "louvain_community_id": pl.Int64(),
    "cycle_count": pl.Int64(),
    "cycle_max_amount": pl.Float64(),
    "cycle_txn_ids": pl.List(pl.String()),
    "complaint_link_count": pl.Int64(),
    "complaint_ack_nos": pl.List(pl.String()),
    "trail_taint_max": pl.Float64(),
    "trail_hops_max": pl.Int64(),
    "terminal_chain_count": pl.Int64(),
}

RISK_REPORT_SCHEMA: dict[str, pl.DataType] = {
    "account_no": pl.String(),
    "risk_score": pl.Float64(),
    "domain_score": pl.Float64(),
    "anomaly_score": pl.Float64(),
    "risk_band": pl.String(),
    "top_reasons": pl.List(pl.String()),
    "explanation_kinds": pl.List(pl.String()),
    "evidence_txn_ids": pl.List(pl.String()),
}

RISK_DISCLAIMER: str = (
    "Risk score is an investigative lead-ranking aid computed from observed "
    "ledger behaviour. It is not a probability of guilt, not legal evidence, "
    "and not a substitute for human investigation."
)

# Monetary comparison tolerance (rupees). Used wherever a computed monetary
# value is compared against a limit or against another monetary value.
MONEY_EPSILON: float = 1e-6


class FeatureConfig(BaseModel):
    """Configurable thresholds for the behavioural feature extractor."""

    pass_through_window_minutes: int = Field(default=15, ge=1, le=1440)
    fan_out_window_minutes: int = Field(default=60, ge=1, le=1440)
    dormancy_threshold_days: int = Field(default=180, ge=0)
    burst_amount_threshold: float = Field(default=25_000.0, gt=0)
    smurf_bulk_threshold: float = Field(default=50_000.0, gt=0)
    smurf_min_splits: int = Field(default=8, ge=1)
    smurf_split_min: float = Field(default=5_000.0, ge=0)
    smurf_split_max: float = Field(default=9_999.99, gt=0)
    aggregator_min_senders: int = Field(default=10, ge=1)
    branch_cohort_hours: int = Field(default=48, ge=1)
    terminal_modes: tuple[str, ...] = ("AEPS", "ATM", "CASH")
    burst_window_hours: int = Field(default=24, ge=1)


class GraphConfig(BaseModel):
    """Configurable limits for graph tracing, cycle detection and correlation."""

    max_hops: int = Field(default=3, ge=1, le=6)
    cycle_max_length: int = Field(default=5, ge=3, le=5)
    cycle_window_hours: int = Field(default=24, ge=1)
    cycle_max_decay: float = Field(default=0.15, ge=0.0, le=1.0)
    community_seed: int = 42
    # How far back a complaint-scoped trace looks for the disputed transfer when
    # the complaint names no suspect.
    complaint_lookback_hours: int = Field(default=24, ge=1)
    # Exact betweenness is O(V*E); `betweenness_k` switches to networkx's
    # deterministic sampled approximation (k sampled source nodes).
    betweenness_k: int | None = None
    pagerank_alpha: float = Field(default=0.85, gt=0.0, lt=1.0)
    trail_window_hours: int = Field(default=168, ge=1)


class RiskConfig(BaseModel):
    """Configurable weights and thresholds for the hybrid risk engine."""

    domain_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    anomaly_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    pass_through_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    pass_through_weight: float = Field(default=25.0, ge=0.0)
    smurfing_weight: float = Field(default=20.0, ge=0.0)
    cycle_weight: float = Field(default=20.0, ge=0.0)
    shared_metadata_weight: float = Field(default=10.0, ge=0.0)
    terminal_cashout_weight: float = Field(default=10.0, ge=0.0)
    complaint_weight: float = Field(default=15.0, ge=0.0)
    shared_device_threshold: int = Field(default=1, ge=0)
    branch_cohort_threshold: int = Field(default=2, ge=0)
    terminal_cashout_threshold: int = Field(default=2, ge=0)
    complaint_link_threshold: int = Field(default=1, ge=0)
    cycle_threshold: int = Field(default=1, ge=0)
    min_rows_for_anomaly: int = Field(default=10, ge=1)
    max_explanations: int = Field(default=3, ge=1, le=10)
    high_risk_band: float = Field(default=70.0, ge=0.0, le=100.0)
    medium_risk_band: float = Field(default=40.0, ge=0.0, le=100.0)
    # Mirrors the feature extractor's thresholds; used to phrase the applied
    # threshold in the explanation text.
    pass_through_window_minutes: int = Field(default=15, ge=1)
    smurf_min_splits: int = Field(default=8, ge=1)
    smurf_split_max: float = Field(default=9_999.99, gt=0)
    branch_cohort_hours: int = Field(default=48, ge=1)

    @property
    def total_domain_weight(self) -> float:
        return (
            self.pass_through_weight
            + self.smurfing_weight
            + self.cycle_weight
            + self.shared_metadata_weight
            + self.terminal_cashout_weight
            + self.complaint_weight
        )


class MoneyTrailLeg(BaseModel):
    """One transaction leg on a traced money trail, with its taint share.

    `taint_amount` is the part of *this* transaction that is attributable to
    the traced funds. It can never exceed the transaction's own amount.
    """

    hop: int = Field(ge=1)
    txn_id: str
    sender_account: str
    receiver_account: str
    amount: float = Field(ge=0)
    timestamp: datetime
    payment_mode: str
    taint_amount: float = Field(ge=0)

    @model_validator(mode="after")
    def _taint_within_amount(self) -> MoneyTrailLeg:
        if self.taint_amount > self.amount + MONEY_EPSILON:
            raise ValueError(
                f"taint_amount {self.taint_amount} exceeds the transaction "
                f"amount {self.amount} for txn {self.txn_id}"
            )
        return self


class MoneyTrail(BaseModel):
    """A bounded, temporally ordered forward trace of disputed funds.

    Accounting semantics: tainted funds are held by an account as a *tainted
    balance* and are drawn down first-in-first-out as that account's
    outgoing transactions are processed in chronological order. A transaction
    therefore carries at most the tainted funds the sender actually held when
    the transfer happened, and never more than the transfer's own amount.

    `total_taint` is the tainted amount that left the origin (the sum of the
    first-hop legs). Taint that was not propagated forward stays with the
    account that held it and is reported nowhere else: it is never invented
    into a downstream transfer.
    """

    origin_account: str
    origin_amount: float = Field(ge=0)
    origin_txn_id: str | None = None
    max_hops: int = Field(ge=1, le=6)
    legs: list[MoneyTrailLeg] = Field(default_factory=list)
    reached_accounts: list[str] = Field(default_factory=list)
    account_taint: dict[str, float] = Field(default_factory=dict)
    total_taint: float = Field(ge=0)

    @property
    def taint_conserved(self) -> bool:
        """No attributed taint may exceed the originating amount or the
        amount of the transaction that carries it."""
        if self.total_taint > self.origin_amount + MONEY_EPSILON:
            return False
        if any(
            leg.taint_amount > leg.amount + MONEY_EPSILON for leg in self.legs
        ):
            return False
        if any(
            amount > self.origin_amount + MONEY_EPSILON
            for amount in self.account_taint.values()
        ):
            return False
        first_hop = sum(
            leg.taint_amount for leg in self.legs if leg.hop == 1
        )
        return abs(first_hop - self.total_taint) <= MONEY_EPSILON


class WashCycle(BaseModel):
    """A closed directed money cycle with monotone amount decay."""

    cycle_accounts: list[str]
    txn_ids: list[str]
    amounts: list[float]
    timestamps: list[datetime]
    start_timestamp: datetime
    end_timestamp: datetime
    total_amount: float = Field(ge=0)
    max_decay_ratio: float = Field(ge=0.0)


class ComplaintLink(BaseModel):
    """Two NCRP cases linked by ledger/device/terminal evidence."""

    ack_no_a: str
    ack_no_b: str
    state_a: str
    state_b: str
    shared_accounts: list[str] = Field(default_factory=list)
    shared_device_ids: list[str] = Field(default_factory=list)
    shared_terminal_accounts: list[str] = Field(default_factory=list)
    link_basis: str = Field(min_length=1)


class RiskExplanation(BaseModel):
    """One investigator-facing explanation line, typed as fact or inference."""

    kind: str = Field(min_length=1)          # "FACT" or "INDICATOR"
    indicator: str = Field(min_length=1)
    measured_value: float | None = None
    threshold: float | None = None
    points: float = 0.0
    detail: str = Field(min_length=1)
    evidence_txn_ids: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Full risk assessment for one account (frame row + typed detail)."""

    account_no: str
    risk_score: float = Field(ge=0.0, le=100.0)
    domain_score: float = Field(ge=0.0, le=100.0)
    anomaly_score: float = Field(ge=0.0, le=100.0)
    risk_band: str
    explanations: list[RiskExplanation] = Field(default_factory=list)
    domain_components: dict[str, float] = Field(default_factory=dict)  # 0..100 each
    top_archetype: str = ""
    disclaimer: str = RISK_DISCLAIMER


# --- Phase 3: case dossier ---------------------------------------------------
DossierFactCategory = Literal["ACCOUNT", "TRANSACTION", "COMPLAINT", "SHARED_KYC"]
DossierMethod = Literal["DOMAIN_RULE", "ISOLATION_FOREST_SHAP", "FIFO_TAINT", "TOPOLOGY"]
DossierScope = Literal["ACCOUNT", "CLUSTER", "COMPLAINT"]
DomainKey = Literal["pass_through", "smurfing", "cycle", "shared_metadata",
                    "terminal_cashout", "complaint"]

DOMAIN_LABELS: dict[str, str] = {
    "pass_through": "Rapid pass-through",
    "smurfing": "Structuring / fan-out",
    "cycle": "Circular wash",
    "shared_metadata": "Shared KYC / branch ring",
    "terminal_cashout": "Terminal cash-out",
    "complaint": "Complaint linkage",
    "anomaly": "ML anomaly (IsolationForest)",
}
DOMAIN_KEYS: tuple[str, ...] = (
    "pass_through", "smurfing", "cycle", "shared_metadata",
    "terminal_cashout", "complaint",
)

BNSS_SECTION_94: str = (
    "Section 94 of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)"
)
BNSS_SECTION_106: str = "Section 106 of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)"

DOSSIER_DISCLAIMER: str = (
    "Section A reproduces records supplied by the source systems and is not a "
    "certified copy. Section B is analytical output of an automated model: it "
    "is an investigative lead, not evidence and not a probability of guilt. "
    "Section C is a draft requisition for review and signature by the competent "
    "investigating officer; it is not legal advice and does not by itself "
    "authorise lien marking, freezing or seizure."
)

REQUISITION_DOCUMENTS: tuple[str, ...] = (
    "Certified copy of the account opening form (AOF) and KYC documents",
    (
        "Customer master details, including registered mobile number, e-mail, "
        "PAN (if recorded) and introducer details"
    ),
    (
        "Certified account statement covering every transaction reference listed "
        "against this bank"
    ),
    (
        "IP address, device identifier/IMEI and login/access logs for the listed "
        "transaction references"
    ),
    "Branch and opening-official details for the listed accounts",
    "Current account status (active/frozen/closed) and any lien already marked",
)


class DossierFact(BaseModel):
    """Section A: one verifiable record, tied to its source."""

    category: DossierFactCategory
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: Literal["ledger", "account_master", "ncrp", "generator"]
    account_no: str | None = None
    txn_id: str | None = None
    ack_no: str | None = None
    timestamp: datetime | None = None
    amount: float | None = None


class DossierSubScore(BaseModel):
    domain: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: float = Field(ge=0.0, le=100.0)
    weight: float = Field(ge=0.0)
    points: float = Field(ge=0.0)


class DossierIndicator(BaseModel):
    """Section B: one analytical measurement. Never evidence."""

    indicator: str = Field(min_length=1)
    method: DossierMethod
    value: float
    unit: Literal["score", "ratio", "count", "rupees"] = "score"
    threshold: float | None = None
    points: float = 0.0
    explanation: str = Field(min_length=1)
    txn_ids: list[str] = Field(default_factory=list)


class RequisitionRequest(BaseModel):
    bank_name: str = Field(min_length=1)
    ifsc_prefix: str = Field(min_length=4, max_length=12)
    accounts: list[str] = Field(default_factory=list)
    utrs: list[str] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)


class DossierRequisition(BaseModel):
    statute_reference: str = Field(min_length=1)
    addressee: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    case_reference: str = Field(min_length=1)
    paragraphs: list[str] = Field(default_factory=list)
    bank_requests: list[RequisitionRequest] = Field(default_factory=list)
    legal_note: str = Field(min_length=1)


class DossierRiskSummary(BaseModel):
    account_no: str
    risk_score: float = Field(ge=0.0, le=100.0)
    domain_score: float = Field(ge=0.0, le=100.0)
    anomaly_score: float = Field(ge=0.0, le=100.0)
    risk_band: str
    top_archetype: str
    sub_scores: list[DossierSubScore] = Field(default_factory=list)


class InvestigationDossier(BaseModel):
    case_reference: str = Field(min_length=1)
    scope: DossierScope
    subject_accounts: list[str] = Field(default_factory=list)
    facts: list[DossierFact] = Field(default_factory=list)
    indicators: list[DossierIndicator] = Field(default_factory=list)
    risk_summary: DossierRiskSummary | None = None
    requisition: DossierRequisition
    disclaimer: str = DOSSIER_DISCLAIMER
