"""Deterministic case-evidence dossier and Section 94 BNSS requisition draft.

Pure Python: Polars/Pydantic models in, Markdown/HTML/JSON out. This module
never imports Streamlit, never reads the clock and never touches the network.
Section A reproduces supplied records; Section B carries analytical output and
is labelled as such; Section C is a draft requisition for legal review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from core.coerce import as_date, as_datetime, as_float, as_str
from core.contracts import (
    BNSS_SECTION_94,
    DOMAIN_KEYS,
    DOMAIN_LABELS,
    DOSSIER_DISCLAIMER,
    REQUISITION_DOCUMENTS,
    DossierFact,
    DossierIndicator,
    DossierRequisition,
    DossierRiskSummary,
    DossierScope,
    DossierSubScore,
    IngestionReport,
    InvestigationDossier,
    MoneyTrail,
    RiskAssessment,
    WashCycle,
)

DEFAULT_BANK_NAMES: dict[str, str] = {
    "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank", "PUNB": "Punjab National Bank", "KKBK": "Kotak Mahindra Bank",
    "BARB": "Bank of Baroda", "CNRB": "Canara Bank", "IOBA": "Indian Overseas Bank",
    "UCBA": "UCO Bank", "UBIN": "Union Bank of India", "IDIB": "Indian Bank",
    "PSIB": "Punjab & Sind Bank", "MAHB": "Bank of Maharashtra",
    "CBIN": "Central Bank of India", "BDBL": "Bandhan Bank",
    "FINO": "Fino Payments Bank", "IPPB": "India Post Payments Bank",
    "AIRP": "Airtel Payments Bank", "PYTM": "Paytm Payments Bank",
}

_FACT_ORDER: dict[str, int] = {
    "ACCOUNT": 0, "TRANSACTION": 1, "COMPLAINT": 2, "SHARED_KYC": 3,
}

IndicatorUnit = Literal["score", "ratio", "count", "rupees"]

_INDICATOR_UNITS: dict[str, IndicatorUnit] = {
    "pass_through": "ratio", "smurfing": "count", "cycle_participation": "count",
    "shared_metadata": "count", "terminal_cashout": "count",
    "complaint_relationship": "count", "isolation_forest_anomaly": "score",
}
_TOPOLOGY_FIELDS: tuple[tuple[str, str, IndicatorUnit], ...] = (
    ("pagerank", "PageRank centrality", "score"),
    ("betweenness_centrality", "Betweenness centrality (bridge role)", "score"),
    ("cycle_count", "Closed money cycles involving this account", "count"),
    ("terminal_chain_count", "Independent chains terminating here", "count"),
    ("complaint_link_count", "NCRP cases linked through shared evidence", "count"),
)


def format_inr(amount: float) -> str:
    """Indian digit grouping with 2 dp, locale-independent and deterministic."""
    negative = amount < 0
    whole, _, paise = f"{abs(amount):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups) + "," + tail
    return f"{'-' if negative else ''}₹{whole}.{paise}"


def format_timestamp(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else "not recorded"


def bank_name_for(ifsc: str | None, table: Mapping[str, str]) -> str:
    prefix = ifsc[:4].upper() if ifsc else ""
    if not prefix:
        return "Bank not identified (no IFSC in supplied records)"
    return table.get(prefix, f"Bank with IFSC prefix {prefix} (name not in reference table)")


def rejected_rows_csv(report: IngestionReport) -> bytes:
    """Audit CSV of rejected rows, exactly as captured during ingestion."""
    return report.rejected_rows.write_csv().encode("utf-8")


@dataclass(frozen=True)
class DossierAccount:
    account_no: str
    metadata: Mapping[str, object] | None = None
    risk: RiskAssessment | None = None
    feature_row: Mapping[str, object] | None = None
    graph_row: Mapping[str, object] | None = None
    transactions: Sequence[Mapping[str, object]] = ()
    complaints: Sequence[Mapping[str, object]] = ()
    trail: MoneyTrail | None = None
    cycles: Sequence[WashCycle] = ()
    imei_peers: Sequence[Mapping[str, object]] = ()
    branch_peers: Sequence[Mapping[str, object]] = ()
    ledger_txn_ids: frozenset[str] | None = None   # None = every leg


_DEFAULT_DOMAIN_WEIGHTS: dict[str, float] = {
    "pass_through": 25.0,
    "smurfing": 20.0,
    "cycle": 20.0,
    "shared_metadata": 10.0,
    "terminal_cashout": 10.0,
    "complaint": 15.0,
    "anomaly": 30.0,
}


class DossierConfig(BaseModel):
    case_reference: str = "PG-CASE-0001"
    scope: DossierScope = "ACCOUNT"
    generated_on: date | None = None
    max_ledger_rows: int = Field(default=200, ge=1)
    deadline_days: int = Field(default=7, ge=1)
    bank_names: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_BANK_NAMES))
    requested_documents: tuple[str, ...] = REQUISITION_DOCUMENTS
    domain_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_DOMAIN_WEIGHTS)
    )


class InvestigationDossierGenerator:
    def __init__(self, accounts: Sequence[DossierAccount],
                 config: DossierConfig | None = None) -> None:
        self._accounts = tuple(sorted(accounts, key=lambda item: item.account_no))
        self._config = config or DossierConfig()
        self._built: InvestigationDossier | None = None

    # --- Section A --------------------------------------------------------
    def _facts_for(self, account: DossierAccount) -> list[DossierFact]:
        facts: list[DossierFact] = [self._account_fact(account)]
        facts.extend(self._ledger_facts(account))
        facts.extend(self._complaint_facts(account))
        facts.extend(self._shared_kyc_facts(account))
        return facts

    def _account_fact(self, account: DossierAccount) -> DossierFact:
        metadata = account.metadata
        if not metadata:
            return DossierFact(
                category="ACCOUNT", label="Account record",
                value="No account master record supplied for this account",
                source="generator", account_no=account.account_no,
            )
        ifsc = as_str(metadata.get("ifsc")) or "not recorded"
        opened = as_date(metadata.get("account_opening_date"))
        value = (
            f"A/C {account.account_no} · IFSC {ifsc} · "
            f"branch {as_str(metadata.get('branch_code')) or 'not recorded'} · "
            f"{as_str(metadata.get('account_type')) or 'type not recorded'} · "
            f"opened {opened.isoformat() if opened else 'not recorded'}"
        )
        return DossierFact(category="ACCOUNT", label="Account record", value=value,
                           source="account_master", account_no=account.account_no)

    def _ledger_facts(self, account: DossierAccount) -> list[DossierFact]:
        legs = [leg for leg in account.transactions
                if account.ledger_txn_ids is None
                or as_str(leg.get("txn_id")) in account.ledger_txn_ids]
        legs.sort(key=lambda leg: (as_datetime(leg.get("timestamp")) or datetime.min,
                                   as_str(leg.get("txn_id"))))
        limit = self._config.max_ledger_rows
        shown = legs[:limit]
        facts: list[DossierFact] = []
        for leg in shown:
            outbound = as_str(leg.get("sender_account")) == account.account_no
            counterparty = as_str(leg.get("receiver_account") if outbound
                                  else leg.get("sender_account"))
            ifsc_key = "receiver_ifsc" if outbound else "sender_ifsc"
            counterparty_ifsc = as_str(leg.get(ifsc_key)) or "IFSC not recorded"
            amount = as_float(leg.get("amount"))
            timestamp = as_datetime(leg.get("timestamp"))
            txn_id = as_str(leg.get("txn_id"))
            mode = as_str(leg.get("payment_mode")) or "mode not recorded"
            facts.append(DossierFact(
                category="TRANSACTION",
                label=(f"Outflow to {counterparty} ({counterparty_ifsc})" if outbound
                       else f"Inflow from {counterparty} ({counterparty_ifsc})"),
                value=(f"{format_inr(amount)} via {mode} on "
                       f"{format_timestamp(timestamp)} (UTR {txn_id})"),
                source="ledger", account_no=account.account_no, txn_id=txn_id,
                timestamp=timestamp, amount=amount,
            ))
        if len(legs) > limit:
            facts.append(DossierFact(
                category="TRANSACTION", label="Ledger listing truncated",
                value=(f"Showing the first {limit} of {len(legs)} legs for "
                       f"A/C {account.account_no}; the complete statement must be "
                       "obtained under the requisition."),
                source="generator", account_no=account.account_no,
            ))
        return facts

    def _complaint_facts(self, account: DossierAccount) -> list[DossierFact]:
        rows = sorted(
            account.complaints,
            key=lambda row: as_str(row.get("ack_no")),
        )
        facts: list[DossierFact] = []
        for row in rows:
            ack_no = as_str(row.get("ack_no"))
            reported = as_datetime(row.get("reported_timestamp"))
            amount = as_float(row.get("disputed_amount"))
            facts.append(DossierFact(
                category="COMPLAINT",
                label=(
                    f"NCRP complaint {ack_no} "
                    f"({as_str(row.get('fraud_category')) or 'category not recorded'})"
                ),
                value=(
                    f"Victim {as_str(row.get('victim_account')) or 'not recorded'} · "
                    f"suspect {as_str(row.get('suspect_account')) or 'not named'} · "
                    f"disputed {format_inr(amount)} · "
                    f"UTR {as_str(row.get('disputed_utr')) or 'not supplied'} · "
                    f"reported {format_timestamp(reported)} · "
                    f"state {as_str(row.get('state')) or 'not recorded'}"
                ),
                source="ncrp",
                account_no=account.account_no,
                ack_no=ack_no or None,
                timestamp=reported,
                amount=amount,
            ))
        return facts

    def _shared_kyc_facts(self, account: DossierAccount) -> list[DossierFact]:
        facts: list[DossierFact] = []
        imei = as_str((account.metadata or {}).get("imei"))
        for peer in account.imei_peers:
            peer_no = as_str(peer.get("account_no"))
            if not imei or not peer_no:
                continue
            facts.append(DossierFact(
                category="SHARED_KYC",
                label="Shared IMEI in account master",
                value=(
                    f"IMEI {imei} is also recorded for A/C {peer_no} "
                    f"({as_str(peer.get('ifsc')) or 'IFSC not recorded'})"
                ),
                source="account_master",
                account_no=account.account_no,
            ))
        branch = as_str((account.metadata or {}).get("branch_code"))
        opened = as_date((account.metadata or {}).get("account_opening_date"))
        for peer in account.branch_peers:
            peer_no = as_str(peer.get("account_no"))
            peer_opened = as_date(peer.get("account_opening_date"))
            if not branch or opened is None or peer_opened is None or not peer_no:
                continue
            hours = abs((peer_opened - opened).total_seconds()) / 3600.0
            if hours > 48.0:
                continue
            facts.append(DossierFact(
                category="SHARED_KYC",
                label="Same-branch account opened within 48 h",
                value=(
                    f"A/C {peer_no} was opened at branch {branch} on "
                    f"{peer_opened.isoformat()}, {hours:.1f} h from this account"
                ),
                source="account_master",
                account_no=account.account_no,
            ))
        return facts

    # --- Section B --------------------------------------------------------
    def _indicators_for(self, account: DossierAccount) -> list[DossierIndicator]:
        indicators: list[DossierIndicator] = []
        risk = account.risk
        if risk is not None:
            for item in risk.explanations:
                indicators.append(DossierIndicator(
                    indicator=item.indicator,
                    method=("ISOLATION_FOREST_SHAP"
                            if item.indicator == "isolation_forest_anomaly"
                            else "DOMAIN_RULE"),
                    value=item.measured_value if item.measured_value is not None else 0.0,
                    unit=_INDICATOR_UNITS.get(item.indicator, "score"),
                    threshold=item.threshold,
                    points=item.points,
                    explanation=item.detail,
                    txn_ids=list(item.evidence_txn_ids),
                ))
        graph = account.graph_row or {}
        for indicator, label, unit in _TOPOLOGY_FIELDS:
            value = as_float(graph.get(indicator))
            if value <= 0:
                continue
            indicators.append(DossierIndicator(
                indicator=indicator, method="TOPOLOGY", value=value, unit=unit,
                explanation=(f"Graph topology measurement: {label.lower()} = "
                             f"{value:.4f}. Computed from the supplied ledger only."),
            ))
        trail = account.trail
        if trail is not None and trail.legs:
            indicators.append(DossierIndicator(
                indicator="fifo_taint_reached", method="FIFO_TAINT",
                value=trail.total_taint, unit="rupees",
                explanation=(
                    f"{format_inr(trail.total_taint)} of the "
                    f"{format_inr(trail.origin_amount)} traced from "
                    f"A/C {trail.origin_account} left it within {trail.max_hops} "
                    "hops under the first-in-first-out tainted-balance "
                    "assumption. Funds held by the account are not counted as "
                    "having moved."),
                txn_ids=[leg.txn_id for leg in trail.legs],
            ))
        return indicators

    def _risk_summary(self) -> DossierRiskSummary | None:
        candidates: list[tuple[DossierAccount, RiskAssessment]] = [
            (account, account.risk)
            for account in self._accounts
            if account.risk is not None
        ]
        if not candidates:
            return None
        _, risk = min(
            candidates, key=lambda pair: (-pair[1].risk_score, pair[0].account_no)
        )
        components = risk.domain_components
        weights = self._config.domain_weights
        sub_scores: list[DossierSubScore] = []
        for key in DOMAIN_KEYS:
            value = max(0.0, min(100.0, components.get(key, 0.0)))
            weight = weights.get(key, 0.0)
            sub_scores.append(DossierSubScore(
                domain=key, label=DOMAIN_LABELS[key], value=value,
                weight=weight, points=weight * value / 100.0,
            ))
        anomaly_weight = weights.get("anomaly", 0.0)
        sub_scores.append(DossierSubScore(
            domain="anomaly", label=DOMAIN_LABELS["anomaly"],
            value=risk.anomaly_score, weight=anomaly_weight,
            points=anomaly_weight * risk.anomaly_score / 100.0,
        ))
        return DossierRiskSummary(
            account_no=risk.account_no,
            risk_score=risk.risk_score,
            domain_score=risk.domain_score,
            anomaly_score=risk.anomaly_score,
            risk_band=risk.risk_band,
            top_archetype=risk.top_archetype,
            sub_scores=sub_scores,
        )

    # --- assembly ---------------------------------------------------------
    def build(self) -> InvestigationDossier:
        if self._built is not None:
            return self._built
        facts = [fact for account in self._accounts for fact in self._facts_for(account)]
        facts.sort(key=lambda item: (
            _FACT_ORDER[item.category],
            item.account_no or "",
            item.timestamp or datetime.min,
            item.txn_id or "",
            item.ack_no or "",
            item.label,
        ))
        dossier = InvestigationDossier(
            case_reference=self._config.case_reference,
            scope=self._config.scope,
            subject_accounts=[account.account_no for account in self._accounts],
            facts=facts,
            indicators=[
                indicator
                for account in self._accounts
                for indicator in self._indicators_for(account)
            ],
            risk_summary=self._risk_summary(),
            # Minimal valid requisition: Task 4 replaces this inline literal with
            # `requisition=self._requisition()` (bank-grouped, Section 94 BNSS).
            requisition=DossierRequisition(
                statute_reference=BNSS_SECTION_94,
                addressee="The Nodal Officer of each bank listed below",
                subject=(
                    f"Production of records under {BNSS_SECTION_94} — "
                    f"case {self._config.case_reference}"
                ),
                case_reference=self._config.case_reference,
                legal_note="Draft for review by the competent investigating officer.",
            ),
            disclaimer=DOSSIER_DISCLAIMER,
        )
        self._built = dossier
        return dossier


__all__ = [
    "DEFAULT_BANK_NAMES",
    "DossierAccount",
    "DossierConfig",
    "InvestigationDossierGenerator",
    "bank_name_for",
    "format_inr",
    "format_timestamp",
    "rejected_rows_csv",
]
