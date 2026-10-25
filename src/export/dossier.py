"""Deterministic case-evidence dossier and Section 94 BNSS requisition draft.

Pure Python: Polars/Pydantic models in, Markdown/HTML/JSON out. This module
never imports Streamlit, never reads the clock and never touches the network.
Section A reproduces supplied records; Section B carries analytical output and
is labelled as such; Section C is a draft requisition for legal review.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from core.coerce import as_date, as_datetime, as_float, as_str
from core.contracts import (
    BNSS_SECTION_94,
    BNSS_SECTION_106,
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
    RequisitionRequest,
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


def _md_text(value: object) -> str:
    """Escape one interpolated value so it cannot alter Markdown structure.

    Backslashes are escaped before pipes: a value ending in a backslash must
    not turn the following cell delimiter into an escaped (inert) pipe.
    Angle brackets are entity-escaped so a Markdown viewer that allows raw
    HTML cannot execute attacker-supplied markup.
    """
    text = (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
    )
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{_html_text(cell)}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_html_text(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_PRINT_CSS: str = (
    "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
    "max-width:960px;margin:2rem auto;padding:0 1rem;color:#111;line-height:1.45}"
    "h1{font-size:1.6rem}h2{margin-top:2rem;border-bottom:2px solid #111;padding-bottom:.25rem}"
    "table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.5rem 0 1rem}"
    "th,td{border:1px solid #999;padding:.35rem .5rem;text-align:left;vertical-align:top}"
    "th{background:#eee}.meta{color:#444;font-size:.85rem}"
    ".note{border-left:4px solid #b00;padding:.5rem .75rem;background:#fdf3f3;font-size:.85rem}"
    "@media print{body{margin:0}h2{page-break-after:avoid}table{page-break-inside:auto}}"
)


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
    primary_ack_no: str | None = None              # complaint dossiers: subject ACK


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
    max_kyc_peers: int = Field(default=25, ge=1)
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
            f"opened {opened.isoformat() if opened else 'not recorded'} · "
            f"mobile {as_str(metadata.get('mobile_no')) or 'not recorded'} · "
            f"IMEI {as_str(metadata.get('imei')) or 'not recorded'}"
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
        primary = account.primary_ack_no
        facts: list[DossierFact] = []
        for row in rows:
            ack_no = as_str(row.get("ack_no"))
            reported = as_datetime(row.get("reported_timestamp"))
            amount = as_float(row.get("disputed_amount"))
            related = primary is not None and ack_no != primary
            prefix = (
                "Related corroborating NCRP complaint" if related
                else "NCRP complaint"
            )
            facts.append(DossierFact(
                category="COMPLAINT",
                label=(
                    f"{prefix} {ack_no} "
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
        limit = self._config.max_kyc_peers
        if len(facts) > limit:
            total = len(facts)
            facts = facts[:limit]
            facts.append(DossierFact(
                category="SHARED_KYC", label="Shared-KYC listing truncated",
                value=(
                    f"Showing the first {limit} of {total} shared-KYC peer records "
                    f"for A/C {account.account_no}; bulk branch openings are "
                    "bounded here and remain available in the account master."
                ),
                source="generator", account_no=account.account_no,
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
            operational_role=risk.operational_role,
            retained_tainted_amount=risk.retained_tainted_amount,
            freeze_priority_score=risk.freeze_priority_score,
        )

    # --- Section C --------------------------------------------------------
    def _requisition(self) -> DossierRequisition:
        config = self._config
        groups: dict[str, dict[str, set[str]]] = {}
        names: dict[str, str] = {}
        for account in self._accounts:
            ifsc = as_str((account.metadata or {}).get("ifsc"))
            prefix = ifsc[:4].upper() if ifsc else "UNKNOWN"
            names.setdefault(prefix, bank_name_for(ifsc, config.bank_names))
            bucket = groups.setdefault(prefix, {"accounts": set(), "utrs": set()})
            bucket["accounts"].add(account.account_no)
            allowed = account.ledger_txn_ids
            for leg in account.transactions:
                txn_id = as_str(leg.get("txn_id"))
                if not txn_id or (allowed is not None and txn_id not in allowed):
                    continue
                bucket["utrs"].add(txn_id)
            for row in account.complaints:
                txn_id = as_str(row.get("disputed_utr"))
                if txn_id and (allowed is None or txn_id in allowed):
                    bucket["utrs"].add(txn_id)

        bank_requests = [
            RequisitionRequest(
                ifsc_prefix=prefix,
                bank_name=names[prefix],
                accounts=sorted(groups[prefix]["accounts"]),
                utrs=sorted(groups[prefix]["utrs"]),
                documents=list(config.requested_documents),
            )
            for prefix in sorted(groups)
        ]
        case = config.case_reference
        paragraphs = [
            (f"Whereas an investigation is being conducted in case {case} into "
             "offences relating to cheating and dishonest misappropriation "
             "committed through electronic payment channels, and the accounts "
             "and transaction references listed below are reasonably connected "
             "with that investigation;"),
            ("And whereas the records sought below are required for the purpose "
             "of the investigation and are expected to be in your custody or "
             "control in electronic form;"),
            (f"Now therefore, in exercise of the powers under {BNSS_SECTION_94}, "
             "you are hereby directed to cause production of the documents "
             "listed against your bank, in electronic form where available, "
             f"within {config.deadline_days} days of receipt of this "
             "requisition, duly certified by an authorised officer;"),
            ("This requisition is limited to the production of records. Any "
             "requirement of lien marking, freezing or seizure shall be pursued "
             f"separately through the CFCFRMS process and {BNSS_SECTION_106}, "
             "as applicable, and is not covered by this communication;"),
            ("A compliance report identifying the officer who certified the "
             "records, the date of production and any record that could not be "
             "produced (with reasons) shall accompany the response."),
        ]
        return DossierRequisition(
            statute_reference=BNSS_SECTION_94,
            addressee="The Nodal Officer of each bank listed below",
            subject=f"Production of records under {BNSS_SECTION_94} — case {case}",
            case_reference=case,
            paragraphs=paragraphs,
            bank_requests=bank_requests,
            legal_note=(
                "Draft for review by the competent investigating officer. This "
                "document is not legal advice, does not certify the records "
                "listed in Section A, and must be signed by an officer "
                "authorised to issue it."
            ),
        )

    # --- renderers --------------------------------------------------------
    def _generated_on_text(self) -> str:
        generated_on = self._config.generated_on
        return generated_on.isoformat() if generated_on is not None else "not recorded"

    def render_markdown(self, dossier: InvestigationDossier | None = None) -> str:
        data = dossier or self.build()
        lines = [
            f"# Case Evidence Dossier — {_md_text(data.case_reference)}", "",
            f"**Scope:** {_md_text(data.scope)}  ",
            (
                "**Subject accounts:** "
                f"{_md_text(', '.join(data.subject_accounts) or 'none')}  "
            ),
            f"**Generated on:** {_md_text(self._generated_on_text())}  ", "",
            "## Section A — Verified Evidentiary Facts", "",
            "> Records exactly as supplied by the source systems. Not a certified copy.", "",
            "| Category | Record | Detail | Source |", "| :--- | :--- | :--- | :--- |",
        ]
        for fact in data.facts:
            lines.append(
                f"| {_md_text(fact.category)} | {_md_text(fact.label)} | "
                f"{_md_text(fact.value)} | {_md_text(fact.source)} |"
            )
        lines += ["", "## Section B — Analytical & AI Risk Indicators", "",
                  "> Automated analysis. Investigative leads, not evidence.", ""]
        if data.risk_summary is not None:
            summary = data.risk_summary
            lines += [
                (
                    f"**Risk score:** {summary.risk_score:.2f}/100 "
                    f"({_md_text(summary.risk_band)}) · "
                    f"domain {summary.domain_score:.2f} · "
                    f"anomaly {summary.anomaly_score:.2f} · "
                    f"archetype: {_md_text(summary.top_archetype)}"
                ),
                (
                    f"**Operational role:** {_md_text(summary.operational_role)} · "
                    f"retained tainted funds {format_inr(summary.retained_tainted_amount)} · "
                    f"freeze priority {summary.freeze_priority_score:.1f}/100"
                ),
                "",
                "| Sub-score | Value | Weight | Points |",
                "| :--- | ---: | ---: | ---: |",
            ]
            for score in summary.sub_scores:
                lines.append(f"| {_md_text(score.label)} | {score.value:.1f} | "
                             f"{score.weight:.0f} | {score.points:.1f} |")
            lines.append("")
        lines += ["| Indicator | Method | Value | Explanation |",
                  "| :--- | :--- | ---: | :--- |"]
        for indicator in data.indicators:
            lines.append(f"| {_md_text(indicator.indicator)} | {_md_text(indicator.method)} | "
                         f"{indicator.value:.4f} | {_md_text(indicator.explanation)} |")
        lines += ["", "## Section C — Section 94 BNSS Requisition (Draft)", "",
                  f"**{_md_text(data.requisition.subject)}**", "",
                  f"To: {_md_text(data.requisition.addressee)}", ""]
        lines += [
            _md_text(paragraph) for paragraph in data.requisition.paragraphs
        ]
        for group in data.requisition.bank_requests:
            lines += [
                "",
                (
                    f"### {_md_text(group.bank_name)} "
                    f"(IFSC prefix {_md_text(group.ifsc_prefix)})"
                ),
                "",
                f"- Accounts: {_md_text(', '.join(group.accounts) or 'none')}",
                (
                    "- Transaction references: "
                    f"{_md_text(', '.join(group.utrs) or 'none')}"
                ),
                "- Documents required:",
            ]
            lines += [f"  - {_md_text(document)}" for document in group.documents]
        lines += ["", f"*{_md_text(data.requisition.legal_note)}*", "",
                  f"**Disclaimer:** {_md_text(data.disclaimer)}", ""]
        return "\n".join(lines)

    def render_html(self, dossier: InvestigationDossier | None = None) -> bytes:
        data = dossier or self.build()
        fact_rows: list[list[object]] = [
            [fact.category, fact.label, fact.value, fact.source] for fact in data.facts
        ]
        parts: list[str] = [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>Case Evidence Dossier — {_html_text(data.case_reference)}</title>",
            f"<style>{_PRINT_CSS}</style></head><body>",
            f"<h1>Case Evidence Dossier — {_html_text(data.case_reference)}</h1>",
            '<p class="meta">',
            f"<strong>Scope:</strong> {_html_text(data.scope)}<br>",
            (
                "<strong>Subject accounts:</strong> "
                f"{_html_text(', '.join(data.subject_accounts) or 'none')}<br>"
            ),
            f"<strong>Generated on:</strong> {_html_text(self._generated_on_text())}",
            "</p>",
            "<h2>Section A — Verified Evidentiary Facts</h2>",
            (
                '<p class="note">Records exactly as supplied by the source systems. '
                "Not a certified copy.</p>"
            ),
            _html_table(["Category", "Record", "Detail", "Source"], fact_rows),
            "<h2>Section B — Analytical &amp; AI Risk Indicators</h2>",
            '<p class="note">Automated analysis. Investigative leads, not evidence.</p>',
        ]
        if data.risk_summary is not None:
            summary = data.risk_summary
            parts.append(
                f"<p><strong>Risk score:</strong> {summary.risk_score:.2f}/100 "
                f"({_html_text(summary.risk_band)}) · domain "
                f"{summary.domain_score:.2f} · anomaly {summary.anomaly_score:.2f} · "
                f"archetype: {_html_text(summary.top_archetype)}</p>"
            )
            parts.append(
                f"<p><strong>Operational role:</strong> "
                f"{_html_text(summary.operational_role)} · retained tainted funds "
                f"{_html_text(format_inr(summary.retained_tainted_amount))} · "
                f"freeze priority {summary.freeze_priority_score:.1f}/100</p>"
            )
            parts.append(_html_table(
                ["Sub-score", "Value", "Weight", "Points"],
                [[score.label, f"{score.value:.1f}", f"{score.weight:.0f}",
                  f"{score.points:.1f}"] for score in summary.sub_scores],
            ))
        parts.append(_html_table(
            ["Indicator", "Method", "Value", "Explanation"],
            [[indicator.indicator, indicator.method, f"{indicator.value:.4f}",
              indicator.explanation] for indicator in data.indicators],
        ))
        requisition = data.requisition
        parts += [
            "<h2>Section C — Section 94 BNSS Requisition (Draft)</h2>",
            f"<p><strong>{_html_text(requisition.subject)}</strong></p>",
            f"<p>To: {_html_text(requisition.addressee)}</p>",
        ]
        parts += [f"<p>{_html_text(paragraph)}</p>" for paragraph in requisition.paragraphs]
        for group in requisition.bank_requests:
            parts.append(
                f"<h3>{_html_text(group.bank_name)} "
                f"(IFSC prefix {_html_text(group.ifsc_prefix)})</h3>"
            )
            parts.append(_html_table(
                ["Accounts", "Transaction references"],
                [[", ".join(group.accounts) or "none",
                  ", ".join(group.utrs) or "none"]],
            ))
            parts.append(_html_table(["Documents required"], [[doc] for doc in group.documents]))
        parts += [
            f"<p><em>{_html_text(requisition.legal_note)}</em></p>",
            (
                f'<p class="note"><strong>Disclaimer:</strong> '
                f"{_html_text(data.disclaimer)}</p>"
            ),
            "</body></html>",
        ]
        return "".join(parts).encode("utf-8")

    def render_json(self, dossier: InvestigationDossier | None = None) -> bytes:
        data = dossier or self.build()
        return data.model_dump_json(indent=2).encode("utf-8")

    # --- assembly ---------------------------------------------------------
    def build(self) -> InvestigationDossier:
        if self._built is not None:
            return self._built
        facts = [fact for account in self._accounts for fact in self._facts_for(account)]
        primary_ack = next(
            (
                account.primary_ack_no
                for account in self._accounts
                if account.primary_ack_no is not None
            ),
            None,
        )

        def _fact_key(item: DossierFact) -> tuple[object, ...]:
            # On complaint dossiers the subject ACK is listed before every
            # related/corroborating complaint, regardless of account or time.
            secondary = int(
                primary_ack is not None
                and item.category == "COMPLAINT"
                and item.ack_no != primary_ack
            )
            return (
                _FACT_ORDER[item.category],
                secondary,
                item.account_no or "",
                item.timestamp or datetime.min,
                item.txn_id or "",
                item.ack_no or "",
                item.label,
            )

        facts.sort(key=_fact_key)
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
            requisition=self._requisition(),
            disclaimer=DOSSIER_DISCLAIMER,
        )
        self._built = dossier
        return dossier


__all__ = [
    "DEFAULT_BANK_NAMES",
    "DossierAccount",
    "DossierConfig",
    "DossierFact",
    "DossierIndicator",
    "DossierRequisition",
    "DossierRiskSummary",
    "DossierScope",
    "DossierSubScore",
    "InvestigationDossier",
    "InvestigationDossierGenerator",
    "bank_name_for",
    "format_inr",
    "format_timestamp",
    "rejected_rows_csv",
]
