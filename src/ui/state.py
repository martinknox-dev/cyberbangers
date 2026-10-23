"""Pure view-model builders for the investigator workbench.

No Streamlit import may appear in this module: every function is a deterministic
transformation of Phase 1/2 artefacts into render-ready structures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from core.coerce import as_date, as_datetime, as_float, as_int, as_str
from core.contracts import (
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_SCHEMA,
    ComplaintLink,
    DossierFact,
    DossierIndicator,
    DossierScope,
    FeatureConfig,
    GraphConfig,
    IngestionReport,
    InvestigationDossier,
    MoneyTrail,
    RiskAssessment,
    RiskConfig,
    WashCycle,
)
from export.dossier import (
    DossierAccount,
    DossierConfig,
    InvestigationDossierGenerator,
)
from features.account_features import extract_account_features
from graph.analyzer import GraphAnalyzer
from risk.scorer import RiskScorer


@dataclass(frozen=True)
class PipelineBundle:
    transactions: pl.DataFrame
    accounts: pl.DataFrame
    complaints: pl.DataFrame
    features: pl.DataFrame
    graph: pl.DataFrame
    risk: pl.DataFrame
    assessments: tuple[RiskAssessment, ...]
    analyzer: GraphAnalyzer
    wash_cycles: tuple[WashCycle, ...]
    complaint_links: tuple[ComplaintLink, ...]
    reports: Mapping[str, IngestionReport]


def run_pipeline(
    transactions: pl.DataFrame,
    accounts: pl.DataFrame | None = None,
    complaints: pl.DataFrame | None = None,
    *,
    feature_config: FeatureConfig | None = None,
    graph_config: GraphConfig | None = None,
    risk_config: RiskConfig | None = None,
    reports: Mapping[str, IngestionReport] | None = None,
) -> PipelineBundle:
    features = extract_account_features(transactions, accounts, complaints, feature_config)
    analyzer = GraphAnalyzer(transactions, accounts, complaints, graph_config)
    graph = analyzer.topology_metrics()
    scorer = RiskScorer(risk_config)
    risk, assessments = scorer.score_with_assessments(features, graph)
    return PipelineBundle(
        transactions=transactions,
        accounts=(
            accounts if accounts is not None
            else pl.DataFrame(schema=ACCOUNT_METADATA_SCHEMA)
        ),
        complaints=(
            complaints if complaints is not None
            else pl.DataFrame(schema=COMPLAINT_SCHEMA)
        ),
        features=features,
        graph=graph,
        risk=risk,
        assessments=tuple(assessments),
        analyzer=analyzer,
        wash_cycles=tuple(analyzer.detect_wash_cycles()),
        complaint_links=tuple(analyzer.correlate_complaints()),
        reports=dict(reports or {}),
    )


@dataclass(frozen=True)
class MappingRow:
    raw_name: str
    canonical_name: str | None
    matched_synonym: str | None
    confidence: float


@dataclass(frozen=True)
class IngestionTableView:
    table: str
    source: str
    rows_total: int
    rows_accepted: int
    rows_rejected: int
    failure_counts: tuple[tuple[str, int], ...]
    mappings: tuple[MappingRow, ...]
    rejected_preview: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class IngestionView:
    tables: tuple[IngestionTableView, ...]
    total_rows: int
    total_accepted: int
    total_rejected: int


def build_ingestion_view(reports: Mapping[str, IngestionReport]) -> IngestionView:
    tables: list[IngestionTableView] = []
    for table in sorted(reports):
        report = reports[table]
        mappings = tuple(
            MappingRow(raw_name=item.raw_name, canonical_name=item.canonical_name,
                       matched_synonym=item.matched_synonym, confidence=item.confidence)
            for _, item in sorted(report.column_mapping.items())
        )
        preview = report.rejected_rows.head(20)
        columns = list(preview.columns)
        rejected_preview = tuple(
            tuple(str(row[column]) for column in columns)
            for row in preview.to_dicts()
        )
        tables.append(IngestionTableView(
            table=table, source=report.source or "not recorded",
            rows_total=report.rows_total, rows_accepted=report.rows_accepted,
            rows_rejected=report.rows_rejected,
            failure_counts=tuple(sorted(report.failure_counts.items())),
            mappings=mappings, rejected_preview=rejected_preview,
        ))
    return IngestionView(
        tables=tuple(tables),
        total_rows=sum(item.rows_total for item in tables),
        total_accepted=sum(item.rows_accepted for item in tables),
        total_rejected=sum(item.rows_rejected for item in tables),
    )


@dataclass(frozen=True)
class LeaderboardRow:
    account_no: str
    risk_score: float
    domain_score: float
    anomaly_score: float
    risk_band: str
    top_archetype: str
    complaint_count: int
    top_reason: str


@dataclass(frozen=True)
class LeaderboardView:
    rows: tuple[LeaderboardRow, ...]
    total_accounts: int
    flagged_accounts: int


_FLAGGED_BANDS = frozenset({"HIGH", "CRITICAL"})


def build_leaderboard(
    bundle: PipelineBundle,
    *,
    min_score: float = 0.0,
    band: str | None = None,
    archetype: str | None = None,
) -> LeaderboardView:
    reasons = {
        str(row["account_no"]): row.get("top_reasons") or []
        for row in bundle.risk.to_dicts()
    }
    complaints = {
        str(row["account_no"]): as_int(row.get("direct_complaint_count"))
        for row in bundle.features.select(
            "account_no", "direct_complaint_count"
        ).to_dicts()
    }
    rows: list[LeaderboardRow] = []
    for assessment in bundle.assessments:
        if assessment.risk_score < min_score:
            continue
        if band is not None and assessment.risk_band != band:
            continue
        if archetype is not None and assessment.top_archetype != archetype:
            continue
        top = reasons.get(assessment.account_no) or []
        rows.append(LeaderboardRow(
            account_no=assessment.account_no,
            risk_score=round(assessment.risk_score, 2),
            domain_score=round(assessment.domain_score, 2),
            anomaly_score=round(assessment.anomaly_score, 2),
            risk_band=assessment.risk_band,
            top_archetype=assessment.top_archetype,
            complaint_count=complaints.get(assessment.account_no, 0),
            top_reason=str(top[0]) if top else "",
        ))
    rows.sort(key=lambda row: (-row.risk_score, row.account_no))
    return LeaderboardView(
        rows=tuple(rows),
        total_accounts=len(bundle.assessments),
        flagged_accounts=sum(1 for a in bundle.assessments if a.risk_band in _FLAGGED_BANDS),
    )


def _row_for(frame: pl.DataFrame, account_no: str) -> Mapping[str, object] | None:
    if frame.height == 0:
        return None
    matched = frame.filter(pl.col("account_no") == account_no)
    return matched.to_dicts()[0] if matched.height else None


def _legs_for(
    transactions: pl.DataFrame, account_no: str
) -> tuple[dict[str, object], ...]:
    legs = transactions.filter(
        (pl.col("sender_account") == account_no)
        | (pl.col("receiver_account") == account_no)
    ).sort(["timestamp", "txn_id"])
    return tuple(legs.to_dicts())


def _complaints_for(
    complaints: pl.DataFrame, account_no: str
) -> tuple[dict[str, object], ...]:
    matched = complaints.filter(
        (pl.col("victim_account") == account_no)
        | (pl.col("suspect_account") == account_no)
    ).sort(["ack_no"])
    return tuple(matched.to_dicts())


def _imei_peers(
    accounts: pl.DataFrame, master: Mapping[str, object] | None
) -> tuple[dict[str, object], ...]:
    imei = as_str((master or {}).get("imei"))
    if not imei or accounts.height == 0:
        return ()
    peers = accounts.filter(
        (pl.col("imei") == imei)
        & (pl.col("account_no") != as_str((master or {}).get("account_no")))
    ).sort("account_no")
    return tuple(peers.to_dicts())


def _branch_peers(
    accounts: pl.DataFrame, master: Mapping[str, object] | None
) -> tuple[dict[str, object], ...]:
    if master is None:
        return ()
    branch = as_str(master.get("branch_code"))
    opened = as_date(master.get("account_opening_date"))
    if not branch or opened is None or accounts.height == 0:
        return ()
    peers: list[tuple[float, dict[str, object]]] = []
    for row in accounts.filter(pl.col("branch_code") == branch).to_dicts():
        if as_str(row.get("account_no")) == as_str(master.get("account_no")):
            continue
        peer_opened = as_date(row.get("account_opening_date"))
        if peer_opened is None:
            continue
        hours = abs((peer_opened - opened).total_seconds()) / 3600.0
        if hours <= 48.0:
            peers.append((hours, row))
    peers.sort(key=lambda item: (item[0], as_str(item[1].get("account_no"))))
    return tuple(row for _, row in peers)


def build_dossier_account(
    bundle: PipelineBundle,
    account_no: str,
    *,
    ledger_txn_ids: frozenset[str] | None = None,
    max_hops: int | None = None,
) -> DossierAccount:
    master = _row_for(bundle.accounts, account_no)
    assessment = next(
        (item for item in bundle.assessments if item.account_no == account_no), None
    )
    trail = (
        bundle.analyzer.trace_money_trail(account_no, max_hops=max_hops)
        if account_no in set(bundle.features["account_no"].to_list())
        else None
    )
    return DossierAccount(
        account_no=account_no, metadata=master, risk=assessment,
        feature_row=_row_for(bundle.features, account_no),
        graph_row=_row_for(bundle.graph, account_no),
        transactions=_legs_for(bundle.transactions, account_no),
        complaints=_complaints_for(bundle.complaints, account_no),
        trail=trail,
        cycles=tuple(
            cycle for cycle in bundle.wash_cycles
            if account_no in cycle.cycle_accounts
        ),
        imei_peers=_imei_peers(bundle.accounts, master),
        branch_peers=_branch_peers(bundle.accounts, master),
        ledger_txn_ids=ledger_txn_ids,
    )


def _dossier_config(scope: DossierScope, case_reference: str) -> DossierConfig:
    return DossierConfig(scope=scope, case_reference=case_reference)


def build_account_dossier(
    bundle: PipelineBundle, account_no: str
) -> InvestigationDossier:
    return InvestigationDossierGenerator(
        [build_dossier_account(bundle, account_no)],
        _dossier_config("ACCOUNT", f"PG-AC-{account_no}"),
    ).build()


def build_cluster_dossier(
    bundle: PipelineBundle, account_nos: Sequence[str]
) -> InvestigationDossier:
    accounts = [
        build_dossier_account(bundle, number) for number in sorted(set(account_nos))
    ]
    return InvestigationDossierGenerator(
        accounts,
        _dossier_config(
            "CLUSTER", "PG-CL-" + "-".join(sorted(set(account_nos))[:3])
        ),
    ).build()


def build_complaint_dossier(
    bundle: PipelineBundle, ack_no: str, *, max_hops: int = 3
) -> InvestigationDossier:
    trail = bundle.analyzer.trace_complaint_trail(ack_no, max_hops=max_hops)
    involved: set[str] = set()
    txn_ids: set[str] = {leg.txn_id for leg in trail.legs}
    if trail.origin_account:
        involved.add(trail.origin_account)
    for leg in trail.legs:
        involved.add(leg.sender_account)
        involved.add(leg.receiver_account)
    rows = bundle.complaints.filter(pl.col("ack_no") == ack_no).to_dicts()
    if rows:
        row = rows[0]
        for key in ("victim_account", "suspect_account"):
            value = as_str(row.get(key))
            if value:
                involved.add(value)
        disputed = as_str(row.get("disputed_utr"))
        if disputed:
            txn_ids.add(disputed)
    else:
        return InvestigationDossierGenerator(
            [], _dossier_config("COMPLAINT", f"PG-ACK-{ack_no}")
        ).build()
    scoped = frozenset(txn_ids)
    accounts = [
        build_dossier_account(bundle, number, ledger_txn_ids=scoped)
        for number in sorted(involved)
    ]
    return InvestigationDossierGenerator(
        accounts, _dossier_config("COMPLAINT", f"PG-ACK-{ack_no}")
    ).build()


@dataclass(frozen=True)
class SubScoreRow:
    domain: str
    label: str
    value: float
    weight: float
    points: float


@dataclass(frozen=True)
class TimelineRow:
    timestamp: datetime
    direction: str          # "IN" | "OUT"
    counterparty: str
    amount: float
    txn_id: str
    payment_mode: str


@dataclass(frozen=True)
class AccountDetailView:
    account_no: str
    metadata: Mapping[str, str]
    risk: RiskAssessment | None
    sub_scores: tuple[SubScoreRow, ...]
    timeline: tuple[TimelineRow, ...]
    facts: tuple[DossierFact, ...]
    indicators: tuple[DossierIndicator, ...]
    dossier: InvestigationDossier


def build_account_detail(
    bundle: PipelineBundle, account_no: str
) -> AccountDetailView:
    dossier = build_account_dossier(bundle, account_no)
    account = build_dossier_account(bundle, account_no)
    sub_scores: tuple[SubScoreRow, ...] = ()
    if dossier.risk_summary is not None:
        sub_scores = tuple(
            SubScoreRow(domain=score.domain, label=score.label, value=score.value,
                        weight=score.weight, points=score.points)
            for score in dossier.risk_summary.sub_scores
        )
    timeline: list[TimelineRow] = []
    for leg in account.transactions:
        outbound = as_str(leg.get("sender_account")) == account_no
        moment = as_datetime(leg.get("timestamp"))
        if moment is None:
            continue
        timeline.append(TimelineRow(
            timestamp=moment,
            direction="OUT" if outbound else "IN",
            counterparty=as_str(
                leg.get("receiver_account") if outbound
                else leg.get("sender_account")
            ),
            amount=as_float(leg.get("amount")), txn_id=as_str(leg.get("txn_id")),
            payment_mode=as_str(leg.get("payment_mode")),
        ))
    timeline.sort(key=lambda row: (row.timestamp, row.txn_id))
    metadata = {
        key: as_str(value)
        for key, value in (account.metadata or {}).items()
        if isinstance(value, str)
    }
    return AccountDetailView(
        account_no=account_no, metadata=metadata, risk=account.risk,
        sub_scores=sub_scores, timeline=tuple(timeline),
        facts=tuple(dossier.facts), indicators=tuple(dossier.indicators),
        dossier=dossier,
    )


_TERMINAL_MODES = frozenset({"AEPS", "ATM", "CASH"})


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    kind: str                 # "ACCOUNT" | "TERMINAL"
    risk_score: float
    risk_band: str
    community: int
    caption: str


@dataclass(frozen=True)
class GraphEdge:
    id: str
    source: str
    target: str
    txn_id: str
    amount: float
    taint_amount: float
    hop: int
    timestamp: datetime
    payment_mode: str


@dataclass(frozen=True)
class GraphView:
    origin_label: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    elements: dict[str, list[dict[str, dict[str, str]]]]


def _node_lookup(bundle: PipelineBundle) -> dict[str, tuple[float, str, int]]:
    risk = {str(row["account_no"]): row for row in bundle.risk.to_dicts()}
    graph = {str(row["account_no"]): row for row in bundle.graph.to_dicts()}
    return {
        account: (
            as_float(risk.get(account, {}).get("risk_score")),
            as_str(risk.get(account, {}).get("risk_band")) or "UNSCORED",
            as_int(graph.get(account, {}).get("louvain_community_id")),
        )
        for account in set(risk) | set(graph)
    }


def _resolve_trail(bundle: PipelineBundle, start: str, max_hops: int) -> MoneyTrail:
    """Route by complaint membership, never by an id prefix.

    Real NCRP acknowledgement numbers are not guaranteed to start with "ACK",
    and an account id may. Membership in the complaint frame is the only
    reliable signal that the reference names a complaint.
    """
    ack_nos = set(bundle.complaints["ack_no"].to_list())
    if start in ack_nos:
        return bundle.analyzer.trace_complaint_trail(start, max_hops=max_hops)
    return bundle.analyzer.trace_money_trail(start, max_hops=max_hops)


def build_trail_graph(
    bundle: PipelineBundle, *, start: str, max_hops: int = 3
) -> GraphView:
    trail = _resolve_trail(bundle, start, max_hops)
    lookup = _node_lookup(bundle)
    legs = sorted(trail.legs, key=lambda leg: (leg.hop, leg.timestamp, leg.txn_id))
    terminal_nodes = {
        leg.receiver_account for leg in legs if leg.payment_mode in _TERMINAL_MODES
    }
    order: list[str] = []
    if trail.origin_account and legs:
        order.append(trail.origin_account)
    for leg in legs:
        for account in (leg.sender_account, leg.receiver_account):
            if account not in order:
                order.append(account)
    nodes = tuple(
        GraphNode(
            id=account,
            label=account,
            kind="TERMINAL" if account in terminal_nodes else "ACCOUNT",
            risk_score=lookup.get(account, (0.0, "UNSCORED", 0))[0],
            risk_band=lookup.get(account, (0.0, "UNSCORED", 0))[1],
            community=lookup.get(account, (0.0, "UNSCORED", 0))[2],
            caption=f"{account} · {lookup.get(account, (0.0, 'UNSCORED', 0))[1]}",
        )
        for account in order
    )
    edges = tuple(
        GraphEdge(
            id=leg.txn_id, source=leg.sender_account, target=leg.receiver_account,
            txn_id=leg.txn_id, amount=leg.amount, taint_amount=leg.taint_amount,
            hop=leg.hop, timestamp=leg.timestamp, payment_mode=leg.payment_mode,
        )
        for leg in legs
    )
    elements: dict[str, list[dict[str, dict[str, str]]]] = {
        "nodes": [
            {"data": {"id": node.id, "label": node.kind.lower(), "caption": node.caption,
                      "band": node.risk_band}}
            for node in nodes
        ],
        "edges": [
            {"data": {"id": edge.id, "source": edge.source, "target": edge.target,
                      "label": f"hop {edge.hop}",
                      "caption": (f"{edge.txn_id} · ₹{edge.amount:.2f} · "
                                  f"tainted ₹{edge.taint_amount:.2f}")}}
            for edge in edges
        ],
    }
    origin_label = (
        f"UTR {trail.origin_txn_id}" if trail.origin_txn_id
        else f"A/C {trail.origin_account}"
    )
    return GraphView(origin_label=origin_label, nodes=nodes, edges=edges, elements=elements)


@dataclass(frozen=True)
class CycleRow:
    accounts: tuple[str, ...]        # cycle order, starting at the smallest id
    txn_ids: tuple[str, ...]
    total_amount: float
    max_decay_ratio: float
    start_timestamp: datetime


@dataclass(frozen=True)
class CycleView:
    rows: tuple[CycleRow, ...]       # sorted by (-max_decay_ratio, accounts)


def build_cycle_view(bundle: PipelineBundle) -> CycleView:
    rows = [
        CycleRow(
            accounts=tuple(cycle.cycle_accounts),
            txn_ids=tuple(cycle.txn_ids),
            total_amount=round(cycle.total_amount, 2),
            max_decay_ratio=cycle.max_decay_ratio,
            start_timestamp=cycle.start_timestamp,
        )
        for cycle in bundle.wash_cycles
    ]
    rows.sort(key=lambda row: (-row.max_decay_ratio, row.accounts))
    return CycleView(rows=tuple(rows))


@dataclass(frozen=True)
class CollusionGroup:
    basis: str                       # "IMEI" | "BRANCH"
    key: str                         # the IMEI, or the branch code
    accounts: tuple[str, ...]        # sorted
    opening_dates: tuple[str, ...]   # ISO dates or "not recorded"


@dataclass(frozen=True)
class CollusionView:
    groups: tuple[CollusionGroup, ...]   # sorted by (basis, key)


def _opening_dates(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    dates: list[str] = []
    for row in rows:
        opened = as_date(row.get("account_opening_date"))
        dates.append(opened.isoformat() if opened is not None else "not recorded")
    return tuple(dates)


def _imei_groups(accounts: pl.DataFrame) -> list[CollusionGroup]:
    groups: list[CollusionGroup] = []
    if accounts.height == 0 or "imei" not in accounts.columns:
        return groups
    rows = accounts.filter(
        pl.col("imei").is_not_null() & (pl.col("imei") != "")
    ).sort(["imei", "account_no"])
    for key, subset in rows.group_by("imei", maintain_order=True):
        imei = as_str(key[0] if isinstance(key, tuple) else key)
        members = subset.sort("account_no")
        if len(members) < 2:
            continue
        groups.append(CollusionGroup(
            basis="IMEI", key=imei,
            accounts=tuple(as_str(item.get("account_no")) for item in members.to_dicts()),
            opening_dates=_opening_dates(members.to_dicts()),
        ))
    return groups


def _branch_groups(accounts: pl.DataFrame) -> list[CollusionGroup]:
    groups: list[CollusionGroup] = []
    if accounts.height == 0 or "branch_code" not in accounts.columns:
        return groups
    rows = accounts.filter(
        pl.col("branch_code").is_not_null() & (pl.col("branch_code") != "")
        & pl.col("account_opening_date").is_not_null()
    ).sort(["branch_code", "account_opening_date", "account_no"])
    for key, subset in rows.group_by("branch_code", maintain_order=True):
        branch = as_str(key[0] if isinstance(key, tuple) else key)
        members = subset.sort(["account_opening_date", "account_no"]).to_dicts()
        cluster: list[dict[str, object]] = []
        for row in members:
            opened = as_date(row.get("account_opening_date"))
            if opened is None:
                continue
            if cluster:
                first = as_date(cluster[0].get("account_opening_date"))
                if first is None or (opened - first).total_seconds() > 48 * 3600:
                    if len(cluster) >= 2:
                        groups.append(_branch_group(branch, cluster))
                    cluster = []
            cluster.append(row)
        if len(cluster) >= 2:
            groups.append(_branch_group(branch, cluster))
    return groups


def _branch_group(branch: str, cluster: Sequence[Mapping[str, object]]) -> CollusionGroup:
    ordered = sorted(cluster, key=lambda row: as_str(row.get("account_no")))
    return CollusionGroup(
        basis="BRANCH", key=branch,
        accounts=tuple(as_str(row.get("account_no")) for row in ordered),
        opening_dates=_opening_dates(ordered),
    )


def build_collusion_view(bundle: PipelineBundle) -> CollusionView:
    groups = _imei_groups(bundle.accounts) + _branch_groups(bundle.accounts)
    groups.sort(key=lambda group: (group.basis, group.key))
    return CollusionView(groups=tuple(groups))


@dataclass(frozen=True)
class TerminalRow:
    account_no: str
    terminal_inflow: float
    cashout_count: int
    upstream_accounts: tuple[str, ...]
    terminal_chain_count: int


@dataclass(frozen=True)
class TerminalView:
    rows: tuple[TerminalRow, ...]    # sorted by (-terminal_inflow, account_no)


def build_terminal_view(bundle: PipelineBundle) -> TerminalView:
    terminal = bundle.transactions.filter(
        pl.col("payment_mode").is_in(sorted(_TERMINAL_MODES))
    )
    if terminal.height == 0:
        return TerminalView(rows=())
    grouped = terminal.group_by("receiver_account").agg(
        pl.col("amount").sum().round(2).alias("inflow"),
        pl.col("sender_account").unique().sort().alias("senders"),
    )
    features = {
        str(row["account_no"]): row for row in bundle.features.to_dicts()
    }
    graph = {str(row["account_no"]): row for row in bundle.graph.to_dicts()}
    rows: list[TerminalRow] = []
    for item in grouped.to_dicts():
        account = as_str(item.get("receiver_account"))
        senders = item.get("senders")
        rows.append(TerminalRow(
            account_no=account,
            terminal_inflow=as_float(item.get("inflow")),
            cashout_count=as_int(features.get(account, {}).get("aeps_cashout_inflow_count")),
            upstream_accounts=tuple(
                sorted(as_str(sender) for sender in senders)
            ) if isinstance(senders, list) else (),
            terminal_chain_count=as_int(
                graph.get(account, {}).get("terminal_chain_count")
            ),
        ))
    rows.sort(key=lambda row: (-row.terminal_inflow, row.account_no))
    return TerminalView(rows=tuple(rows))


@dataclass(frozen=True)
class CorrelationRow:
    ack_a: str
    state_a: str
    ack_b: str
    state_b: str
    link_basis: str
    shared_accounts: tuple[str, ...]
    shared_devices: tuple[str, ...]
    shared_terminals: tuple[str, ...]
    interstate: bool


@dataclass(frozen=True)
class CorrelationHotspot:
    key: str
    ack_nos: tuple[str, ...]         # sorted


@dataclass(frozen=True)
class CorrelationView:
    rows: tuple[CorrelationRow, ...]      # sorted by (ack_a, ack_b)
    hotspots: tuple[CorrelationHotspot, ...]  # sorted by (-len(ack_nos), key)


def build_correlation_view(bundle: PipelineBundle) -> CorrelationView:
    rows = [
        CorrelationRow(
            ack_a=link.ack_no_a, state_a=link.state_a,
            ack_b=link.ack_no_b, state_b=link.state_b,
            link_basis=link.link_basis,
            shared_accounts=tuple(sorted(link.shared_accounts)),
            shared_devices=tuple(sorted(link.shared_device_ids)),
            shared_terminals=tuple(sorted(link.shared_terminal_accounts)),
            interstate=link.state_a != link.state_b,
        )
        for link in bundle.complaint_links
    ]
    rows.sort(key=lambda row: (row.ack_a, row.ack_b))
    hotspot_acks: dict[str, set[str]] = {}
    for link in bundle.complaint_links:
        keys = [*link.shared_accounts, *link.shared_terminal_accounts]
        for key in keys:
            hotspot_acks.setdefault(key, set()).update(
                {link.ack_no_a, link.ack_no_b}
            )
    hotspots = [
        CorrelationHotspot(key=key, ack_nos=tuple(sorted(acks)))
        for key, acks in hotspot_acks.items()
    ]
    hotspots.sort(key=lambda item: (-len(item.ack_nos), item.key))
    return CorrelationView(rows=tuple(rows), hotspots=tuple(hotspots))


def dossier_markdown(dossier: InvestigationDossier) -> str:
    return InvestigationDossierGenerator([]).render_markdown(dossier)


def dossier_html(dossier: InvestigationDossier) -> bytes:
    return InvestigationDossierGenerator([]).render_html(dossier)


def dossier_json(dossier: InvestigationDossier) -> bytes:
    return InvestigationDossierGenerator([]).render_json(dossier)


__all__ = [
    "AccountDetailView",
    "CollusionGroup",
    "CollusionView",
    "CorrelationHotspot",
    "CorrelationRow",
    "CorrelationView",
    "CycleRow",
    "CycleView",
    "GraphEdge",
    "GraphNode",
    "GraphView",
    "IngestionTableView",
    "IngestionView",
    "LeaderboardRow",
    "LeaderboardView",
    "MappingRow",
    "PipelineBundle",
    "SubScoreRow",
    "TerminalRow",
    "TerminalView",
    "TimelineRow",
    "build_account_detail",
    "build_account_dossier",
    "build_cluster_dossier",
    "build_collusion_view",
    "build_complaint_dossier",
    "build_correlation_view",
    "build_cycle_view",
    "build_dossier_account",
    "build_ingestion_view",
    "build_leaderboard",
    "build_terminal_view",
    "build_trail_graph",
    "dossier_html",
    "dossier_json",
    "dossier_markdown",
    "run_pipeline",
]
