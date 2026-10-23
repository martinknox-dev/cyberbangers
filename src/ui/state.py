"""Pure view-model builders for the investigator workbench.

No Streamlit import may appear in this module: every function is a deterministic
transformation of Phase 1/2 artefacts into render-ready structures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from core.coerce import as_int
from core.contracts import (
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_SCHEMA,
    ComplaintLink,
    FeatureConfig,
    GraphConfig,
    IngestionReport,
    RiskAssessment,
    RiskConfig,
    WashCycle,
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


__all__ = [
    "IngestionTableView",
    "IngestionView",
    "LeaderboardRow",
    "LeaderboardView",
    "MappingRow",
    "PipelineBundle",
    "build_ingestion_view",
    "build_leaderboard",
    "run_pipeline",
]
