"""Phase 3 tests: workbench view models, dossiers and the Streamlit app."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from core.contracts import (
    ACCOUNT_FEATURE_SCHEMA,
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_SCHEMA,
    DOMAIN_KEYS,
    GRAPH_INDICATOR_SCHEMA,
    REQUISITION_DOCUMENTS,
    TRANSACTION_SCHEMA,
    IngestionReport,
    InvestigationDossier,
    MoneyTrail,
    MoneyTrailLeg,
    RiskAssessment,
)
from export.dossier import (
    DossierAccount,
    DossierConfig,
    InvestigationDossierGenerator,
    format_inr,
    rejected_rows_csv,
)
from ingestion.mapper import DatasetIngester
from risk.scorer import RiskScorer
from simulator.generator import SyndicateDataSimulator
from ui import state
from ui.state import (
    build_account_detail,
    build_account_dossier,
    build_cluster_dossier,
    build_collusion_view,
    build_complaint_dossier,
    build_correlation_view,
    build_cycle_view,
    build_ingestion_view,
    build_leaderboard,
    build_terminal_view,
    build_trail_graph,
    run_pipeline,
)

BASE = datetime(2024, 6, 1, 10, 0)
APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"


def _feature_row(account_no: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in ACCOUNT_FEATURE_SCHEMA}
    row.update(
        account_no=account_no, txn_count=4, inflow_count=2, outflow_count=2,
        inflow_total=100_000.0, outflow_total=0.0, net_flow=0.0,
        unique_senders=1, unique_receivers=2, fan_in_fan_out_asymmetry=1.0,
        pass_through_ratio_15m=0.95, median_retention_minutes=3.0,
        smurf_bulk_inflow_count=1, smurf_split_count=10, large_payout_count=0,
        burst_24h_max_total=0.0, dormancy_days_before_burst=200.0,
        dormancy_burst_score=1.0, shared_device_account_count=0,
        branch_cohort_48h_count=0, aeps_cashout_inflow_count=0,
        direct_complaint_count=0, pass_through_utrs=[], smurf_utrs=[],
        first_txn_timestamp=BASE, last_txn_timestamp=BASE,
    )
    row.update(overrides)
    return row


def _graph_row(account_no: str) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in GRAPH_INDICATOR_SCHEMA}
    row.update(
        account_no=account_no, pagerank=0.1, betweenness_centrality=0.0,
        louvain_community_id=1, cycle_count=0, cycle_max_amount=0.0,
        cycle_txn_ids=[], complaint_link_count=0, complaint_ack_nos=[],
        trail_taint_max=0.0, trail_hops_max=0, terminal_chain_count=0,
    )
    return row


def test_score_with_assessments_returns_the_same_frame_as_score() -> None:
    features = pl.DataFrame(
        [_feature_row(f"A{i:02d}") for i in range(12)],
        schema=ACCOUNT_FEATURE_SCHEMA,
    )
    graph = pl.DataFrame([_graph_row(f"A{i:02d}") for i in range(12)],
                         schema=GRAPH_INDICATOR_SCHEMA)
    scorer = RiskScorer()
    frame, assessments = scorer.score_with_assessments(features, graph)
    assert frame.equals(scorer.score(features, graph))
    assert len(assessments) == 12


def test_assessment_exposes_six_domain_components_and_an_archetype() -> None:
    features = pl.DataFrame([_feature_row("MULE")], schema=ACCOUNT_FEATURE_SCHEMA)
    graph = pl.DataFrame([_graph_row("MULE")], schema=GRAPH_INDICATOR_SCHEMA)
    _, assessments = RiskScorer().score_with_assessments(features, graph)
    assessment = assessments[0]
    assert set(assessment.domain_components) == set(DOMAIN_KEYS)
    assert all(0.0 <= value <= 100.0 for value in assessment.domain_components.values())
    # 0.95 pass-through ratio with 3-minute retention saturates the rule
    assert assessment.domain_components["pass_through"] == 100.0
    assert assessment.top_archetype == "Rapid pass-through mule"


def _account_row(account_no: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "account_no": account_no, "ifsc": "UTIB0000123",
        "account_opening_date": date(2024, 5, 1), "branch_code": "BR042",
        "mobile_no": "9876543210", "imei": "351104510123456",
        "kyc_pincode": "800001", "account_type": "SAVINGS",
    }
    row.update(overrides)
    return row


def _txn(txn_id: str, sender: str, receiver: str, amount: float,
         minutes: int, mode: str = "UPI") -> dict[str, object]:
    return {
        "txn_id": txn_id, "sender_account": sender, "sender_ifsc": "UTIB0000123",
        "receiver_account": receiver, "receiver_ifsc": "SBIN0000421",
        "amount": amount, "timestamp": BASE + timedelta(minutes=minutes),
        "payment_mode": mode, "device_id": "DEV-1", "ip_address": "10.0.0.1",
    }


def _generator(account_no: str = "MULE", **kwargs: object) -> InvestigationDossierGenerator:
    account = DossierAccount(
        account_no=account_no,
        metadata=_account_row(account_no),
        transactions=(
            _txn("U1", "VICTIM", account_no, 100.0, 0),
            _txn("U2", account_no, "MULE2", 90.0, 5),
            _txn("U3", account_no, "MULE2", 10.0, 6, mode="AEPS"),
        ),
        complaints=(
            {"ack_no": "ACK1", "victim_account": "VICTIM", "suspect_account": account_no,
             "fraud_category": "UPI_FRAUD", "reported_timestamp": BASE + timedelta(hours=2),
             "disputed_amount": 100.0, "disputed_utr": "U1", "state": "Bihar"},
        ),
        imei_peers=(_account_row("PEER1", branch_code="BR099"),),
        branch_peers=(_account_row("PEER2", imei="359999999999999",
                                   account_opening_date=date(2024, 5, 2)),),
    )
    return InvestigationDossierGenerator([account], DossierConfig(**kwargs))


def test_section_a_lists_ledger_account_ncrp_and_shared_kyc_facts() -> None:
    dossier = _generator().build()
    categories = [fact.category for fact in dossier.facts]
    assert categories == sorted(
        categories, key=["ACCOUNT", "TRANSACTION", "COMPLAINT", "SHARED_KYC"].index
    )
    assert {"ACCOUNT", "TRANSACTION", "COMPLAINT", "SHARED_KYC"} <= set(categories)

    utrs = [fact.txn_id for fact in dossier.facts if fact.category == "TRANSACTION"]
    assert utrs == ["U1", "U2", "U3"]                       # chronological
    inflow = next(fact for fact in dossier.facts if fact.txn_id == "U1")
    assert inflow.label == "Inflow from VICTIM (UTIB0000123)"
    assert "₹100.00" in inflow.value and "UTR U1" in inflow.value
    assert inflow.source == "ledger"

    ack = next(fact for fact in dossier.facts if fact.ack_no == "ACK1")
    assert ack.category == "COMPLAINT" and ack.source == "ncrp"
    assert "Bihar" in ack.value and "₹100.00" in ack.value

    imei = next(fact for fact in dossier.facts if fact.label == "Shared IMEI in account master")
    assert "351104510123456" in imei.value and "PEER1" in imei.value
    branch = next(fact for fact in dossier.facts if fact.label.startswith("Same-branch account"))
    assert "PEER2" in branch.value and "BR042" in branch.value


def test_missing_metadata_and_imei_are_stated_not_invented() -> None:
    account = DossierAccount(account_no="GHOST", transactions=())
    dossier = InvestigationDossierGenerator([account]).build()
    record = next(fact for fact in dossier.facts if fact.category == "ACCOUNT")
    assert record.value == "No account master record supplied for this account"
    assert record.source == "generator"
    assert not any(fact.category == "SHARED_KYC" for fact in dossier.facts)


def test_ledger_is_bounded_and_says_so() -> None:
    legs = tuple(_txn(f"T{i:03d}", "VICTIM", "MULE", 1.0, i) for i in range(12))
    account = DossierAccount(account_no="MULE", metadata=_account_row("MULE"),
                             transactions=legs)
    dossier = InvestigationDossierGenerator(
        [account], DossierConfig(max_ledger_rows=5)
    ).build()
    ledger = [fact for fact in dossier.facts if fact.category == "TRANSACTION"]
    assert len(ledger) == 6                       # 5 legs + the truncation notice
    truncated = next(fact for fact in ledger if fact.label == "Ledger listing truncated")
    assert "5 of 12" in truncated.value
    assert truncated.source == "generator"


def test_unknown_account_selection_values_never_invent_optional_fields() -> None:
    account = DossierAccount(
        account_no="MULE",
        metadata=_account_row("MULE", ifsc=None, imei=None, account_opening_date=None),
        transactions=(_txn("U1", "VICTIM", "MULE", 50.0, 0),),
    )
    dossier = InvestigationDossierGenerator([account]).build()
    text = " ".join(fact.value for fact in dossier.facts)
    assert "IFSC not recorded" in text or "IFSC not supplied" in text
    assert "351104510123456" not in text


def test_rejected_rows_csv_is_deterministic_and_keeps_reasons() -> None:
    rejected = pl.DataFrame(
        {"utr": ["A", "B"], "amt": ["x", "10"],
         "rejection_reason": ["unparseable_amount", "negative_amount"]}
    )
    report = IngestionReport(rows_total=2, rows_accepted=0, rows_rejected=2,
                             rejected_rows=rejected)
    payload = rejected_rows_csv(report)
    assert payload == rejected_rows_csv(report)
    text = payload.decode("utf-8")
    assert text.splitlines()[0] == "utr,amt,rejection_reason"
    assert "unparseable_amount" in text


def test_format_inr_uses_indian_grouping() -> None:
    assert format_inr(0) == "₹0.00"
    assert format_inr(999) == "₹999.00"
    assert format_inr(1_000) == "₹1,000.00"
    assert format_inr(100_000) == "₹1,00,000.00"
    assert format_inr(12_345_678.9) == "₹1,23,45,678.90"


def _scored_assessment(account_no: str = "MULE") -> RiskAssessment:
    """A real assessment from a fitted model (the anomaly line needs >= 10 rows)."""
    features = pl.DataFrame(
        [_feature_row(account_no)]
        + [
            _feature_row(f"B{i:02d}", pass_through_ratio_15m=0.05,
                         median_retention_minutes=600.0, smurf_bulk_inflow_count=0,
                         smurf_split_count=0, dormancy_days_before_burst=2.0)
            for i in range(11)
        ],
        schema=ACCOUNT_FEATURE_SCHEMA,
    )
    graph = pl.DataFrame(
        [_graph_row(account_no)] + [_graph_row(f"B{i:02d}") for i in range(11)],
        schema=GRAPH_INDICATOR_SCHEMA,
    )
    _, assessments = RiskScorer().score_with_assessments(features, graph)
    return next(item for item in assessments if item.account_no == account_no)


def test_section_b_separates_indicators_from_facts() -> None:
    account = DossierAccount(
        account_no="MULE",
        metadata=_account_row("MULE"),
        risk=_scored_assessment("MULE"),
        feature_row=_feature_row("MULE"),
        graph_row=_graph_row("MULE"),
        transactions=(_txn("U1", "VICTIM", "MULE", 100.0, 0),),
        trail=MoneyTrail(
            origin_account="MULE", origin_amount=100.0, origin_txn_id="U1",
            max_hops=3, total_taint=100.0, account_taint={"MULE2": 90.0},
            legs=[MoneyTrailLeg(hop=1, txn_id="U2", sender_account="MULE",
                                receiver_account="MULE2", amount=90.0,
                                timestamp=BASE, payment_mode="UPI", taint_amount=90.0)],
        ),
    )
    generator = InvestigationDossierGenerator([account])
    dossier = generator.build()

    methods = {indicator.method for indicator in dossier.indicators}
    assert "DOMAIN_RULE" in methods and "FIFO_TAINT" in methods
    assert "ISOLATION_FOREST_SHAP" in methods
    assert all(indicator.explanation for indicator in dossier.indicators)
    # nothing analytical leaks into the evidentiary section
    fact_text = " ".join(f"{f.label} {f.value}" for f in dossier.facts).lower()
    assert "shap" not in fact_text
    assert "taint" not in fact_text
    assert "risk" not in fact_text

    assert dossier.risk_summary is not None
    assert len(dossier.risk_summary.sub_scores) == 7          # 6 domains + anomaly
    assert {score.domain for score in dossier.risk_summary.sub_scores} >= set(DOMAIN_KEYS)


def test_fifo_taint_indicator_states_the_assumption() -> None:
    account = DossierAccount(
        account_no="MULE", metadata=_account_row("MULE"),
        trail=MoneyTrail(
            origin_account="MULE", origin_amount=100.0, origin_txn_id="U1",
            max_hops=3, total_taint=90.0, account_taint={"MULE2": 90.0},
            legs=[MoneyTrailLeg(hop=1, txn_id="U2", sender_account="MULE",
                                receiver_account="MULE2", amount=90.0,
                                timestamp=BASE, payment_mode="UPI", taint_amount=90.0)],
        ),
    )
    dossier = InvestigationDossierGenerator([account]).build()
    taint = next(i for i in dossier.indicators if i.method == "FIFO_TAINT")
    assert taint.unit == "rupees"
    assert taint.value == 90.0
    assert "FIFO" in taint.explanation or "first-in-first-out" in taint.explanation
    assert taint.txn_ids == ["U2"]


def test_account_without_risk_still_produces_an_empty_section_b() -> None:
    dossier = InvestigationDossierGenerator([DossierAccount(account_no="GHOST")]).build()
    assert dossier.indicators == []
    assert dossier.risk_summary is None


def test_requisition_cites_section_94_and_keeps_lien_marking_separate() -> None:
    dossier = _generator().build()
    req = dossier.requisition
    assert req.statute_reference == (
        "Section 94 of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)"
    )
    assert "Nodal Officer" in req.addressee
    body = " ".join(req.paragraphs)
    assert "Section 94" in body
    assert "Section 106" in body and "CFCFRMS" in body
    assert "separate" in body.lower()
    assert "not legal advice" in req.legal_note.lower()
    assert dossier.disclaimer.startswith("Section A reproduces records")


def test_requisition_groups_accounts_by_bank_and_lists_only_their_utrs() -> None:
    axis = DossierAccount(
        account_no="A1", metadata=_account_row("A1", ifsc="UTIB0000123"),
        # both legs touch A1: it receives U-AXIS and sends U-SBI
        transactions=(_txn("U-AXIS", "VICTIM", "A1", 100.0, 0),
                      _txn("U-SBI", "A1", "S1", 90.0, 5)),
    )
    sbi = DossierAccount(
        account_no="S1", metadata=_account_row("S1", ifsc="SBIN0000421"),
        transactions=(_txn("U-SBI", "A1", "S1", 90.0, 5),),
    )
    dossier = InvestigationDossierGenerator([axis, sbi]).build()
    groups = {group.ifsc_prefix: group for group in dossier.requisition.bank_requests}
    assert groups["UTIB"].bank_name == "Axis Bank"
    assert groups["SBIN"].bank_name == "State Bank of India"
    assert groups["UTIB"].accounts == ["A1"]
    assert groups["SBIN"].accounts == ["S1"]
    # A1 is the sender of U-SBI, so both banks must be asked about that leg
    assert {"U-AXIS", "U-SBI"} <= set(groups["UTIB"].utrs)
    assert "U-AXIS" not in groups["SBIN"].utrs      # S1 never touches the Axis leg
    assert groups["SBIN"].utrs == ["U-SBI"]
    assert groups["SBIN"].documents == list(REQUISITION_DOCUMENTS)


def test_requisition_handles_an_account_without_ifsc() -> None:
    account = DossierAccount(account_no="NOIFSC",
                             metadata=_account_row("NOIFSC", ifsc=None))
    dossier = InvestigationDossierGenerator([account]).build()
    group = dossier.requisition.bank_requests[0]
    assert group.ifsc_prefix == "UNKNOWN"
    assert "not identified" in group.bank_name.lower()


def test_renderers_are_deterministic() -> None:
    first = _generator().render_json()
    second = _generator().render_json()
    assert first == second
    assert _generator().render_markdown() == _generator().render_markdown()
    assert _generator().render_html() == _generator().render_html()


def test_json_round_trips_through_the_contract() -> None:
    dossier = _generator().build()
    restored = InvestigationDossier.model_validate_json(_generator().render_json())
    assert restored == dossier


def test_html_is_self_contained_and_print_ready() -> None:
    payload = _generator().render_html().decode("utf-8")
    assert payload.startswith("<!DOCTYPE html>")
    assert "http://" not in payload and "https://" not in payload
    assert "<script" not in payload.lower()
    assert "@media print" in payload
    assert "Section A" in payload and "Section B" in payload and "Section C" in payload


def test_values_are_escaped_in_html_and_markdown() -> None:
    hostile = DossierAccount(
        account_no="<script>alert(1)</script>",
        metadata=_account_row("A1", branch_code="BR|042"),
        transactions=(_txn("U1", "VICTIM", "A1", 10.0, 0),),
    )
    generator = InvestigationDossierGenerator([hostile])
    html_payload = generator.render_html().decode("utf-8")
    assert "<script>alert(1)</script>" not in html_payload
    assert "&lt;script&gt;" in html_payload
    markdown = generator.render_markdown()
    table_rows = [line for line in markdown.splitlines() if line.startswith("|")]
    widths = {line.count("|") - line.count("\\|") for line in table_rows}
    assert len(widths) == 1                 # no value can break the table structure
    assert "BR\\|042" in markdown           # the pipe is escaped, not dropped


def _pipeline_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    transactions = pl.DataFrame([
        _txn("T0", "V2", "HUB", 50_000.0, -10),
        _txn("T1", "V", "HUB", 100_000.0, 0),
        _txn("T2", "HUB", "M2", 95_000.0, 5),
        _txn("T3", "M2", "TERM", 93_000.0, 10, mode="AEPS"),
        _txn("T4", "OTHER", "SHOP", 500.0, 20),
    ], schema=TRANSACTION_SCHEMA)
    accounts = pl.DataFrame([_account_row("HUB"), _account_row("M2"),
                             _account_row("TERM")], schema=ACCOUNT_METADATA_SCHEMA)
    complaints = pl.DataFrame([
        {"ack_no": "ACK1", "victim_account": "V", "suspect_account": "HUB",
         "fraud_category": "UPI_FRAUD", "reported_timestamp": BASE + timedelta(hours=1),
         "disputed_amount": 100_000.0, "disputed_utr": "T1", "state": "Bihar"},
        {"ack_no": "ACK2", "victim_account": "V2", "suspect_account": "HUB",
         "fraud_category": "UPI_FRAUD", "reported_timestamp": BASE + timedelta(hours=2),
         "disputed_amount": 50_000.0, "disputed_utr": None, "state": "Jharkhand"},
    ], schema=COMPLAINT_SCHEMA)
    return transactions, accounts, complaints


def test_pipeline_is_deterministic_and_reuses_phase2_layers() -> None:
    frames = _pipeline_frames()
    first = run_pipeline(*frames)
    second = run_pipeline(*frames)
    assert first.features.equals(second.features)
    assert first.graph.equals(second.graph)
    assert first.risk.equals(second.risk)
    assert [a.account_no for a in first.assessments] == sorted(
        a.account_no for a in first.assessments
    )
    assert first.wash_cycles == ()
    assert first.complaint_links == ()


def test_leaderboard_ranks_by_score_then_account() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    view = build_leaderboard(bundle)
    scores = [row.risk_score for row in view.rows]
    assert scores == sorted(scores, reverse=True)
    assert view.rows[0].account_no == "HUB"
    assert view.rows[0].risk_band
    assert view.rows[0].top_archetype == "Rapid pass-through mule"
    filtered = build_leaderboard(bundle, min_score=100.0)
    assert filtered.rows == ()
    assert filtered.total_accounts == len(bundle.assessments)


def test_ingestion_view_summarises_failures_and_mappings() -> None:
    csv_bytes = (
        b"UTR,From Account,To Account,Amount,Txn Date,Mode\n"
        b"T1,V,HUB,100,2024-06-01 10:00:00,upi\n"
        b"T2,V,HUB,not-a-number,2024-06-01 10:05:00,upi\n"
        b",V,HUB,50,2024-06-01 10:06:00,upi\n"
    )
    _, report = DatasetIngester().ingest_transactions(csv_bytes)
    view = build_ingestion_view({"transactions": report})
    table = view.tables[0]
    assert table.table == "transactions"
    assert (table.rows_total, table.rows_accepted, table.rows_rejected) == (3, 1, 2)
    failures = dict(table.failure_counts)
    assert failures["unparseable_amount"] == 1
    assert failures["missing:txn_id"] == 1        # note the `missing:<column>` form
    mapped = {row.raw_name: row for row in table.mappings}
    assert mapped["UTR"].canonical_name == "txn_id"
    assert mapped["UTR"].matched_synonym == "utr"
    assert mapped["UTR"].confidence == 1.0
    assert mapped["Txn Date"].canonical_name == "timestamp"
    assert view.total_rejected == 2 and view.total_rows == 3


def test_account_dossier_facts_and_indicators_are_separated() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    dossier = build_account_dossier(bundle, "HUB")
    assert dossier.scope == "ACCOUNT"
    assert dossier.subject_accounts == ["HUB"]
    assert any(fact.txn_id == "T1" for fact in dossier.facts)
    assert any(indicator.method == "FIFO_TAINT" for indicator in dossier.indicators)
    assert dossier.risk_summary is not None
    assert dossier.risk_summary.account_no == "HUB"
    assert all(fact.source != "ledger" or fact.txn_id for fact in dossier.facts)


def test_account_detail_exposes_seven_sub_scores_and_a_chronological_timeline() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    detail = build_account_detail(bundle, "HUB")
    assert [score.domain for score in detail.sub_scores][:6] == list(DOMAIN_KEYS)
    assert detail.sub_scores[-1].domain == "anomaly"
    assert [row.txn_id for row in detail.timeline] == ["T0", "T1", "T2"]
    assert [row.direction for row in detail.timeline] == ["IN", "IN", "OUT"]
    assert detail.metadata["ifsc"] == "UTIB0000123"


def test_complaint_dossier_does_not_list_unrelated_victim_spending() -> None:
    transactions, accounts, complaints = _pipeline_frames()
    transactions = pl.concat([
        transactions,
        pl.DataFrame([_txn("V-LEGIT", "V", "SHOP2", 700.0, 30)],
                     schema=TRANSACTION_SCHEMA),
    ])
    bundle = run_pipeline(transactions, accounts, complaints)
    dossier = build_complaint_dossier(bundle, "ACK1")
    assert dossier.scope == "COMPLAINT"
    assert "V-LEGIT" not in [fact.txn_id for fact in dossier.facts]
    assert "T1" in [fact.txn_id for fact in dossier.facts]
    assert "ACK1" in [fact.ack_no for fact in dossier.facts]


def test_complaint_dossier_without_utr_uses_the_suspect_leg() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    dossier = build_complaint_dossier(bundle, "ACK2")
    # ACK2 has no disputed_utr: the inferred origin is V2's leg to the named suspect
    assert "T0" in [fact.txn_id for fact in dossier.facts]
    assert "ACK2" in [fact.ack_no for fact in dossier.facts]
    assert dossier.indicators          # trail-based indicators exist


def test_unknown_account_and_ack_produce_valid_empty_documents() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    ghost = build_account_dossier(bundle, "NO-SUCH-ACCOUNT")
    assert ghost.subject_accounts == ["NO-SUCH-ACCOUNT"]
    assert ghost.facts and ghost.facts[0].source == "generator"
    missing = build_complaint_dossier(bundle, "ACK-NOT-THERE")
    assert missing.scope == "COMPLAINT" and missing.facts == []


def test_cluster_dossier_covers_every_selected_account() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    dossier = build_cluster_dossier(bundle, ["M2", "HUB"])
    assert dossier.scope == "CLUSTER"
    assert dossier.subject_accounts == ["HUB", "M2"]
    account_facts = [fact for fact in dossier.facts if fact.category == "ACCOUNT"]
    assert [fact.account_no for fact in account_facts] == ["HUB", "M2"]
    assert dossier.risk_summary is not None
    assert dossier.risk_summary.account_no == "HUB"      # highest score


def test_trail_graph_is_traceable_by_account_utr_and_complaint() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    by_account = build_trail_graph(bundle, start="HUB", max_hops=2)
    assert by_account.edges and all(edge.taint_amount > 0 for edge in by_account.edges)
    assert {node.id for node in by_account.nodes} >= {"HUB", "M2", "TERM"}

    by_utr = build_trail_graph(bundle, start="T1", max_hops=2)
    assert by_utr.origin_label.endswith("T1")
    assert all(edge.txn_id != "T1" for edge in by_utr.edges)   # origin leg, not a trail leg

    by_ack = build_trail_graph(bundle, start="ACK1", max_hops=2)
    assert any(edge.txn_id == "T1" for edge in by_ack.edges)  # complaint-scoped seed leg


def test_trail_graph_elements_are_renderable_and_deterministic() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    view = build_trail_graph(bundle, start="HUB", max_hops=3)
    assert view.elements == build_trail_graph(bundle, start="HUB", max_hops=3).elements
    node_ids = {node["data"]["id"] for node in view.elements["nodes"]}
    assert all(edge["data"]["source"] in node_ids for edge in view.elements["edges"])
    assert all(edge["data"]["target"] in node_ids for edge in view.elements["edges"])
    terminal = next(node for node in view.nodes if node.id == "TERM")
    assert terminal.kind == "TERMINAL"
    assert terminal.risk_band


def test_unknown_trace_start_yields_an_empty_canvas() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    view = build_trail_graph(bundle, start="NO-SUCH-ID", max_hops=3)
    assert view.nodes == () and view.edges == ()
    assert view.elements == {"nodes": [], "edges": []}


def test_collusion_and_terminal_and_correlation_views() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    collusion = build_collusion_view(bundle)
    imei_groups = [group for group in collusion.groups if group.basis == "IMEI"]
    assert imei_groups and all(len(group.accounts) >= 2 for group in collusion.groups)

    terminals = build_terminal_view(bundle)
    assert [row.account_no for row in terminals.rows] == ["TERM"]
    assert terminals.rows[0].terminal_inflow == 93_000.0

    correlation = build_correlation_view(bundle)
    assert correlation.rows == ()           # no shared evidence in this ledger
    assert correlation.hotspots == ()


def test_cycle_view_lists_a_closed_wash_loop() -> None:
    # every hop loses <= 15 %, so Phase 2 reports this as a wash cycle
    transactions = pl.DataFrame([
        _txn("C1", "A", "B", 100_000.0, 0),
        _txn("C2", "B", "C", 97_000.0, 10),
        _txn("C3", "C", "A", 95_000.0, 20),
    ], schema=TRANSACTION_SCHEMA)
    view = build_cycle_view(run_pipeline(transactions))
    assert len(view.rows) == 1
    row = view.rows[0]
    assert row.accounts == ("A", "B", "C")
    assert row.txn_ids == ("C1", "C2", "C3")
    assert row.total_amount == 292_000.0      # Phase 2 sums the legs
    assert 0.0 < row.max_decay_ratio <= 0.15


def test_correlation_view_flags_interstate_links() -> None:
    sim = SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)
    bundle = run_pipeline(sim.transactions, sim.accounts, sim.complaints)
    view = build_correlation_view(bundle)
    # the seeded scenario routes three state-distinct complaints through one
    # AEPS terminal, so links must exist (verified: 3 links, 1 shared terminal)
    assert view.rows
    assert any(row.interstate for row in view.rows)
    assert any(row.shared_terminals for row in view.rows)
    hotspot = view.hotspots[0]
    assert len(hotspot.ack_nos) >= 2
    assert all(item.ack_nos == tuple(sorted(item.ack_nos)) for item in view.hotspots)
    assert view.hotspots == tuple(
        sorted(view.hotspots, key=lambda item: (-len(item.ack_nos), item.key))
    )
    assert view.rows == tuple(sorted(view.rows, key=lambda item: (item.ack_a, item.ack_b)))


def test_dossier_format_wrappers_delegate() -> None:
    dossier = _generator().build()
    assert state.dossier_markdown(dossier) == _generator().render_markdown(dossier)
    assert state.dossier_html(dossier) == _generator().render_html(dossier)
    assert state.dossier_json(dossier) == _generator().render_json(dossier)


def test_app_renders_empty_state_without_a_dataset() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not app.exception, [str(item) for item in app.exception]
    titles = [title.value for title in app.title]
    assert any("PraharGraph" in title for title in titles)
    labels = [tab.label for tab in app.tabs]
    assert labels == [
        "1 · Ingestion & Data Hygiene",
        "2 · Mule Risk Ranking & XAI",
        "3 · Money Trail & Syndicates",
        "4 · Complaint Correlation & Dossier",
    ]


def test_offline_config_disables_telemetry_and_remote_fonts() -> None:
    text = Path(".streamlit/config.toml").read_text()
    assert "gatherUsageStats = false" in text
    assert 'font = "sans serif"' in text


def test_app_benchmark_load_populates_ingestion_metrics() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=300)
    app.run()
    assert not app.exception, [str(item) for item in app.exception]
    app.button(key="load_benchmark").click().run()
    assert not app.exception, [str(item) for item in app.exception]

    metrics = {
        item.label: int(str(item.value).replace(",", ""))
        for item in app.metric
    }
    # the seeded benchmark is ingested, so Tab 1 shows real numbers
    assert metrics["Rows total"] > 0
    assert metrics["Accepted"] > 0
    assert metrics["Rejected"] >= 3
    assert metrics["Rows total"] == metrics["Accepted"] + metrics["Rejected"]


def test_benchmark_load_returns_ingestion_reports() -> None:
    from ui.app import load_benchmark

    transactions, accounts, complaints, reports = load_benchmark()
    assert transactions.height > 0 and accounts.height > 0 and complaints.height > 0
    assert set(reports) == {"transactions", "accounts", "complaints"}
    txn_report = reports["transactions"]
    assert txn_report.rows_rejected == 3
    assert set(txn_report.failure_counts) == {
        "unparseable_amount", "missing:txn_id", "duplicate_utr",
    }
    assert txn_report.rows_accepted == transactions.height
    assert txn_report.rows_total == transactions.height + 3
    assert "rejection_reason" in txn_report.rejected_rows.columns
    assert reports["accounts"].rows_rejected == 0
    assert reports["complaints"].rows_rejected == 0
