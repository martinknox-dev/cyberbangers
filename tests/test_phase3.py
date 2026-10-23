"""Phase 3 tests: workbench view models, dossiers and the Streamlit app."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from core.contracts import (
    ACCOUNT_FEATURE_SCHEMA,
    DOMAIN_KEYS,
    GRAPH_INDICATOR_SCHEMA,
    REQUISITION_DOCUMENTS,
    IngestionReport,
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
from risk.scorer import RiskScorer

BASE = datetime(2024, 6, 1, 10, 0)


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
