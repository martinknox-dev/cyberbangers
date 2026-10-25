"""Phase 5 regulatory export tests: SAR/STR formats."""

import json

import polars as pl

from core.contracts import RiskConfig
from export.regulatory import SARExporter, STRExporter
from ui import state


def test_sar_export_produces_valid_json() -> None:
    txns = pl.DataFrame({
        "txn_id": ["T1"], "sender_account": ["A"], "receiver_account": ["B"],
        "amount": [500000.0], "timestamp": ["2024-06-01 10:00:00"],
        "payment_mode": ["UPI"],
    }).with_columns(pl.col("timestamp").str.to_datetime())

    bundle = state.run_pipeline(txns, risk_config=RiskConfig())
    dossier = state.build_account_dossier(bundle, "A")
    exporter = SARExporter()
    sar_json = exporter.export_sar(dossier, reporting_entity="BANK001")

    data = json.loads(sar_json)
    assert data["report_type"] == "SAR"
    assert data["reporting_entity"] == "BANK001"
    assert "subject_accounts" in data
    assert "suspicious_activity" in data
    assert "grounds_for_suspicion" in data


def test_str_export_produces_valid_csv() -> None:
    txns = pl.DataFrame({
        "txn_id": ["T1"], "sender_account": ["A"], "receiver_account": ["B"],
        "amount": [1000000.0], "timestamp": ["2024-06-01 10:00:00"],
        "payment_mode": ["UPI"],
    }).with_columns(pl.col("timestamp").str.to_datetime())

    bundle = state.run_pipeline(txns, risk_config=RiskConfig())
    dossier = state.build_account_dossier(bundle, "A")
    exporter = STRExporter()
    str_csv = exporter.export_str(dossier, reporting_entity="BANK001")

    # Parse CSV
    import io
    df = pl.read_csv(io.StringIO(str_csv))
    # Should have at least 1 row (summary row)
    assert df.height >= 1
    assert "report_type" in df.columns
    assert df["report_type"][0] == "STR"
    assert "reporting_entity" in df.columns
    assert df["reporting_entity"][0] == "BANK001"
    assert "subject_account" in df.columns
    # Check for summary row
    summary_rows = df.filter(pl.col("indicator_type") == "SUMMARY")
    assert summary_rows.height >= 1
    assert summary_rows["subject_account"][0] == "A"
    assert "value" in df.columns
    assert summary_rows["value"][0] == 1000000.0


def test_sar_export_includes_risk_details() -> None:
    txns = pl.DataFrame({
        "txn_id": ["T1", "T2"], "sender_account": ["A", "B"], "receiver_account": ["B", "C"],
        "amount": [500000.0, 400000.0], "timestamp": ["2024-06-01 10:00:00", "2024-06-01 10:05:00"],
        "payment_mode": ["UPI", "UPI"],
    }).with_columns(pl.col("timestamp").str.to_datetime())

    bundle = state.run_pipeline(txns, risk_config=RiskConfig())
    dossier = state.build_account_dossier(bundle, "A")
    exporter = SARExporter()
    sar_json = exporter.export_sar(dossier, reporting_entity="BANK001")

    data = json.loads(sar_json)
    assert data["suspicious_activity"]["risk_score"] >= 0
    assert data["suspicious_activity"]["risk_band"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert data["suspicious_activity"]["operational_role"] in (
        "LAYER_1_COLLECTOR", "LAYERING_CONDUIT", "SMURF_DISTRIBUTOR",
        "WASH_CYCLE_NODE", "CASH_OUT_TERMINAL", "COLLUSION_RING_MEMBER",
        "STANDARD_ACCOUNT"
    )
    assert isinstance(data["grounds_for_suspicion"], list)
    assert len(data["grounds_for_suspicion"]) > 0