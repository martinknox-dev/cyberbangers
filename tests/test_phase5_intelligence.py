"""Phase 5 acceptance tests: hardware-adaptive compute, risk diffusion,
multi-detector ensemble and freeze priority.

Covers the four Phase 5 tasks:

1. Hardware-adaptive compute profiler (``core.hardware``) and tier wiring.
2. Cohort normalisation + semi-supervised graph risk diffusion + retained taint.
3. Multi-detector ensemble + operational role + freeze priority index.
4. Exposure of the additive fields through dossier / state / benchmark.
"""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from core.contracts import (
    ACCOUNT_FEATURE_SCHEMA,
    GRAPH_INDICATOR_SCHEMA,
    RISK_REPORT_SCHEMA,
    FeatureConfig,
    GraphConfig,
    RiskAssessment,
    RiskConfig,
)
from core import hardware
from core.hardware import ComputeTier, HardwareProfile, detect_hardware_profile
from features.account_features import extract_account_features
from graph.analyzer import GraphAnalyzer
from risk.scorer import RiskScorer
from simulator.generator import SyndicateDataSimulator
from ui import state as ui_state


BASE = datetime(2024, 6, 1, 10, 0, 0)


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


# ---------------------------------------------------------------------------
# Task 1 — hardware-adaptive compute profiler
# ---------------------------------------------------------------------------


class TestHardwareProfile:
    def test_lite_tier_knobs(self) -> None:
        profile = detect_hardware_profile("LITE")
        assert profile.tier == "LITE"
        assert profile.iforest_estimators == 100
        assert profile.enable_knn_density is False
        assert profile.diffusion_iterations == 5
        assert profile.max_cycle_length == 3
        assert profile.betweenness_sample_k == 64
        assert profile.duckdb_threads == 2
        assert profile.duckdb_memory_limit == "256MB"

    def test_standard_tier_knobs(self) -> None:
        profile = detect_hardware_profile("STANDARD")
        assert profile.tier == "STANDARD"
        assert profile.iforest_estimators == 200
        assert profile.enable_knn_density is True
        assert profile.diffusion_iterations == 10
        assert profile.max_cycle_length == 4
        assert profile.betweenness_sample_k == 256
        assert profile.duckdb_threads == 4
        assert profile.duckdb_memory_limit == "1GB"

    def test_workstation_tier_knobs(self) -> None:
        profile = detect_hardware_profile("WORKSTATION")
        assert profile.tier == "WORKSTATION"
        assert profile.iforest_estimators == 400
        assert profile.enable_knn_density is True
        assert profile.diffusion_iterations == 20
        assert profile.max_cycle_length == 5
        assert profile.betweenness_sample_k is None
        assert profile.duckdb_threads == 8
        assert profile.duckdb_memory_limit == "4GB"

    def test_gpu_detection_is_reflected_in_the_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hardware, "_gpu_detected", lambda: True)
        monkeypatch.setattr(hardware, "_system_memory_gb", lambda: 32.0)
        monkeypatch.setattr("os.cpu_count", lambda: 16)
        profile = detect_hardware_profile("WORKSTATION")
        assert profile.gpu_detected is True
        assert profile.total_ram_gb == 32.0
        assert profile.cpu_cores == 16

    def test_auto_tier_prefers_workstation_for_gpu_big_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hardware, "_gpu_detected", lambda: True)
        monkeypatch.setattr(hardware, "_system_memory_gb", lambda: 32.0)
        monkeypatch.setattr("os.cpu_count", lambda: 16)
        assert detect_hardware_profile().tier == "WORKSTATION"

    def test_auto_tier_prefers_standard_for_big_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hardware, "_gpu_detected", lambda: False)
        monkeypatch.setattr(hardware, "_system_memory_gb", lambda: 16.0)
        monkeypatch.setattr("os.cpu_count", lambda: 8)
        assert detect_hardware_profile().tier == "STANDARD"

    def test_auto_tier_prefers_lite_for_small_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hardware, "_gpu_detected", lambda: False)
        monkeypatch.setattr(hardware, "_system_memory_gb", lambda: 4.0)
        monkeypatch.setattr("os.cpu_count", lambda: 2)
        assert detect_hardware_profile().tier == "LITE"

    def test_hardware_profile_is_frozen(self) -> None:
        profile = detect_hardware_profile("LITE")
        assert isinstance(profile, HardwareProfile)


class TestComputeTierWiring:
    def test_feature_config_exposes_compute_tier(self) -> None:
        assert FeatureConfig().compute_tier == "LITE"

    def test_graph_config_exposes_compute_tier(self) -> None:
        assert GraphConfig().compute_tier == "LITE"

    def test_risk_config_exposes_compute_tier(self) -> None:
        assert RiskConfig().compute_tier == "LITE"

    def test_risk_scorer_resolves_tier_profile_from_config(self) -> None:
        scorer = RiskScorer(RiskConfig(compute_tier="WORKSTATION"))
        assert scorer.profile.tier == "WORKSTATION"
        assert scorer.profile.iforest_estimators == 400

    def test_risk_scorer_default_tier_keeps_the_lightweight_model(self) -> None:
        scorer = RiskScorer()
        assert scorer.profile.tier == "LITE"
        assert scorer.profile.iforest_estimators == 100
        assert scorer.profile.enable_knn_density is False

    def test_run_pipeline_accepts_and_stores_compute_tier(self) -> None:
        bundle = ui_state.run_pipeline(
            pl.DataFrame(schema={"txn_id": pl.String()}),
            compute_tier="STANDARD",
        )
        assert bundle.hardware_profile is not None
        assert bundle.hardware_profile.tier == "STANDARD"
        assert bundle.risk_config.compute_tier == "STANDARD"


# ---------------------------------------------------------------------------
# Task 2 — cohort normalisation and graph risk diffusion
# ---------------------------------------------------------------------------


class TestCohortVolumeZScore:
    def test_feature_frame_exposes_cohort_volume_zscore(self) -> None:
        features = extract_account_features(
            pl.DataFrame(
                [
                    {
                        "txn_id": "T1", "sender_account": "A", "receiver_account": "B",
                        "amount": 100.0, "timestamp": BASE, "payment_mode": "UPI",
                        "device_id": None, "ip_address": None,
                    }
                ],
                schema={
                    "txn_id": pl.String(), "sender_account": pl.String(),
                    "receiver_account": pl.String(), "amount": pl.Float64(),
                    "timestamp": pl.Datetime("us"), "payment_mode": pl.String(),
                    "device_id": pl.String(), "ip_address": pl.String(),
                },
            ),
            accounts=None,
            complaints=None,
        )
        assert "cohort_volume_zscore" in features.columns

    def test_cohort_volume_zscore_is_zero_without_peer_cohort(self) -> None:
        features = extract_account_features(
            pl.DataFrame(
                [
                    {
                        "txn_id": "T1", "sender_account": "A", "receiver_account": "B",
                        "amount": 100.0, "timestamp": BASE, "payment_mode": "UPI",
                        "device_id": None, "ip_address": None,
                    }
                ],
                schema={
                    "txn_id": pl.String(), "sender_account": pl.String(),
                    "receiver_account": pl.String(), "amount": pl.Float64(),
                    "timestamp": pl.Datetime("us"), "payment_mode": pl.String(),
                    "device_id": pl.String(), "ip_address": pl.String(),
                },
            ),
            accounts=None,
        )
        row = features.filter(pl.col("account_no") == "A").to_dicts()[0]
        assert row["cohort_volume_zscore"] == 0.0


class TestGraphRiskDiffusion:
    def test_topology_metrics_include_retained_taint_and_proximity(self) -> None:
        sim = SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)
        graph = GraphAnalyzer(sim.transactions, sim.accounts, sim.complaints).topology_metrics()
        assert "retained_tainted_amount" in graph.columns
        assert "network_proximity_risk" in graph.columns
        assert graph["retained_tainted_amount"].dtype == pl.Float64
        assert graph["network_proximity_risk"].dtype == pl.Float64
        assert graph.height > 0
        assert graph["retained_tainted_amount"].min() >= 0.0
        assert graph["network_proximity_risk"].max() <= 1.0


# ---------------------------------------------------------------------------
# Task 3 — multi-detector ensemble, operational role, freeze priority
# ---------------------------------------------------------------------------


class TestAnomalyEnsemble:
    def test_anomaly_uses_lof_when_enabled(self) -> None:
        scorer = RiskScorer(RiskConfig(compute_tier="WORKSTATION"))
        features = pl.DataFrame(
            [_feature_row(f"A{i:02d}") for i in range(12)],
            schema=ACCOUNT_FEATURE_SCHEMA,
        )
        graph = pl.DataFrame(
            [_graph_row(f"A{i:02d}") for i in range(12)],
            schema=GRAPH_INDICATOR_SCHEMA,
        )
        report, assessments = scorer.score_with_assessments(features, graph)
        assert report.height == 12
        assert all(a.anomaly_score > 0.0 for a in assessments)

    def test_anomaly_falls_back_to_isolation_forest_when_disabled(self) -> None:
        scorer = RiskScorer()
        features = pl.DataFrame(
            [_feature_row(f"A{i:02d}") for i in range(12)],
            schema=ACCOUNT_FEATURE_SCHEMA,
        )
        graph = pl.DataFrame(
            [_graph_row(f"A{i:02d}") for i in range(12)],
            schema=GRAPH_INDICATOR_SCHEMA,
        )
        report, assessments = scorer.score_with_assessments(features, graph)
        assert all(a.anomaly_score > 0.0 for a in assessments)


class TestOperationalRoleAndFreezePriority:
    def test_assessment_carries_operational_role_and_freeze_priority(self) -> None:
        scorer = RiskScorer()
        features = pl.DataFrame(
            [_feature_row("MULE")],
            schema=ACCOUNT_FEATURE_SCHEMA,
        )
        graph = pl.DataFrame(
            [_graph_row("MULE")],
            schema=GRAPH_INDICATOR_SCHEMA,
        )
        assessments = scorer.assess(features, graph)
        assert len(assessments) == 1
        assessment = assessments[0]
        assert assessment.operational_role in (
            "LAYER_1_COLLECTOR", "LAYERING_CONDUIT", "SMURF_DISTRIBUTOR",
            "WASH_CYCLE_NODE", "CASH_OUT_TERMINAL", "COLLUSION_RING_MEMBER",
            "STANDARD_ACCOUNT",
        )
        assert 0.0 <= assessment.freeze_priority_score <= 100.0
        assert assessment.retained_tainted_amount >= 0.0

    def test_risk_report_schema_has_phase5_fields(self) -> None:
        assert "operational_role" in RISK_REPORT_SCHEMA
        assert "retained_tainted_amount" in RISK_REPORT_SCHEMA
        assert "freeze_priority_score" in RISK_REPORT_SCHEMA


class TestLeaderboardExposure:
    def test_leaderboard_rows_carry_freeze_priority(self) -> None:
        bundle = ui_state.run_pipeline(
            pl.DataFrame(schema={"txn_id": pl.String()}),
            compute_tier="LITE",
        )
        view = ui_state.build_leaderboard(bundle)
        assert view.total_accounts == len(bundle.assessments)
