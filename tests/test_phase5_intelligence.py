"""Phase 5 acceptance tests: hardware-adaptive compute, risk diffusion,
multi-detector ensemble and freeze priority.

Covers the four Phase 5 tasks:

1. Hardware-adaptive compute profiler (``core.hardware``) and tier wiring.
2. Cohort normalisation + semi-supervised graph risk diffusion + retained taint.
3. Multi-detector ensemble + operational role + freeze priority index.
4. Exposure of the additive fields through dossier / state / benchmark.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from core import hardware
from core.contracts import (
    ACCOUNT_FEATURE_SCHEMA,
    GRAPH_INDICATOR_SCHEMA,
    RISK_REPORT_SCHEMA,
    FeatureConfig,
    GraphConfig,
    RiskAssessment,
    RiskConfig,
)
from core.hardware import HardwareProfile, detect_hardware_profile
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
    def test_graph_analyzer_resolves_tier_profile(self) -> None:
        analyzer = GraphAnalyzer(
            pl.DataFrame(schema={"txn_id": pl.String()}),
            config=GraphConfig(compute_tier="WORKSTATION"),
        )
        assert analyzer.profile.tier == "WORKSTATION"
        assert analyzer.profile.max_cycle_length == 5
        assert analyzer.profile.diffusion_iterations == 20

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
        assert bundle.analyzer.profile.tier == "STANDARD"

    def test_feature_config_tier_sets_duckdb_capacity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from features import account_features

        class _RecordingConnection:
            def __init__(self, inner: object) -> None:
                self._inner = inner
                self.statements: list[str] = []

            def execute(self, sql: str, *args: object, **kwargs: object) -> object:
                self.statements.append(sql)
                return self._inner.execute(sql, *args, **kwargs)  # type: ignore[attr-defined]

            def __getattr__(self, name: str) -> object:
                return getattr(self._inner, name)

        recorders: list[_RecordingConnection] = []
        original_connect = account_features.duckdb.connect

        def _connect() -> _RecordingConnection:
            recorder = _RecordingConnection(original_connect())
            recorders.append(recorder)
            return recorder

        monkeypatch.setattr(account_features.duckdb, "connect", _connect)
        transactions = pl.DataFrame(
            [
                {"txn_id": "T1", "sender_account": "A", "receiver_account": "B",
                 "amount": 100.0, "timestamp": BASE, "payment_mode": "UPI"},
            ],
            schema={
                "txn_id": pl.String(), "sender_account": pl.String(),
                "receiver_account": pl.String(), "amount": pl.Float64(),
                "timestamp": pl.Datetime("us"), "payment_mode": pl.String(),
            },
        )
        extract_account_features(
            transactions, config=FeatureConfig(compute_tier="WORKSTATION")
        )
        statements = recorders[0].statements
        assert "SET threads = 8" in statements
        assert "SET memory_limit = '4GB'" in statements


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

    def test_cohort_volume_zscore_compares_within_account_type(self) -> None:
        accounts = pl.DataFrame(
            [
                {"account_no": "A", "account_type": "SAVINGS"},
                {"account_no": "B", "account_type": "SAVINGS"},
                {"account_no": "C", "account_type": "SAVINGS"},
                {"account_no": "D", "account_type": "PMJDY"},
            ],
            schema={"account_no": pl.String(), "account_type": pl.String()},
        )
        transactions = pl.DataFrame(
            [
                {
                    "txn_id": f"T{index}", "sender_account": sender,
                    "receiver_account": "D", "amount": amount, "timestamp": BASE,
                    "payment_mode": "UPI", "device_id": None, "ip_address": None,
                }
                for index, (sender, amount) in enumerate(
                    [("A", 100.0), ("B", 100.0), ("C", 100.0), ("A", 100.0)]
                )
            ],
            schema={
                "txn_id": pl.String(), "sender_account": pl.String(),
                "receiver_account": pl.String(), "amount": pl.Float64(),
                "timestamp": pl.Datetime("us"), "payment_mode": pl.String(),
                "device_id": pl.String(), "ip_address": pl.String(),
            },
        )
        features = extract_account_features(transactions, accounts=accounts)
        by_account = {
            row["account_no"]: row["cohort_volume_zscore"]
            for row in features.to_dicts()
        }
        # A moved 200 against savings peers at 100; B and C sit below the mean.
        assert by_account["A"] > 0.0
        assert by_account["B"] < 0.0
        assert by_account["C"] < 0.0
        # D is the only PMJDY account, so it has no peer cohort.
        assert by_account["D"] == 0.0


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

    def test_fifo_residual_is_the_unsent_taint(self) -> None:
        transactions = pl.DataFrame(
            [
                {"txn_id": "T1", "sender_account": "V", "receiver_account": "A",
                 "amount": 1000.0, "timestamp": BASE, "payment_mode": "UPI"},
                {"txn_id": "T2", "sender_account": "A", "receiver_account": "B",
                 "amount": 400.0, "timestamp": BASE + timedelta(minutes=5),
                 "payment_mode": "UPI"},
                {"txn_id": "T3", "sender_account": "A", "receiver_account": "X",
                 "amount": 200.0, "timestamp": BASE + timedelta(minutes=10),
                 "payment_mode": "UPI"},
            ],
            schema={
                "txn_id": pl.String(), "sender_account": pl.String(),
                "receiver_account": pl.String(), "amount": pl.Float64(),
                "timestamp": pl.Datetime("us"), "payment_mode": pl.String(),
            },
        )
        trail = GraphAnalyzer(transactions).trace_money_trail("T1", max_hops=3)
        assert trail.retained_taint["A"] == pytest.approx(400.0)
        assert trail.retained_taint["B"] == pytest.approx(400.0)
        assert trail.retained_taint["X"] == pytest.approx(200.0)

    def test_retained_taint_sums_distinct_disputes_without_double_counting(self) -> None:
        transactions = pl.DataFrame(
            [
                {"txn_id": "T1", "sender_account": "V1", "receiver_account": "A",
                 "amount": 1000.0, "timestamp": BASE, "payment_mode": "UPI"},
                {"txn_id": "T2", "sender_account": "V2", "receiver_account": "A",
                 "amount": 500.0, "timestamp": BASE + timedelta(minutes=1),
                 "payment_mode": "UPI"},
            ],
            schema={
                "txn_id": pl.String(), "sender_account": pl.String(),
                "receiver_account": pl.String(), "amount": pl.Float64(),
                "timestamp": pl.Datetime("us"), "payment_mode": pl.String(),
            },
        )
        complaints = pl.DataFrame(
            [
                {"ack_no": "ACK1", "victim_account": "V1", "suspect_account": "A",
                 "fraud_category": "UPI fraud", "reported_timestamp": BASE,
                 "disputed_amount": 1000.0, "disputed_utr": "T1", "state": "BR"},
                {"ack_no": "ACK2", "victim_account": "V2", "suspect_account": "A",
                 "fraud_category": "UPI fraud", "reported_timestamp": BASE,
                 "disputed_amount": 500.0, "disputed_utr": "T2", "state": "BR"},
                {"ack_no": "ACK3", "victim_account": "V2", "suspect_account": "A",
                 "fraud_category": "UPI fraud", "reported_timestamp": BASE,
                 "disputed_amount": 500.0, "disputed_utr": "T2", "state": "BR"},
            ],
            schema={
                "ack_no": pl.String(), "victim_account": pl.String(),
                "suspect_account": pl.String(), "fraud_category": pl.String(),
                "reported_timestamp": pl.Datetime("us"),
                "disputed_amount": pl.Float64(), "disputed_utr": pl.String(),
                "state": pl.String(),
            },
        )
        graph = GraphAnalyzer(transactions, complaints=complaints).topology_metrics()
        retained = dict(zip(
            graph["account_no"].to_list(),
            graph["retained_tainted_amount"].to_list(),
        ))
        # T1 + T2 are distinct tainted inflows; the repeated T2 is skipped.
        assert retained["A"] == pytest.approx(1_500.0)

    def test_network_proximity_decays_from_the_complaint_seed(self) -> None:
        transactions = pl.DataFrame(
            [
                {"txn_id": "T1", "sender_account": "V", "receiver_account": "A",
                 "amount": 1000.0, "timestamp": BASE, "payment_mode": "UPI"},
                {"txn_id": "T2", "sender_account": "A", "receiver_account": "B",
                 "amount": 900.0, "timestamp": BASE + timedelta(minutes=5),
                 "payment_mode": "UPI"},
                {"txn_id": "T3", "sender_account": "B", "receiver_account": "C",
                 "amount": 800.0, "timestamp": BASE + timedelta(minutes=10),
                 "payment_mode": "UPI"},
            ],
            schema={
                "txn_id": pl.String(), "sender_account": pl.String(),
                "receiver_account": pl.String(), "amount": pl.Float64(),
                "timestamp": pl.Datetime("us"), "payment_mode": pl.String(),
            },
        )
        complaints = pl.DataFrame(
            [
                {"ack_no": "ACK1", "victim_account": "V", "suspect_account": "A",
                 "fraud_category": "UPI fraud", "reported_timestamp": BASE,
                 "disputed_amount": 1000.0, "disputed_utr": "T1", "state": "BR"},
            ],
            schema={
                "ack_no": pl.String(), "victim_account": pl.String(),
                "suspect_account": pl.String(), "fraud_category": pl.String(),
                "reported_timestamp": pl.Datetime("us"),
                "disputed_amount": pl.Float64(), "disputed_utr": pl.String(),
                "state": pl.String(),
            },
        )
        graph = GraphAnalyzer(transactions, complaints=complaints).topology_metrics()
        proximity = dict(zip(
            graph["account_no"].to_list(),
            graph["network_proximity_risk"].to_list(),
        ))
        assert proximity["A"] == pytest.approx(1.0)
        assert 0.0 < proximity["B"] < proximity["A"]
        assert 0.0 < proximity["C"] < proximity["B"]


# ---------------------------------------------------------------------------
# Task 3 — multi-detector ensemble, operational role, freeze priority
# ---------------------------------------------------------------------------


def _varied_feature_rows(count: int = 12) -> list[dict[str, object]]:
    """A small population with one deliberate behavioural outlier."""
    rows = []
    for index in range(count):
        rows.append(
            _feature_row(
                f"A{index:02d}",
                txn_count=4 + index,
                inflow_total=10_000.0 + index * 500.0,
                outflow_total=9_000.0 + index * 450.0,
                unique_senders=2 + index,
                unique_receivers=2 + index,
            )
        )
    rows[-1]["inflow_total"] = 5_000_000.0
    rows[-1]["outflow_total"] = 4_900_000.0
    rows[-1]["unique_senders"] = 200
    rows[-1]["unique_receivers"] = 200
    return rows


class TestAnomalyEnsemble:
    def test_anomaly_uses_lof_when_enabled(self) -> None:
        scorer = RiskScorer(RiskConfig(compute_tier="WORKSTATION"))
        features = pl.DataFrame(_varied_feature_rows(), schema=ACCOUNT_FEATURE_SCHEMA)
        graph = pl.DataFrame(
            [_graph_row(f"A{i:02d}") for i in range(12)],
            schema=GRAPH_INDICATOR_SCHEMA,
        )
        report, assessments = scorer.score_with_assessments(features, graph)
        assert report.height == 12
        assert all(0.0 <= a.anomaly_score <= 100.0 for a in assessments)
        assert any(a.anomaly_score > 0.0 for a in assessments)
        assert assessments[-1].anomaly_score == max(
            a.anomaly_score for a in assessments
        )

    def test_anomaly_falls_back_to_isolation_forest_when_disabled(self) -> None:
        scorer = RiskScorer()
        features = pl.DataFrame(_varied_feature_rows(), schema=ACCOUNT_FEATURE_SCHEMA)
        graph = pl.DataFrame(
            [_graph_row(f"A{i:02d}") for i in range(12)],
            schema=GRAPH_INDICATOR_SCHEMA,
        )
        _report, assessments = scorer.score_with_assessments(features, graph)
        assert all(0.0 <= a.anomaly_score <= 100.0 for a in assessments)
        assert assessments[-1].anomaly_score == max(
            a.anomaly_score for a in assessments
        )


class TestOperationalRoleAndFreezePriority:
    def _assess(self, **overrides: object) -> RiskAssessment:
        scorer = RiskScorer()
        features = pl.DataFrame(
            [_feature_row("MULE", **overrides)], schema=ACCOUNT_FEATURE_SCHEMA
        )
        graph = pl.DataFrame([_graph_row("MULE")], schema=GRAPH_INDICATOR_SCHEMA)
        return scorer.assess(features, graph)[0]

    def test_assessment_carries_operational_role_and_freeze_priority(self) -> None:
        assessment = self._assess()
        assert assessment.operational_role in (
            "LAYER_1_COLLECTOR", "LAYERING_CONDUIT", "SMURF_DISTRIBUTOR",
            "WASH_CYCLE_NODE", "CASH_OUT_TERMINAL", "COLLUSION_RING_MEMBER",
            "STANDARD_ACCOUNT",
        )
        assert 0.0 <= assessment.freeze_priority_score <= 100.0
        assert assessment.retained_tainted_amount >= 0.0

    def test_operational_role_classifies_the_topology(self) -> None:
        assert self._assess(
            aeps_cashout_inflow_count=8, pass_through_ratio_15m=0.1
        ).operational_role == "CASH_OUT_TERMINAL"
        assert self._assess(
            smurf_bulk_inflow_count=2, smurf_split_count=12,
            pass_through_ratio_15m=0.1,
        ).operational_role == "SMURF_DISTRIBUTOR"
        assert self._assess(
            smurf_bulk_inflow_count=0, smurf_split_count=0,
            pass_through_ratio_15m=0.95, median_retention_minutes=2.0,
        ).operational_role == "LAYERING_CONDUIT"
        assert self._assess(
            smurf_bulk_inflow_count=0, smurf_split_count=0,
            shared_device_account_count=4, pass_through_ratio_15m=0.1,
        ).operational_role == "COLLUSION_RING_MEMBER"
        assert self._assess(
            smurf_bulk_inflow_count=0, smurf_split_count=0,
            direct_complaint_count=1, unique_senders=15, inflow_total=500_000.0,
            outflow_total=10_000.0, pass_through_ratio_15m=0.0,
        ).operational_role == "LAYER_1_COLLECTOR"
        assert self._assess(
            pass_through_ratio_15m=0.0, median_retention_minutes=None,
            smurf_bulk_inflow_count=0, shared_device_account_count=0,
            direct_complaint_count=0,
        ).operational_role == "STANDARD_ACCOUNT"

    def test_wash_cycle_role_comes_from_the_graph(self) -> None:
        scorer = RiskScorer()
        features = pl.DataFrame(
            [_feature_row("LOOP", pass_through_ratio_15m=0.1)],
            schema=ACCOUNT_FEATURE_SCHEMA,
        )
        graph = pl.DataFrame(
            [{**_graph_row("LOOP"), "cycle_count": 2, "cycle_max_amount": 50_000.0,
              "cycle_txn_ids": ["T1", "T2", "T3"]}],
            schema=GRAPH_INDICATOR_SCHEMA,
        )
        assessment = scorer.assess(features, graph)[0]
        assert assessment.operational_role == "WASH_CYCLE_NODE"

    def test_freeze_priority_rises_with_recoverable_taint(self) -> None:
        rows = [_feature_row(f"A{index:02d}") for index in range(12)]
        features = pl.DataFrame(rows, schema=ACCOUNT_FEATURE_SCHEMA)
        graph_rows = [_graph_row(f"A{index:02d}") for index in range(12)]
        graph_rows[0]["retained_tainted_amount"] = 1_000.0
        graph_rows[-1]["retained_tainted_amount"] = 10_000_000.0
        graph_rows[0]["network_proximity_risk"] = 0.8
        graph_rows[-1]["network_proximity_risk"] = 0.8
        graph = pl.DataFrame(graph_rows, schema=GRAPH_INDICATOR_SCHEMA)
        _, assessments = RiskScorer().score_with_assessments(features, graph)
        by_account = {item.account_no: item for item in assessments}
        assert (
            by_account["A11"].freeze_priority_score
            > by_account["A00"].freeze_priority_score
        )
        assert all(
            0.0 <= item.freeze_priority_score <= 100.0 for item in assessments
        )

    def test_risk_report_schema_has_phase5_fields(self) -> None:
        assert "operational_role" in RISK_REPORT_SCHEMA
        assert "retained_tainted_amount" in RISK_REPORT_SCHEMA
        assert "freeze_priority_score" in RISK_REPORT_SCHEMA

    def test_risk_report_emits_phase5_fields(self) -> None:
        features = pl.DataFrame(
            [_feature_row("MULE")], schema=ACCOUNT_FEATURE_SCHEMA
        )
        graph = pl.DataFrame(
            [{**_graph_row("MULE"), "retained_tainted_amount": 5_000.0,
              "network_proximity_risk": 0.9}],
            schema=GRAPH_INDICATOR_SCHEMA,
        )
        report = RiskScorer().score(features, graph)
        row = report.to_dicts()[0]
        assert row["operational_role"] != ""
        assert row["retained_tainted_amount"] == pytest.approx(5_000.0)
        assert 0.0 <= row["freeze_priority_score"] <= 100.0


class TestLeaderboardExposure:
    def test_leaderboard_rows_carry_freeze_priority(self) -> None:
        bundle = ui_state.run_pipeline(
            pl.DataFrame(schema={"txn_id": pl.String()}),
            compute_tier="LITE",
        )
        view = ui_state.build_leaderboard(bundle)
        assert view.total_accounts == len(bundle.assessments)

    def test_leaderboard_and_detail_expose_phase5_fields(self) -> None:
        transactions = pl.DataFrame(
            [
                {"txn_id": "T1", "sender_account": "V", "receiver_account": "A",
                 "amount": 10_000.0, "timestamp": BASE, "payment_mode": "UPI"},
                {"txn_id": "T2", "sender_account": "A", "receiver_account": "B",
                 "amount": 9_500.0, "timestamp": BASE + timedelta(minutes=2),
                 "payment_mode": "UPI"},
            ],
            schema={
                "txn_id": pl.String(), "sender_account": pl.String(),
                "receiver_account": pl.String(), "amount": pl.Float64(),
                "timestamp": pl.Datetime("us"), "payment_mode": pl.String(),
            },
        )
        complaints = pl.DataFrame(
            [
                {"ack_no": "ACK1", "victim_account": "V", "suspect_account": "A",
                 "fraud_category": "UPI fraud", "reported_timestamp": BASE,
                 "disputed_amount": 10_000.0, "disputed_utr": "T1", "state": "BR"},
            ],
            schema={
                "ack_no": pl.String(), "victim_account": pl.String(),
                "suspect_account": pl.String(), "fraud_category": pl.String(),
                "reported_timestamp": pl.Datetime("us"),
                "disputed_amount": pl.Float64(), "disputed_utr": pl.String(),
                "state": pl.String(),
            },
        )
        bundle = ui_state.run_pipeline(transactions, complaints=complaints)
        view = ui_state.build_leaderboard(bundle)
        assert view.rows
        assert all(hasattr(row, "operational_role") for row in view.rows)
        assert all(hasattr(row, "freeze_priority_score") for row in view.rows)
        detail = ui_state.build_account_detail(bundle, "A")
        assert detail.operational_role != ""
        assert detail.freeze_priority_score >= 0.0
        dossier = ui_state.build_account_dossier(bundle, "A")
        assert dossier.risk_summary is not None
        assert dossier.risk_summary.operational_role != ""
        assert dossier.risk_summary.freeze_priority_score >= 0.0
