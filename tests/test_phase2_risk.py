"""Phase 2 acceptance tests: behavioural features, hybrid risk, explanations."""
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from core.contracts import (
    ACCOUNT_FEATURE_SCHEMA,
    RISK_REPORT_SCHEMA,
    FeatureConfig,
)
from features.account_features import extract_account_features
from simulator.generator import SyndicateDataSimulator

BASE = datetime(2024, 6, 1, 10, 0, 0)


def leg(
    txn_id: str,
    sender: str,
    receiver: str,
    amount: float,
    minutes: float,
    mode: str = "UPI",
    device: str | None = None,
) -> dict[str, object]:
    return {
        "txn_id": txn_id,
        "sender_account": sender,
        "sender_ifsc": "SBIN0000001",
        "receiver_account": receiver,
        "receiver_ifsc": "SBIN0000002",
        "amount": amount,
        "timestamp": BASE + timedelta(minutes=minutes),
        "payment_mode": mode,
        "device_id": device,
        "ip_address": "10.0.0.1",
    }


def txns(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def account_row(
    account_no: str,
    opened: date | None = None,
    branch: str = "BR001",
    imei: str | None = None,
) -> dict[str, object]:
    return {
        "account_no": account_no,
        "ifsc": "SBIN0000001",
        "account_opening_date": opened,
        "branch_code": branch,
        "mobile_no": "9812000000",
        "imei": imei,
        "kyc_pincode": "800013",
        "account_type": "SAVINGS",
    }


def feature_row(features: pl.DataFrame, account_no: str) -> dict[str, object]:
    rows = features.filter(pl.col("account_no") == account_no).to_dicts()
    assert len(rows) == 1, f"{account_no} not found in features"
    return rows[0]


class TestAccountFeatures:
    def test_one_row_per_distinct_account(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 100.0, 0),
            leg("T2", "B", "C", 40.0, 5),
        ])
        features = extract_account_features(frame)
        assert features.height == 3
        assert set(features["account_no"].to_list()) == {"A", "B", "C"}
        assert list(features.columns) == list(ACCOUNT_FEATURE_SCHEMA)
        assert features.dtypes == [
            ACCOUNT_FEATURE_SCHEMA[name] for name in features.columns
        ]

    def test_counts_totals_and_counterparties(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 100.0, 0),
            leg("T2", "C", "B", 50.0, 5),
            leg("T3", "B", "D", 30.0, 10),
        ])
        row = feature_row(extract_account_features(frame), "B")
        assert row["txn_count"] == 3
        assert row["inflow_count"] == 2
        assert row["outflow_count"] == 1
        assert row["inflow_total"] == pytest.approx(150.0)
        assert row["outflow_total"] == pytest.approx(30.0)
        assert row["net_flow"] == pytest.approx(120.0)
        assert row["unique_senders"] == 2
        assert row["unique_receivers"] == 1

    def test_fan_in_fan_out_asymmetry_signs(self) -> None:
        frame = txns([
            leg("T1", "A1", "AGG", 100.0, 0),
            leg("T2", "A2", "AGG", 100.0, 1),
            leg("T3", "A3", "AGG", 100.0, 2),
            leg("T4", "AGG", "Z1", 250.0, 3),
        ])
        aggregator = feature_row(extract_account_features(frame), "AGG")
        assert aggregator["fan_in_fan_out_asymmetry"] < 0

        frame2 = txns([
            leg("T1", "SRC", "DIST", 500.0, 0),
            leg("T2", "DIST", "R1", 100.0, 1),
            leg("T3", "DIST", "R2", 100.0, 2),
        ])
        distributor = feature_row(extract_account_features(frame2), "DIST")
        assert distributor["fan_in_fan_out_asymmetry"] > 0

    def test_pass_through_uses_own_legs_only(self) -> None:
        # A receives 100 and pays 60 + 30 out. B receives 60 and forwards all of
        # it to D: B is a pass-through, but A's ratio must not absorb B's leg.
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 60.0, 1),
            leg("T3", "A", "C", 30.0, 2),
            leg("T4", "B", "D", 60.0, 3),
        ])
        features = extract_account_features(frame)
        a_row = feature_row(features, "A")
        assert a_row["pass_through_ratio_15m"] == pytest.approx(0.9)
        # retention of A's two outflows: +1 min and +2 min -> median 1.5 min
        assert a_row["median_retention_minutes"] == pytest.approx(1.5)
        b_row = feature_row(features, "B")
        assert b_row["pass_through_ratio_15m"] == pytest.approx(1.0)
        # B received at +1 min and forwarded at +3 min -> 2 min retention
        assert b_row["median_retention_minutes"] == pytest.approx(2.0)
        d_row = feature_row(features, "D")
        assert d_row["pass_through_ratio_15m"] == 0.0
        assert d_row["median_retention_minutes"] is None

    def test_pass_through_respects_window(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 90.0, 30),     # 30 min later: outside the window
        ])
        row = feature_row(extract_account_features(frame), "A")
        assert row["pass_through_ratio_15m"] == 0.0
        assert row["median_retention_minutes"] is None

    def test_pass_through_ratio_capped_at_one(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 80.0, 1),
            leg("T3", "A", "C", 80.0, 2),
        ])
        row = feature_row(extract_account_features(frame), "A")
        assert row["pass_through_ratio_15m"] == 1.0

    def test_smurfing_episode_counts(self) -> None:
        rows = [leg("T0", "SRC", "COLL", 95_000.0, 0, mode="IMPS")]
        for i in range(9):
            rows.append(leg(f"T{i + 1}", "COLL", f"W{i}", 6_000.0 + i, 5 + i))
        features = extract_account_features(txns(rows))
        coll = feature_row(features, "COLL")
        assert coll["smurf_bulk_inflow_count"] == 1
        assert coll["smurf_split_count"] == 9
        # a wallet that only receives is not a smurfer
        wallet = feature_row(features, "W0")
        assert wallet["smurf_bulk_inflow_count"] == 0
        assert wallet["smurf_split_count"] == 0

    def test_smurfing_requires_minimum_split_count(self) -> None:
        rows = [leg("T0", "SRC", "COLL", 95_000.0, 0, mode="IMPS")]
        for i in range(4):
            rows.append(leg(f"T{i + 1}", "COLL", f"W{i}", 6_000.0, 5 + i))
        coll = feature_row(extract_account_features(txns(rows)), "COLL")
        assert coll["smurf_bulk_inflow_count"] == 0
        assert coll["smurf_split_count"] == 0

    def test_split_band_respects_bounds(self) -> None:
        rows = [leg("T0", "SRC", "COLL", 95_000.0, 0, mode="IMPS")]
        for i in range(9):
            amount = 12_000.0 if i == 0 else 6_000.0   # one leg above the band
            rows.append(leg(f"T{i + 1}", "COLL", f"W{i}", amount, 5 + i))
        coll = feature_row(extract_account_features(txns(rows)), "COLL")
        assert coll["smurf_split_count"] == 8

    def test_dormancy_unknown_when_no_opening_date(self) -> None:
        frame = txns([leg("T1", "V", "A", 40_000.0, 0)])
        row = feature_row(extract_account_features(frame), "A")
        assert row["dormancy_days_before_burst"] is None
        assert row["dormancy_burst_score"] is None

    def test_dormancy_burst_detected_with_opening_date(self) -> None:
        frame = txns([leg("T1", "V", "A", 40_000.0, 0)])
        accounts = pl.DataFrame([account_row("A", opened=date(2023, 4, 1))])
        row = feature_row(extract_account_features(frame, accounts), "A")
        assert row["dormancy_days_before_burst"] == pytest.approx(427.0, abs=1.0)
        assert row["dormancy_burst_score"] == 1.0
        assert row["burst_24h_max_total"] == pytest.approx(40_000.0)

    def test_recent_account_is_not_dormant(self) -> None:
        frame = txns([leg("T1", "V", "A", 40_000.0, 0)])
        accounts = pl.DataFrame([account_row("A", opened=date(2024, 5, 30))])
        row = feature_row(extract_account_features(frame, accounts), "A")
        assert row["dormancy_burst_score"] == 0.0

    def test_shared_device_and_branch_cohort(self) -> None:
        frame = txns([
            leg("T1", "A", "X", 10.0, 0, device="DEV1"),
            leg("T2", "B", "Y", 10.0, 1, device="DEV1"),
        ])
        accounts = pl.DataFrame([
            account_row("A", opened=date(2024, 5, 1), branch="BR042"),
            account_row("B", opened=date(2024, 5, 2), branch="BR042"),
            account_row("C", opened=date(2023, 1, 1), branch="BR099"),
        ])
        features = extract_account_features(frame, accounts)
        a_row = feature_row(features, "A")
        assert a_row["shared_device_account_count"] == 1
        assert a_row["branch_cohort_48h_count"] == 1
        c_row = feature_row(features, "C")
        assert c_row["shared_device_account_count"] == 0
        assert c_row["branch_cohort_48h_count"] == 0

    def test_terminal_cashout_indicator(self) -> None:
        frame = txns([
            leg("T1", "A", "TERM", 20_000.0, 0, mode="AEPS"),
            leg("T2", "B", "TERM", 25_000.0, 1, mode="AEPS"),
            leg("T3", "C", "TERM", 10.0, 2, mode="UPI"),
        ])
        term = feature_row(extract_account_features(frame), "TERM")
        assert term["aeps_cashout_inflow_count"] == 2

    def test_direct_complaint_count(self) -> None:
        from core.contracts import COMPLAINT_SCHEMA

        frame = txns([
            leg("T1", "VICTIM", "MULE", 5_000.0, 0),
            leg("T2", "X", "Y", 1.0, 1),
        ])
        complaints = pl.DataFrame(
            [{
                "ack_no": "ACK1",
                "victim_account": "VICTIM",
                "suspect_account": "MULE",
                "fraud_category": "UPI_FRAUD",
                "reported_timestamp": BASE,
                "disputed_amount": 5_000.0,
                "disputed_utr": "T1",
                "state": "Bihar",
            }],
            schema=COMPLAINT_SCHEMA,
        )
        features = extract_account_features(frame, None, complaints)
        assert feature_row(features, "VICTIM")["direct_complaint_count"] == 1
        assert feature_row(features, "MULE")["direct_complaint_count"] == 1
        assert feature_row(features, "X")["direct_complaint_count"] == 0

    def test_empty_transactions_yield_conforming_empty_frame(self) -> None:
        features = extract_account_features(pl.DataFrame(schema={
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
        }))
        assert features.height == 0
        assert list(features.columns) == list(ACCOUNT_FEATURE_SCHEMA)

    def test_accounts_without_transactions_still_appear(self) -> None:
        frame = txns([leg("T1", "A", "B", 10.0, 0)])
        accounts = pl.DataFrame([account_row("C", opened=date(2020, 1, 1))])
        features = extract_account_features(frame, accounts)
        assert set(features["account_no"].to_list()) == {"A", "B", "C"}
        row = feature_row(features, "C")
        assert row["txn_count"] == 0
        assert row["inflow_total"] == 0.0

    def test_duplicate_transaction_ids_are_tolerated(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 100.0, 0),
            leg("T1", "A", "B", 100.0, 0),
        ])
        row = feature_row(extract_account_features(frame), "B")
        assert row["inflow_count"] == 2   # counted as observed legs, no crash

    def test_features_are_deterministic(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 60.0, 1),
        ])
        first = extract_account_features(frame)
        second = extract_account_features(frame)
        assert first.equals(second)

    def test_features_are_deterministic_on_a_large_ledger(self) -> None:
        # Money aggregates are rounded to paise: a parallel float summation
        # order must never change the frame.
        sim = SyndicateDataSimulator(seed=42).generate(background_transactions=20_000)
        first = extract_account_features(sim.transactions, sim.accounts, sim.complaints)
        second = extract_account_features(sim.transactions, sim.accounts, sim.complaints)
        assert first.equals(second)

    def test_configurable_thresholds_change_results(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 60.0, 20),   # 20 minutes later
        ])
        default_row = feature_row(extract_account_features(frame), "A")
        assert default_row["pass_through_ratio_15m"] == 0.0
        wide = extract_account_features(
            frame, config=FeatureConfig(pass_through_window_minutes=60)
        )
        assert feature_row(wide, "A")["pass_through_ratio_15m"] == pytest.approx(0.6)


class TestSimulatorTopologyFeatures:
    @pytest.fixture()
    def sim(self):
        return SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)

    def test_rapid_hub_features(self, sim) -> None:
        features = extract_account_features(sim.transactions, sim.accounts)
        hubs = (
            sim.transactions.filter(pl.col("txn_id").str.starts_with("S1-"))
            .filter(pl.col("txn_id").str.ends_with("-0"))["receiver_account"]
            .to_list()
        )
        assert len(hubs) == 20
        for hub in hubs:
            row = feature_row(features, hub)
            assert row["pass_through_ratio_15m"] >= 0.9
            assert row["dormancy_burst_score"] == 1.0
            assert row["median_retention_minutes"] is not None
            assert row["median_retention_minutes"] <= 15.0

    def test_fan_out_collector_features(self, sim) -> None:
        features = extract_account_features(sim.transactions, sim.accounts)
        collectors = (
            sim.transactions.filter(pl.col("txn_id").str.starts_with("S2-"))
            .filter(pl.col("txn_id").str.ends_with("-0"))["receiver_account"]
            .to_list()
        )
        assert len(collectors) == 10
        for collector in collectors:
            row = feature_row(features, collector)
            assert row["smurf_bulk_inflow_count"] == 1
            assert row["smurf_split_count"] == 12
            assert row["unique_receivers"] == 12

    def test_terminal_cashout_features(self, sim) -> None:
        features = extract_account_features(sim.transactions, sim.accounts)
        terminal = sim.accounts.filter(
            pl.col("account_no") == "555500000010"
        )["account_no"].item()
        row = feature_row(features, terminal)
        assert row["aeps_cashout_inflow_count"] == 3

    def test_branch_ring_features(self, sim) -> None:
        features = extract_account_features(sim.transactions, sim.accounts)
        ring = sim.accounts.filter(pl.col("imei") == "351104510123456")
        assert ring.height == 6
        for account_no in ring["account_no"].to_list():
            row = feature_row(features, account_no)
            assert row["shared_device_account_count"] >= 5
            assert row["branch_cohort_48h_count"] == 5

    def test_background_traffic_is_ordinary(self, sim) -> None:
        features = extract_account_features(sim.transactions, sim.accounts)
        background_accounts = features.filter(
            ~pl.col("account_no").is_in(
                sim.transactions.filter(pl.col("is_fraud"))["sender_account"].to_list()
                + sim.transactions.filter(pl.col("is_fraud"))["receiver_account"].to_list()
            )
        )
        assert background_accounts.height > 500
        median_pass_through = background_accounts["pass_through_ratio_15m"].median()
        assert median_pass_through == 0.0
        assert background_accounts["smurf_bulk_inflow_count"].max() == 0

    def test_risk_report_schema_is_typed(self) -> None:
        assert list(RISK_REPORT_SCHEMA) == [
            "account_no", "risk_score", "domain_score", "anomaly_score",
            "risk_band", "top_reasons", "explanation_kinds", "evidence_txn_ids",
            "operational_role", "retained_tainted_amount", "freeze_priority_score",
        ]


class TestHybridRiskScorer:
    @pytest.fixture()
    def sim(self):
        return SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)

    def _features(self, sim):
        from features.account_features import extract_account_features

        return extract_account_features(sim.transactions, sim.accounts, sim.complaints)

    def _graph(self, sim):
        from graph.analyzer import GraphAnalyzer

        return GraphAnalyzer(sim.transactions, sim.accounts, sim.complaints).topology_metrics()

    def test_report_conforms_to_schema(self, sim) -> None:
        from risk.scorer import RiskScorer

        report = RiskScorer().score(self._features(sim), self._graph(sim))
        assert list(report.columns) == list(RISK_REPORT_SCHEMA)
        assert report.dtypes == [
            RISK_REPORT_SCHEMA[name] for name in report.columns
        ]
        assert report.height == self._features(sim).height

    def test_scores_are_bounded(self, sim) -> None:
        from risk.scorer import RiskScorer

        report = RiskScorer().score(self._features(sim), self._graph(sim))
        assert report["risk_score"].min() >= 0.0
        assert report["risk_score"].max() <= 100.0
        assert report["domain_score"].min() >= 0.0
        assert report["domain_score"].max() <= 100.0
        assert report["anomaly_score"].min() >= 0.0
        assert report["anomaly_score"].max() <= 100.0

    def test_scores_are_deterministic(self, sim) -> None:
        from risk.scorer import RiskScorer

        features = self._features(sim)
        graph = self._graph(sim)
        first = RiskScorer().score(features, graph)
        second = RiskScorer().score(features, graph)
        assert first.equals(second)

    def test_injected_mules_outrank_background_accounts(self, sim) -> None:
        from risk.scorer import RiskScorer

        features = self._features(sim)
        graph = self._graph(sim)
        report = RiskScorer().score(features, graph)
        hubs = (
            sim.transactions.filter(pl.col("txn_id").str.starts_with("S1-"))
            .filter(pl.col("txn_id").str.ends_with("-0"))["receiver_account"]
            .unique()
            .to_list()
        )
        mule_scores = report.filter(pl.col("account_no").is_in(hubs))["risk_score"].to_list()
        background = report.filter(~pl.col("account_no").is_in(hubs))["risk_score"]
        # every seeded hub outranks the 95th percentile of ordinary traffic
        assert min(mule_scores) > background.quantile(0.95)
        # and the ranking is dominated by seeded culprits
        top = report.sort("risk_score", descending=True).head(30)["account_no"].to_list()
        all_culprits = set().union(*culprit_accounts(sim).values())
        assert sum(1 for account in top if account in all_culprits) >= 20
        assert sum(1 for account in top if account in set(hubs)) >= 5

    def test_explanations_carry_value_threshold_and_utrs(self, sim) -> None:
        from risk.scorer import RiskScorer

        features = self._features(sim)
        report = RiskScorer().score(features, self._graph(sim))
        hub = (
            sim.transactions.filter(pl.col("txn_id").str.starts_with("S1-"))
            .filter(pl.col("txn_id").str.ends_with("-0"))["receiver_account"][0]
        )
        row = report.filter(pl.col("account_no") == hub).to_dicts()[0]
        reasons = row["top_reasons"]
        assert reasons
        assert len(reasons) <= 3
        assert any("Pass-through" in reason for reason in reasons)
        assert any("0.85" in reason or "85" in reason for reason in reasons)
        assert row["evidence_txn_ids"]
        assert all(utr.startswith("S1-") for utr in row["evidence_txn_ids"])

    def test_explanation_kinds_separate_facts_from_inferences(self, sim) -> None:
        from risk.scorer import RiskScorer

        report = RiskScorer().score(self._features(sim), self._graph(sim))
        for kinds in report["explanation_kinds"].to_list():
            assert set(kinds) <= {"FACT", "INDICATOR"}
            assert kinds

    def test_no_penalty_for_account_type_or_state(self) -> None:
        from risk.scorer import RiskScorer

        base = {
            "account_no": "A",
            "txn_count": 10,
            "inflow_count": 5,
            "outflow_count": 5,
            "inflow_total": 100.0,
            "outflow_total": 100.0,
            "net_flow": 0.0,
            "unique_senders": 5,
            "unique_receivers": 5,
            "fan_in_fan_out_asymmetry": 0.0,
            "pass_through_ratio_15m": 0.9,
            "median_retention_minutes": 2.0,
            "smurf_bulk_inflow_count": 0,
            "smurf_split_count": 0,
            "large_payout_count": 0,
            "burst_24h_max_total": 0.0,
            "dormancy_days_before_burst": None,
            "dormancy_burst_score": None,
            "shared_device_account_count": 0,
            "branch_cohort_48h_count": 0,
            "aeps_cashout_inflow_count": 0,
            "direct_complaint_count": 0,
            "pass_through_utrs": ["T1"],
            "smurf_utrs": [],
            "first_txn_timestamp": BASE,
            "last_txn_timestamp": BASE,
        }
        frames = []
        for index in range(6):
            row = dict(base)
            row["account_no"] = f"A{index}"
            frames.append(row)
        features = pl.DataFrame(frames)
        report = RiskScorer().score(features)
        assert report.height == 6
        assert len(set(report["risk_score"].to_list())) == 1

    def test_model_inputs_exclude_ground_truth_labels(self) -> None:
        from risk.scorer import ANOMALY_FEATURE_COLUMNS

        assert "is_fraud" not in ANOMALY_FEATURE_COLUMNS
        assert "fraud_scheme" not in ANOMALY_FEATURE_COLUMNS
        assert "account_type" not in ANOMALY_FEATURE_COLUMNS
        assert "kyc_pincode" not in ANOMALY_FEATURE_COLUMNS

    def test_shap_tree_explainer_works_on_isolation_forest(self) -> None:
        import numpy
        import shap
        from sklearn.ensemble import IsolationForest

        rng = numpy.random.default_rng(0)
        matrix = rng.normal(size=(120, 4))
        model = IsolationForest(random_state=42).fit(matrix)
        explainer = shap.TreeExplainer(model)
        values = explainer(matrix[:5]).values
        assert values.shape == (5, 4)
        assert numpy.isfinite(values).all()
        # deterministic: a second explainer gives the same attributions
        again = shap.TreeExplainer(model).shap_values(matrix[:5])
        assert numpy.allclose(values, again)

    def test_empty_features_return_empty_report(self) -> None:
        from risk.scorer import RiskScorer

        report = RiskScorer().score(pl.DataFrame(schema=ACCOUNT_FEATURE_SCHEMA))
        assert report.height == 0
        assert list(report.columns) == list(RISK_REPORT_SCHEMA)

    def test_small_dataset_skips_the_anomaly_component(self) -> None:
        from risk.scorer import RiskScorer

        features = pl.DataFrame([
            {
                "account_no": f"A{index}",
                "txn_count": index,
                "inflow_count": index,
                "outflow_count": index,
                "inflow_total": float(index),
                "outflow_total": float(index),
                "net_flow": 0.0,
                "unique_senders": index,
                "unique_receivers": index,
                "fan_in_fan_out_asymmetry": 0.0,
                "pass_through_ratio_15m": 0.0,
                "median_retention_minutes": None,
                "smurf_bulk_inflow_count": 0,
                "smurf_split_count": 0,
                "large_payout_count": 0,
                "burst_24h_max_total": 0.0,
                "dormancy_days_before_burst": None,
                "dormancy_burst_score": None,
                "shared_device_account_count": 0,
                "branch_cohort_48h_count": 0,
                "aeps_cashout_inflow_count": 0,
                "direct_complaint_count": 0,
                "pass_through_utrs": [],
                "smurf_utrs": [],
                "first_txn_timestamp": BASE,
                "last_txn_timestamp": BASE,
            }
            for index in range(3)
        ])
        report = RiskScorer().score(features)
        assert report.height == 3
        assert (report["anomaly_score"] == 0.0).all()

    def test_constant_and_non_finite_features_are_safe(self) -> None:
        from risk.scorer import RiskScorer

        features = pl.DataFrame([
            {
                "account_no": f"A{index}",
                "txn_count": 5,
                "inflow_count": 5,
                "outflow_count": 5,
                "inflow_total": 100.0,
                "outflow_total": 100.0,
                "net_flow": 0.0,
                "unique_senders": 2,
                "unique_receivers": 2,
                "fan_in_fan_out_asymmetry": 0.0,
                "pass_through_ratio_15m": 0.0,
                "median_retention_minutes": None,
                "smurf_bulk_inflow_count": 0,
                "smurf_split_count": 0,
                "large_payout_count": 0,
                "burst_24h_max_total": float("inf"),
                "dormancy_days_before_burst": float("nan"),
                "dormancy_burst_score": None,
                "shared_device_account_count": 0,
                "branch_cohort_48h_count": 0,
                "aeps_cashout_inflow_count": 0,
                "direct_complaint_count": 0,
                "pass_through_utrs": [],
                "smurf_utrs": [],
                "first_txn_timestamp": BASE,
                "last_txn_timestamp": BASE,
            }
            for index in range(12)
        ])
        report = RiskScorer().score(features)
        assert report.height == 12
        assert report["risk_score"].is_finite().all()
        assert report["anomaly_score"].is_finite().all()

    def test_risk_bands_follow_thresholds(self, sim) -> None:
        from core.contracts import RiskConfig
        from risk.scorer import RiskScorer

        cfg = RiskConfig()
        report = RiskScorer().score(self._features(sim), self._graph(sim))
        high = cfg.high_risk_band
        medium = cfg.medium_risk_band
        mid = (high + medium) / 2.0
        for row in report.to_dicts():
            score = row["risk_score"]
            if score >= high:
                assert row["risk_band"] == "CRITICAL"
            elif score >= mid:
                assert row["risk_band"] == "HIGH"
            elif score >= medium:
                assert row["risk_band"] == "MEDIUM"
            else:
                assert row["risk_band"] == "LOW"

    def test_typed_assessments_carry_the_disclaimer(self, sim) -> None:
        from risk.scorer import RiskScorer

        assessments = RiskScorer().assess(self._features(sim), self._graph(sim))
        assert assessments
        for assessment in assessments:
            assert 0.0 <= assessment.risk_score <= 100.0
            assert "not a probability of guilt" in assessment.disclaimer


def culprit_accounts(sim) -> dict[str, set[str]]:
    """Per-scheme culprit accounts, derived from the seeded topology.

    Culprits are the accounts that *drive* a scheme (hubs, collectors, cash-out
    mules, colluders, loop nodes). Victims, origin payers and one-off tools
    (micro wallets, hop relays) are involved but are not the fraud origin, so
    they are excluded from the culprit set and reported separately.
    """
    t = sim.transactions
    s1 = t.filter(pl.col("txn_id").str.starts_with("S1-"))
    s2 = t.filter(pl.col("txn_id").str.starts_with("S2-"))
    s3 = t.filter(pl.col("txn_id").str.starts_with("S3-"))
    s4 = t.filter(pl.col("txn_id").str.starts_with("S4-"))
    s5 = t.filter(pl.col("txn_id").str.starts_with("S5-"))
    return {
        "rapid_pass_through": set(
            s1.filter(pl.col("txn_id").str.ends_with("-0"))["receiver_account"].to_list()
        ),
        "fan_out_smurfing": set(
            s2.filter(pl.col("txn_id").str.ends_with("-0"))["receiver_account"].to_list()
        ),
        "aeps_terminal_cash_out": set(
            s3.filter(~pl.col("txn_id").str.ends_with("-0"))["sender_account"].to_list()
        ),
        "branch_collusion": set(s4["sender_account"].to_list()),
        "circular_wash": set(s5["sender_account"].to_list()),
    }


class TestSyntheticBenchmark:
    """Synthetic-only evaluation — these are NOT real-world accuracy numbers.

    Ground-truth labels are used here and nowhere else in the test suite.
    """

    @pytest.fixture()
    def sim(self):
        return SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)

    def _report(self, sim):
        from features.account_features import extract_account_features
        from graph.analyzer import GraphAnalyzer
        from risk.scorer import RiskScorer

        features = extract_account_features(sim.transactions, sim.accounts, sim.complaints)
        graph = GraphAnalyzer(
            sim.transactions, sim.accounts, sim.complaints
        ).topology_metrics()
        return RiskScorer().score(features, graph), features

    def test_recall_and_false_positives_on_synthetic_data(self, sim) -> None:
        report, features = self._report(sim)
        culprits = culprit_accounts(sim)
        all_culprits = set().union(*culprits.values())
        involved = (
            set(sim.transactions.filter(pl.col("is_fraud"))["sender_account"].to_list())
            | set(sim.transactions.filter(pl.col("is_fraud"))["receiver_account"].to_list())
        )
        background = set(features["account_no"].to_list()) - involved
        threshold = 40.0
        flagged = set(
            report.filter(pl.col("risk_score") >= threshold)["account_no"].to_list()
        )
        true_positives = flagged & all_culprits
        background_false_positives = flagged & background
        per_scheme = {
            name: len(flagged & accounts) for name, accounts in culprits.items()
        }
        print(
            "\nSYNTHETIC BENCHMARK (seed=42, 5k background transactions) — "
            "not a real-world accuracy claim"
        )
        print(f"  culprits={len(all_culprits)} flagged>={threshold}: "
              f"{len(true_positives)} (recall {len(true_positives) / len(all_culprits):.0%})")
        print(f"  background accounts={len(background)} "
              f"false positives={len(background_false_positives)}")
        print(f"  per scheme: {per_scheme}")
        top = report.sort("risk_score", descending=True).head(60)["account_no"].to_list()
        precision_at_60 = sum(1 for account in top if account in all_culprits)
        print(f"  precision@60 = {precision_at_60}/60")

        assert len(background_false_positives) <= 1, (
            f"Background false positives {len(background_false_positives)} > 1"
        )
        assert len(true_positives) >= 30
        assert per_scheme["rapid_pass_through"] >= 8
        assert per_scheme["fan_out_smurfing"] >= 8
        assert per_scheme["aeps_terminal_cash_out"] >= 2
        assert per_scheme["branch_collusion"] >= 4
        assert per_scheme["circular_wash"] >= 8
        assert precision_at_60 >= 40

    def test_no_ground_truth_label_reaches_the_scorer(self, sim) -> None:
        # The feature table is derived from ledger data only: it must not carry
        # the simulator's labels at all.
        from features.account_features import extract_account_features

        features = extract_account_features(sim.transactions, sim.accounts, sim.complaints)
        assert "is_fraud" not in features.columns
        assert "fraud_scheme" not in features.columns


def test_high_critical_bands_catch_all_five_schemes_with_strong_recall():
    """Calibration test: HIGH/CRITICAL bands must surface all 5 fraud schemes.

    After recalibration, the peak-plus-corroboration domain aggregation (or
    equivalently calibrated band thresholds) should ensure:
    - Each of the 5 synthetic schemes has non-zero recall at HIGH/CRITICAL
    - Overall actor-level recall at HIGH/CRITICAL >= 70%
    - Background false positives at HIGH/CRITICAL <= 1 (near-zero)
    - Precision against fraud-involved accounts (culprits + victims + hops) >= 95%
    """
    from features.account_features import extract_account_features
    from graph.analyzer import GraphAnalyzer
    from risk.scorer import RiskScorer
    from simulator.generator import SyndicateDataSimulator

    sim = SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)
    features = extract_account_features(sim.transactions, sim.accounts, sim.complaints)
    graph = GraphAnalyzer(sim.transactions, sim.accounts, sim.complaints).topology_metrics()
    report = RiskScorer().score(features, graph)

    culprits = culprit_accounts(sim)
    all_culprits = set().union(*culprits.values())
    involved = (
        set(sim.transactions.filter(pl.col("is_fraud"))["sender_account"].to_list())
        | set(sim.transactions.filter(pl.col("is_fraud"))["receiver_account"].to_list())
    )
    background = set(features["account_no"].to_list()) - involved

    high_critical = set(
        report.filter(pl.col("risk_band").is_in(["HIGH", "CRITICAL"]))["account_no"].to_list()
    )
    true_positives = high_critical & all_culprits
    background_fps = high_critical & background
    involved_tp = high_critical & involved

    per_scheme_recall = {
        name: len(high_critical & accounts) / len(accounts) if accounts else 0.0
        for name, accounts in culprits.items()
    }
    overall_recall = len(true_positives) / len(all_culprits) if all_culprits else 0.0
    # Precision against all fraud-involved accounts (culprits + involved non-culprits)
    involved_precision = len(involved_tp) / len(high_critical) if high_critical else 0.0

    print("\nHIGH/CRITICAL calibration check:")
    print(f"  flagged={len(high_critical)} culprits={len(true_positives)} involved={len(involved_tp)} bg_fps={len(background_fps)}")
    print(f"  overall_recall={overall_recall:.1%} involved_precision={involved_precision:.1%} bg_fps={len(background_fps)}")
    for name, recall in per_scheme_recall.items():
        print(f"    {name}: recall={recall:.1%} ({len(high_critical & culprits[name])}/{len(culprits[name])})")

    # After calibration, all 5 schemes should have non-zero recall at HIGH/CRITICAL
    for name, recall in per_scheme_recall.items():
        assert recall > 0.0, f"Scheme {name} has zero recall at HIGH/CRITICAL"

    # Overall actor-level recall at HIGH/CRITICAL >= 70%
    assert overall_recall >= 0.70, f"Overall HIGH/CRITICAL recall {overall_recall:.1%} < 70%"

    # Background false positives at HIGH/CRITICAL <= 1 (near-zero)
    assert len(background_fps) <= 1, f"Background false positives {len(background_fps)} > 1"

    # Precision against fraud-involved accounts >= 95%
    assert involved_precision >= 0.95, f"Involved precision {involved_precision:.1%} < 95%"


class TestDuplicateAccountMetadata:
    """I1: duplicate account metadata must not duplicate accounts."""

    def _accounts(self) -> pl.DataFrame:
        return pl.DataFrame([
            account_row("A", opened=date(2020, 1, 1)),
            account_row("A", opened=date(2020, 1, 1)),   # duplicate (keep-first)
            account_row("C", opened=date(2021, 1, 1)),
        ])

    def test_duplicate_metadata_cannot_duplicate_feature_rows(self) -> None:
        frame = txns([leg("T1", "A", "B", 100.0, 0)])
        features = extract_account_features(frame, self._accounts())
        assert features.height == 3                      # A, B, C
        assert features["account_no"].n_unique() == 3
        assert features["account_no"].to_list() == ["A", "B", "C"]

    def test_duplicate_metadata_cannot_duplicate_risk_rows(self) -> None:
        from risk.scorer import RiskScorer

        frame = txns([leg("T1", "A", "B", 100.0, 0)])
        features = extract_account_features(frame, self._accounts())
        report = RiskScorer().score(features)
        assert report.height == 3
        assert report["account_no"].n_unique() == report.height


class TestShapExplanationSemantics:
    """I3: the SHAP line must stay honest about what it explains."""

    @pytest.fixture()
    def sim(self):
        return SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)

    def test_shap_attribution_direction_matches_the_isolation_forest(self) -> None:
        import numpy
        import shap
        from sklearn.ensemble import IsolationForest

        from risk.scorer import ANOMALY_FEATURE_COLUMNS

        rng = numpy.random.default_rng(0)
        width = len(ANOMALY_FEATURE_COLUMNS)
        matrix = rng.normal(loc=10.0, scale=2.0, size=(300, width))
        matrix[0] = 10_000.0            # an obvious outlier
        model = IsolationForest(random_state=42).fit(matrix)
        explainer = shap.TreeExplainer(model)
        values = numpy.asarray(explainer(matrix).values, dtype=float)
        base = float(numpy.asarray(explainer.expected_value).reshape(-1)[0])
        raw = values.sum(axis=1) + base
        anomaly = -model.score_samples(matrix)
        # the explained output falls as the account becomes more anomalous
        assert numpy.corrcoef(raw, anomaly)[0, 1] < -0.9
        # ... and the scorer's sign convention (contribution = -shap) makes an
        # anomalous account's driver positive
        assert raw[0] < base
        assert float((-values[0]).max()) > 0

    def test_anomaly_explanation_states_the_scale_it_is_on(self, sim) -> None:
        from features.account_features import extract_account_features
        from risk.scorer import RiskScorer

        features = extract_account_features(sim.transactions, sim.accounts, sim.complaints)
        report = RiskScorer().score(features)
        lines = [
            reason
            for reasons in report["top_reasons"].to_list()
            for reason in reasons
            if "Isolation Forest" in reason
        ]
        assert lines
        for line in lines:
            assert "not points on the 0-100" in line
            assert "internal tree path-length" in line
