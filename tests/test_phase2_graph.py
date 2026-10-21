"""Phase 2 acceptance tests: graph analytics, money trails, complaint links."""
from datetime import datetime, timedelta

import polars as pl
import pytest

from core.contracts import GRAPH_INDICATOR_SCHEMA, GraphConfig
from graph.analyzer import GraphAnalyzer
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


def complaint(
    ack_no: str,
    victim: str,
    suspect: str | None,
    utr: str | None,
    amount: float,
    state: str,
    minutes: float = 0.0,
) -> dict[str, object]:
    return {
        "ack_no": ack_no,
        "victim_account": victim,
        "suspect_account": suspect,
        "fraud_category": "UPI_FRAUD",
        "reported_timestamp": BASE + timedelta(minutes=minutes),
        "disputed_amount": amount,
        "disputed_utr": utr,
        "state": state,
    }


class TestGraphBasics:
    def test_parallel_transactions_stay_distinct(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 100.0, 0),
            leg("T2", "A", "B", 250.0, 5),
        ])
        graph = GraphAnalyzer(frame).build_multigraph()
        assert graph.number_of_edges() == 2
        edges = sorted(
            (data["txn_id"], data["amount"]) for _, _, data in graph.edges(data=True)
        )
        assert edges == [("T1", 100.0), ("T2", 250.0)]

    def test_edge_attributes_are_preserved(self) -> None:
        frame = txns([leg("T1", "A", "B", 100.0, 0, mode="IMPS")])
        graph = GraphAnalyzer(frame).build_multigraph()
        data = graph.edges["A", "B", "T1"]
        assert data["txn_id"] == "T1"
        assert data["amount"] == 100.0
        assert data["timestamp"] == BASE
        assert data["payment_mode"] == "IMPS"

    def test_self_loop_is_kept(self) -> None:
        frame = txns([leg("T1", "A", "A", 10.0, 0)])
        graph = GraphAnalyzer(frame).build_multigraph()
        assert graph.number_of_edges() == 1
        assert graph.has_edge("A", "A")

    def test_empty_graph_is_safe(self) -> None:
        empty = pl.DataFrame(schema={
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
        })
        analyzer = GraphAnalyzer(empty)
        assert analyzer.build_multigraph().number_of_edges() == 0
        metrics = analyzer.topology_metrics()
        assert metrics.height == 0
        assert list(metrics.columns) == list(GRAPH_INDICATOR_SCHEMA)

    def test_disconnected_components_are_kept(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 10.0, 0),
            leg("T2", "C", "D", 10.0, 1),
        ])
        graph = GraphAnalyzer(frame).build_multigraph()
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 2


class TestMoneyTrail:
    def test_trail_hops_and_chronological_order(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 60.0, 5),
            leg("T3", "B", "C", 60.0, 10),
            leg("T4", "C", "D", 60.0, 15),
        ])
        trail = GraphAnalyzer(frame).trace_money_trail("V", max_hops=3)
        # hop 1 is the victim's own outgoing leg, then one hop per layer
        assert [leg_.txn_id for leg_ in trail.legs] == ["T1", "T2", "T3"]
        assert [leg_.hop for leg_ in trail.legs] == [1, 2, 3]
        assert trail.legs[0].timestamp < trail.legs[1].timestamp

    def test_trail_from_disputed_utr(self) -> None:
        frame = txns([
            leg("T1", "V", "MULE", 100.0, 0),
            leg("T2", "MULE", "NEXT", 90.0, 5),
        ])
        trail = GraphAnalyzer(frame).trace_money_trail("T1", max_hops=2)
        assert trail.origin_txn_id == "T1"
        assert trail.origin_account == "MULE"
        assert trail.origin_amount == 100.0
        assert [leg_.txn_id for leg_ in trail.legs] == ["T2"]

    def test_taint_is_proportional_and_conserved(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 60.0, 1),
            leg("T3", "A", "C", 40.0, 2),
            leg("T4", "B", "D", 60.0, 3),
        ])
        trail = GraphAnalyzer(frame).trace_money_trail("V", max_hops=3)
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint["T1"] == pytest.approx(100.0)
        assert taint["T2"] == pytest.approx(60.0)
        assert taint["T3"] == pytest.approx(40.0)
        assert taint["T4"] == pytest.approx(60.0)
        # the origin's outflow carries the whole disputed amount, never more
        assert trail.total_taint == pytest.approx(100.0)
        assert trail.taint_conserved
        for leg_ in trail.legs:
            assert leg_.taint_amount <= trail.origin_amount + 1e-9

    def test_taint_never_exceeds_origin_on_cycles(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 100.0, 1),
            leg("T3", "B", "A", 100.0, 2),
        ])
        trail = GraphAnalyzer(frame).trace_money_trail("V", max_hops=3)
        for leg_ in trail.legs:
            assert leg_.taint_amount <= 100.0 + 1e-9
        assert trail.taint_conserved

    def test_trail_respects_max_hops(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 100.0, 1),
            leg("T3", "B", "C", 100.0, 2),
            leg("T4", "C", "D", 100.0, 3),
        ])
        trail = GraphAnalyzer(frame).trace_money_trail("V", max_hops=1)
        assert [leg_.txn_id for leg_ in trail.legs] == ["T1"]

    def test_trail_never_walks_backwards_in_time(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 10),
            leg("T2", "A", "B", 100.0, 20),
            leg("T3", "B", "C", 100.0, 5),   # earlier than B's arrival
        ])
        trail = GraphAnalyzer(frame).trace_money_trail("V", max_hops=3)
        # T3 happened before the tainted money reached B, so it is not followed
        assert [leg_.txn_id for leg_ in trail.legs] == ["T1", "T2"]

    def test_trail_is_deterministic(self) -> None:
        frame = txns([
            leg("T1", "V", "A", 100.0, 0),
            leg("T2", "A", "B", 60.0, 1),
            leg("T3", "A", "C", 40.0, 2),
        ])
        analyzer = GraphAnalyzer(frame)
        first = analyzer.trace_money_trail("V", max_hops=2)
        second = analyzer.trace_money_trail("V", max_hops=2)
        assert [leg_.txn_id for leg_ in first.legs] == [
            leg_.txn_id for leg_ in second.legs
        ]
        assert [leg_.taint_amount for leg_ in first.legs] == [
            leg_.taint_amount for leg_ in second.legs
        ]


class TestWashCycles:
    def test_wash_cycle_detected(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 100.0, 0),
            leg("T2", "B", "C", 97.0, 10),
            leg("T3", "C", "A", 94.0, 20),
        ])
        cycles = GraphAnalyzer(frame).detect_wash_cycles()
        assert len(cycles) == 1
        cycle = cycles[0]
        assert cycle.cycle_accounts == ["A", "B", "C"]
        assert cycle.txn_ids == ["T1", "T2", "T3"]
        assert cycle.max_decay_ratio <= 0.15

    def test_cycle_requires_monotone_decay(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 100.0, 0),
            leg("T2", "B", "C", 50.0, 10),   # 50% decay: outside the band
            leg("T3", "C", "A", 50.0, 20),
        ])
        assert GraphAnalyzer(frame).detect_wash_cycles() == []

    def test_cycle_requires_time_window(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 100.0, 0),
            leg("T2", "B", "C", 97.0, 60 * 30),   # 30 hours later
            leg("T3", "C", "A", 94.0, 60 * 40),
        ])
        assert GraphAnalyzer(frame).detect_wash_cycles() == []

    def test_cycles_are_deduplicated(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 100.0, 0),
            leg("T2", "B", "C", 97.0, 10),
            leg("T3", "C", "A", 94.0, 20),
        ])
        analyzer = GraphAnalyzer(frame)
        assert len(analyzer.detect_wash_cycles()) == 1
        assert len(analyzer.detect_wash_cycles()) == 1


class TestComplaintCorrelation:
    def _frame(self) -> pl.DataFrame:
        return txns([
            leg("T1", "V1", "M1", 100.0, 0, device="D1"),
            leg("T2", "M1", "M2", 90.0, 5),
            leg("T3", "V2", "M3", 200.0, 60, device="D2"),
            leg("T4", "M3", "M2", 180.0, 65),
        ])

    def test_complaints_link_through_shared_account(self) -> None:
        complaints = pl.DataFrame([
            complaint("ACK1", "V1", "M1", "T1", 100.0, "Bihar"),
            complaint("ACK2", "V2", "M3", "T3", 200.0, "Delhi"),
        ])
        analyzer = GraphAnalyzer(self._frame(), complaints=complaints)
        links = analyzer.correlate_complaints()
        assert len(links) == 1
        link = links[0]
        assert link.ack_no_a == "ACK1"
        assert link.ack_no_b == "ACK2"
        assert link.state_a == "Bihar" and link.state_b == "Delhi"
        assert "M2" in link.shared_accounts

    def test_complaints_without_shared_evidence_are_not_linked(self) -> None:
        frame = txns([
            leg("T1", "V1", "M1", 100.0, 0),
            leg("T2", "V2", "M9", 200.0, 60),
        ])
        complaints = pl.DataFrame([
            complaint("ACK1", "V1", "M1", "T1", 100.0, "Bihar"),
            complaint("ACK2", "V2", "M9", "T2", 200.0, "Bihar"),
        ])
        links = GraphAnalyzer(frame, complaints=complaints).correlate_complaints()
        assert links == []

    def test_same_state_alone_never_links_complaints(self) -> None:
        frame = txns([
            leg("T1", "V1", "M1", 100.0, 0),
            leg("T2", "V2", "M9", 200.0, 60),
        ])
        complaints = pl.DataFrame([
            complaint("ACK1", "V1", "M1", "T1", 100.0, "Bihar"),
            complaint("ACK2", "V2", "M9", "T2", 200.0, "Bihar"),
        ])
        assert GraphAnalyzer(frame, complaints=complaints).correlate_complaints() == []

    def test_shared_device_links_complaints(self) -> None:
        frame = txns([
            leg("T1", "V1", "M1", 100.0, 0, device="DEV9"),
            leg("T2", "V2", "M9", 200.0, 60, device="DEV9"),
        ])
        complaints = pl.DataFrame([
            complaint("ACK1", "V1", "M1", "T1", 100.0, "Bihar"),
            complaint("ACK2", "V2", "M9", "T2", 200.0, "Delhi"),
        ])
        links = GraphAnalyzer(frame, complaints=complaints).correlate_complaints()
        assert len(links) == 1
        assert links[0].shared_device_ids == ["DEV9"]
        assert links[0].link_basis == "shared_device"

    def test_ack_numbers_are_preserved(self) -> None:
        complaints = pl.DataFrame([
            complaint("ACK2024000001", "V1", "M1", "T1", 100.0, "Bihar"),
        ])
        analyzer = GraphAnalyzer(self._frame(), complaints=complaints)
        links = analyzer.correlate_complaints()
        assert all("ACK" in link.ack_no_a for link in links)


class TestTopologyMetrics:
    def test_metrics_cover_every_account(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 10.0, 0),
            leg("T2", "B", "C", 10.0, 1),
        ])
        metrics = GraphAnalyzer(frame).topology_metrics()
        assert set(metrics["account_no"].to_list()) == {"A", "B", "C"}
        assert list(metrics.columns) == list(GRAPH_INDICATOR_SCHEMA)

    def test_pagerank_sums_to_one(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 10.0, 0),
            leg("T2", "B", "C", 10.0, 1),
            leg("T3", "A", "C", 10.0, 2),
        ])
        metrics = GraphAnalyzer(frame).topology_metrics()
        assert metrics["pagerank"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_bridge_account_has_high_betweenness(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 10.0, 0),
            leg("T2", "C", "B", 10.0, 1),
            leg("T3", "B", "D", 10.0, 2),
        ])
        metrics = GraphAnalyzer(frame).topology_metrics()
        betweenness = dict(zip(
            metrics["account_no"].to_list(),
            metrics["betweenness_centrality"].to_list(),
        ))
        assert betweenness["B"] > betweenness["A"]

    def test_community_ids_are_stable(self) -> None:
        frame = txns([
            leg("T1", "A", "B", 10.0, 0),
            leg("T2", "B", "A", 10.0, 1),
            leg("T3", "X", "Y", 10.0, 2),
            leg("T4", "Y", "X", 10.0, 3),
        ])
        first = GraphAnalyzer(frame).topology_metrics()
        second = GraphAnalyzer(frame).topology_metrics()
        assert first.equals(second)
        communities = dict(zip(
            first["account_no"].to_list(),
            first["louvain_community_id"].to_list(),
        ))
        assert communities["A"] == communities["B"]
        assert communities["X"] == communities["Y"]
        assert communities["A"] != communities["X"]


class TestSimulatorGraph:
    @pytest.fixture()
    def sim(self):
        return SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)

    def test_circular_wash_cycles_are_found(self, sim) -> None:
        analyzer = GraphAnalyzer(sim.transactions, sim.accounts, sim.complaints)
        cycles = analyzer.detect_wash_cycles()
        # 6 three-hop circular-wash loops, plus the branch ring's reciprocal
        # two-hop ping-pong (also a decaying closed loop)
        three_hop = [cycle for cycle in cycles if len(cycle.cycle_accounts) == 3]
        assert len(three_hop) == 6
        for cycle in three_hop:
            assert cycle.max_decay_ratio <= 0.15
        assert all(cycle.max_decay_ratio <= 0.15 for cycle in cycles)

    def test_complaints_correlate_across_states(self, sim) -> None:
        analyzer = GraphAnalyzer(sim.transactions, sim.accounts, sim.complaints)
        links = analyzer.correlate_complaints()
        assert links, "the seeded complaints should share downstream mules"
        for link in links:
            assert link.ack_no_a and link.ack_no_b
            assert link.link_basis

    def test_trail_from_simulated_dispute(self, sim) -> None:
        analyzer = GraphAnalyzer(sim.transactions, sim.accounts, sim.complaints)
        utr = sim.complaints["disputed_utr"][0]
        trail = analyzer.trace_money_trail(utr, max_hops=3)
        assert trail.legs
        assert trail.origin_amount > 0
        assert trail.taint_conserved

    def test_metrics_are_deterministic(self, sim) -> None:
        first = GraphAnalyzer(sim.transactions, sim.accounts, sim.complaints).topology_metrics()
        second = GraphAnalyzer(sim.transactions, sim.accounts, sim.complaints).topology_metrics()
        assert first.equals(second)

    def test_sampled_betweenness_is_deterministic(self, sim) -> None:
        from core.contracts import GraphConfig

        config = GraphConfig(betweenness_k=50)
        first = GraphAnalyzer(
            sim.transactions, sim.accounts, sim.complaints, config=config
        ).topology_metrics()
        second = GraphAnalyzer(
            sim.transactions, sim.accounts, sim.complaints, config=config
        ).topology_metrics()
        assert first.equals(second)


class TestTaintAccounting:
    """C1/M1/M2: taint is a per-transaction amount that never exceeds the
    transaction, is drawn down chronologically from a tainted balance, and is
    never attributed to a transfer that predates the arrival of the funds."""

    def _trail(self, rows, start, max_hops=3):
        return GraphAnalyzer(pl.DataFrame(rows)).trace_money_trail(start, max_hops=max_hops)

    def test_small_legs_cannot_carry_more_taint_than_they_moved(self) -> None:
        # 5,000 arrives, only 1 leaves to each of two accounts.
        rows = [
            leg("T1", "V", "A", 5000, 0),
            leg("T2", "A", "X", 1, 1),
            leg("T3", "A", "Y", 1, 2),
        ]
        trail = self._trail(rows, "T1")
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint["T2"] == pytest.approx(1.0)
        assert taint["T3"] == pytest.approx(1.0)
        assert sum(taint.values()) == pytest.approx(2.0)
        for leg_ in trail.legs:
            assert leg_.taint_amount <= leg_.amount + 1e-9

    def test_large_legs_carry_their_own_amount_and_leave_a_remainder(self) -> None:
        # 40,000 arrives, 18,400 + 18,800 leave: 2,800 is never propagated.
        rows = [
            leg("T1", "V", "A", 40_000, 0),
            leg("T2", "A", "B", 18_400, 1),
            leg("T3", "A", "C", 18_800, 2),
        ]
        trail = self._trail(rows, "T1")
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint["T2"] == pytest.approx(18_400.0)
        assert taint["T3"] == pytest.approx(18_800.0)
        assert sum(taint.values()) == pytest.approx(37_200.0)
        assert sum(taint.values()) <= trail.origin_amount + 1e-9

    def test_fan_out_with_competing_outgoing_transfers(self) -> None:
        # 1,000 tainted funds, four transfers of 300: the tainted balance is
        # drawn down in order and never exceeds the pool.
        rows = [
            leg("T1", "V", "A", 1000, 0),
            leg("T2", "A", "B", 300, 1),
            leg("T3", "A", "C", 300, 2),
            leg("T4", "A", "D", 300, 3),
            leg("T5", "A", "E", 300, 4),
        ]
        trail = self._trail(rows, "T1")
        taint = [leg_.taint_amount for leg_ in trail.legs]
        assert taint == pytest.approx([300.0, 300.0, 300.0, 100.0])
        assert sum(taint) == pytest.approx(1000.0)

    def test_cycle_accounting(self) -> None:
        rows = [
            leg("T1", "V", "A", 100, 0),
            leg("T2", "A", "B", 90, 5),
            leg("T3", "B", "A", 80, 10),
            leg("T4", "A", "C", 70, 15),
        ]
        trail = self._trail(rows, "T1")
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint["T2"] == pytest.approx(90.0)
        assert taint["T3"] == pytest.approx(80.0)
        assert taint["T4"] == pytest.approx(70.0)
        for leg_ in trail.legs:
            assert leg_.taint_amount <= leg_.amount + 1e-9

    def test_converging_paths(self) -> None:
        # A holds 200 tainted and splits it; B and C both fund D.
        rows = [
            leg("T1", "V", "A", 200, 0),
            leg("T2", "A", "B", 100, 1),
            leg("T3", "A", "C", 100, 2),
            leg("T4", "B", "D", 60, 5),
            leg("T5", "C", "D", 40, 8),
            leg("T6", "D", "E", 90, 10),
        ]
        trail = self._trail(rows, "T1")
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint["T2"] == pytest.approx(100.0)
        assert taint["T3"] == pytest.approx(100.0)
        assert taint["T4"] == pytest.approx(60.0)
        assert taint["T5"] == pytest.approx(40.0)
        # D received 60 + 40 of tainted money and can pass on at most that.
        assert taint["T6"] == pytest.approx(90.0)
        assert trail.account_taint["D"] == pytest.approx(100.0)

    def test_earlier_transfer_excludes_a_later_arrival(self) -> None:
        # A is tainted at t=0 and sends at t=10; it only receives more tainted
        # funds at t=60, so the t=10 transfer must not carry them.
        rows = [
            leg("T1", "V", "A", 100, 0),
            leg("T2", "A", "X", 50, 10),
            leg("T3", "A", "W", 50, 20),
            leg("T4", "W", "A", 100, 60),
            leg("T5", "A", "Y", 100, 70),
        ]
        trail = self._trail(rows, "T1")
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint["T2"] == pytest.approx(50.0)   # funded by the t=0 arrival
        assert taint["T4"] == pytest.approx(50.0)   # W held 50 tainted
        # the t=70 transfer draws on the balance left plus the t=60 arrival
        assert taint["T5"] == pytest.approx(50.0)

    def test_later_arrival_is_processed_at_an_already_visited_account(self) -> None:
        # A is visited at t=0 and again at t=60; the later arrival must still
        # be credited so the t=70 transfer can carry it.
        rows = [
            leg("T1", "V", "A", 100, 0),
            leg("T2", "A", "W", 100, 20),
            leg("T3", "W", "A", 100, 60),
            leg("T4", "A", "Y", 100, 70),
        ]
        trail = self._trail(rows, "T1")
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint["T2"] == pytest.approx(100.0)
        assert taint["T3"] == pytest.approx(100.0)
        assert taint["T4"] == pytest.approx(100.0)

    def test_same_timestamp_transactions_are_deterministic(self) -> None:
        rows = [
            leg("T1", "V", "A", 100, 0),
            leg("L2", "A", "Y", 40, 1),
            leg("L1", "A", "X", 60, 1),
        ]
        first = self._trail(rows, "T1")
        second = self._trail(rows, "T1")
        assert [leg_.txn_id for leg_ in first.legs] == ["L1", "L2"]
        assert [leg_.taint_amount for leg_ in first.legs] == [
            leg_.taint_amount for leg_ in second.legs
        ]
        assert first.legs[0].taint_amount == pytest.approx(60.0)
        assert first.legs[1].taint_amount == pytest.approx(40.0)

    def test_unpropagated_taint_is_not_fabricated(self) -> None:
        rows = [
            leg("T1", "V", "A", 5000, 0),
            leg("T2", "A", "X", 1, 1),
        ]
        trail = self._trail(rows, "T1")
        assert sum(leg_.taint_amount for leg_ in trail.legs) == pytest.approx(1.0)
        assert trail.total_taint == pytest.approx(1.0)
        assert trail.taint_conserved

    def test_contract_rejects_taint_above_the_transaction_amount(self) -> None:
        from pydantic import ValidationError

        from core.contracts import MoneyTrail, MoneyTrailLeg

        with pytest.raises(ValidationError):
            MoneyTrailLeg(
                hop=1, txn_id="T1", sender_account="A", receiver_account="B",
                amount=10.0, timestamp=BASE, payment_mode="UPI", taint_amount=50.0,
            )
        with pytest.raises(ValidationError):
            MoneyTrail(
                origin_account="A", origin_amount=100.0, total_taint=50.0,
                legs=[MoneyTrailLeg(
                    hop=1, txn_id="T1", sender_account="A", receiver_account="B",
                    amount=10.0, timestamp=BASE, payment_mode="UPI",
                    taint_amount=50.0,
                )],
            )


class TestGraphInputValidation:
    """I2: required vs optional transaction columns, and clear failures."""

    def test_missing_required_column_raises_a_clear_error(self) -> None:
        from graph.analyzer import GraphInputError

        with pytest.raises(GraphInputError) as excinfo:
            GraphAnalyzer(pl.DataFrame({"txn_id": ["T1"]}))
        message = str(excinfo.value)
        assert "sender_account" in message
        assert "missing" in message.lower()

    def test_optional_columns_default_to_null(self) -> None:
        frame = pl.DataFrame({
            "txn_id": ["T1"],
            "sender_account": ["A"],
            "receiver_account": ["B"],
            "amount": [10.0],
            "timestamp": [BASE],
            "payment_mode": ["UPI"],
        })
        analyzer = GraphAnalyzer(frame)
        graph = analyzer.build_multigraph()
        assert graph.number_of_edges() == 1
        data = graph.edges["A", "B", "T1"]
        assert data["device_id"] is None
        assert data["amount"] == 10.0

    def test_empty_frame_is_safe(self) -> None:
        analyzer = GraphAnalyzer(pl.DataFrame())
        assert analyzer.build_multigraph().number_of_edges() == 0
        assert analyzer.topology_metrics().height == 0

    def test_frame_with_only_the_required_columns_is_usable(self) -> None:
        frame = pl.DataFrame({
            "txn_id": ["T1", "T2"],
            "sender_account": ["A", "A"],
            "receiver_account": ["B", "C"],
            "amount": [10.0, 20.0],
            "timestamp": [BASE, BASE + timedelta(minutes=1)],
            "payment_mode": ["UPI", "IMPS"],
        })
        analyzer = GraphAnalyzer(frame)
        assert analyzer.build_multigraph().number_of_edges() == 2
        trail = analyzer.trace_money_trail("A", max_hops=1)
        assert [leg_.txn_id for leg_ in trail.legs] == ["T1", "T2"]


class TestComplaintScopedTrail:
    """Step 1: a complaint-scoped trace must never taint unrelated victim
    spending, and must work with and without a disputed UTR."""

    def _ledger(self) -> pl.DataFrame:
        return pl.DataFrame([
            # the victim's ordinary spending BEFORE the fraud
            leg("P1", "V", "SHOP", 2_000, 0),
            leg("P2", "V", "SHOP", 3_000, 30),
            # the disputed transfer
            leg("D1", "V", "MULE", 100, 60),
            # the mule moves the money on
            leg("D2", "MULE", "MULE2", 90, 90),
            # unrelated spending AFTER the fraud
            leg("P3", "V", "SHOP", 1_500, 120),
        ])

    def _complaint(self, utr: str | None = "D1", amount: float = 100.0):
        return {
            "ack_no": "ACK1",
            "victim_account": "V",
            "suspect_account": "MULE",
            "fraud_category": "UPI_FRAUD",
            "reported_timestamp": BASE + timedelta(minutes=180),
            "disputed_amount": amount,
            "disputed_utr": utr,
            "state": "Bihar",
        }

    def test_complaint_trail_does_not_taint_pre_fraud_victim_spending(self) -> None:
        analyzer = GraphAnalyzer(self._ledger())
        trail = analyzer.trace_complaint_trail(self._complaint(), max_hops=3)
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert trail.origin_amount == pytest.approx(100.0)
        # the disputed leg carries the tainted money, the mule's onward leg too
        assert taint["D1"] == pytest.approx(100.0)
        assert taint["D2"] == pytest.approx(90.0)
        # the victim's unrelated spending is never tainted
        assert "P1" not in taint
        assert "P2" not in taint
        assert "P3" not in taint
        assert trail.total_taint == pytest.approx(100.0)

    def test_complaint_trail_uses_the_utr_leg(self) -> None:
        analyzer = GraphAnalyzer(self._ledger())
        trail = analyzer.trace_complaint_trail(self._complaint(), max_hops=3)
        assert trail.origin_txn_id == "D1"
        assert trail.origin_account == "V"
        assert [leg_.txn_id for leg_ in trail.legs][:2] == ["D1", "D2"]

    def test_complaint_trail_bounds_the_claim_to_the_ledger_amount(self) -> None:
        analyzer = GraphAnalyzer(self._ledger())
        # the complaint claims more than the ledger leg moved
        trail = analyzer.trace_complaint_trail(self._complaint(amount=5_000.0), max_hops=3)
        assert trail.origin_amount == pytest.approx(100.0)
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint["D1"] == pytest.approx(100.0)

    def test_complaint_trail_without_utr_falls_back_to_the_suspect_leg(self) -> None:
        analyzer = GraphAnalyzer(self._ledger())
        trail = analyzer.trace_complaint_trail(self._complaint(utr=None), max_hops=3)
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        # the victim's payment to the named suspect is the inferred origin
        assert taint.get("D1") == pytest.approx(100.0)
        assert taint.get("D2") == pytest.approx(90.0)
        assert "P1" not in taint and "P2" not in taint and "P3" not in taint
        assert trail.origin_amount == pytest.approx(100.0)

    def test_complaint_trail_without_utr_uses_the_lookback_window(self) -> None:
        # No suspect: only the victim's transfers inside the lookback window
        # before the report can be the disputed one. The earlier spending
        # (300 minutes before the report) is outside a 2-hour window.
        rows = [
            leg("P1", "V", "SHOP", 2_000, 0),
            leg("D1", "V", "MULE", 100, 180),
            leg("D2", "MULE", "MULE2", 90, 190),
        ]
        analyzer = GraphAnalyzer(
            pl.DataFrame(rows), config=GraphConfig(complaint_lookback_hours=2)
        )
        complaint = self._complaint(utr=None)
        complaint["suspect_account"] = None
        complaint["reported_timestamp"] = BASE + timedelta(minutes=300)
        trail = analyzer.trace_complaint_trail(complaint, max_hops=3)
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint.get("D1") == pytest.approx(100.0)
        assert taint.get("D2") == pytest.approx(90.0)
        assert "P1" not in taint

    def test_complaint_trail_never_taints_spending_after_the_report(self) -> None:
        analyzer = GraphAnalyzer(self._ledger())
        complaint = self._complaint(utr=None)
        complaint["reported_timestamp"] = BASE + timedelta(minutes=90)
        trail = analyzer.trace_complaint_trail(complaint, max_hops=3)
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert "P3" not in taint     # P3 is after the report

    def test_complaint_trail_for_an_unknown_complaint_is_empty(self) -> None:
        analyzer = GraphAnalyzer(self._ledger())
        trail = analyzer.trace_complaint_trail("ACK-NOT-THERE", max_hops=3)
        assert trail.legs == []
        assert trail.total_taint == 0.0

    def test_account_trace_can_be_bounded_by_utr(self) -> None:
        analyzer = GraphAnalyzer(self._ledger())
        trail = analyzer.trace_money_trail(
            "V", max_hops=3, origin_amount=100.0, seed_txn_ids=["D1"],
        )
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint.get("D1") == pytest.approx(100.0)
        assert taint.get("D2") == pytest.approx(90.0)
        assert "P1" not in taint and "P2" not in taint and "P3" not in taint

    def test_account_trace_can_be_bounded_by_timestamp(self) -> None:
        analyzer = GraphAnalyzer(self._ledger())
        trail = analyzer.trace_money_trail(
            "V", max_hops=3, start_timestamp=BASE + timedelta(minutes=60),
        )
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        # spending before the bound is neither traversed nor tainted
        assert "P1" not in taint and "P2" not in taint
        assert taint.get("D1") == pytest.approx(100.0)

    def test_generic_account_trace_still_seeds_all_outgoing_volume(self) -> None:
        analyzer = GraphAnalyzer(self._ledger())
        trail = analyzer.trace_money_trail("V", max_hops=3)
        assert trail.origin_amount == pytest.approx(6_600.0)
        taint = {leg_.txn_id: leg_.taint_amount for leg_ in trail.legs}
        assert taint.get("P1") == pytest.approx(2_000.0)
        assert taint.get("D1") == pytest.approx(100.0)
