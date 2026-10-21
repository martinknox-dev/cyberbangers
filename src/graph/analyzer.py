"""LAYER 3 — graph analytics, money trails and complaint correlation.

The transaction ledger is represented as a `networkx.MultiDiGraph`: every
transaction stays a distinct edge (keyed by its UTR) carrying amount,
timestamp and payment mode, so parallel transfers and self-loops are never
collapsed. A collapsed `DiGraph` is derived for centrality and community
metrics.

All traversals are temporally ordered (a leg is only followed when it is not
earlier than the arrival of the tainted money) and every ordering is
deterministic: edges are visited in `(timestamp, txn_id)` order and no
unordered set iteration ever reaches the output.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from heapq import heappop, heappush
from typing import Final

import networkx as nx
import polars as pl

from core.contracts import (
    GRAPH_INDICATOR_SCHEMA,
    MONEY_EPSILON,
    ComplaintLink,
    GraphConfig,
    MoneyTrail,
    MoneyTrailLeg,
    WashCycle,
)

_TERMINAL_MODES: Final[tuple[str, ...]] = ("AEPS", "ATM", "CASH")

# Above this many nodes, betweenness centrality switches automatically to a
# deterministic sampled approximation (networkx `k` + `seed`). Exact betweenness
# costs O(V*E), so a work budget covers the other case: a graph with many edges
# but few accounts (a dense 100k-row ledger) is sampled too.
_BETWEENNESS_EXACT_NODE_LIMIT: Final[int] = 2_000
_BETWEENNESS_EXACT_WORK_BUDGET: Final[int] = 20_000_000
_BETWEENNESS_SAMPLE_K: Final[int] = 256

# Required vs optional transaction columns. Missing required columns are a
# caller error and raise `GraphInputError`; missing optional columns are
# supplied as nulls.
_REQUIRED_TRANSACTION_COLUMNS: Final[tuple[str, ...]] = (
    "txn_id",
    "sender_account",
    "receiver_account",
    "amount",
    "timestamp",
    "payment_mode",
)
_OPTIONAL_TRANSACTION_COLUMNS: Final[tuple[str, ...]] = (
    "sender_ifsc",
    "receiver_ifsc",
    "device_id",
    "ip_address",
)


class GraphInputError(ValueError):
    """Raised when the transaction frame cannot be interpreted.

    Carries the list of missing required columns so a caller can report the
    problem instead of leaking a raw Polars `ColumnNotFoundError`.
    """

    def __init__(self, message: str, missing_columns: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing_columns = missing_columns


@dataclass(frozen=True)
class _Edge:
    """One transaction edge, with the attributes the graph layer needs."""

    txn_id: str
    sender: str
    receiver: str
    amount: float
    timestamp: datetime
    payment_mode: str
    device_id: str | None


@dataclass
class _TaintLot:
    """A quantity of tainted funds held by an account.

    `hop` is the tainted-route depth at which the funds arrived; `arrived` is
    the timestamp from which the funds may be spent. `restricted_to`, when
    set, limits the lot to the listed transaction ids: it may only fund
    those legs (used to bound a trace to specific transfers).
    """

    amount: float
    hop: int
    arrived: datetime
    restricted_to: frozenset[str] | None = None


@dataclass(frozen=True)
class _ComplaintEvidence:
    """Ledger evidence reachable from one complaint (trail, devices, terminals)."""

    ack_no: str
    state: str
    accounts: frozenset[str]
    devices: frozenset[str]
    terminals: frozenset[str]


class GraphAnalyzer:
    """Money-trail, wash-cycle, complaint-correlation and topology analytics."""

    def __init__(
        self,
        transactions: pl.DataFrame,
        accounts: pl.DataFrame | None = None,
        complaints: pl.DataFrame | None = None,
        config: GraphConfig | None = None,
    ) -> None:
        self._config = config or GraphConfig()
        self._edges = self._prepare_edges(transactions)
        self._txn_index = {edge.txn_id: edge for edge in self._edges}
        # Keep-first lookup (Phase 1 ingestion dedupes UTRs the same way).
        self._edge_by_id: dict[str, _Edge] = {}
        for edge in self._edges:
            self._edge_by_id.setdefault(edge.txn_id, edge)
        self._accounts = accounts
        self._complaints = self._prepare_complaints(complaints)
        self._graph = self._build_multigraph()
        self._collapsed = self._build_collapsed_graph()
        self._outgoing: dict[str, list[_Edge]] = {}
        for edge in self._edges:
            self._outgoing.setdefault(edge.sender, []).append(edge)
        for node in self._outgoing:
            self._outgoing[node].sort(key=lambda e: (e.timestamp, e.txn_id))
        # Amount-ordered view per node: the decay band of a wash cycle is
        # narrow, so the cycle search jumps straight to the candidate edges
        # instead of scanning every outgoing leg of every account.
        self._amount_index: dict[str, tuple[list[float], list[_Edge]]] = {}
        for node, edges in self._outgoing.items():
            ordered = sorted(edges, key=lambda e: (e.amount, e.timestamp, e.txn_id))
            self._amount_index[node] = ([edge.amount for edge in ordered], ordered)

    # --- construction -----------------------------------------------------

    @staticmethod
    def _prepare_edges(transactions: pl.DataFrame) -> list[_Edge]:
        """Transaction edges in deterministic `(timestamp, txn_id)` order.

        Required columns must be present; optional ones default to null. An
        empty frame yields no edges rather than an error.
        """
        if transactions.height == 0:
            return []
        missing = tuple(
            name for name in _REQUIRED_TRANSACTION_COLUMNS
            if name not in transactions.columns
        )
        if missing:
            raise GraphInputError(
                "transactions frame is missing required column(s): "
                + ", ".join(missing)
                + "; required columns are "
                + ", ".join(_REQUIRED_TRANSACTION_COLUMNS),
                missing_columns=missing,
            )
        columns: list[pl.Expr] = [
            pl.col("txn_id").cast(pl.String()),
            pl.col("sender_account").cast(pl.String()),
            pl.col("receiver_account").cast(pl.String()),
            pl.col("amount").cast(pl.Float64()),
            pl.col("timestamp").cast(pl.Datetime("us")),
            pl.col("payment_mode").cast(pl.String()),
        ]
        columns.extend(
            pl.col(name).cast(pl.String())
            if name in transactions.columns
            else pl.lit(None, dtype=pl.String()).alias(name)
            for name in _OPTIONAL_TRANSACTION_COLUMNS
        )
        frame = transactions.select(columns).filter(
            pl.col("sender_account").is_not_null() & pl.col("receiver_account").is_not_null()
        )
        frame = frame.sort(["timestamp", "txn_id"])
        return [
            _Edge(
                txn_id=row["txn_id"] or "",
                sender=row["sender_account"],
                receiver=row["receiver_account"],
                amount=float(row["amount"] or 0.0),
                timestamp=row["timestamp"],
                payment_mode=row["payment_mode"] or "",
                device_id=row["device_id"],
            )
            for row in frame.iter_rows(named=True)
        ]

    @staticmethod
    def _prepare_complaints(complaints: pl.DataFrame | None) -> pl.DataFrame:
        if complaints is None or complaints.height == 0:
            return pl.DataFrame(
                schema={
                    "ack_no": pl.String(),
                    "victim_account": pl.String(),
                    "suspect_account": pl.String(),
                    "disputed_utr": pl.String(),
                    "state": pl.String(),
                    "disputed_amount": pl.Float64(),
                    "reported_timestamp": pl.Datetime("us"),
                }
            )
        required: dict[str, pl.DataType] = {
            "ack_no": pl.String(),
            "victim_account": pl.String(),
            "suspect_account": pl.String(),
            "disputed_utr": pl.String(),
            "state": pl.String(),
            "disputed_amount": pl.Float64(),
            "reported_timestamp": pl.Datetime("us"),
        }
        columns: list[pl.Expr] = []
        for name, dtype in required.items():
            if name in complaints.columns:
                columns.append(pl.col(name).cast(dtype, strict=False).alias(name))
            else:
                columns.append(pl.lit(None, dtype=dtype).alias(name))
        return complaints.select(columns).sort("ack_no")

    def _build_multigraph(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for edge in self._edges:
            key = edge.txn_id
            suffix = 1
            while graph.has_edge(edge.sender, edge.receiver, key):
                suffix += 1
                key = f"{edge.txn_id}#{suffix}"
            graph.add_edge(
                edge.sender,
                edge.receiver,
                key=key,
                txn_id=edge.txn_id,
                amount=edge.amount,
                timestamp=edge.timestamp,
                payment_mode=edge.payment_mode,
                device_id=edge.device_id,
            )
        return graph

    def _build_collapsed_graph(self) -> nx.DiGraph:
        collapsed = nx.DiGraph()
        for edge in self._edges:
            if collapsed.has_edge(edge.sender, edge.receiver):
                collapsed[edge.sender][edge.receiver]["amount"] += edge.amount
                collapsed[edge.sender][edge.receiver]["txn_count"] += 1
            else:
                collapsed.add_edge(
                    edge.sender,
                    edge.receiver,
                    amount=edge.amount,
                    txn_count=1,
                )
        return collapsed

    # --- public API -------------------------------------------------------

    def build_multigraph(self) -> nx.MultiDiGraph:
        """The transaction multigraph (one edge per transaction)."""
        return self._graph

    def build_collapsed_graph(self) -> nx.DiGraph:
        """The collapsed directed graph used for centrality and communities."""
        return self._collapsed

    def trace_money_trail(
        self,
        start: str,
        *,
        max_hops: int | None = None,
        origin_amount: float | None = None,
        start_timestamp: datetime | None = None,
        seed_txn_ids: Sequence[str] | None = None,
    ) -> MoneyTrail:
        """Trace tainted funds forward from a disputed UTR or an account.

        **Accounting model — FIFO draw-down on a tainted balance.**

        * Tainted funds are *held* by an account as a tainted balance. A
          disputed UTR seeds the balance of that transaction's receiver; an
          account seeds it with the declared amount (by default the volume
          that left the account), which becomes spendable from that account's
          first outgoing transfer.
        * The ledger is swept in chronological order. When an account sends a
          transaction, the tainted balance it holds *at that moment* is drawn
          down first-in-first-out: the transfer carries at most the tainted
          funds the sender actually held, and never more than the transfer's
          own amount.
        * Tainted funds that arrive later are credited at their arrival time,
          so a transfer can never carry funds that had not arrived yet, and a
          later arrival at an already-visited account is still processed.
        * Taint that is not propagated stays with the account that held it. It
          is never invented into a downstream transfer, and no leg is ever
          attributed twice.

        **What cannot be concluded.** The ledger does not say *which* rupees
        left an account, so the FIFO draw-down is an explicit assumption
        (tainted funds are spent first). Funds that entered the account from
        outside the traced path are *not* counted as tainted, and balances
        cannot be established for accounts whose history is not in the frame.
        The tainted amount is therefore an upper bound on the disputed money
        that can be shown to have moved along a leg.

        Args:
            start: a transaction id (disputed UTR) or an account number.
            max_hops: forward hops to follow (1..6).
            origin_amount: tainted amount to seed; for an account this is the
                volume that left it (the money a victim account is traced by),
                for a UTR it bounds the amount of that leg.
            start_timestamp: the trail starts no earlier than this time; legs
                before it are neither traversed nor tainted.
            seed_txn_ids: restrict an account-origin trace to these specific
                transfers — the tainted balance may only fund them, so the
                account's unrelated spending is never tainted.
        """
        hops = max_hops if max_hops is not None else self._config.max_hops
        hops = max(1, min(6, hops))
        restricted = (
            frozenset(seed_txn_ids) if seed_txn_ids is not None else None
        )
        origin_edge = self._txn_index.get(start)
        if origin_edge is not None:
            origin_account = origin_edge.receiver
            amount = (
                min(origin_edge.amount, origin_amount)
                if origin_amount is not None
                else origin_edge.amount
            )
            origin_txn_id = origin_edge.txn_id
            seed_time = origin_edge.timestamp
        else:
            origin_account = start
            outgoing = self._outgoing.get(start, [])
            eligible = (
                [edge for edge in outgoing if edge.txn_id in restricted]
                if restricted is not None
                else list(outgoing)
            )
            available = sum(edge.amount for edge in eligible)
            amount = (
                origin_amount if origin_amount is not None else available
            )
            if restricted is not None:
                # the pool can never exceed the transfers it may fund
                amount = min(amount, available)
            origin_txn_id = None
            earliest = eligible[0].timestamp if eligible else datetime.min
            seed_time = (
                max(start_timestamp, earliest)
                if start_timestamp is not None
                else earliest
            )

        legs: list[MoneyTrailLeg] = []
        account_taint: dict[str, float] = {origin_account: amount}
        reached: list[str] = []
        if amount <= 0:
            return MoneyTrail(
                origin_account=origin_account,
                origin_amount=max(0.0, amount),
                origin_txn_id=origin_txn_id,
                max_hops=hops,
                legs=[],
                reached_accounts=[],
                account_taint=account_taint,
                total_taint=0.0,
            )

        lots: dict[str, deque[_TaintLot]] = defaultdict(deque)
        lots[origin_account].append(
            _TaintLot(
                amount=amount,
                hop=0,
                arrived=seed_time,
                restricted_to=restricted,
            )
        )

        queue: list[tuple[datetime, str, int]] = []
        counter = 0

        def enqueue(account: str, since: datetime) -> None:
            """Queue the account's outgoing legs that are not in the past."""
            nonlocal counter
            for edge in self._outgoing.get(account, []):
                if edge.timestamp >= since:
                    heappush(queue, (edge.timestamp, edge.txn_id, counter))
                    counter += 1

        enqueue(origin_account, seed_time)
        processed: set[str] = set()
        first_hop_taint = 0.0

        while queue:
            _, txn_id, _ = heappop(queue)
            if txn_id in processed:
                continue
            processed.add(txn_id)
            edge = self._edge_by_id.get(txn_id)
            if edge is None:
                continue
            sender_lots = lots.get(edge.sender)
            if not sender_lots:
                continue
            # Plan the draw-down before applying it: the hop of the leg
            # depends on which tainted lots fund it.
            plan: list[tuple[_TaintLot, float]] = []
            remaining = edge.amount
            for lot in sender_lots:
                if remaining <= 0:
                    break
                if lot.arrived > edge.timestamp:
                    break
                if lot.restricted_to is not None and edge.txn_id not in lot.restricted_to:
                    # this lot may only fund its seed legs
                    continue
                take = min(lot.amount, remaining)
                if take > 0:
                    plan.append((lot, take))
                    remaining -= take
            if not plan:
                continue
            leg_hop = max(lot.hop for lot, _ in plan) + 1
            if leg_hop > hops:
                # Beyond the hop limit: the trail stops here and the tainted
                # balance is left untouched.
                continue
            taint = sum(take for _, take in plan)
            for lot, take in plan:
                lot.amount -= take
            while sender_lots and sender_lots[0].amount <= MONEY_EPSILON:
                sender_lots.popleft()
            legs.append(
                MoneyTrailLeg(
                    hop=leg_hop,
                    txn_id=edge.txn_id,
                    sender_account=edge.sender,
                    receiver_account=edge.receiver,
                    amount=edge.amount,
                    timestamp=edge.timestamp,
                    payment_mode=edge.payment_mode,
                    taint_amount=taint,
                )
            )
            if leg_hop == 1:
                first_hop_taint += taint
            if edge.receiver not in reached:
                reached.append(edge.receiver)
            account_taint[edge.receiver] = min(
                account_taint.get(edge.receiver, 0.0) + taint, amount
            )
            # The tainted funds arrive now: they may only be spent later.
            lots[edge.receiver].append(
                _TaintLot(amount=taint, hop=leg_hop, arrived=edge.timestamp)
            )
            enqueue(edge.receiver, edge.timestamp)

        return MoneyTrail(
            origin_account=origin_account,
            origin_amount=amount,
            origin_txn_id=origin_txn_id,
            max_hops=hops,
            legs=legs,
            reached_accounts=reached,
            account_taint=account_taint,
            total_taint=min(first_hop_taint, amount),
        )

    def _complaint_row(
        self, complaint: str | Mapping[str, object]
    ) -> Mapping[str, object] | None:
        """Resolve a complaint by acknowledgement number or take it as given."""
        if isinstance(complaint, Mapping):
            return complaint
        wanted = str(complaint)
        for row in self._complaints.to_dicts():
            if str(row.get("ack_no") or "") == wanted:
                return row
        return None

    def trace_complaint_trail(
        self,
        complaint: str | Mapping[str, object],
        *,
        max_hops: int | None = None,
    ) -> MoneyTrail:
        """Trace the money one specific complaint disputes.

        * **With a `disputed_utr` that resolves to a transaction**: the tainted
          money is that exact leg, bounded by the claimed amount
          (`min(leg.amount, disputed_amount)`); the trace continues from
          there. Unrelated victim spending is never tainted.
        * **Without a UTR**: the disputed leg is inferred from the victim's
          outgoing transfers — those to the named suspect if there is one,
          otherwise those inside `complaint_lookback_hours` before the report.
          Transfers *after* the report are never tainted.

        The returned trail is empty when the complaint is unknown or when no
        plausible origin leg exists.
        """
        hops = max_hops if max_hops is not None else self._config.max_hops
        hops = max(1, min(6, hops))
        row = self._complaint_row(complaint)
        if row is None:
            return MoneyTrail(
                origin_account="",
                origin_amount=0.0,
                max_hops=hops,
                legs=[],
                reached_accounts=[],
                account_taint={},
                total_taint=0.0,
            )

        claimed_raw = row.get("disputed_amount")
        claimed = (
            float(claimed_raw)
            if isinstance(claimed_raw, (int, float)) and not isinstance(claimed_raw, bool)
            else 0.0
        )
        utr = row.get("disputed_utr")
        edge = self._edge_by_id.get(str(utr)) if utr else None
        seed_legs: list[_Edge] = []
        if edge is not None:
            seed_legs = [edge]
        else:
            victim = str(row.get("victim_account") or "")
            suspect = str(row.get("suspect_account") or "")
            reported_raw = row.get("reported_timestamp")
            reported: datetime | None = (
                reported_raw if isinstance(reported_raw, datetime) else None
            )
            candidates = [
                candidate
                for candidate in self._outgoing.get(victim, [])
                if reported is None or candidate.timestamp <= reported
            ]
            if suspect:
                seed_legs = [
                    candidate for candidate in candidates
                    if candidate.receiver == suspect
                ]
            else:
                floor = (
                    reported - timedelta(hours=self._config.complaint_lookback_hours)
                    if reported is not None
                    else None
                )
                seed_legs = [
                    candidate for candidate in candidates
                    if floor is None or candidate.timestamp >= floor
                ]
        if not seed_legs:
            return MoneyTrail(
                origin_account=str(row.get("victim_account") or ""),
                origin_amount=0.0,
                max_hops=hops,
                legs=[],
                reached_accounts=[],
                account_taint={},
                total_taint=0.0,
            )

        available = sum(seed.amount for seed in seed_legs)
        amount = min(claimed, available) if claimed > 0 else available
        sender = seed_legs[0].sender
        seed_time = min(seed.timestamp for seed in seed_legs)
        trail = self.trace_money_trail(
            sender,
            max_hops=hops,
            origin_amount=amount,
            start_timestamp=seed_time,
            seed_txn_ids=[seed.txn_id for seed in seed_legs],
        )
        if len(seed_legs) == 1:
            trail = trail.model_copy(update={"origin_txn_id": seed_legs[0].txn_id})
        return trail

    def _total_outflow(self, account: str) -> float:
        """Total volume that left an account (its outgoing legs)."""
        return sum(
            edge.amount for edge in self._edges if edge.sender == account
        )

    def detect_wash_cycles(self) -> list[WashCycle]:
        """Closed directed cycles (<= 5 edges, <= 24 h, decaying amounts).

        A cycle is reported when every hop loses at most
        `cycle_max_decay` (15 %) of the previous hop's amount, so a genuine
        commission-style wash is separated from ordinary circular payments.
        """
        window = timedelta(hours=self._config.cycle_window_hours)
        max_length = self._config.cycle_max_length
        max_decay = self._config.cycle_max_decay
        found: dict[tuple[str, ...], WashCycle] = {}

        for start_edge in self._edges:
            cycle = self._extend_cycle(
                start_edge, window, max_length, max_decay
            )
            if cycle is None:
                continue
            key = tuple(cycle.cycle_accounts)
            if key not in found:
                found[key] = cycle
        return [found[key] for key in sorted(found)]

    def _extend_cycle(
        self,
        start_edge: _Edge,
        window: timedelta,
        max_length: int,
        max_decay: float,
    ) -> WashCycle | None:
        """Depth-first search for a decaying cycle that closes on the start."""
        horizon = start_edge.timestamp + window
        path: list[_Edge] = [start_edge]
        visited = {start_edge.sender, start_edge.receiver}
        result = self._search_cycle(
            start_edge.receiver,
            start_edge.sender,
            start_edge.amount,
            path,
            visited,
            horizon,
            max_length,
            max_decay,
        )
        if result is None:
            return None
        return self._canonical_cycle(result)

    def _search_cycle(
        self,
        node: str,
        origin: str,
        current_amount: float,
        path: list[_Edge],
        visited: set[str],
        horizon: datetime,
        max_length: int,
        max_decay: float,
    ) -> list[_Edge] | None:
        if len(path) >= max_length:
            return None
        amounts, ordered = self._amount_index.get(node, ([], []))
        low = current_amount * (1.0 - max_decay)
        start = bisect_left(amounts, low)
        end = bisect_right(amounts, current_amount)
        for edge in ordered[start:end]:
            if edge.timestamp < path[0].timestamp or edge.timestamp > horizon:
                continue
            if edge.receiver == origin:
                return [*path, edge]
            if edge.receiver in visited:
                continue
            path.append(edge)
            visited.add(edge.receiver)
            found = self._search_cycle(
                edge.receiver,
                origin,
                edge.amount,
                path,
                visited,
                horizon,
                max_length,
                max_decay,
            )
            path.pop()
            visited.discard(edge.receiver)
            if found is not None:
                return found
        return None

    def _canonical_cycle(self, edges: list[_Edge]) -> WashCycle:
        """Rotate a cycle so it always starts at its smallest account id."""
        nodes = [edge.sender for edge in edges]
        rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(len(nodes))]
        best = min(range(len(rotations)), key=lambda i: rotations[i])
        rotated = edges[best:] + edges[:best]
        rotated_nodes = list(rotations[best])
        amounts = [edge.amount for edge in rotated]
        timestamps = [edge.timestamp for edge in rotated]
        decay = 0.0
        for index in range(1, len(rotated)):
            previous = rotated[index - 1].amount
            if previous > 0:
                decay = max(decay, (previous - rotated[index].amount) / previous)
        return WashCycle(
            cycle_accounts=rotated_nodes,
            txn_ids=[edge.txn_id for edge in rotated],
            amounts=amounts,
            timestamps=timestamps,
            start_timestamp=min(timestamps),
            end_timestamp=max(timestamps),
            total_amount=sum(amounts),
            max_decay_ratio=decay,
        )

    def correlate_complaints(self) -> list[ComplaintLink]:
        """Link complaints that share ledger, device or terminal evidence.

        Only evidence present in the data creates a link: a shared downstream
        account, a shared device, or a shared cash-out terminal. The reporting
        state alone is never a link.
        """
        rows = self._complaints.to_dicts()
        trails: dict[str, _ComplaintEvidence] = {}
        for row in rows:
            ack_no = row.get("ack_no")
            if not ack_no:
                continue
            accounts: set[str] = set()
            devices: set[str] = set()
            terminals: set[str] = set()
            utr = row.get("disputed_utr")
            origin_edge = self._txn_index.get(utr) if utr else None
            if origin_edge is not None and origin_edge.device_id:
                devices.add(origin_edge.device_id)
            if utr:
                trail = self.trace_money_trail(utr, max_hops=3)
                for leg in trail.legs:
                    accounts.add(leg.receiver_account)
                    if leg.payment_mode in _TERMINAL_MODES:
                        terminals.add(leg.receiver_account)
            trails[str(ack_no)] = _ComplaintEvidence(
                ack_no=str(ack_no),
                state=str(row.get("state") or ""),
                accounts=frozenset(accounts),
                devices=frozenset(devices),
                terminals=frozenset(terminals),
            )

        links: list[ComplaintLink] = []
        keys = sorted(trails)
        for index, key_a in enumerate(keys):
            for key_b in keys[index + 1:]:
                first = trails[key_a]
                second = trails[key_b]
                shared_accounts = sorted(first.accounts & second.accounts)
                shared_devices = sorted(first.devices & second.devices)
                shared_terminals = sorted(first.terminals & second.terminals)
                if not (shared_accounts or shared_devices or shared_terminals):
                    continue
                if shared_accounts:
                    basis = "shared_account"
                elif shared_devices:
                    basis = "shared_device"
                else:
                    basis = "shared_terminal"
                links.append(
                    ComplaintLink(
                        ack_no_a=key_a,
                        ack_no_b=key_b,
                        state_a=first.state,
                        state_b=second.state,
                        shared_accounts=shared_accounts,
                        shared_device_ids=shared_devices,
                        shared_terminal_accounts=shared_terminals,
                        link_basis=basis,
                    )
                )
        return links

    def topology_metrics(self) -> pl.DataFrame:
        """Per-account centrality, community, cycle and complaint indicators."""
        nodes = sorted(self._collapsed.nodes)
        if not nodes:
            return pl.DataFrame(schema=GRAPH_INDICATOR_SCHEMA)

        pagerank = self._pagerank()
        betweenness = self._betweenness()
        communities = self._communities()

        cycle_count: dict[str, int] = {node: 0 for node in nodes}
        cycle_max_amount: dict[str, float] = {node: 0.0 for node in nodes}
        cycle_txn_ids: dict[str, list[str]] = {node: [] for node in nodes}
        for cycle in self.detect_wash_cycles():
            for account in cycle.cycle_accounts:
                cycle_count[account] = cycle_count.get(account, 0) + 1
                cycle_max_amount[account] = max(
                    cycle_max_amount.get(account, 0.0), cycle.total_amount
                )
                cycle_txn_ids[account].extend(cycle.txn_ids)

        complaint_links: dict[str, int] = {node: 0 for node in nodes}
        complaint_acks: dict[str, list[str]] = {node: [] for node in nodes}
        taint_max: dict[str, float] = {node: 0.0 for node in nodes}
        hops_max: dict[str, int] = {node: 0 for node in nodes}
        terminal_chains: dict[str, int] = {node: 0 for node in nodes}
        for link in self.correlate_complaints():
            for account in (
                *link.shared_accounts,
                *link.shared_terminal_accounts,
            ):
                complaint_links[account] = complaint_links.get(account, 0) + 1
                complaint_acks[account].extend([link.ack_no_a, link.ack_no_b])
        for row in self._complaints.to_dicts():
            ack_no = str(row.get("ack_no") or "")
            for field in ("victim_account", "suspect_account"):
                account = str(row.get(field) or "")
                if account:
                    complaint_links[account] = complaint_links.get(account, 0) + 1
                    if ack_no:
                        complaint_acks[account].append(ack_no)
            utr = row.get("disputed_utr")
            if not utr:
                continue
            trail = self.trace_money_trail(utr, max_hops=3)
            for account, taint in trail.account_taint.items():
                taint_max[account] = max(taint_max.get(account, 0.0), taint)
            for leg in trail.legs:
                hops_max[leg.receiver_account] = max(
                    hops_max.get(leg.receiver_account, 0), leg.hop
                )
                if leg.payment_mode in _TERMINAL_MODES:
                    terminal_chains[leg.receiver_account] = (
                        terminal_chains.get(leg.receiver_account, 0) + 1
                    )

        frame = pl.DataFrame(
            {
                "account_no": nodes,
                "pagerank": [pagerank.get(node, 0.0) for node in nodes],
                "betweenness_centrality": [
                    betweenness.get(node, 0.0) for node in nodes
                ],
                "louvain_community_id": [
                    communities.get(node, 0) for node in nodes
                ],
                "cycle_count": [cycle_count.get(node, 0) for node in nodes],
                "cycle_max_amount": [cycle_max_amount.get(node, 0.0) for node in nodes],
                "cycle_txn_ids": [
                    sorted(set(cycle_txn_ids.get(node, []))) for node in nodes
                ],
                "complaint_link_count": [complaint_links.get(node, 0) for node in nodes],
                "complaint_ack_nos": [
                    sorted(set(complaint_acks.get(node, []))) for node in nodes
                ],
                "trail_taint_max": [taint_max.get(node, 0.0) for node in nodes],
                "trail_hops_max": [hops_max.get(node, 0) for node in nodes],
                "terminal_chain_count": [
                    terminal_chains.get(node, 0) for node in nodes
                ],
            }
        )
        return frame.select(
            [
                pl.col(name).cast(dtype, strict=False).alias(name)
                for name, dtype in GRAPH_INDICATOR_SCHEMA.items()
            ]
        ).sort("account_no")

    def _pagerank(self) -> dict[str, float]:
        if self._collapsed.number_of_nodes() == 0:
            return {}
        raw = nx.pagerank(
            self._collapsed, alpha=self._config.pagerank_alpha, weight=None
        )
        return {str(node): float(value) for node, value in raw.items()}

    def _betweenness(self) -> dict[str, float]:
        if self._collapsed.number_of_nodes() == 0:
            return {}
        nodes = self._collapsed.number_of_nodes()
        edges = self._collapsed.number_of_edges()
        k = self._config.betweenness_k
        if k is None and (
            nodes > _BETWEENNESS_EXACT_NODE_LIMIT
            or nodes * edges > _BETWEENNESS_EXACT_WORK_BUDGET
        ):
            # Exact betweenness is O(V*E); on a large graph the deterministic
            # sampled approximation keeps a 100k-row run within a few seconds.
            k = min(_BETWEENNESS_SAMPLE_K, nodes)
        raw = nx.betweenness_centrality(
            self._collapsed,
            weight=None,
            k=k,
            seed=self._config.community_seed,
        )
        return {str(node): float(value) for node, value in raw.items()}

    def _communities(self) -> dict[str, int]:
        """Louvain communities with a fixed seed and stable community ids."""
        if self._collapsed.number_of_nodes() == 0:
            return {}
        undirected = self._collapsed.to_undirected()
        communities = nx.community.louvain_communities(
            undirected, seed=self._config.community_seed
        )
        ordered = sorted((sorted(group) for group in communities), key=lambda g: g)
        assignment: dict[str, int] = {}
        for community_id, group in enumerate(ordered):
            for member in group:
                assignment[member] = community_id
        return assignment
