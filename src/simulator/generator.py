"""Deterministic synthetic syndicate environment with ground-truth labels.

Same seed => identical frames, every run, every machine. The five fraud
schemes are pure arithmetic (no RNG); only the legitimate background and
the account master consume the seeded NumPy generator, in fixed order:
accounts first, then background. Task 7 completes the scheme builders at
the marked seam.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Final

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from core.contracts import (
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_SCHEMA,
    TRANSACTION_SCHEMA,
)

SCHEME_RAPID_PASS_THROUGH: Final[str] = "rapid_pass_through"
SCHEME_FAN_OUT_SMURFING: Final[str] = "fan_out_smurfing"
SCHEME_AEPS_TERMINAL_CASH_OUT: Final[str] = "aeps_terminal_cash_out"
SCHEME_BRANCH_COLLUSION: Final[str] = "branch_collusion_shared_device"
SCHEME_CIRCULAR_WASH: Final[str] = "circular_wash"

SCHEME_LABELS: Final[tuple[str, ...]] = (
    SCHEME_RAPID_PASS_THROUGH,
    SCHEME_FAN_OUT_SMURFING,
    SCHEME_AEPS_TERMINAL_CASH_OUT,
    SCHEME_BRANCH_COLLUSION,
    SCHEME_CIRCULAR_WASH,
)

SCHEME_TXN_COUNTS: Final[dict[str, int]] = {
    SCHEME_RAPID_PASS_THROUGH: 80,       # 20 episodes x (1 inflow + 3 hops)
    SCHEME_FAN_OUT_SMURFING: 130,        # 10 collectors x (1 bulk + 12 micro)
    SCHEME_AEPS_TERMINAL_CASH_OUT: 9,    # 3 complaint trails x 3 legs
    SCHEME_BRANCH_COLLUSION: 12,         # 6 colluders x 2 directions
    SCHEME_CIRCULAR_WASH: 18,            # 6 loops x 3 hops (A->B->C->A)
}

LABELED_TRANSACTION_SCHEMA: Final[dict[str, pl.DataType]] = {
    **TRANSACTION_SCHEMA,
    "is_fraud": pl.Boolean(),
    "fraud_scheme": pl.String(),
}

# 2024-06-01T00:00:00Z epoch seconds, fixed at import (no wall clock/tz table).
_BASE_EPOCH_S: Final[int] = 1717200000
_RETAIL_ACCOUNTS: Final[int] = 900
_MERCHANT_ACCOUNTS: Final[int] = 100
_SCHEME_IFSC: Final[str] = "UTIB0042000"

SHARED_BRANCH_IMEI: Final[str] = "351104510123456"
TERMINAL_MERCHANT_ACCOUNT: Final[str] = "555500000010"

COMPLAINT_STATES: Final[tuple[str, ...]] = (
    "Bihar", "Delhi", "Maharashtra", "Tamil Nadu", "Jharkhand",
)

# Role <-> account-block mapping shared by builders, master rows and complaints.
_BLOCK_RAPID_ORIGIN: Final[int] = 0
_BLOCK_RAPID_HUB: Final[int] = 1
_BLOCK_RAPID_HOP1: Final[int] = 2
_BLOCK_RAPID_HOP2: Final[int] = 3
_BLOCK_RAPID_HOP3: Final[int] = 4
_BLOCK_FAN_PAYER: Final[int] = 5
_BLOCK_FAN_COLL: Final[int] = 6
_BLOCK_FAN_MICRO: Final[int] = 7
_BLOCK_AEPS_VICT: Final[int] = 8
_BLOCK_AEPS_SM1: Final[int] = 9
_BLOCK_AEPS_SM2: Final[int] = 10
_BLOCK_BRANCH_COLLUDER: Final[int] = 11
_BLOCK_LOOP_A: Final[int] = 12
_BLOCK_LOOP_B: Final[int] = 13
_BLOCK_LOOP_C: Final[int] = 14

# Account-master IFSCs by scheme block (account rows only; txn legs keep
# _SCHEME_IFSC, graph joins key on account_no).
_SCHEME_IFSC_BY_BLOCK: Final[dict[int, str]] = {
    0: "UTIB0003331",   # rapid origins (legitimate long-standing payers)
    1: "SBIN0009001",   # rapid hubs (aged, low-tier)
    2: "UTIB0004401",   # rapid hop relay 1
    3: "UTIB0004402",   # rapid hop relay 2
    4: "UTIB0004403",   # rapid hop relay 3
    5: "UTIB0003332",   # fan-out payers
    6: "UTIB0003302",   # fan-out collectors
    7: "SBIN0007701",   # micro wallets
    8: "UTIB0002222",   # AEPS victims (long-standing customers)
    9: "UTIB0004409",   # AEPS mule tier 1
    10: "UTIB0004410",  # AEPS mule tier 2
    11: "SBIN0000421",  # branch colluders (BR042)
    12: "UTIB0005512",  # circular-wash node A
    13: "UTIB0005513",  # circular-wash node B
    14: "UTIB0005514",  # circular-wash node C
}


def _scheme_account(block: int, seq: int) -> str:
    """Reserved scheme-account number: "29" + suffix >= 90_000_000.

    Numeric-only (Decision 13), disjoint from background retail suffixes
    (0..899 -> "2900000000..2900000899") and merchant numbers ("55…").
    """
    return f"29{90_000_000 + block * 10_000 + seq:08d}"


def _scheme_entries() -> list[dict[str, object]]:
    """Master rows for every scheme participant (273 rows + merchant, appended
    by `_accounts`). Attributes are ground truth for Phase-2 signals:
    rapid hubs are aged >= 180 days before the base epoch with low-tier types;
    micro wallets/fresh collectors are recently opened; colluders share one
    branch, a shared device IMEI, and phone-prefix kinship opened within 48h.
    """
    entries: list[dict[str, object]] = []

    def entry(block: int, seq: int, *, opened: date, atype: str, branch: str) -> None:
        entries.append({
            "account_no": _scheme_account(block, seq),
            "ifsc": _SCHEME_IFSC_BY_BLOCK[block],
            "account_opening_date": opened,
            "branch_code": branch,
            "mobile_no": f"98{block:02d}{seq:06d}",          # 10 digits, entry-unique
            "imei": f"35{block:02d}{seq:06d}00000",          # 15 digits
            "kyc_pincode": "800013" if block % 2 == 0 else "110001",
            "account_type": atype,
        })

    for e in range(20):
        entry(_BLOCK_RAPID_ORIGIN, e, opened=date(2021, 10, 1) + timedelta(days=e),
              atype="SAVINGS", branch=f"BR{40 + e // 5:03d}")
        entry(_BLOCK_RAPID_HUB, e, opened=date(2023, 5, 1) + timedelta(days=e // 2),
              atype="PMJDY" if e % 2 == 0 else "SAVINGS", branch=f"BR9{e // 2:02d}")
        for hop_block in (_BLOCK_RAPID_HOP1, _BLOCK_RAPID_HOP2, _BLOCK_RAPID_HOP3):
            entry(hop_block, e, opened=date(2024, 5, 20) + timedelta(days=e % 11),
                  atype="CURRENT", branch=f"BR{60 + e:03d}")
    for e in range(10):
        entry(_BLOCK_FAN_PAYER, e, opened=date(2024, 1, 10) + timedelta(days=e),
              atype="SAVINGS", branch=f"BR{50 + e:03d}")
        entry(_BLOCK_FAN_COLL, e, opened=date(2024, 4, 28) + timedelta(days=e),
              atype="CURRENT", branch=f"BR{80 + e:03d}")
        for i in range(12):
            entry(_BLOCK_FAN_MICRO, e * 100 + i, opened=date(2024, 5, 25) + timedelta(days=i),
                  atype="PMJDY", branch=f"BR{90 + i:03d}")
    for t in range(3):
        entry(_BLOCK_AEPS_VICT, t, opened=date(2023, 2, 15) + timedelta(days=t),
              atype="SAVINGS", branch="BR102")
        entry(_BLOCK_AEPS_SM1, t, opened=date(2024, 4, 2) + timedelta(days=t),
              atype="CURRENT", branch="BR330")
        entry(_BLOCK_AEPS_SM2, t, opened=date(2024, 4, 2) + timedelta(days=t),
              atype="CURRENT", branch="BR331")
    for i in range(6):
        opened_at = datetime(2024, 5, 20, 11, 0) + timedelta(hours=7 * i)
        colluder = _scheme_account(_BLOCK_BRANCH_COLLUDER, i)
        entries.append({
            "account_no": colluder,
            "ifsc": _SCHEME_IFSC_BY_BLOCK[_BLOCK_BRANCH_COLLUDER],       # SBIN0000421
            "account_opening_date": opened_at.date(),                    # spans <= 48h
            "branch_code": "BR042",
            "mobile_no": f"9162{901000 + i * 731}",                      # shared prefix
            "imei": SHARED_BRANCH_IMEI,                                  # shared device
            "kyc_pincode": "800013",
            "account_type": "CURRENT",
        })
    for loop in range(6):
        for block in (_BLOCK_LOOP_A, _BLOCK_LOOP_B, _BLOCK_LOOP_C):
            entry(block, loop, opened=date(2023, 11, 20) + timedelta(days=loop),
                  atype="SAVINGS", branch=f"BR{20 + loop:03d}")
    assert len(entries) == 273
    return entries


def _cast_labeled(df: pl.DataFrame) -> pl.DataFrame:
    """Order + type a frame as the labeled canonical transaction frame."""
    return df.select(
        [
            pl.col(name).cast(dtype, strict=False).alias(name)
            for name, dtype in LABELED_TRANSACTION_SCHEMA.items()
        ]
    )


def _txn_frame(cols: dict[str, Any]) -> pl.DataFrame:
    """Round-trip raw column payload into the labeled canonical frame.

    `cols["timestamp"]` holds epoch **seconds** (int64); ids are Strings,
    amounts floats, flags booleans. Scheme rows are already confluent with
    the anti-fraud label convention.
    """
    df = pl.DataFrame(cols)
    df = df.with_columns(pl.from_epoch("timestamp", time_unit="s"))
    return _cast_labeled(df)


class SimulationResult(BaseModel):
    """Frames produced by one simulation (labeled txns + accounts + complaints)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transactions: pl.DataFrame
    accounts: pl.DataFrame
    complaints: pl.DataFrame


class SyndicateDataSimulator:
    """Seeded generator: legit background + 5 fraud topologies w/ truth labels."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def generate(self, background_transactions: int = 12_000) -> SimulationResult:
        if not 5_000 <= background_transactions <= 20_000:
            raise ValueError(
                "background_transactions must be in [5000, 20000], "
                f"got {background_transactions}"
            )
        rng = np.random.default_rng(self._seed)
        accounts = self._accounts(rng)
        background = self._background(rng, background_transactions)
        schemes: list[pl.DataFrame] = self._scheme_frames()
        transactions = _cast_labeled(
            pl.concat([background, *schemes], how="vertical")
        ).sort(["timestamp", "txn_id"])
        return SimulationResult(
            transactions=transactions,
            accounts=accounts,
            complaints=self._complaints(),
        )

    def _accounts(self, rng: np.random.Generator) -> pl.DataFrame:
        """Retail + merchant master records (scalar arrays; build once)."""
        n = _RETAIL_ACCOUNTS + _MERCHANT_ACCOUNTS
        types = rng.choice(
            np.array(["SAVINGS", "PMJDY", "CURRENT"], dtype=object),
            size=n, p=[0.72, 0.18, 0.10],
        )
        offsets = rng.integers(0, 1_500, size=n)
        pincodes = rng.integers(110_000, 899_999, size=n)
        mobiles = rng.integers(0, 300_000_000, size=n)
        imeis = rng.integers(10**14, 10**15, size=n)
        cols: dict[str, list[object]] = {
            "account_no": [],
            "ifsc": [],
            "account_opening_date": [],
            "branch_code": [],
            "mobile_no": [],
            "imei": [],
            "kyc_pincode": [],
            "account_type": [],
        }
        for i in range(n):
            retail = i < _RETAIL_ACCOUNTS
            cols["account_no"].append(f"29{i:08d}" if retail else f"55{i - _RETAIL_ACCOUNTS:08d}")
            cols["ifsc"].append(f"UTIB0{(i * 97) % 1_000_000:06d}")
            cols["account_opening_date"].append(date(2020, 1, 1) + timedelta(days=int(offsets[i])))
            cols["branch_code"].append(f"BR{i // 25 + 1:03d}")
            cols["mobile_no"].append(f"9{700_000_000 + int(mobiles[i])}")
            cols["imei"].append(str(int(imeis[i])))
            cols["kyc_pincode"].append(f"{int(pincodes[i]):06d}")
            cols["account_type"].append(str(types[i]))
        for participant in _scheme_entries():
            for field, value in participant.items():
                cols[field].append(value)
        cols["account_no"].append(TERMINAL_MERCHANT_ACCOUNT)
        cols["ifsc"].append("UTIB0000777")
        cols["account_opening_date"].append(date(2022, 3, 10))
        cols["branch_code"].append("BR077")
        cols["mobile_no"].append("9170001111")
        cols["imei"].append("351104510998871")
        cols["kyc_pincode"].append("800013")
        cols["account_type"].append("CURRENT")
        return pl.DataFrame(cols, schema=ACCOUNT_METADATA_SCHEMA)

    def _background(self, rng: np.random.Generator, n: int) -> pl.DataFrame:
        """Legit P2P + retail-merchant flows; every step vectorized."""
        senders = rng.integers(0, _RETAIL_ACCOUNTS, size=n)
        retail_recv = (senders + rng.integers(1, _RETAIL_ACCOUNTS, size=n)) % _RETAIL_ACCOUNTS
        merch_pick = _RETAIL_ACCOUNTS + rng.integers(0, _MERCHANT_ACCOUNTS, size=n)
        recv = np.where(rng.random(n) < 0.12, merch_pick, retail_recv)
        amounts = np.clip(np.round(rng.lognormal(6.5, 1.1, n), 2), 10.0, 250_000.0)
        offsets = rng.integers(0, 30 * 24 * 60, size=n)
        modes = rng.choice(
            np.array(["UPI", "IMPS", "NEFT", "CASH"], dtype=object),
            size=n, p=[0.75, 0.15, 0.07, 0.03],
        )
        devices = rng.integers(1, 5_000, size=n)
        ip4 = rng.integers(1, 255, size=(n, 4))
        base = pl.DataFrame(
            {
                "sid": senders,
                "rid": recv,
                "amount": amounts,
                "epoch": (_BASE_EPOCH_S + offsets * 60).astype(np.int64),
                "mode": modes,
                "dev": devices,
                "i0": ip4[:, 0],
                "i1": ip4[:, 1],
                "i2": ip4[:, 2],
                "i3": ip4[:, 3],
            }
        )
        with_cols = base.with_columns(
            [
                pl.concat_str(
                    [pl.lit("BG"), pl.int_range(0, pl.len(), dtype=pl.Int64).cast(pl.String).str.pad_start(9, "0")]
                ).alias("txn_id"),
                pl.concat_str(
                    [pl.lit("29"), pl.col("sid").cast(pl.String).str.pad_start(8, "0")]
                ).alias("sender_account"),
                pl.concat_str(
                    [pl.lit("UTIB0"), (((pl.col("sid").cast(pl.Int64)) * 97) % 1_000_000).cast(pl.String).str.pad_start(6, "0")]
                ).alias("sender_ifsc"),
                pl.when(pl.col("rid") < _RETAIL_ACCOUNTS)
                .then(
                    pl.concat_str(
                        [pl.lit("29"), pl.col("rid").cast(pl.String).str.pad_start(8, "0")]
                    )
                )
                .otherwise(
                    pl.concat_str(
                        [pl.lit("55"), (pl.col("rid") - _RETAIL_ACCOUNTS).cast(pl.String).str.pad_start(8, "0")]
                    )
                )
                .alias("receiver_account"),
                pl.concat_str(
                    [pl.lit("UTIB0"), ((pl.col("rid").cast(pl.Int64)) * 97 % 1_000_000).cast(pl.String).str.pad_start(6, "0")]
                ).alias("receiver_ifsc"),
                pl.from_epoch(pl.col("epoch"), time_unit="s").alias("timestamp"),
                pl.col("mode").cast(pl.String).alias("payment_mode"),
                pl.concat_str(
                    [pl.lit("D"), pl.col("dev").cast(pl.String).str.pad_start(6, "0")]
                ).alias("device_id"),
                pl.concat_str(
                    [
                        pl.col("i0").cast(pl.String), pl.lit("."),
                        pl.col("i1").cast(pl.String), pl.lit("."),
                        pl.col("i2").cast(pl.String), pl.lit("."),
                        pl.col("i3").cast(pl.String),
                    ]
                ).alias("ip_address"),
                pl.lit(False).alias("is_fraud"),
                pl.lit(None, dtype=pl.String()).alias("fraud_scheme"),
            ]
        )
        return _cast_labeled(with_cols)

    # === fraud schemes (Task 7) ===========================================

    def _scheme_frames(self) -> list[pl.DataFrame]:
        return [
            self._scheme_rapid_pass_through(),
            self._scheme_fan_out_smurfing(),
            self._scheme_aeps_terminal_cash_out(),
            self._scheme_branch_collusion(),
            self._scheme_circular_wash(),
        ]

    def _row(
        self,
        *,
        txn_id: str,
        sender: str,
        receiver: str,
        amount: float,
        epoch_s: int,
        mode: str,
        device: str,
        ip: str = "10.10.0.1",
        is_fraud: bool = True,
        scheme: str | None = None,
    ) -> dict[str, object]:
        return {
            "txn_id": txn_id,
            "sender_account": sender,
            "sender_ifsc": _SCHEME_IFSC,
            "receiver_account": receiver,
            "receiver_ifsc": _SCHEME_IFSC,
            "amount": round(amount, 2),
            "timestamp": epoch_s,
            "payment_mode": mode,
            "device_id": device,
            "ip_address": ip,
            "is_fraud": is_fraud,
            "fraud_scheme": scheme,
        }

    def _scheme_rapid_pass_through(self) -> pl.DataFrame:
        """20 episodes: hub inflow drained >=90% across hops within 15 min."""
        rows: list[dict[str, object]] = []
        for e in range(20):
            t0 = _BASE_EPOCH_S + 120 + e * 37 * 60
            hub = _scheme_account(_BLOCK_RAPID_HUB, e)
            inflow = 40_000.0 + e * 1_000.0
            rows.append(self._row(   # victim pays the mule hub
                txn_id=f"S1-{e:03d}-0",
                sender=_scheme_account(_BLOCK_RAPID_ORIGIN, e), receiver=hub,
                amount=inflow, epoch_s=t0, mode="UPI", device=f"S1DEV{e:03d}",
                scheme=SCHEME_RAPID_PASS_THROUGH,
            ))
            rows.append(self._row(   # hop 1: 48% of inflow
                txn_id=f"S1-{e:03d}-1", sender=hub,
                receiver=_scheme_account(_BLOCK_RAPID_HOP1, e),
                amount=inflow * 0.48, epoch_s=t0 + 120, mode="UPI",
                device=f"S1DEV{e:03d}", scheme=SCHEME_RAPID_PASS_THROUGH,
            ))
            rows.append(self._row(   # hop 2: 45% of inflow (hub total 0.93)
                txn_id=f"S1-{e:03d}-2", sender=hub,
                receiver=_scheme_account(_BLOCK_RAPID_HOP2, e),
                amount=inflow * 0.45, epoch_s=t0 + 240, mode="UPI",
                device=f"S1DEV{e:03d}", scheme=SCHEME_RAPID_PASS_THROUGH,
            ))
            rows.append(self._row(   # relay from hop 1's address
                txn_id=f"S1-{e:03d}-3", sender=_scheme_account(_BLOCK_RAPID_HOP1, e),
                receiver=_scheme_account(_BLOCK_RAPID_HOP3, e),
                amount=inflow * 0.46, epoch_s=t0 + 420, mode="UPI",
                device=f"S1DEV{e:03d}", scheme=SCHEME_RAPID_PASS_THROUGH,
            ))
        return _txn_frame({
            key: [row[key] for row in rows] for key in rows[0]
        })

    def _scheme_fan_out_smurfing(self) -> pl.DataFrame:
        """10 collectors distributing to 12 micro wallets under the 10k cap."""
        rows: list[dict[str, object]] = []
        for e in range(10):
            t0 = _BASE_EPOCH_S + 1_800 + e * 53 * 60
            coll = _scheme_account(_BLOCK_FAN_COLL, e)
            rows.append(self._row(
                txn_id=f"S2-{e:03d}-0", sender=_scheme_account(_BLOCK_FAN_PAYER, e),
                receiver=coll, amount=95_000.0, epoch_s=t0, mode="IMPS",
                device=f"S2DEV{e:03d}", scheme=SCHEME_FAN_OUT_SMURFING,
            ))
            for i in range(12):
                amount = 6_000.0 + ((i * 911) % 3_000) + 0.37   # always < 10_000
                rows.append(self._row(
                    txn_id=f"S2-{e:03d}-{i + 1:02d}", sender=coll,
                    receiver=_scheme_account(_BLOCK_FAN_MICRO, e * 100 + i),
                    amount=amount, epoch_s=t0 + 180 + i * 120, mode="UPI",
                    device=f"S2DEV{e:03d}", scheme=SCHEME_FAN_OUT_SMURFING,
                ))
        return _txn_frame({
            key: [row[key] for row in rows] for key in rows[0]
        })

    def _scheme_aeps_terminal_cash_out(self) -> pl.DataFrame:
        """3 distinct complaint trails all cashing out at one AEPS terminal."""
        rows: list[dict[str, object]] = []
        for trail in range(3):
            t0 = _BASE_EPOCH_S + (2 + trail * 3) * 86_400   # June 3/6/9
            victim = _scheme_account(_BLOCK_AEPS_VICT, trail)
            tier1 = _scheme_account(_BLOCK_AEPS_SM1, trail)
            tier2 = _scheme_account(_BLOCK_AEPS_SM2, trail)
            rows.append(self._row(   # victim -> mule tier 1 (disputed leg; Decision 14)
                txn_id=f"S3-{trail:02d}-0", sender=victim, receiver=tier1,
                amount=40_000.0, epoch_s=t0, mode="IMPS", device=f"SMDEV{trail:02d}",
                scheme=SCHEME_AEPS_TERMINAL_CASH_OUT,
            ))
            rows.append(self._row(
                txn_id=f"S3-{trail:02d}-1", sender=tier1, receiver=tier2,
                amount=36_000.0, epoch_s=t0 + 180, mode="IMPS",
                device=f"SMDEV{trail:02d}", scheme=SCHEME_AEPS_TERMINAL_CASH_OUT,
            ))
            rows.append(self._row(
                txn_id=f"S3-{trail:02d}-2", sender=tier2,
                receiver=TERMINAL_MERCHANT_ACCOUNT, amount=25_000.0,
                epoch_s=t0 + 360, mode="AEPS", device="AEPSDEV000001",
                scheme=SCHEME_AEPS_TERMINAL_CASH_OUT,
            ))
        return _txn_frame({
            key: [row[key] for row in rows] for key in rows[0]
        })

    def _scheme_branch_collusion(self) -> pl.DataFrame:
        """Same-branch, opened-within-48h, shared IMEI; 12 transfers among them."""
        rows: list[dict[str, object]] = []
        base = _BASE_EPOCH_S + 4 * 86_400   # one shared demo day
        for i in range(6):
            sender = _scheme_account(_BLOCK_BRANCH_COLLUDER, i)
            peer = _scheme_account(_BLOCK_BRANCH_COLLUDER, (i + 1) % 6)
            rows.append(self._row(
                txn_id=f"S4-{i:02d}-F", sender=sender, receiver=peer,
                amount=2_000.0 + i * 317.25, epoch_s=base + i * 30,
                mode="UPI", device=SHARED_BRANCH_IMEI,
                scheme=SCHEME_BRANCH_COLLUSION,
            ))
            rows.append(self._row(
                txn_id=f"S4-{i:02d}-B", sender=peer, receiver=sender,
                amount=1_800.0 + i * 210.5, epoch_s=base + i * 45,
                mode="UPI", device=SHARED_BRANCH_IMEI,
                scheme=SCHEME_BRANCH_COLLUSION,
            ))
        return _txn_frame({key: [row[key] for row in rows] for key in rows[0]})

    def _scheme_circular_wash(self) -> pl.DataFrame:
        """6 closed loops A->B->C->A with 3% commission decay per hop."""
        rows: list[dict[str, object]] = []
        for loop in range(6):
            base = _BASE_EPOCH_S + 30_000 + loop * 11 * 60
            a_node = _scheme_account(_BLOCK_LOOP_A, loop)
            b_node = _scheme_account(_BLOCK_LOOP_B, loop)
            c_node = _scheme_account(_BLOCK_LOOP_C, loop)
            x0 = 120_000.0 + loop * 5_000.0
            hop1 = x0 * 0.97
            hop2 = hop1 * 0.97
            rows.append(self._row(
                txn_id=f"S5-{loop:02d}-0", sender=a_node, receiver=b_node,
                amount=x0, epoch_s=base, mode="UPI", device=f"S5DEV{loop:02d}",
                scheme=SCHEME_CIRCULAR_WASH,
            ))
            rows.append(self._row(
                txn_id=f"S5-{loop:02d}-1", sender=b_node, receiver=c_node,
                amount=hop1, epoch_s=base + 60, mode="UPI", device=f"S5DEV{loop:02d}",
                scheme=SCHEME_CIRCULAR_WASH,
            ))
            rows.append(self._row(
                txn_id=f"S5-{loop:02d}-2", sender=c_node, receiver=a_node,
                amount=hop2, epoch_s=base + 120, mode="UPI", device=f"S5DEV{loop:02d}",
                scheme=SCHEME_CIRCULAR_WASH,
            ))
        return _txn_frame({key: [row[key] for row in rows] for key in rows[0]})

    def _complaints(self) -> pl.DataFrame:
        """Ground truth: 8 rapid + 5 fan-out + 3 AEPS NCRP complaints.

        Every row's disputed_utr cites a real transaction (the initial
        victim->mule inflow leg); victim_account is that leg's payer and
        suspect_account its receiver, so a join against transactions
        verifies both (TestComplaints pins this).
        """
        rows: list[dict[str, object]] = []

        def add(
            victim: str, suspect: str, disputed_utr: str,
            disputed_amount: float, fraud_category: str,
            reported_at: datetime,
        ) -> None:
            index = len(rows)
            rows.append({
                "ack_no": f"ACK2024{index + 1:06d}",
                "victim_account": victim,
                "suspect_account": suspect,
                "fraud_category": fraud_category,
                "reported_timestamp": reported_at,
                "disputed_amount": disputed_amount,
                "disputed_utr": disputed_utr,
                "state": COMPLAINT_STATES[index % len(COMPLAINT_STATES)],
            })

        # rapid pass-through: the origin account disputes its UPI inflow to the hub
        for e in (0, 2, 4, 6, 8, 10, 12, 14):
            add(
                victim=_scheme_account(_BLOCK_RAPID_ORIGIN, e),
                suspect=_scheme_account(_BLOCK_RAPID_HUB, e),
                disputed_utr=f"S1-{e:03d}-0",
                disputed_amount=40_000.0 + e * 1_000.0,
                fraud_category="RAPID_PASS_THROUGH",
                reported_at=datetime(2024, 6, 1, 14, 0) + timedelta(hours=e),
            )
        # fan-out smurfing: the payer disputes the bulk transfer to its collector
        for e in (0, 2, 4, 6, 8):
            add(
                victim=_scheme_account(_BLOCK_FAN_PAYER, e),
                suspect=_scheme_account(_BLOCK_FAN_COLL, e),
                disputed_utr=f"S2-{e:03d}-0",
                disputed_amount=95_000.0,
                fraud_category="SMURFING_FAN_OUT",
                reported_at=datetime(2024, 6, 2, 9, 0) + timedelta(hours=e),
            )
        # AEPS cash-out ring: the victim disputes the inflow sent to mule tier 1
        for t in range(3):
            add(
                victim=_scheme_account(_BLOCK_AEPS_VICT, t),
                suspect=_scheme_account(_BLOCK_AEPS_SM1, t),
                disputed_utr=f"S3-{t:02d}-0",
                disputed_amount=40_000.0,
                fraud_category="AEPS_CASHOUT",
                reported_at=datetime(2024, 6, 3 + t * 3, 20, 0),
            )
        return pl.DataFrame(rows, schema=COMPLAINT_SCHEMA)
