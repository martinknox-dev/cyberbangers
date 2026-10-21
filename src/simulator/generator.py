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
        """Returns [] in Task 6; the five builders are appended here by Task 7."""
        return []

    def _complaints(self) -> pl.DataFrame:
        """Task 7 wires the ground truth: 16 rows (episodes of schemes 1/2/3)."""
        rows: list[dict[str, object]] = []
        return pl.DataFrame(rows, schema=COMPLAINT_SCHEMA)
