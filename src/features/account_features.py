"""LAYER 2 — behavioural feature extraction.

One deterministic row per distinct account seen in the transaction ledger or
in the account metadata. Heavy aggregation runs as in-process DuckDB SQL over
canonical Polars frames; assembly is Polars. There are no row-level Python
loops, no `iter_rows` and no Pandas anywhere in this module.

Time semantics: every temporal predicate is evaluated on integer epoch seconds
(``t``) inside SQL, and the calendar fields (opening date, dormancy) are
derived in Polars. DuckDB's Arrow scan maps Polars temporal columns to BLOB, so
timestamps are converted to epoch seconds before they cross the SQL boundary.

Dormancy is never inferred from the *absence* of transactions: an account with
no known opening date has unknown dormancy (``null``), never zero.
"""

from __future__ import annotations

from typing import Final

import duckdb
import polars as pl

from core.contracts import ACCOUNT_FEATURE_SCHEMA, FeatureConfig
from core.hardware import detect_hardware_profile

_EMPTY_FRAME: Final[pl.DataFrame] = pl.DataFrame(schema=ACCOUNT_FEATURE_SCHEMA)

# Intermediate aggregate columns (some queries return no rows at all).
_INTERMEDIATE_COLUMNS: Final[dict[str, pl.DataType]] = {
    "txn_count": pl.Int64(),
    "inflow_count": pl.Int64(),
    "outflow_count": pl.Int64(),
    "inflow_total": pl.Float64(),
    "outflow_total": pl.Float64(),
    "unique_senders": pl.Int64(),
    "unique_receivers": pl.Int64(),
    "first_ts": pl.Int64(),
    "last_ts": pl.Int64(),
    "attributed_out": pl.Float64(),
    "median_retention_minutes": pl.Float64(),
    "pass_through_utrs": pl.List(pl.String()),
    "smurf_bulk_inflow_count": pl.Int64(),
    "smurf_split_count": pl.Int64(),
    "smurf_utrs": pl.List(pl.String()),
    "large_payout_count": pl.Int64(),
    "burst_total": pl.Float64(),
    "burst_start": pl.Int64(),
    "device_peers": pl.Int64(),
    "imei_peers": pl.Int64(),
    "branch_cohort_count": pl.Int64(),
    "aeps_cashout_inflow_count": pl.Int64(),
    "direct_complaint_count": pl.Int64(),
}

_LEGS_SQL: Final[str] = """
CREATE OR REPLACE TEMP TABLE legs AS
SELECT
    receiver_account AS account_no,
    sender_account   AS counterparty,
    txn_id,
    ts_epoch         AS t,
    amount,
    'in'              AS direction,
    payment_mode,
    CAST(NULL AS VARCHAR) AS device_id
FROM tx
WHERE receiver_account IS NOT NULL AND sender_account IS NOT NULL
UNION ALL
SELECT
    sender_account,
    receiver_account,
    txn_id,
    ts_epoch,
    amount,
    'out',
    payment_mode,
    device_id
FROM tx
WHERE receiver_account IS NOT NULL AND sender_account IS NOT NULL
"""

_BASE_SQL: Final[str] = """
SELECT
    account_no,
    count(*)::BIGINT                                              AS txn_count,
    count(*) FILTER (WHERE direction = 'in')::BIGINT              AS inflow_count,
    count(*) FILTER (WHERE direction = 'out')::BIGINT             AS outflow_count,
    round(coalesce(sum(amount) FILTER (WHERE direction = 'in'), 0), 2)::DOUBLE  AS inflow_total,
    round(coalesce(sum(amount) FILTER (WHERE direction = 'out'), 0), 2)::DOUBLE AS outflow_total,
    count(DISTINCT counterparty) FILTER (WHERE direction = 'in')::BIGINT  AS unique_senders,
    count(DISTINCT counterparty) FILTER (WHERE direction = 'out')::BIGINT AS unique_receivers,
    min(t)::BIGINT AS first_ts,
    max(t)::BIGINT AS last_ts
FROM legs
GROUP BY account_no
"""

# Each outflow leg is attributed to the EARLIEST preceding inflow of the same
# account inside the window (FIFO, tie-broken by txn_id): an outflow is counted
# once, and only the account's own legs participate.
_PASS_THROUGH_SQL: Final[str] = """
WITH pairs AS (
    SELECT
        o.account_no,
        o.txn_id            AS out_txn,
        o.t                 AS out_t,
        o.amount            AS out_amt,
        i.t                 AS in_t,
        ROW_NUMBER() OVER (
            PARTITION BY o.account_no, o.txn_id ORDER BY i.t, i.txn_id
        ) AS rn
    FROM legs o
    JOIN legs i
      ON o.account_no = i.account_no
     AND o.direction = 'out'
     AND i.direction = 'in'
     AND o.t >= i.t
     AND o.t <= i.t + {window_s}
),
attributed AS (
    SELECT * FROM pairs WHERE rn = 1
)
SELECT
    account_no,
    round(sum(out_amt), 2)::DOUBLE AS attributed_out,
    median((out_t - in_t)::DOUBLE) / 60.0 AS median_retention_minutes,
    list(out_txn ORDER BY out_txn) AS pass_through_utrs
FROM attributed
GROUP BY account_no
"""

_SMURF_SQL: Final[str] = """
WITH bulk AS (
    SELECT account_no, txn_id, t
    FROM legs
    WHERE direction = 'in' AND amount >= {bulk_threshold}
),
splits AS (
    SELECT account_no, txn_id, t
    FROM legs
    WHERE direction = 'out'
      AND amount >= {split_min}
      AND amount <= {split_max}
),
joined AS (
    SELECT b.account_no, b.txn_id AS bulk_txn, s.txn_id AS split_txn
    FROM bulk b
    JOIN splits s
      ON s.account_no = b.account_no
     AND s.t >= b.t
     AND s.t <= b.t + {fan_out_s}
),
counts AS (
    SELECT account_no, bulk_txn, count(DISTINCT split_txn) AS n
    FROM joined
    GROUP BY account_no, bulk_txn
),
qualifying AS (
    SELECT account_no, bulk_txn FROM counts WHERE n >= {min_splits}
),
with_splits AS (
    SELECT q.account_no, q.bulk_txn, j.split_txn
    FROM qualifying q
    JOIN joined j
      ON j.account_no = q.account_no
     AND j.bulk_txn = q.bulk_txn
)
SELECT
    account_no,
    count(DISTINCT bulk_txn)::BIGINT  AS smurf_bulk_inflow_count,
    count(DISTINCT split_txn)::BIGINT AS smurf_split_count,
    list(DISTINCT split_txn ORDER BY split_txn) AS smurf_utrs
FROM with_splits
GROUP BY account_no
"""

_LARGE_PAYOUT_SQL: Final[str] = """
SELECT account_no, count(*)::BIGINT AS large_payout_count
FROM legs
WHERE direction = 'out'
  AND amount >= {bulk_threshold}
  AND payment_mode IN ('IMPS', 'RTGS', 'NEFT')
GROUP BY account_no
"""

_BURST_SQL: Final[str] = """
WITH win AS (
    SELECT
        account_no,
        txn_id,
        t,
        sum(amount) OVER (
            PARTITION BY account_no ORDER BY t
            RANGE BETWEEN {burst_s} PRECEDING AND CURRENT ROW
        ) AS window_total,
        min(t) OVER (
            PARTITION BY account_no ORDER BY t
            RANGE BETWEEN {burst_s} PRECEDING AND CURRENT ROW
        ) AS window_start
    FROM legs
    WHERE direction = 'in'
),
ranked AS (
    SELECT
        account_no, window_total, window_start,
        row_number() OVER (
            PARTITION BY account_no
            ORDER BY window_total DESC, t ASC, txn_id ASC
        ) AS rn
    FROM win
)
SELECT account_no, round(window_total, 2)::DOUBLE AS burst_total, window_start::BIGINT AS burst_start
FROM ranked
WHERE rn = 1
"""

_DEVICE_PEERS_SQL: Final[str] = """
WITH devs AS (
    SELECT device_id, count(DISTINCT account_no) AS n
    FROM legs
    WHERE device_id IS NOT NULL AND device_id <> ''
    GROUP BY device_id
),
peer AS (
    SELECT DISTINCT l.account_no, d.n
    FROM legs l
    JOIN devs d ON d.device_id = l.device_id
    WHERE l.device_id IS NOT NULL AND l.device_id <> ''
)
SELECT account_no, max(n)::BIGINT AS device_peers
FROM peer
GROUP BY account_no
"""

_IMEI_PEERS_SQL: Final[str] = """
WITH imeis AS (
    SELECT imei, count(*) AS n
    FROM acc
    WHERE imei IS NOT NULL AND imei <> ''
    GROUP BY imei
),
peer AS (
    SELECT DISTINCT a.account_no, i.n
    FROM acc a
    JOIN imeis i ON i.imei = a.imei
    WHERE a.imei IS NOT NULL AND a.imei <> ''
)
SELECT account_no, max(n)::BIGINT AS imei_peers
FROM peer
GROUP BY account_no
"""

_BRANCH_COHORT_SQL: Final[str] = """
WITH cohort AS (
    SELECT account_no, branch_code, open_epoch
    FROM acc
    WHERE branch_code IS NOT NULL AND open_epoch IS NOT NULL
)
SELECT
    x.account_no,
    count(*) FILTER (WHERE y.account_no <> x.account_no)::BIGINT AS branch_cohort_count
FROM cohort x
JOIN cohort y
  ON y.branch_code = x.branch_code
 AND abs(y.open_epoch - x.open_epoch) <= {cohort_s}
GROUP BY x.account_no
"""

_TERMINAL_SQL: Final[str] = """
SELECT account_no, count(*)::BIGINT AS aeps_cashout_inflow_count
FROM legs
WHERE direction = 'in' AND payment_mode IN ({terminal_modes})
GROUP BY account_no
"""

_COMPLAINT_SQL: Final[str] = """
WITH linked AS (
    SELECT ack_no, victim_account AS account_no
    FROM cx WHERE victim_account IS NOT NULL
    UNION ALL
    SELECT ack_no, suspect_account
    FROM cx WHERE suspect_account IS NOT NULL
    UNION ALL
    SELECT c.ack_no, t.receiver_account
    FROM cx c
    JOIN tx t ON t.txn_id = c.disputed_utr
)
SELECT account_no, count(DISTINCT ack_no)::BIGINT AS direct_complaint_count
FROM linked
WHERE account_no IS NOT NULL
GROUP BY account_no
"""


def _with_missing_columns(
    frame: pl.DataFrame | None, required: dict[str, pl.DataType]
) -> pl.DataFrame:
    """Return a frame that has every required column (missing ones are null)."""
    if frame is None:
        return pl.DataFrame(schema=required)
    columns: list[pl.Expr] = []
    for name, dtype in required.items():
        if name in frame.columns:
            columns.append(pl.col(name).cast(dtype, strict=False).alias(name))
        else:
            columns.append(pl.lit(None, dtype=dtype).alias(name))
    return frame.select(columns)


def _prepare_transactions(transactions: pl.DataFrame | None) -> pl.DataFrame:
    """Canonical transaction columns plus integer epoch seconds."""
    prepared = _with_missing_columns(
        transactions,
        {
            "txn_id": pl.String(),
            "sender_account": pl.String(),
            "receiver_account": pl.String(),
            "amount": pl.Float64(),
            "timestamp": pl.Datetime("us"),
            "payment_mode": pl.String(),
            "device_id": pl.String(),
        },
    )
    return prepared.with_columns(
        pl.col("timestamp").dt.epoch("s").fill_null(0).cast(pl.Int64).alias("ts_epoch")
    )


def _prepare_accounts(accounts: pl.DataFrame | None) -> pl.DataFrame:
    """Account metadata with an integer epoch for the opening date.

    Duplicate account numbers are collapsed with the documented keep-first
    policy (the same policy Phase 1 ingestion applies), so a repeated
    metadata row can never produce a second feature row for one account.
    """
    prepared = _with_missing_columns(
        accounts,
        {
            "account_no": pl.String(),
            "branch_code": pl.String(),
            "imei": pl.String(),
            "account_opening_date": pl.Date(),
            "account_type": pl.String(),
        },
    )
    if prepared.height:
        prepared = prepared.unique(subset=["account_no"], keep="first")
    return prepared.with_columns(
        pl.col("account_opening_date")
        .dt.epoch("s")
        .cast(pl.Int64)
        .alias("open_epoch")
    )


def _prepare_complaints(complaints: pl.DataFrame | None) -> pl.DataFrame:
    return _with_missing_columns(
        complaints,
        {
            "ack_no": pl.String(),
            "victim_account": pl.String(),
            "suspect_account": pl.String(),
            "disputed_utr": pl.String(),
        },
    )


def _account_universe(
    transactions: pl.DataFrame, accounts: pl.DataFrame | None
) -> pl.DataFrame:
    """Every distinct account seen in the ledger or the metadata."""
    from_ledger = pl.concat(
        [
            transactions.select(pl.col("sender_account").alias("account_no")),
            transactions.select(pl.col("receiver_account").alias("account_no")),
        ],
        how="vertical",
    )
    from_metadata = (
        accounts.select(pl.col("account_no"))
        if accounts is not None and "account_no" in accounts.columns
        else pl.DataFrame(schema={"account_no": pl.String()})
    )
    return (
        pl.concat([from_ledger, from_metadata], how="vertical")
        .filter(pl.col("account_no").is_not_null())
        .unique()
        .sort("account_no")
    )


def extract_account_features(
    transactions: pl.DataFrame,
    accounts: pl.DataFrame | None = None,
    complaints: pl.DataFrame | None = None,
    config: FeatureConfig | None = None,
) -> pl.DataFrame:
    """One behavioural feature row per distinct account.

    Args:
        transactions: canonical transaction ledger (`TRANSACTION_SCHEMA`).
        accounts: optional account metadata (`ACCOUNT_METADATA_SCHEMA`).
        complaints: optional NCRP complaints (`COMPLAINT_SCHEMA`).
        config: configurable thresholds; defaults follow `ARCHITECTURE.md` §5.1.

    Returns:
        A frame conforming exactly to `ACCOUNT_FEATURE_SCHEMA`, sorted by
        `account_no` (deterministic for identical inputs).
    """
    cfg = config or FeatureConfig()
    tx = _prepare_transactions(transactions)
    acc = _prepare_accounts(accounts)
    cx = _prepare_complaints(complaints)

    universe = _account_universe(tx, acc)
    if universe.height == 0:
        return _EMPTY_FRAME

    connection = duckdb.connect()
    try:
        profile = detect_hardware_profile(cfg.compute_tier)
        connection.execute(f"SET threads = {profile.duckdb_threads}")
        connection.execute(
            f"SET memory_limit = '{profile.duckdb_memory_limit}'"
        )
        connection.register("tx", tx.to_arrow())
        connection.register("acc", acc.to_arrow())
        connection.register("cx", cx.to_arrow())
        connection.execute(_LEGS_SQL)

        window_s = cfg.pass_through_window_minutes * 60
        fan_out_s = cfg.fan_out_window_minutes * 60
        burst_s = cfg.burst_window_hours * 3600
        cohort_s = cfg.branch_cohort_hours * 3600
        terminal_modes = ", ".join(f"'{mode}'" for mode in cfg.terminal_modes)

        base = connection.execute(_BASE_SQL).pl()
        pass_through = connection.execute(
            _PASS_THROUGH_SQL.format(window_s=window_s)
        ).pl()
        smurf = connection.execute(
            _SMURF_SQL.format(
                bulk_threshold=cfg.smurf_bulk_threshold,
                split_min=cfg.smurf_split_min,
                split_max=cfg.smurf_split_max,
                fan_out_s=fan_out_s,
                min_splits=cfg.smurf_min_splits,
            )
        ).pl()
        large_payouts = connection.execute(
            _LARGE_PAYOUT_SQL.format(bulk_threshold=cfg.smurf_bulk_threshold)
        ).pl()
        burst = connection.execute(_BURST_SQL.format(burst_s=burst_s)).pl()
        device_peers = connection.execute(_DEVICE_PEERS_SQL).pl()
        imei_peers = connection.execute(_IMEI_PEERS_SQL).pl()
        branch_cohort = connection.execute(
            _BRANCH_COHORT_SQL.format(cohort_s=cohort_s)
        ).pl()
        terminal = connection.execute(
            _TERMINAL_SQL.format(terminal_modes=terminal_modes)
        ).pl()
        complaints_linked = connection.execute(_COMPLAINT_SQL).pl()
    finally:
        connection.close()

    frame = universe
    for part in (
        base,
        pass_through,
        smurf,
        large_payouts,
        burst,
        device_peers,
        imei_peers,
        branch_cohort,
        terminal,
        complaints_linked,
    ):
        if part.height:
            frame = frame.join(part, on="account_no", how="left")

    # An aggregate that matched nothing contributes no column at all; add the
    # missing ones as nulls so the assembly below is uniform.
    missing = [
        pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in _INTERMEDIATE_COLUMNS.items()
        if name not in frame.columns
    ]
    if missing:
        frame = frame.with_columns(missing)

    zero_int = [
        "txn_count",
        "inflow_count",
        "outflow_count",
        "unique_senders",
        "unique_receivers",
        "smurf_bulk_inflow_count",
        "smurf_split_count",
        "large_payout_count",
        "aeps_cashout_inflow_count",
        "direct_complaint_count",
        "device_peers",
        "imei_peers",
        "branch_cohort_count",
    ]
    zero_float = ["inflow_total", "outflow_total", "attributed_out"]
    for name in ("pass_through_utrs", "smurf_utrs"):
        frame = frame.with_columns(
            pl.col(name).fill_null(pl.lit([]).cast(pl.List(pl.String()))).alias(name)
        )
    frame = frame.with_columns(
        [pl.col(name).fill_null(0).cast(pl.Int64()) for name in zero_int]
        + [pl.col(name).fill_null(0.0).cast(pl.Float64()) for name in zero_float]
    )

    # Peer counts exclude the account itself.
    frame = frame.with_columns(
        pl.max_horizontal(
            pl.col("device_peers") - 1,
            pl.col("imei_peers") - 1,
        )
        .clip(lower_bound=0)
        .cast(pl.Int64())
        .alias("shared_device_account_count"),
        pl.col("branch_cohort_count").cast(pl.Int64()).alias("branch_cohort_48h_count"),
    )

    frame = frame.with_columns(
        (
            pl.col("attributed_out").fill_null(0.0).cast(pl.Float64())
            / pl.when(pl.col("inflow_total") > 0)
            .then(pl.col("inflow_total"))
            .otherwise(1.0)
        )
        .clip(upper_bound=1.0)
        .round(12)
        .cast(pl.Float64())
        .alias("pass_through_ratio_15m")
    )

    frame = frame.with_columns(
        (
            (pl.col("unique_receivers").cast(pl.Float64()) - pl.col("unique_senders").cast(pl.Float64()))
            / pl.max_horizontal(
                pl.col("unique_receivers").cast(pl.Float64())
                + pl.col("unique_senders").cast(pl.Float64()),
                pl.lit(1.0),
            )
        )
        .cast(pl.Float64())
        .alias("fan_in_fan_out_asymmetry")
    )

    # Dormancy: measured against a known opening date only.
    opening = acc.select(
        pl.col("account_no"),
        pl.col("open_epoch").alias("opening_epoch"),
    )
    frame = frame.join(opening, on="account_no", how="left").with_columns(
        pl.when(
            pl.col("opening_epoch").is_not_null() & pl.col("burst_start").is_not_null()
        )
        .then(
            ((pl.col("burst_start").cast(pl.Float64()) - pl.col("opening_epoch").cast(pl.Float64()))
             / 86_400.0).floor()
        )
        .otherwise(None)
        .cast(pl.Float64())
        .alias("dormancy_days_before_burst"),
        pl.col("burst_total").fill_null(0.0).cast(pl.Float64()).alias("burst_24h_max_total"),
    )

    frame = frame.with_columns(
        pl.when(pl.col("dormancy_days_before_burst").is_null())
        .then(None)
        .when(
            (pl.col("dormancy_days_before_burst") >= cfg.dormancy_threshold_days)
            & (pl.col("burst_24h_max_total") > cfg.burst_amount_threshold)
        )
        .then(1.0)
        .otherwise(0.0)
        .cast(pl.Float64())
        .alias("dormancy_burst_score")
    )

    frame = frame.with_columns(
        (pl.col("inflow_total") - pl.col("outflow_total"))
        .cast(pl.Float64())
        .alias("net_flow"),
        pl.col("first_ts").cast(pl.Int64()).dt.epoch("s").alias("_first_dt"),
    )

    frame = frame.with_columns(
        pl.from_epoch(pl.col("_first_dt"), time_unit="s").alias("first_txn_timestamp"),
        pl.from_epoch(pl.col("last_ts").cast(pl.Int64()), time_unit="s").alias("last_txn_timestamp"),
    )

    # Cohort-normalised volume: compare each account with peers of the same
    # account type (PMJDY / SAVINGS / CURRENT, or UNKNOWN when metadata is
    # absent). A cohort without a usable spread (fewer than two accounts, or
    # all volumes equal) contributes z = 0, never an invented outlier.
    frame = frame.join(
        acc.select("account_no", "account_type"), on="account_no", how="left"
    ).with_columns(
        (pl.col("inflow_total") + pl.col("outflow_total"))
        .cast(pl.Float64())
        .alias("_volume"),
        pl.col("account_type").fill_null("UNKNOWN").alias("_cohort"),
    )
    cohort_stats = frame.group_by("_cohort").agg(
        pl.col("_volume").mean().alias("_cohort_mean"),
        pl.col("_volume").std(ddof=0).alias("_cohort_std"),
        pl.len().alias("_cohort_size"),
    )
    frame = frame.join(cohort_stats, on="_cohort", how="left").with_columns(
        pl.when(
            (pl.col("_cohort_size") >= 2) & (pl.col("_cohort_std") > 0.0)
        )
        .then(
            (pl.col("_volume") - pl.col("_cohort_mean")) / pl.col("_cohort_std")
        )
        .otherwise(0.0)
        .cast(pl.Float64())
        .round(6)
        .alias("cohort_volume_zscore")
    )

    return frame.select(
        [
            pl.col(name).cast(dtype, strict=False).alias(name)
            for name, dtype in ACCOUNT_FEATURE_SCHEMA.items()
        ]
    ).sort("account_no")
