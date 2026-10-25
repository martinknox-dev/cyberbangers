"""Phase 5 temporal pattern mining.

Extract temporal behavioral features from transaction timestamps.
"""

from __future__ import annotations

import polars as pl


def compute_temporal_features(transactions: pl.DataFrame) -> pl.DataFrame:
    """Compute temporal behavioral features per account.

    Args:
        transactions: DataFrame with columns sender_account, receiver_account,
            timestamp, amount, payment_mode.

    Returns:
        DataFrame with one row per account and temporal features:
        - night_txn_ratio: fraction of transactions in night hours (22-06)
        - peak_hour: hour of day with most activity (0-23)
        - velocity_txns_per_hour: transactions per active hour
        - dow_concentration: max day-of-week concentration (0-1)
        - active_days: number of distinct days with activity
    """
    if transactions.height == 0:
        return pl.DataFrame(schema={
            "account_no": pl.String(),
            "night_txn_ratio": pl.Float64(),
            "peak_hour": pl.Int64(),
            "velocity_txns_per_hour": pl.Float64(),
            "dow_concentration": pl.Float64(),
            "active_days": pl.Int64(),
        })

    # Ensure timestamp is datetime
    txns = transactions.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us"))
    )

    # Combine sender and receiver accounts with their timestamps
    sender_txns = txns.select([
        pl.col("sender_account").alias("account_no"),
        pl.col("timestamp"),
        pl.lit(1).alias("txn_count"),
    ])
    receiver_txns = txns.select([
        pl.col("receiver_account").alias("account_no"),
        pl.col("timestamp"),
        pl.lit(1).alias("txn_count"),
    ])

    all_txns = pl.concat([sender_txns, receiver_txns])

    # Extract temporal components
    all_txns = all_txns.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("weekday"),  # 1=Mon, 7=Sun
        pl.col("timestamp").dt.date().alias("date"),
    ])

    # Night ratio: fraction of transactions in night hours (22-23, 0-5)
    night_hours = list(range(22, 24)) + list(range(6))

    # Per-account aggregations
    result = all_txns.group_by("account_no").agg([
        # Night transaction ratio
        (pl.col("hour").is_in(night_hours).sum() / pl.len()).alias("night_txn_ratio"),
        # Peak hour (mode of hour, with deterministic tie-breaker: smallest hour)
        pl.col("hour").mode().min().alias("peak_hour"),
        # Velocity: transactions per active hour
        (pl.len().cast(pl.Float64) / pl.col("hour").n_unique().cast(pl.Float64)).alias("velocity_txns_per_hour"),
        # Day-of-week concentration (max frequency / total)
        _dow_concentration_expr().alias("dow_concentration"),
        # Active days
        pl.col("date").n_unique().alias("active_days"),
    ])

    return result.sort("account_no")


def _dow_concentration_expr() -> pl.Expr:
    """Compute day-of-week concentration as max frequency / total."""
    # value_counts returns a struct with "weekday" and "count" fields
    # We need to extract the count and find the max
    return (
        pl.col("weekday").value_counts()
        .struct.field("count")
        .max()
        / pl.len()
    )