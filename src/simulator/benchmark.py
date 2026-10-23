"""Seeded, fully offline benchmark report for the ``prahar-verify`` command.

Ground-truth labels are consumed *after* the blind pipeline run, only to
measure detection. They never reach feature extraction, graph analytics or the
risk scorer. Every number describes the simulator's five synthetic topologies;
this is not a real-world accuracy claim.
"""

from __future__ import annotations

import time
from typing import Final

import polars as pl

from core.contracts import (
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_SCHEMA,
    TRANSACTION_SCHEMA,
)
from core.hardware import detect_hardware_profile
from features.account_features import extract_account_features
from graph.analyzer import GraphAnalyzer
from ingestion.mapper import DatasetIngester
from risk.scorer import RiskScorer
from simulator.generator import (
    SCHEME_AEPS_TERMINAL_CASH_OUT,
    SCHEME_BRANCH_COLLUSION,
    SCHEME_CIRCULAR_WASH,
    SCHEME_FAN_OUT_SMURFING,
    SCHEME_LABELS,
    SCHEME_RAPID_PASS_THROUGH,
    SimulationResult,
    SyndicateDataSimulator,
)

SEED: Final[int] = 42
BACKGROUND_TRANSACTIONS: Final[int] = 5_000
BANDS: Final[tuple[str, ...]] = ("HIGH", "CRITICAL")
SCREEN_THRESHOLD: Final[float] = 40.0


def _culprit_accounts(sim: SimulationResult) -> dict[str, set[str]]:
    """Per-scheme accounts that drive a topology (mirrors Phase 2 evaluation).

    Victims, origin payers and one-off tools (micro wallets, hop relays) are
    involved in the fraud but are not its origin, so they are excluded.
    """
    txns = sim.transactions

    def scheme(number: int) -> pl.DataFrame:
        return txns.filter(pl.col("txn_id").str.starts_with(f"S{number}-"))

    s1, s2, s3, s4, s5 = (scheme(number) for number in range(1, 6))
    return {
        SCHEME_RAPID_PASS_THROUGH: set(
            s1.filter(pl.col("txn_id").str.ends_with("-0"))[
                "receiver_account"
            ].to_list()
        ),
        SCHEME_FAN_OUT_SMURFING: set(
            s2.filter(pl.col("txn_id").str.ends_with("-0"))[
                "receiver_account"
            ].to_list()
        ),
        SCHEME_AEPS_TERMINAL_CASH_OUT: set(
            s3.filter(~pl.col("txn_id").str.ends_with("-0"))[
                "sender_account"
            ].to_list()
        ),
        SCHEME_BRANCH_COLLUSION: set(s4["sender_account"].to_list()),
        SCHEME_CIRCULAR_WASH: set(s5["sender_account"].to_list()),
    }


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


def _inr(amount: float) -> str:
    return f"₹{amount:,.0f}"


def _ingest(sim: SimulationResult) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, int, int
]:
    """Blind ingestion: drop labels, round-trip the frames through the ingester."""
    ingester = DatasetIngester()
    transactions, txn_report = ingester.ingest_transactions(
        sim.transactions.select(list(TRANSACTION_SCHEMA)).write_csv().encode()
    )
    accounts, acc_report = ingester.ingest_accounts(
        sim.accounts.select(list(ACCOUNT_METADATA_SCHEMA)).write_csv().encode()
    )
    complaints, cmp_report = ingester.ingest_complaints(
        sim.complaints.select(list(COMPLAINT_SCHEMA)).write_csv().encode()
    )
    total = (
        txn_report.rows_total + acc_report.rows_total + cmp_report.rows_total
    )
    rejected = (
        txn_report.rows_rejected
        + acc_report.rows_rejected
        + cmp_report.rows_rejected
    )
    return transactions, accounts, complaints, total, rejected


def main() -> int:
    started = time.perf_counter()
    sim = SyndicateDataSimulator(seed=SEED).generate(
        background_transactions=BACKGROUND_TRANSACTIONS
    )
    transactions, accounts, complaints, total_rows, rejected_rows = _ingest(sim)

    features = extract_account_features(transactions, accounts, complaints)
    graph = GraphAnalyzer(transactions, accounts, complaints).topology_metrics()
    scorer = RiskScorer()
    risk, assessments = scorer.score_with_assessments(features, graph)
    runtime = time.perf_counter() - started
    profile = scorer.profile
    auto_profile = detect_hardware_profile(None)

    culprits = _culprit_accounts(sim)
    all_culprits = set().union(*culprits.values())
    involved = (
        set(sim.transactions.filter(pl.col("is_fraud"))["sender_account"].to_list())
        | set(
            sim.transactions.filter(pl.col("is_fraud"))["receiver_account"].to_list()
        )
    )
    flagged: dict[str, set[str]] = {
        band: set(
            risk.filter(pl.col("risk_band") == band)["account_no"].to_list()
        )
        for band in BANDS
    }
    combined = set().union(*(flagged[band] for band in BANDS))
    combined_hits = combined & all_culprits
    involved_hits = combined & involved
    background_fps = combined - involved
    screened = set(
        risk.filter(pl.col("risk_score") >= SCREEN_THRESHOLD)[
            "account_no"
        ].to_list()
    )
    screened_hits = screened & all_culprits

    print("PraharGraph seeded benchmark - synthetic, offline, deterministic")
    print(
        f"seed={SEED} - background_transactions={BACKGROUND_TRANSACTIONS} - "
        "labels dropped before ingestion"
    )
    print(
        f"  hardware: compute_tier={profile.tier} "
        f"(duckdb_threads={profile.duckdb_threads}, "
        f"memory_limit={profile.duckdb_memory_limit}, "
        f"iforest_estimators={profile.iforest_estimators}, "
        f"knn_density={'on' if profile.enable_knn_density else 'off'}, "
        f"diffusion_iterations={profile.diffusion_iterations}; "
        f"auto-detected={auto_profile.tier})"
    )
    print()
    print("Ingestion (blind round-trip through the CSV ingester):")
    print(
        f"  rows total={total_rows} accepted={total_rows - rejected_rows} "
        f"rejected={rejected_rows}"
    )
    print(
        f"  transactions={transactions.height} accounts={accounts.height} "
        f"complaints={complaints.height} risk assessments={len(assessments)}"
    )
    print()
    print(
        f"{'Band':<10}{'Flagged':>8}{'True positives':>16}"
        f"{'Precision':>11}{'Recall':>9}"
    )
    for band in BANDS:
        hits = flagged[band] & all_culprits
        print(
            f"{band:<10}{len(flagged[band]):>8}{len(hits):>16}"
            f"{_pct(len(hits), len(flagged[band])):>11}"
            f"{_pct(len(hits), len(all_culprits)):>9}"
        )
    print(
        f"{'HIGH+CRIT':<10}{len(combined):>8}{len(combined_hits):>16}"
        f"{_pct(len(combined_hits), len(combined)):>11}"
        f"{_pct(len(combined_hits), len(all_culprits)):>9}"
    )
    print(
        f"  combined involved precision={_pct(len(involved_hits), len(combined))} "
        f"(denominator includes victims and relays) - "
        f"background false positives={len(background_fps)}"
    )
    print()
    print(
        "Per-scheme detection at HIGH or CRITICAL (precision is the scheme's "
        "share of the combined band's flags, so the columns sum to overall "
        "combined-band precision):"
    )
    header = (
        f"{'Scheme':<32}{'Culprits':>9}{'HIGH rec':>11}{'CRIT rec':>11}"
        f"{'H+C recall':>13}{'H+C prec':>11}"
    )
    print(header)
    for scheme in SCHEME_LABELS:
        members = culprits[scheme]
        line = f"{scheme:<32}{len(members):>9}"
        for band in BANDS:
            detected = members & flagged[band]
            line += f"{_pct(len(detected), len(members)):>11}"
        detected = members & combined
        line += (
            f"{_pct(len(detected), len(members)):>13}"
            f"{_pct(len(detected), len(combined)):>11}"
        )
        print(line)
    print()
    print("Top 5 accounts by freeze priority (recoverable tainted rupees):")
    top = risk.sort("freeze_priority_score", descending=True).head(5)
    print(
        f"  {'Account':<16}{'Band':<10}{'Role':<24}{'Risk':>6}"
        f"{'Freeze':>8}{'Recoverable taint':>20}"
    )
    for row in top.to_dicts():
        print(
            f"  {row['account_no']!s:<16}{row['risk_band']!s:<10}"
            f"{row['operational_role']!s:<24}"
            f"{float(row['risk_score']):>6.1f}"
            f"{float(row['freeze_priority_score']):>8.1f}"
            f"{_inr(float(row['retained_tainted_amount'])):>20}"
        )
    print()
    print(
        f"Screening band (risk_score >= {SCREEN_THRESHOLD:.0f}, the Phase 2 "
        "evaluation threshold):"
    )
    print(
        f"  flagged={len(screened)} of {len(assessments)} accounts - "
        f"precision={_pct(len(screened_hits), len(screened))} - "
        f"recall={_pct(len(screened_hits), len(all_culprits))}"
    )
    print(
        "  per-scheme recall: "
        + " - ".join(
            f"{scheme} {len(culprits[scheme] & screened)}/{len(culprits[scheme])}"
            for scheme in SCHEME_LABELS
        )
    )
    print()
    print(
        f"Runtime: {runtime:.2f} s "
        "(simulate -> ingest -> features -> graph -> risk)"
    )
    print(
        "Synthetic topologies only: these numbers are not real-world "
        "accuracy claims."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
