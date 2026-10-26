"""PraharGraph investigator workbench (offline Streamlit front end)."""

from __future__ import annotations

from typing import Any, cast

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from st_link_analysis import EdgeStyle, NodeStyle, st_link_analysis

from core.contracts import (
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_SCHEMA,
    RISK_DISCLAIMER,
    TRANSACTION_SCHEMA,
    IngestionReport,
    TableKind,
)
from core.hardware import ComputeTier
from export.dossier import format_inr, rejected_rows_csv
from ingestion.mapper import DatasetIngester
from simulator.generator import SyndicateDataSimulator
from ui import state

_TABS = (
    "1 · Ingestion & Data Hygiene",
    "2 · Mule Risk Ranking & XAI",
    "3 · Money Trail & Syndicates",
    "4 · Complaint Correlation & Dossier",
)

_CUSTOM_CSS = """
<style>
/* Main typography and container refinement */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
    max-width: 98%;
}

/* Metric card styling */
div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
div[data-testid="stMetric"] label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.025em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0f172a;
}

/* Status pills and badges */
.badge-critical {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #f87171;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-high {
    background-color: #ffedd5;
    color: #c2410c;
    border: 1px solid #fb923c;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-medium {
    background-color: #dbeafe;
    color: #1d4ed8;
    border: 1px solid #60a5fa;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-low {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-role {
    background-color: #f0fdf4;
    color: #166534;
    border: 1px solid #86efac;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Hardware profile card */
.hw-card {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 10px 12px;
    margin-top: 12px;
    font-size: 0.82rem;
    color: #334155;
    line-height: 1.5;
}
.hw-card-title {
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 6px;
    font-size: 0.88rem;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
}

/* Legend strip */
.legend-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    font-size: 0.82rem;
    color: #334155;
    padding: 8px 12px;
    background: #f1f5f9;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    margin-bottom: 8px;
}
</style>
"""


def _audit_rows(duplicate_utr: str) -> str:
    """Three deterministic malformed legs so the audit tab is never empty.

    They are rejected during ingestion, so they never reach features, scoring or
    the dossier — they exist only to demonstrate rejection handling.
    """
    return (
        "AUDIT-BAD-AMOUNT,VICTIM-A,UTIB0000001,MULE-A,SBIN0000001,not-a-number,"
        "2024-06-01 10:00:00,UPI,DEV-AUDIT,10.9.9.1\n"
        ",VICTIM-B,UTIB0000002,MULE-B,SBIN0000002,500,2024-06-01 10:01:00,"
        "UPI,DEV-AUDIT,10.9.9.1\n"
        f"{duplicate_utr},VICTIM-C,UTIB0000003,MULE-C,SBIN0000003,750,"
        "2024-06-01 10:02:00,UPI,DEV-AUDIT,10.9.9.1\n"
    )


def load_benchmark(
    background: int = 5_000,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, IngestionReport]]:
    """Seeded benchmark, ingested so the audit tab shows real numbers.

    Ground-truth label columns (`is_fraud`, `fraud_scheme`) are dropped before
    ingestion, so the workbench demonstrates a genuinely blind run. Canonical
    columns round-trip exactly through the ingester (verified: frames are
    `.equals()` after ingestion), which is what makes Tab 1 show true mapping
    confidences and rejection counts for the demo dataset.
    """
    result = SyndicateDataSimulator(seed=42).generate(
        background_transactions=background
    )
    transactions = result.transactions.select(list(TRANSACTION_SCHEMA))
    accounts = result.accounts.select(list(ACCOUNT_METADATA_SCHEMA))
    complaints = result.complaints.select(list(COMPLAINT_SCHEMA))

    first_utr = str(transactions["txn_id"][0])
    ingester = DatasetIngester()
    transactions, txn_report = ingester.ingest_transactions(
        (transactions.write_csv() + _audit_rows(first_utr)).encode("utf-8")
    )
    accounts, acc_report = ingester.ingest_accounts(
        accounts.write_csv().encode("utf-8")
    )
    complaints, cmp_report = ingester.ingest_complaints(
        complaints.write_csv().encode("utf-8")
    )
    return transactions, accounts, complaints, {
        "transactions": txn_report,
        "accounts": acc_report,
        "complaints": cmp_report,
    }


def _load_uploaded(file: Any, table: TableKind) -> tuple[pl.DataFrame, IngestionReport]:
    """One uploader file -> (frame, report). Uploaders are single-file by design."""
    return DatasetIngester().ingest(file.getvalue(), table=table)


def _control_bar() -> state.PipelineBundle | None:
    """Dataset and compute control bar.

    Passes the user-selected compute tier into `state.run_pipeline` and caches
    the resulting bundle in `st.session_state`.
    """
    st.sidebar.header("Dataset & Compute Engine")
    tier_option = st.sidebar.selectbox(
        "Compute Tier",
        ["AUTO", "LITE", "STANDARD", "WORKSTATION"],
        index=0,
        key="compute_tier",
        help="Hardware-adaptive profiling (DuckDB threads/RAM, cycle depth, anomaly ensemble)",
    )
    compute_tier: ComputeTier | None = (
        None if tier_option == "AUTO" else cast(ComputeTier, tier_option)
    )

    benchmark_clicked = st.sidebar.button(
        "Load seeded benchmark dataset", key="load_benchmark", use_container_width=True
    )
    st.sidebar.caption("Or upload multi-bank investigation files:")
    uploads: dict[TableKind, Any] = {
        "transactions": st.sidebar.file_uploader(
            "Transactions (CSV/XLSX)", type=["csv", "xlsx"],
            accept_multiple_files=False, key="up_txn"),
        "accounts": st.sidebar.file_uploader(
            "Account metadata (CSV/XLSX)", type=["csv", "xlsx"],
            accept_multiple_files=False, key="up_acc"),
        "complaints": st.sidebar.file_uploader(
            "NCRP complaints (CSV/XLSX)", type=["csv", "xlsx"],
            accept_multiple_files=False, key="up_cmp"),
    }
    upload_clicked = st.sidebar.button(
        "Build from uploads", key="load_uploads", use_container_width=True
    )

    if benchmark_clicked:
        transactions, accounts, complaints, reports = load_benchmark()
        st.session_state["pg_bundle"] = state.run_pipeline(
            transactions, accounts, complaints, reports=reports, compute_tier=compute_tier
        )
    if upload_clicked:
        frames: dict[TableKind, pl.DataFrame] = {}
        upload_reports: dict[str, IngestionReport] = {}
        for table, file in uploads.items():
            if file is None:
                continue
            frame, report = _load_uploaded(file, table)
            frames[table] = frame
            upload_reports[table] = report
        if "transactions" in frames:
            st.session_state["pg_bundle"] = state.run_pipeline(
                frames["transactions"],
                frames.get(
                    "accounts", pl.DataFrame(schema=ACCOUNT_METADATA_SCHEMA)
                ),
                frames.get(
                    "complaints", pl.DataFrame(schema=COMPLAINT_SCHEMA)
                ),
                reports=upload_reports,
                compute_tier=compute_tier,
            )
        else:
            st.sidebar.error("A transactions file is required.")

    bundle: state.PipelineBundle | None = st.session_state.get("pg_bundle")

    if bundle is not None and bundle.hardware_profile is not None:
        hp = bundle.hardware_profile
        st.sidebar.markdown(
            f"""
            <div class="hw-card">
              <div class="hw-card-title">Hardware Profile ({hp.tier})</div>
              <div><b>CPU Cores:</b> {hp.cpu_cores}</div>
              <div><b>System RAM:</b> {hp.total_ram_gb} GB</div>
              <div><b>DuckDB Threads:</b> {hp.duckdb_threads}</div>
              <div><b>DuckDB RAM Limit:</b> {hp.duckdb_memory_limit}</div>
              <div><b>Max Cycle Length:</b> {hp.max_cycle_length}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if bundle is None:
        st.info("Load the seeded benchmark or upload CSV/XLSX files to begin.")
    return bundle


def _render_ingestion(bundle: state.PipelineBundle | None) -> None:
    if bundle is None:
        return
    view = state.build_ingestion_view(bundle.reports)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", view.total_rows)
    c2.metric("Accepted & Normalized", view.total_accepted)
    c3.metric("Quarantined (Rejected)", view.total_rejected)
    c4.metric("Tables Loaded", len(view.tables))

    for table in view.tables:
        st.subheader(f"Table: {table.table} ({table.source})")
        col_map, col_fail = st.columns(2)
        with col_map:
            st.markdown("**Fuzzy Schema Mapping Confidence**")
            st.dataframe(
                [
                    {
                        "Raw Header": row.raw_name,
                        "Canonical Field": row.canonical_name,
                        "Matched Synonym": row.matched_synonym,
                        "Confidence": round(row.confidence, 3),
                    }
                    for row in table.mappings
                ],
                use_container_width=True,
                hide_index=True,
            )
        with col_fail:
            st.markdown("**Quarantine & Failure Reasons**")
            st.dataframe(
                [
                    {"Failure Reason": reason, "Quarantined Rows": count}
                    for reason, count in table.failure_counts
                ],
                use_container_width=True,
                hide_index=True,
            )
        if table.rejected_preview:
            st.caption(f"Quarantined rows preview ({len(table.rejected_preview)} sample rows)")
            st.dataframe(table.rejected_preview, use_container_width=True)
        report = bundle.reports.get(table.table)
        if report is not None:
            st.download_button(
                f"Download Quarantined Rows — {table.table} (CSV)",
                data=rejected_rows_csv(report),
                file_name=f"quarantined_{table.table}.csv",
                mime="text/csv",
                key=f"rej_{table.table}",
            )


def _render_ranking(bundle: state.PipelineBundle | None) -> None:
    if bundle is None:
        return

    # Filter & Sort Bar (single 4-column horizontal row)
    f1, f2, f3, f4 = st.columns(4)
    bands = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    band = f1.selectbox("Risk band", bands, index=0, key="filter_band")
    min_score = f2.slider("Minimum risk score", 0, 100, 0, 5, key="filter_min_score")

    available_roles = sorted({
        a.operational_role for a in bundle.assessments
        if a.operational_role and a.operational_role != "STANDARD_ACCOUNT"
    })
    role_options = ["ALL"] + available_roles
    selected_role = f3.selectbox("Archetype / Role", role_options, index=0, key="filter_role")

    sort_by = f4.selectbox(
        "Sort by",
        ["Risk Score", "Freeze Priority (Recoverable ₹)"],
        index=0,
        key="sort_by",
    )

    leaderboard = state.build_leaderboard(
        bundle, min_score=float(min_score), band=None if band == "ALL" else band
    )
    rows = list(leaderboard.rows)
    if selected_role != "ALL":
        rows = [
            r for r in rows
            if r.operational_role == selected_role or r.top_archetype == selected_role
        ]

    if sort_by == "Freeze Priority (Recoverable ₹)":
        rows.sort(
            key=lambda r: (
                -r.freeze_priority_score,
                -r.retained_tainted_amount,
                -r.risk_score,
                r.account_no,
            )
        )
    else:
        rows.sort(
            key=lambda r: (-r.risk_score, -r.freeze_priority_score, r.account_no)
        )

    st.caption(
        f"{len(rows)} of {leaderboard.total_accounts} accounts matching filters · "
        f"{leaderboard.flagged_accounts} in HIGH/CRITICAL alert tier"
    )

    st.dataframe(
        [
            {
                "Account": row.account_no,
                "Risk Score": row.risk_score,
                "Band": row.risk_band,
                "Role": row.operational_role,
                "Freeze Priority": row.freeze_priority_score,
                "Retained Tainted (₹)": format_inr(row.retained_tainted_amount),
                "Domain": row.domain_score,
                "ML Anomaly": row.anomaly_score,
                "Complaints": row.complaint_count,
                "Top Indicator": row.top_reason,
            }
            for row in rows
        ],
        use_container_width=True,
        hide_index=True,
    )

    if not rows:
        st.info("No accounts match the current filter criteria.")
        return

    st.markdown("---")
    st.subheader("Selected Account Inspector & XAI Attribution")
    selected = st.selectbox(
        "Inspect account", [row.account_no for row in rows], key="inspect_account"
    )
    detail = state.build_account_detail(bundle, selected)

    # 4-card KPI strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Composite Risk Score",
        f"{detail.risk.risk_score:.1f}" if detail.risk else "N/A",
        delta=f"{detail.risk.risk_band} BAND" if detail.risk else None,
    )
    k2.metric("Operational Role", detail.operational_role)
    k3.metric("Recoverable Tainted Funds (₹)", format_inr(detail.retained_tainted_amount))
    k4.metric("Freeze Priority Index", f"{detail.freeze_priority_score:.1f}")

    # Radar chart and timeline side-by-side
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Behavioral Risk Sub-Scores (70/30)")
        if detail.sub_scores:
            radar_fig = go.Figure(
                go.Scatterpolar(
                    r=[score.value for score in detail.sub_scores] + [detail.sub_scores[0].value],
                    theta=[score.label for score in detail.sub_scores] + [detail.sub_scores[0].label],
                    fill="toself",
                    fillcolor="rgba(37, 99, 235, 0.2)",
                    line={"color": "#2563eb", "width": 2},
                    name="Sub-scores",
                )
            )
            radar_fig.update_layout(
                polar={
                    "radialaxis": {"range": [0, 100], "visible": True, "tickfont": {"size": 9}},
                    "angularaxis": {"tickfont": {"size": 10}},
                },
                showlegend=False,
                height=320,
                margin={"l": 20, "r": 20, "t": 35, "b": 20},
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.caption("No sub-score breakdown available.")

    with c2:
        st.markdown("##### Inflow / Outflow Chronological Timeline")
        if detail.timeline:
            time_fig = go.Figure()
            for row in detail.timeline:
                time_fig.add_trace(
                    go.Scatter(
                        x=[row.timestamp],
                        y=[row.amount],
                        mode="markers+text",
                        text=[row.txn_id],
                        textposition="top center",
                        marker={
                            "symbol": "triangle-down" if row.direction == "OUT" else "circle",
                            "color": "#dc2626" if row.direction == "OUT" else "#16a34a",
                            "size": 10,
                        },
                        name=row.direction,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{row.txn_id}</b><br>"
                            + f"{row.direction} {row.counterparty}<br>"
                            + f"Amount: {format_inr(row.amount)}<br>"
                            + f"Mode: {row.payment_mode}<extra></extra>"
                        ),
                    )
                )
            time_fig.update_layout(
                yaxis_title="Amount (₹)",
                xaxis_title="Time",
                height=320,
                margin={"l": 20, "r": 20, "t": 35, "b": 20},
            )
            st.plotly_chart(time_fig, use_container_width=True)
        else:
            st.caption("No transaction records found for this account.")

    # Section A and Section B side-by-side
    facts_column, indicators_column = st.columns(2)
    with facts_column:
        st.markdown("##### Section A — Verified Facts (Supplied Records)")
        st.dataframe(
            [
                {
                    "Category": fact.category,
                    "Record": fact.label,
                    "Detail": fact.value,
                    "Source": fact.source,
                }
                for fact in detail.facts
            ],
            use_container_width=True,
            hide_index=True,
        )
    with indicators_column:
        st.markdown("##### Section B — AI & Analytical Indicators (Leads Only)")
        st.dataframe(
            [
                {
                    "Indicator": item.indicator,
                    "Method": item.method,
                    "Value": item.value,
                    "Points": item.points,
                    "Explanation": item.explanation,
                }
                for item in detail.indicators
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_graph(bundle: state.PipelineBundle | None) -> None:
    if bundle is None:
        return

    # Compact single-row controls
    c1, c2, c3, c4 = st.columns([1.5, 2, 2, 1.2])
    source = c1.selectbox("Trace from", ["Account", "UTR", "Complaint ACK"], key="trace_from")
    accounts = [row.account_no for row in state.build_leaderboard(bundle).rows]
    utrs = sorted(set(bundle.transactions["txn_id"].to_list()))
    ack_nos = sorted(set(bundle.complaints["ack_no"].to_list()))

    options = {
        "Account": accounts[:200],
        "UTR": utrs[:500],
        "Complaint ACK": ack_nos,
    }[source]

    entity_select = c2.selectbox("Reference", options if options else [""], key="trace_ref")
    manual = c3.text_input("Custom reference", key="trail_manual")
    hops = c4.slider("Hops", 1, 6, 3, key="trace_hops")

    st.caption(
        "Selection lists are capped for responsiveness: top 200 of "
        f"{len(accounts)} accounts, first 500 of {len(utrs)} transaction "
        f"references, all {len(ack_nos)} complaint ACKs. Enter any reference in Custom reference."
    )

    entity = manual.strip() or entity_select
    if entity:
        view = state.build_trail_graph(bundle, start=entity, max_hops=hops)
        if not view.nodes:
            st.warning(f"No trail found for {entity} within {hops} hops.")
        else:
            # 4-metric summary strip
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trail Origin", view.origin_label)
            m2.metric("Nodes in Trail", len(view.nodes))
            total_tainted = sum(edge.taint_amount for edge in view.edges)
            m3.metric("Total Tainted Flow", format_inr(total_tainted))
            m4.metric("Hops Traced", f"{max((e.hop for e in view.edges), default=0)} of {hops}")

            # Compact color legend
            st.markdown(
                '<div class="legend-strip">'
                '<span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#2563eb; margin-right:4px;"></span>'
                '<strong>Account Node</strong> (#2563eb) &nbsp;&nbsp;|&nbsp;&nbsp; '
                '<span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#dc2626; margin-right:4px;"></span>'
                '<strong>Terminal / Cash-Out</strong> (#dc2626) &nbsp;&nbsp;|&nbsp;&nbsp; '
                '<strong>Directed Legs:</strong> Hops 1–6 color-graduated with ₹ flow and FIFO taint captions'
                '</div>',
                unsafe_allow_html=True,
            )

            st_link_analysis(
                elements=view.elements,
                layout={"name": "cose", "animate": False, "fit": True},
                node_styles=[
                    NodeStyle(label="account", color="#2563eb", caption="caption"),
                    NodeStyle(label="terminal", color="#dc2626", caption="caption"),
                ],
                edge_styles=[
                    EdgeStyle(label=f"hop {hop}", directed=True, caption="caption")
                    for hop in range(1, 7)
                ],
                height=650,
                key="trail_canvas",
            )

    st.markdown("---")
    st.subheader("Syndicate & Network Topologies")
    loops, collusion, terminals = st.tabs(
        ["Circular wash loops", "Branch / IMEI cohorts", "AePS terminal hotspots"]
    )
    with loops:
        st.dataframe(
            [
                {
                    "Accounts": " → ".join(row.accounts),
                    "UTRs": ", ".join(row.txn_ids),
                    "Total Amount (₹)": format_inr(row.total_amount),
                    "Max Decay Ratio": f"{row.max_decay_ratio:.2%}",
                    "Start Timestamp": str(row.start_timestamp),
                }
                for row in state.build_cycle_view(bundle).rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    with collusion:
        st.dataframe(
            [
                {
                    "Collusion Basis": group.basis,
                    "Cluster Key": group.key,
                    "Member Accounts": ", ".join(group.accounts),
                    "Opening Dates": ", ".join(group.opening_dates),
                }
                for group in state.build_collusion_view(bundle).groups
            ],
            use_container_width=True,
            hide_index=True,
        )
    with terminals:
        st.dataframe(
            [
                {
                    "Terminal Account": row.account_no,
                    "Terminal Inflow (₹)": format_inr(row.terminal_inflow),
                    "Cash-Out Legs": row.cashout_count,
                    "Upstream Chains": row.terminal_chain_count,
                    "Upstream Accounts": ", ".join(row.upstream_accounts),
                }
                for row in state.build_terminal_view(bundle).rows
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_case(bundle: state.PipelineBundle | None) -> None:
    if bundle is None:
        return
    correlation = state.build_correlation_view(bundle)

    # Top KPI strip
    k1, k2, k3 = st.columns(3)
    k1.metric("Correlated Complaint Pairs", len(correlation.rows))
    k2.metric("Interstate Links", sum(1 for row in correlation.rows if row.interstate))
    k3.metric("Convergence Hotspots", len(correlation.hotspots))

    # Side-by-side tables
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Cross-State NCRP Complaint Links")
        st.dataframe(
            [
                {
                    "ACK A": row.ack_a,
                    "State A": row.state_a,
                    "ACK B": row.ack_b,
                    "State B": row.state_b,
                    "Basis": row.link_basis,
                    "Interstate": "Yes" if row.interstate else "No",
                }
                for row in correlation.rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown("##### Shared Mule / Terminal Convergence Hotspots")
        st.dataframe(
            [
                {
                    "Shared Hotspot Key": hotspot.key,
                    "Linked NCRP ACKs": ", ".join(hotspot.ack_nos),
                    "Total Complaints Linked": len(hotspot.ack_nos),
                }
                for hotspot in correlation.hotspots
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("Case Evidence Dossier Builder (Section 94 BNSS)")

    # Scope and Target Picker in one row
    r1, r2 = st.columns([1.5, 2.5])
    scope = r1.radio(
        "Dossier Scope", ["Account", "Cluster", "Complaint ACK"], horizontal=True, key="dossier_scope"
    )
    dossier = None
    account_nos = [row.account_no for row in state.build_leaderboard(bundle).rows]

    if scope == "Account":
        chosen_account = r2.selectbox(
            "Select Target Account", account_nos[:200] if account_nos else [""], key="dossier_account"
        )
        if chosen_account:
            dossier = state.build_account_dossier(bundle, chosen_account)
    elif scope == "Cluster":
        chosen = r2.multiselect(
            "Select Cluster Accounts", account_nos[:20], key="dossier_cluster"
        )
        if chosen:
            dossier = state.build_cluster_dossier(bundle, chosen)
    else:
        ack_nos = sorted(set(bundle.complaints["ack_no"].to_list()))
        chosen_ack = r2.selectbox(
            "Select Complaint ACK", ack_nos if ack_nos else [""], key="dossier_ack"
        )
        if chosen_ack:
            dossier = state.build_complaint_dossier(bundle, chosen_ack)

    if dossier is None:
        st.info("Select a target account, cluster, or complaint ACK above to preview and export the dossier.")
        return

    # 3 download buttons in a prominent 3-column bar
    b1, b2, b3 = st.columns(3)
    b1.download_button(
        "Download .html (Print-Ready)",
        data=state.dossier_html(dossier),
        file_name=f"{dossier.case_reference}.html",
        mime="text/html",
        use_container_width=True,
        key="dl_dossier_html",
    )
    b2.download_button(
        "Download .md",
        data=state.dossier_markdown(dossier).encode(),
        file_name=f"{dossier.case_reference}.md",
        mime="text/markdown",
        use_container_width=True,
        key="dl_dossier_md",
    )
    b3.download_button(
        "Download .json",
        data=state.dossier_json(dossier),
        file_name=f"{dossier.case_reference}.json",
        mime="application/json",
        use_container_width=True,
        key="dl_dossier_json",
    )

    with st.container(border=True):
        st.markdown(state.dossier_markdown(dossier))


def main() -> None:
    st.set_page_config(
        page_title="PraharGraph — Investigator Workbench",
        page_icon="🛡",
        layout="wide",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    st.title("PraharGraph — Investigator Workbench")
    st.caption(
        "Offline Mule Account Detection, FIFO Money-Trail Graph & Section 94 BNSS Workbench · "
        + RISK_DISCLAIMER
    )
    bundle = _control_bar()
    tab_ingest, tab_rank, tab_graph, tab_case = st.tabs(list(_TABS))
    with tab_ingest:
        _render_ingestion(bundle)
    with tab_rank:
        _render_ranking(bundle)
    with tab_graph:
        _render_graph(bundle)
    with tab_case:
        _render_case(bundle)


if __name__ == "__main__":
    main()
