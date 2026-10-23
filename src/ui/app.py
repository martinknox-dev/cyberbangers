"""PraharGraph investigator workbench (offline Streamlit front end)."""

from __future__ import annotations

from typing import Any

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
from export.dossier import rejected_rows_csv
from ingestion.mapper import DatasetIngester
from simulator.generator import SyndicateDataSimulator
from ui import state

_TABS = (
    "1 · Ingestion & Data Hygiene",
    "2 · Mule Risk Ranking & XAI",
    "3 · Money Trail & Syndicates",
    "4 · Complaint Correlation & Dossier",
)


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
    """Seeded Bihar benchmark, ingested so the audit tab shows real numbers.

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
    """Dataset control bar. The pipeline runs only when the dataset changes.

    `state.run_pipeline` fits IsolationForest + TreeSHAP and rebuilds DuckDB and
    NetworkX state. Streamlit re-executes this whole script on every widget
    interaction, so the finished bundle is cached in `st.session_state` and
    reused by every slider/selectbox rerun.
    """
    st.sidebar.header("Dataset")
    benchmark_clicked = st.sidebar.button(
        "Load seeded Bihar benchmark dataset", key="load_benchmark"
    )
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
    upload_clicked = st.sidebar.button("Build from uploads", key="load_uploads")

    if benchmark_clicked:
        transactions, accounts, complaints, reports = load_benchmark()
        st.session_state["pg_bundle"] = state.run_pipeline(
            transactions, accounts, complaints, reports=reports
        )
    if upload_clicked:
        frames: dict[TableKind, pl.DataFrame] = {}
        upload_reports: dict[str, IngestionReport] = {}
        for table, file in uploads.items():
            if file is None:
                continue
            frame, report = _load_uploaded(file, table)
            frames[table] = frame
            upload_reports[table] = report   # exactly one report per table
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
            )
        else:
            st.sidebar.error("A transactions file is required.")

    bundle: state.PipelineBundle | None = st.session_state.get("pg_bundle")
    if bundle is None:
        st.info("Load the seeded benchmark or upload CSV/XLSX files to begin.")
    return bundle


def _render_ingestion(bundle: state.PipelineBundle | None) -> None:
    if bundle is None:
        return
    view = state.build_ingestion_view(bundle.reports)
    left, middle, right = st.columns(3)
    left.metric("Rows total", view.total_rows)
    middle.metric("Accepted", view.total_accepted)
    right.metric("Rejected", view.total_rejected)
    for table in view.tables:
        st.subheader(f"{table.table} — {table.source}")
        st.dataframe(
            [{"header": row.raw_name, "canonical": row.canonical_name,
              "synonym": row.matched_synonym, "confidence": round(row.confidence, 3)}
             for row in table.mappings],
            use_container_width=True, hide_index=True,
        )
        st.dataframe(
            [{"reason": reason, "rows": count}
             for reason, count in table.failure_counts],
            use_container_width=True, hide_index=True,
        )
        if table.rejected_preview:
            st.caption(f"First {len(table.rejected_preview)} rejected rows")
            st.dataframe(table.rejected_preview, use_container_width=True)
        report = bundle.reports.get(table.table)
        if report is not None:
            st.download_button(
                f"Download rejected rows — {table.table} (CSV)",
                data=rejected_rows_csv(report),
                file_name=f"rejected_{table.table}.csv",
                mime="text/csv",
                key=f"rej_{table.table}",
            )


def _render_ranking(bundle: state.PipelineBundle | None) -> None:
    if bundle is None:
        return
    bands = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    band = st.selectbox("Risk band", bands, index=0)
    min_score = st.slider("Minimum risk score", 0, 100, 0, 5)
    leaderboard = state.build_leaderboard(
        bundle, min_score=float(min_score), band=None if band == "ALL" else band
    )
    st.caption(f"{len(leaderboard.rows)} of {leaderboard.total_accounts} accounts · "
               f"{leaderboard.flagged_accounts} in HIGH/CRITICAL")
    st.dataframe(
        [{"Account": row.account_no, "Score": row.risk_score,
          "Domain": row.domain_score, "ML anomaly": row.anomaly_score,
          "Band": row.risk_band, "Archetype": row.top_archetype,
          "Complaints": row.complaint_count, "Top reason": row.top_reason}
         for row in leaderboard.rows],
        use_container_width=True, hide_index=True,
    )
    if not leaderboard.rows:
        return
    selected = st.selectbox(
        "Inspect account", [row.account_no for row in leaderboard.rows]
    )
    detail = state.build_account_detail(bundle, selected)
    facts_column, indicators_column = st.columns(2)
    with facts_column:
        st.markdown("### Section A — Verified facts")
        st.dataframe(
            [{"Category": fact.category, "Record": fact.label,
              "Detail": fact.value, "Source": fact.source}
             for fact in detail.facts],
            use_container_width=True, hide_index=True,
        )
    with indicators_column:
        st.markdown("### Section B — Analytical indicators")
        st.dataframe(
            [{"Indicator": item.indicator, "Method": item.method,
              "Value": item.value, "Points": item.points,
              "Explanation": item.explanation}
             for item in detail.indicators],
            use_container_width=True, hide_index=True,
        )
        if detail.sub_scores:
            st.plotly_chart(
                go.Figure(go.Scatterpolar(
                    r=[score.value for score in detail.sub_scores]
                        + [detail.sub_scores[0].value],
                    theta=[score.label for score in detail.sub_scores]
                        + [detail.sub_scores[0].label],
                    fill="toself", name="Sub-scores",
                )).update_layout(
                    polar={"radialaxis": {"range": [0, 100]}}, showlegend=False
                ),
                use_container_width=True,
            )
    if detail.timeline:
        figure = go.Figure()
        for row in detail.timeline:
            figure.add_trace(go.Scatter(
                x=[row.timestamp], y=[row.amount],
                mode="markers+text", text=[row.txn_id], textposition="top center",
                marker={
                    "symbol": "triangle-down" if row.direction == "OUT" else "circle",
                    "color": "#b00020" if row.direction == "OUT" else "#1a7f37",
                    "size": 12,
                },
                name=row.direction, showlegend=False,
                hovertemplate=("%{text}<br>" + row.direction + " " + row.counterparty
                               + "<br>₹%{y:,.2f}<extra></extra>"),
            ))
        st.plotly_chart(
            figure.update_layout(title="Inflow / outflow timeline",
                                 yaxis_title="Amount (₹)"),
            use_container_width=True,
        )


def _render_graph(bundle: state.PipelineBundle | None) -> None:
    if bundle is None:
        return
    source = st.radio("Trace from", ["Account", "UTR", "Complaint ACK"], horizontal=True)
    options = {
        "Account": [row.account_no for row in state.build_leaderboard(bundle).rows[:200]],
        "UTR": sorted(set(bundle.transactions["txn_id"].to_list()))[:500],
        "Complaint ACK": sorted(set(bundle.complaints["ack_no"].to_list())),
    }[source]
    entity = st.selectbox("Reference", options) if options else ""
    hops = st.slider("Hops", 1, 6, 3)
    if entity:
        view = state.build_trail_graph(bundle, start=entity, max_hops=hops)
        st.caption(f"Trail origin: {view.origin_label}")
        st_link_analysis(
            elements=view.elements,
            layout={"name": "cose", "animate": False, "fit": True},
            node_styles=[
                NodeStyle(label="account", color="#345eeb", caption="caption"),
                NodeStyle(label="terminal", color="#b00020", caption="caption"),
            ],
            # one style per hop so multi-hop trails stay visually distinguishable
            edge_styles=[
                EdgeStyle(label=f"hop {hop}", directed=True, caption="caption")
                for hop in range(1, 7)
            ],
            height=650,
            key="trail_canvas",
        )
    loops, collusion, terminals = st.tabs(
        ["Circular wash loops", "Branch / IMEI cohorts", "AePS terminal hotspots"]
    )
    with loops:
        st.dataframe(
            [{"Accounts": " → ".join(row.accounts), "UTRs": ", ".join(row.txn_ids),
              "Total": row.total_amount, "Max decay": row.max_decay_ratio}
             for row in state.build_cycle_view(bundle).rows],
            use_container_width=True, hide_index=True,
        )
    with collusion:
        st.dataframe(
            [{"Basis": group.basis, "Key": group.key,
              "Accounts": ", ".join(group.accounts)}
             for group in state.build_collusion_view(bundle).groups],
            use_container_width=True, hide_index=True,
        )
    with terminals:
        st.dataframe(
            [{"Terminal": row.account_no, "Terminal inflow": row.terminal_inflow,
              "Cash-out legs": row.cashout_count,
              "Upstream chains": row.terminal_chain_count}
             for row in state.build_terminal_view(bundle).rows],
            use_container_width=True, hide_index=True,
        )


def _render_case(bundle: state.PipelineBundle | None) -> None:
    if bundle is None:
        return
    correlation = state.build_correlation_view(bundle)
    st.subheader("Cross-state NCRP correlation")
    st.dataframe(
        [{"ACK A": row.ack_a, "State A": row.state_a,
          "ACK B": row.ack_b, "State B": row.state_b,
          "Basis": row.link_basis, "Interstate": row.interstate}
         for row in correlation.rows],
        use_container_width=True, hide_index=True,
    )
    st.dataframe(
        [{"Shared account/terminal": hotspot.key,
          "ACKs": ", ".join(hotspot.ack_nos)}
         for hotspot in correlation.hotspots],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("Case evidence dossier")
    scope = st.radio(
        "Dossier scope", ["Account", "Cluster", "Complaint ACK"], horizontal=True
    )
    dossier = None
    if scope == "Account":
        account_nos = [
            row.account_no for row in state.build_leaderboard(bundle).rows[:200]
        ]
        if account_nos:
            dossier = state.build_account_dossier(
                bundle, st.selectbox("Account", account_nos, key="dossier_account")
            )
    elif scope == "Cluster":
        account_nos = [
            row.account_no for row in state.build_leaderboard(bundle).rows[:20]
        ]
        chosen = st.multiselect("Cluster accounts", account_nos, key="dossier_cluster")
        if chosen:
            dossier = state.build_cluster_dossier(bundle, chosen)
    else:
        ack_nos = sorted(set(bundle.complaints["ack_no"].to_list()))
        if ack_nos:
            dossier = state.build_complaint_dossier(
                bundle, st.selectbox("Complaint ACK", ack_nos, key="dossier_ack")
            )
    if dossier is None:
        st.info("Select a subject to preview the dossier.")
        return
    st.markdown(state.dossier_markdown(dossier))
    st.download_button(
        "Download .md", data=state.dossier_markdown(dossier).encode(),
        file_name=f"{dossier.case_reference}.md", mime="text/markdown",
    )
    st.download_button(
        "Download .html", data=state.dossier_html(dossier),
        file_name=f"{dossier.case_reference}.html", mime="text/html",
    )
    st.download_button(
        "Download .json", data=state.dossier_json(dossier),
        file_name=f"{dossier.case_reference}.json", mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="PraharGraph — Investigator Workbench",
                       page_icon="🛡", layout="wide")
    st.title("PraharGraph — Investigator Workbench")
    st.caption(
        "Offline mule-account investigation for Bihar Police · " + RISK_DISCLAIMER
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
