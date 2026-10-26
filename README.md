# PraharGraph (प्रहार ग्राफ)

**Offline Mule Account Detection, FIFO Money-Trail Graph & Section 94 BNSS Workbench**  
*Built for Pan-India Law Enforcement, State Cyber Cells, Economic Offences Units (EOU), and Bank Fraud Analytics Teams.*

---

## Executive Summary & Problem Framing

Financial cyber fraud across India (digital arrests, investment Ponzi scams, task-based frauds, and AePS cash-out networks) relies on structured mule account rings that span multiple banks and states. Investigating officers face heterogeneous bank statement formats, delayed responses from nodal bank officers, and black-box ML tools that fail Indian evidentiary standards under the Bharatiya Nagarik Suraksha Sanhita (BNSS).

**PraharGraph** is a 100% offline, deterministic, forensic-grade investigation workbench. It ingests multi-bank CSV/XLSX exports and NCRP complaint records, reconstructs the temporal FIFO money trail, computes explainable risk scores with actionable operational roles, and generates court-admissible Section 94 BNSS requisition dossiers in seconds.

---

## What Problem It Solves vs. Existing Gaps

| Dimension | 1. Manual Excel / CDR Stitching | 2. Static Bank Rule Engines | 3. Black-Box Cloud AI/ML | 4. PraharGraph (This Solution) |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Hop Traversal** | Days to weeks; breaks past Hop-2 across different banks | Fails on split smurfing & circular wash loops | Obscure graph neural networks without temporal audit | **Instant directed FIFO money-flow graph** up to 6 hops; tracks tainted residual amounts |
| **Forensic Hygiene & Rejection** | Silent cell truncations and unhandled date/amount parsing | Hard failures or silent rejection on alternate column headers | Data ingested without data lineage or audit trails | **Fuzzy schema mapping (Jaro-Winkler)** + 100% quarantine audit CSV export (zero data loss) |
| **XAI & Evidentiary Quality** | Subjective officer notes; error-prone calculations | Rigid IF/THEN flags with high false positives | Inadmissible in court (unexplained neural activations) | **70/30 Hybrid XAI**; Section A (Supplied Facts) strictly partitioned from Section B (AI Leads) |
| **Triage & Asset Recovery** | Unranked accounts; officers freeze empty pass-throughs | Flags large transactions, missing small distributor smurfs | Prioritizes score, not recoverable fund balance | **Freeze Priority Index** orders targets by *retained tainted ₹ balance* for maximum recovery |
| **Security & Sovereign Compliance** | Cloud Excel sharing risks data leaks & non-compliance | Siloed within individual banking cores | Cloud API dependent; leaks sensitive PII / KYC data | **100% Air-Gapped / Offline**; zero network calls, zero telemetry, local DuckDB & Polars |

---

## 4-Step Forensic Workflow

```text
 ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
 │ 1. Fuzzy Ingest &    │ ----> │ 2. DuckDB Engine &   │ ----> │ 3. Hybrid 70/30 XAI  │ ----> │ 4. Section 94 BNSS   │
 │    Quarantine Audit  │       │    FIFO Taint Graph  │       │    & Freeze Priority │       │    Requisition Export│
 └──────────────────────┘       └──────────────────────┘       └──────────────────────┘       └──────────────────────┘
   • Tolerates messy CSV/XLSX     • High-speed SQL features      • 70% Rules + 30% iForest      • Standalone HTML / MD
   • Auto-header normalization    • Directed FIFO taint flow     • 7 Operational Muling Roles   • Section A (Facts)
   • Quarantine audit CSV export  • Wash cycle & cohort graph    • Ranked by Recoverable ₹      • Section B (AI Leads)
```

1. **Fuzzy Ingest & Data Hygiene:** Auto-maps heterogeneous bank statement headers using Jaro-Winkler string distance and regex. Isolates malformed rows into an exportable Quarantine Audit CSV with exact failure reasons (`unparseable_amount`, `missing:txn_id`, `duplicate_utr`, `negative_amount`).
2. **DuckDB Features & FIFO Graph Taint:** Executes sub-second feature extraction in DuckDB. Graph engine preserves temporal directionality (money cannot flow backward in time) and tracks tainted fund residuals across multi-hop chains and closed wash loops.
3. **Hybrid 70/30 XAI & Freeze Priority:** Blends deterministic behavioral indicators (70%) with unsupervised IsolationForest/TreeSHAP anomaly scoring (30%). Classifies accounts into concrete operational roles (`SMURF_DISTRIBUTOR`, `LAYERING_CONDUIT`, `CASH_OUT_TERMINAL`, `WASH_CYCLE_NODE`) and computes a **Freeze Priority Score** prioritizing accounts with actionable, retained tainted balances.
4. **Section 94 BNSS Requisition Dossier:** Exports court-ready investigation packages in 1-click standalone HTML, Markdown, and JSON. Strictly separates verified records (Section A) from machine-assisted leads (Section B), providing an automated Section 94 BNSS notice draft for authorized police officers.

---

## 3-Command Quickstart (100% Offline)

Requirements: Python 3.12–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies (air-gapped or pre-resolved cache)
uv sync

# 2. Launch the interactive Pan-India Investigator Workbench
PYTHONPATH=src uv run streamlit run src/ui/app.py

# 3. Run the automated forensic benchmark suite
uv run prahar-verify
```

*In the web UI, click **Load seeded benchmark dataset** or upload multi-bank files from `data/sample_uploads/`.*

---

## Requirements Coverage Matrix

| # | Problem Statement Requirement | Architecture & Implementation | Test Verification |
| :-: | :--- | :--- | :--- |
| **1** | **Transaction Analysis & Ingestion** | `src/ingestion/mapper.py` (`DatasetIngester`, fuzzy header matching, byte-stable XLSX/CSV parser, quarantine audit) | `tests/test_phase1.py` (all header variations, date styles, malformed amounts, duplicate UTRs) |
| **2** | **Mule Account Detection & XAI** | `src/risk/scorer.py` (70/30 hybrid scoring, cohort volume z-score, IsolationForest + LOF + empirical tail, TreeSHAP, Freeze Priority) | `tests/test_phase2_risk.py`, `tests/test_phase5_intelligence.py` (`TestAnomalyEnsemble`, `TestCohortNormalization`) |
| **3** | **Money Trail Analysis** | `src/graph/analyzer.py` (`trace_money_trail`, `trace_complaint_trail`, FIFO taint propagation, retained residual tracking) | `tests/test_phase2_graph.py` (temporal monotonicity, taint conservation, 6-hop depth limits) |
| **4** | **Complaint Correlation** | `src/graph/analyzer.py` (`correlate_complaints`), `src/ui/state.py` (`build_correlation_view`, interstate nexus, shared terminals) | `tests/test_phase2_graph.py`, `tests/test_phase3.py` (`test_correlation_view_flags_interstate_links`) |
| **5** | **Graph-Based Investigation** | `src/ui/state.py` (`build_trail_graph`), `src/ui/app.py` (Cytoscape interactive canvas, circular wash loops, branch/IMEI cohorts) | `tests/test_phase3.py` (`test_trail_graph_is_traceable_by_account_utr_and_complaint`, `test_cycle_view_lists_a_closed_wash_loop`) |
| **6** | **Investigation Dashboard & Dossier** | `src/ui/app.py` (4-tab executive workbench), `src/export/dossier.py` (Section A/B/C dossiers, Section 94 BNSS requisitions) | `tests/test_phase3.py` (`test_app_benchmark_load_populates_ingestion_metrics`, `test_app_upload_branch_builds_a_bundle_from_sample_files`) |

---

## Verified Benchmark Results

Ran automatically via `uv run prahar-verify` (blind evaluation against 6,539 rows, 5,249 transactions, 1,274 accounts, 16 NCRP complaints, seed=42):

| Metric | Result | Operational Significance |
| :--- | :--- | :--- |
| **Overall Culprit Recall** | **100.0% (60/60 culprits)** | Every fraud-involved account identified in HIGH or CRITICAL alert tier |
| **Combined Alert Precision** | **98.7%** | Only 1 false positive across 1,274 accounts; officers never waste time on clean citizens |
| **End-to-End Latency** | **1.56 seconds** | Ingest -> DuckDB Features -> Graph Taint -> ML Ensemble -> Requisition generation |
| **Rapid Pass-Through Recall** | **100.0% (20/20)** | Catches rapid in-and-out conduits (median retention < 15 min) |
| **Fan-Out Smurfing Recall** | **100.0% (10/10)** | Detects 1-to-N micro-splitting across mule distributor rings |
| **AePS Cash-Out Recall** | **100.0% (6/6)** | Flags micro-ATM / AePS CSP terminals absorbing illicit inflows |
| **Branch / Device Collusion Recall** | **100.0% (6/6)** | Identifies sleeper accounts opened within 48h or sharing device IMEIs |
| **Circular Wash Ring Recall** | **100.0% (18/18)** | Traverses closed loops (e.g. A → B → C → A) with <= 15% decay |
| **Automated Test Suite** | **292 Passing Tests** | 100% strict `mypy` and `ruff` verified; zero breaking changes |

### Hardware-Adaptive Profiles

PraharGraph adapts its compute footprint dynamically to any forensic environment:
- **`LITE`** (Field Laptops, 2 cores, 256MB RAM limit): 100 trees, cycle depth 3, sampling betweenness.
- **`STANDARD`** (Station Desktops, 4-8 cores, 1GB RAM limit): 200 trees, cycle depth 4, LOF density active.
- **`WORKSTATION`** (Forensic Labs, 8+ cores, 4GB RAM limit): 400 trees, cycle depth 5, exact betweenness, full graph diffusion.

---

## Legal & Sovereign Architecture Standards

1. **Evidentiary Integrity (Bharatiya Sakshya Adhiniyam, 2023):**
   - **Section A (Verified Facts):** Verbatim reproduction of bank statements and NCRP records. Hashable and audit-proof.
   - **Section B (AI & Analytical Leads):** Labeled explicitly as machine inference and intelligence leads, not primary evidence.
   - **Section C (Section 94 BNSS Draft):** Structured formal notice requesting certified records under Section 2A of the Bankers' Books Evidence Act, 1891.
2. **Deterministic Reproducibility:**
   - No `datetime.now()` or system clock dependencies inside generated dossiers; byte-identical outputs across executions.
   - Zero protected attributes (`state`, `pincode`, `account_type`) in scoring models to eliminate geographical or socio-economic bias.
3. **Air-Gapped Sovereign Privacy:**
   - Completely local processing using DuckDB and Polars. Telemetry disabled (`gatherUsageStats = false`). No remote CDNs, fonts, or external API dependencies.
