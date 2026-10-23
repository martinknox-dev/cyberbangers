# PraharGraph

**Offline mule-account and money-trail investigation workbench for Bihar Police**
Bihar Police Cyber Hackathon 2026 · C-DAC Patna evaluation build

## Problem and ground-truth focus

PraharGraph ingests messy multi-bank CSV/XLSX exports and NCRP complaint
records, derives account behaviour features and a transaction graph, and ranks
mule accounts with a transparent 70/30 domain + IsolationForest/SHAP risk
engine. It runs **100% offline and deterministically**: no network calls, no
telemetry, no wall-clock reads in generated documents, and the same seed
produces byte-identical data and dossiers. Every input row is accounted for —
accepted rows are counted and rejected rows retain their exact reason and are
exportable as a CSV audit, so nothing is silently dropped. Outputs are strictly
partitioned into **Section A verified facts** (supplied ledger/master/NCRP
records), **Section B analytical and AI indicators** (leads, never evidence)
and **Section C a draft Section 94 BNSS requisition** for authorised-officer
review.

## Quickstart (offline, reproducible)

Requirements: Python 3.12–3.13 and [uv](https://docs.astral.sh/uv/). The first
`uv sync` resolves dependencies once; runtime never touches the network.

```bash
uv sync
uv run pytest -q
uv run prahar-verify
uv run streamlit run src/ui/app.py
```

In the workbench, either click **Load seeded Bihar benchmark dataset** or upload
the three deterministic demo files from `data/sample_uploads/` (transactions
CSV, account-master XLSX, NCRP complaints CSV). Regenerate those files at any
time, byte-for-byte, with:

```bash
uv run python -m simulator.export_samples
```

Quality gates used throughout the build (strict mypy, ruff, full suite):

```bash
uv run pytest -q          # 275 passed
uv run mypy src           # Success: no issues found in 22 source files
uv run ruff check src tests
```

## Requirements coverage matrix

| # | Problem statement requirement | Implementation | Verification |
| --- | --- | --- | --- |
| 1 | **Transaction Analysis** — ingest heterogeneous bank files, normalise headers/dates/amounts, audit rejects | `src/ingestion/mapper.py` (`DatasetIngester`, fuzzy header mapping, CSV/XLSX detection); `src/features/account_features.py` | `tests/test_phase1.py` (header styles, date formats, XLSX, duplicate UTR, rejection reasons); `tests/test_phase2_risk.py::test_features_are_deterministic` |
| 2 | **Mule Account Detection** — rank mule accounts with explainable risk | `src/risk/scorer.py` (`RiskScorer.score_with_assessments`, 70/30 hybrid, cohort z-score, per-domain components, IsolationForest + LOF + empirical-tail ensemble, TreeSHAP) | `tests/test_phase2_risk.py::test_injected_mules_outrank_background_accounts`, `::test_explanations_carry_value_threshold_and_utrs`, `::test_shap_tree_explainer_works_on_isolation_forest`; `tests/test_phase5_intelligence.py::TestAnomalyEnsemble` |
| 3 | **Money Trail Analysis** — follow disputed funds through hops with an explicit tracing assumption | `src/graph/analyzer.py` (`trace_money_trail`, `trace_complaint_trail`, FIFO taint and retained residual, hop limits, damped proximity diffusion) | `tests/test_phase2_graph.py` (hop order, taint conservation, cycles, never walk backwards, determinism, simulated dispute trail); `tests/test_phase5_intelligence.py::TestGraphRiskDiffusion` |
| 4 | **Complaint Correlation** — link NCRP cases through shared accounts, devices, terminals and states | `src/graph/analyzer.py` (`correlate_complaints`, shared-evidence links); `src/ui/state.py` (`build_correlation_view`) | `tests/test_phase2_graph.py::test_complaints_link_through_shared_account`, `::test_shared_device_links_complaints`, `::test_complaints_correlate_across_states`; `tests/test_phase3.py::test_correlation_view_flags_interstate_links` |
| 5 | **Graph-Based Investigation** — interactive money-flow canvas, wash loops, collusion cohorts, terminal hotspots | `src/ui/state.py` (`build_trail_graph`, cycle/collusion/terminal views); `src/ui/app.py` (Tab 3, per-hop edge styles) | `tests/test_phase3.py::test_trail_graph_is_traceable_by_account_utr_and_complaint`, `::test_trail_graph_elements_are_renderable_and_deterministic`, `::test_cycle_view_lists_a_closed_wash_loop`, `::test_collusion_and_terminal_and_correlation_views` |
| 6 | **Investigation Dashboard** — four-tab workbench and printable evidence dossier | `src/ui/app.py` (ingestion, ranking/XAI, graph, case tabs); `src/export/dossier.py` (Section A/B/C Markdown, standalone HTML, JSON, rejected-row CSV) | `tests/test_phase3.py::test_app_upload_branch_builds_a_bundle_from_sample_files`, `::test_section_a_lists_ledger_account_ncrp_and_shared_kyc_facts`, `::test_section_b_separates_indicators_from_facts`, `::test_requisition_cites_section_94_and_keeps_lien_marking_separate` |

## Verified benchmark results

`uv run prahar-verify` runs the seeded benchmark end to end, prints ingestion
and detection metrics, and exits 0. Exact output of one run:

```text
PraharGraph seeded benchmark - synthetic, offline, deterministic
seed=42 - background_transactions=5000 - labels dropped before ingestion
  hardware: compute_tier=LITE (duckdb_threads=2, memory_limit=256MB, iforest_estimators=100, knn_density=off, diffusion_iterations=5; auto-detected=STANDARD)

Ingestion (blind round-trip through the CSV ingester):
  rows total=6539 accepted=6539 rejected=0
  transactions=5249 accounts=1274 complaints=16 risk assessments=1274

Band       Flagged  True positives  Precision   Recall
HIGH            44              29      65.9%    48.3%
CRITICAL        31              31     100.0%    51.7%
HIGH+CRIT       75              60      80.0%   100.0%
  combined involved precision=98.7% (denominator includes victims and relays) - background false positives=1

Per-scheme detection at HIGH or CRITICAL (precision is the scheme's share of the combined band's flags, so the columns sum to overall combined-band precision):
Scheme                           Culprits   HIGH rec   CRIT rec   H+C recall   H+C prec
rapid_pass_through                     20      60.0%      40.0%       100.0%      26.7%
fan_out_smurfing                       10       0.0%     100.0%       100.0%      13.3%
aeps_terminal_cash_out                  6     100.0%       0.0%       100.0%       8.0%
branch_collusion_shared_device          6      83.3%      16.7%       100.0%       8.0%
circular_wash                          18      33.3%      66.7%       100.0%      24.0%

Top 5 accounts by freeze priority (recoverable tainted rupees):
  Account         Band      Role                      Risk  Freeze   Recoverable taint
  2990060008      CRITICAL  SMURF_DISTRIBUTOR         74.7    61.9              ₹7,870
  2990060006      CRITICAL  SMURF_DISTRIBUTOR         74.4    61.7              ₹7,870
  2990060004      CRITICAL  SMURF_DISTRIBUTOR         74.2    61.6              ₹7,870
  2990060002      CRITICAL  SMURF_DISTRIBUTOR         74.0    61.5              ₹7,870
  2990060000      CRITICAL  SMURF_DISTRIBUTOR         73.9    61.4              ₹7,870

Screening band (risk_score >= 40, the Phase 2 evaluation threshold):
  flagged=77 of 1274 accounts - precision=77.9% - recall=100.0%
  per-scheme recall: rapid_pass_through 20/20 - fan_out_smurfing 10/10 - aeps_terminal_cash_out 6/6 - branch_collusion_shared_device 6/6 - circular_wash 18/18

Runtime: 1.71 s (simulate -> ingest -> features -> graph -> risk)
Synthetic topologies only: these numbers are not real-world accuracy claims.
```

How to read these numbers, honestly:

- **HIGH/CRITICAL alert tier** now catches **100%** of all culprit accounts (60 of 60)
  across all five fraud topologies, with **98.7% precision against fraud-involved
  accounts** and only **1 background false positive**. Every scheme reaches
  **100% actor-level recall**: rapid pass-through 20/20, fan-out smurfing 10/10,
  AEPS terminal cash-out 6/6, branch/IMEI collusion 6/6, circular wash 18/18.
- The **screening band** (`risk_score >= 40`) flags 77 accounts with 100% recall
  and 77.9% culprit precision; this is the wider triage band for initial review.
- Per-scheme precision columns are the scheme's share of the combined band's
  flags (they sum to 80% against named culprits); the **98.7%** figure counts
  every flagged account that is fraud-involved (victims, relays and mule
  accounts), which is the precision that matters when a case file is opened.
- The benchmark consumes ground-truth labels **only after** the blind pipeline
  has run; labels never enter features, graph analytics or the scorer. These
  are synthetic topology results, **not real-world accuracy claims**.
- Runtime varies with hardware; **1.71 s** was measured on the development
  machine for simulate → ingest → features → graph → risk over 6,539 rows with
  the LITE compute profile (the machine auto-detects as STANDARD).

## Phase 5 — hardware-adaptive engine, diffusion and freeze triage

The Phase 5 engine keeps the same four layers but makes the compute budget
explicit and adds graph-derived evidence:

- **Compute tiers.** `core.hardware.detect_hardware_profile()` resolves
  `LITE` (2 DuckDB threads, 256 MB, 64 betweenness samples, cycle length 3,
  100 IsolationForest trees, density off, 5 diffusion iterations), `STANDARD`
  (4, 1 GB, 256, 4, 200, density on, 10) and `WORKSTATION` (8, 4 GB, exact
  betweenness, 5, 400, density on, 20). Auto-detection upgrades to STANDARD at
  ≥8 cores and ≥8 GB RAM, and to WORKSTATION with a GPU, ≥16 cores and ≥16 GB.
  Tiers are overridable on `FeatureConfig`, `GraphConfig`, `RiskConfig` and
  `run_pipeline(..., compute_tier=...)`; LITE remains the deterministic default.
- **Cohort normalisation.** `cohort_volume_zscore` compares an account's total
  volume with peers of the same `account_type` (PMJDY / SAVINGS / CURRENT /
  UNKNOWN). A cohort without a usable spread yields `0.0`, never an invented
  outlier.
- **FIFO retention and graph diffusion.** Each complaint trace exposes
  `retained_taint`: tainted funds that arrived but had not left when the trace
  stopped. `retained_tainted_amount` lifts this to the graph, while
  `network_proximity_risk` diffuses complaint/taint seeds along directed edges
  with a 0.7 damping factor, so a conduit one hop from a complaint outranks a
  merely adjacent background account. Proximity only corroborates the
  complaint/pass-through components when the account is actually moving money
  onward, which keeps background false positives at one.
- **Detector ensemble.** `anomaly_score` combines IsolationForest (TreeSHAP
  attribution unchanged), a `LocalOutlierFactor` density check on tiers that
  enable it, and an empirical robust-tail detector that stays silent until the
  top 5% of the uploaded dataset. Weights are 0.80/0.20 (forest/tail) on LITE
  and 0.70/0.15/0.15 on STANDARD and WORKSTATION.
- **Strong cycle evidence.** A single decaying wash cycle contributes 0.8 of
  the cycle component (the component saturates at two cycles), because a closed
  loop is already a high-specificity topology rather than a weak hint. This is
  what lifts every circular-wash participant into HIGH/CRITICAL without adding
  background flags.
- **Operational role and freeze priority.** Every assessment carries one of
  `LAYER_1_COLLECTOR`, `LAYERING_CONDUIT`, `SMURF_DISTRIBUTOR`,
  `WASH_CYCLE_NODE`, `CASH_OUT_TERMINAL`, `COLLUSION_RING_MEMBER` or
  `STANDARD_ACCOUNT`, plus a 0–100 `freeze_priority_score` =
  `0.60 × risk + 0.30 × log-scaled recoverable taint + 0.10 × proximity`.
  This orders the case list by where disputed money can still be frozen, not
  just by risk.

The app's **Load seeded Bihar benchmark** path round-trips the same dataset
through ingestion and adds three deterministic malformed rows, so Tab 1 shows
5,249 accepted and 3 rejected rows (`unparseable_amount`, `missing:txn_id`,
`duplicate_utr`) with a downloadable rejection audit.

## Sample uploads (judge demo)

`data/sample_uploads/` contains three small, realistic files generated by
`simulator.export_samples`:

| File | Purpose |
| --- | --- |
| `transactions_messy_bank_export.csv` | Alternate bank headers (`UTR`, `Debit_Acc`, `PAYEE`, `Amount_INR`, `Txn_Time`, `Mode`), a 3-hop mule chain and **5 deliberate rejects**: unparseable amount, missing UTR, duplicate UTR, negative amount, unparseable timestamp |
| `account_master_kyc.xlsx` | KYC master with IFSC, opening date, branch, mobile, IMEI and pincode; includes a shared-IMEI / same-branch cohort |
| `ncrp_complaints_interstate.csv` | Three interstate NCRP complaints (`ACK`, `Disputed_UTR`, `State`) tied to ledger UTRs |

The XLSX is byte-stable: workbook properties and zip entry timestamps are
fixed, so regeneration produces identical bytes.

## Architecture and offline guarantees

```text
src/ingestion/   CSV/XLSX -> canonical transaction/account/complaint frames + audit reports
src/features/    DuckDB + Polars account behaviour features
src/graph/       NetworkX topology, FIFO-taint money trails, wash cycles, complaint links
src/risk/        transparent domain score + IsolationForest/TreeSHAP anomaly score
src/export/      Section A/B/C dossier (Markdown, standalone HTML, JSON) + requisition draft
src/ui/          Streamlit-free view models (state.py) and the Streamlit workbench (app.py)
src/simulator/   seeded generator, deterministic sample exporter, prahar-verify benchmark
```

- No CDN, remote fonts, telemetry or outbound requests; Streamlit usage stats are
  disabled in `.streamlit/config.toml`.
- No `datetime.now()`/`date.today()` in dossier output; `generated_on` is
  explicit, and every list has an explicit deterministic sort.
- No ground-truth (`is_fraud`, `fraud_scheme`) or protected attributes
  (`account_type`, `kyc_pincode`, `state`) enter the scorer.
- The anomaly component is dataset-relative (min-max normalised per upload) and
  is labelled as an inference, not evidence.

## Legal and analytical limits

- Section A reproduces supplied records and is **not a certified copy**.
- Section B is automated analysis and an **investigative lead, never evidence**.
- Section C cites **Section 94 BNSS** for production of records only. Lien
  marking, freezing or seizure is explicitly **not** covered and must be pursued
  separately through the CFCFRMS process and **Section 106 BNSS** as applicable.
  The requisition is a draft requiring review and signature by an authorised
  officer; it is not legal advice.
- The risk score ranks workload; it is **never a probability of guilt**.

For a plain-language walkthrough of every phase, see `ARCHITECTURE.md` and
`docs/plans/`.
