# PraharGraph — System Architecture & Engineering Specification

**Project:** Intelligent Mule Account Detection & Financial Fraud Investigation Platform (PS-01)  
**Context:** Bihar Police & C-DAC Patna Cyber Hackathon 2026  
**Execution Model:** 100% Offline, Local-First, Deterministic, Explainable AI (XAI)

---

## 1. Core Design Philosophy & Guardrails

PraharGraph is an offline-capable financial crime workbench designed for Law Enforcement Agencies (LEAs) and bank fraud analysts working with asynchronous data dumps (NCRP complaint exports, bank statement CSV/XLSX responses, and KYC metadata).

### Non-Negotiable Engineering & Domain Rules
1. **Zero Silent Data Drops:** Messy real-world files must never crash the pipeline or silently discard bad rows. Every rejected record is isolated in its original raw schema alongside a deterministic `rejection_reason` for evidentiary audit.
2. **Vectorized & In-Process Speed:** No `iter_rows()`, Python row loops, or Pandas memory bottlenecks. All tabular transformations use **Polars** expressions and **DuckDB** in-process SQL queries, targeting `< 2.5 seconds` end-to-end processing for 100,000 transactions on consumer hardware.
3. **Facts vs. Inferences Separation:**
   - **Verified Facts:** Raw ledger transfers, UTR matches, NCRP Ack numbers, shared KYC attributes (IMEI, mobile, branch).
   - **Risk Indicators (Inferences):** Pass-through velocity scores, graph centrality, and ML anomaly scores. Scores prioritize leads; they are never presented as standalone legal proof of guilt.
4. **Zero Geographic or Demographic Prejudice:** Location (state/pincode) and account type (`PMJDY`, `SAVINGS`, `CURRENT`) are used strictly for investigative filtering, baseline cohort normalization, and cross-jurisdiction correlation—never as raw additive penalties that bias scores against rural populations.
5. **Legal Precision:** Automated reports distinguish between **Section 94 BNSS** (summoning electronic records/KYC documents from Nodal Officers) and **CFCFRMS / Section 106 BNSS** (lien marking/seizure workflows). No fake live API integrations (e.g., unauthorized RBIH/I4C endpoints) are claimed or mocked.

---

## 2. Repository Structure & Ownership Boundaries

Each directory communicates strictly via typed objects defined in `src/core/contracts.py`. Teammates and coding agents must never import internal private helpers (`_*`) across module boundaries.

```text
prahar-graph/
├── ARCHITECTURE.md                  # Master system blueprint (this file)
├── pyproject.toml                   # Locked dependencies (uv, mypy --strict, pytest, ruff)
├── docs/
│   └── plans/                       # Obra Superpowers TDD implementation plans
│       ├── phase1-foundation.md
│       ├── phase2-risk-and-graph.md
│       └── phase3-ui-and-dossier.md
├── src/
│   ├── core/
│   │   └── contracts.py             # Single source of truth: Pydantic models & Polars schemas
│   ├── ingestion/
│   │   └── mapper.py                # RapidFuzz header mapper, Polars normalizer, audit reporter
│   ├── simulator/
│   │   └── generator.py             # Deterministic seeded background + 5 ground-truth fraud schemes
│   ├── features/                    # [Phase 2] DuckDB + Polars account behavioral feature extractor
│   │   └── account_features.py
│   ├── risk/                        # [Phase 2] Hybrid 0-100 Risk Engine + IsolationForest + TreeSHAP
│   │   └── scorer.py
│   ├── graph/                       # [Phase 2] NetworkX topology, taint tracker, cycle & cluster finder
│   │   └── analyzer.py
│   ├── export/                      # [Phase 3] Case Evidence Dossier & Sec 94 BNSS draft generator
│   │   └── dossier.py
│   └── ui/                          # [Phase 3] Streamlit workbench + interactive graph canvas
│       └── app.py
└── tests/
    ├── test_phase1.py               # Contracts, fuzzy ingestion, synthetic generator tests
    ├── test_phase2_risk.py          # Feature math, hybrid scorer, SHAP explanation tests
    └── test_phase2_graph.py         # Multi-hop trail, cycle, community, complaint link tests
```

---

## 3. Canonical Data Contracts (`src/core/contracts.py`)

All identifier fields (`account_no`, `txn_id`, `ifsc`, `imei`, `mobile_no`, `kyc_pincode`) are strictly `pl.String` to preserve leading zeros. Only monetary values are `pl.Float64`, and timestamps are naive ledger UTC `pl.Datetime("us")`.

### 3.1 Input Entity Schemas

#### `TRANSACTION_SCHEMA`
| Column | Polars Type | Mandatory | Validation / Normalization Rule |
| :--- | :--- | :---: | :--- |
| `txn_id` | `pl.String` | Yes | Unique UTR/Reference. Deduplicated (`keep="first"`, later flagged `duplicate_utr`). |
| `sender_account` | `pl.String` | Yes | Trimmed non-empty string. |
| `sender_ifsc` | `pl.String` | No | Uppercase; if present, must match `^[A-Z]{4}0[0-9A-Z]{6}$` else `invalid_ifsc`. |
| `receiver_account` | `pl.String` | Yes | Trimmed non-empty string. |
| `receiver_ifsc` | `pl.String` | No | Uppercase; if present, must match `^[A-Z]{4}0[0-9A-Z]{6}$` else `invalid_ifsc`. |
| `amount` | `pl.Float64` | Yes | Strip `₹`, `,`, spaces. Must be $\ge 0.0$ (`< 0` flagged `negative_amount`). |
| `timestamp` | `pl.Datetime("us")` | Yes | UNIX epoch (s/ms), ISO-8601, then `DD/MM/YYYY` before `MM/DD/YYYY`. |
| `payment_mode` | `pl.String` | Yes | Canonicalized to `Literal["UPI", "IMPS", "NEFT", "RTGS", "AEPS", "ATM", "CASH"]`. |
| `device_id` | `pl.String` | No | Device fingerprint, IMEI, or terminal ID (e.g., AePS/POS terminal). |
| `ip_address` | `pl.String` | No | Originating IPv4/IPv6 string. |

*(Note: Simulator output appends `is_fraud: pl.Boolean` and `fraud_scheme: pl.String` to form `LABELED_TRANSACTION_SCHEMA` for benchmark verification).*

#### `ACCOUNT_METADATA_SCHEMA`
| Column | Polars Type | Mandatory | Validation / Normalization Rule |
| :--- | :--- | :---: | :--- |
| `account_no` | `pl.String` | Yes | Primary key; deduplicated keep-first. |
| `ifsc` | `pl.String` | Yes | Matches `IFSC_PATTERN`. |
| `account_opening_date` | `pl.Date` | No | Parsed date of onboarding. |
| `branch_code` | `pl.String` | Yes | Bank branch identifier. |
| `mobile_no` | `pl.String` | No | 10–12 digits (`^\d{10,12}$`). |
| `imei` | `pl.String` | No | 15 digits (`^\d{15}$`). |
| `kyc_pincode` | `pl.String` | No | 6 digits (`^\d{6}$`). |
| `account_type` | `pl.String` | Yes | `Literal["SAVINGS", "PMJDY", "CURRENT"]`. |

#### `COMPLAINT_SCHEMA`
| Column | Polars Type | Mandatory | Validation / Normalization Rule |
| :--- | :--- | :---: | :--- |
| `ack_no` | `pl.String` | Yes | NCRP Case Acknowledgement ID (Case key, not a UTR). |
| `disputed_utr` | `pl.String` | No | Links complaint directly to `transactions.txn_id`. |
| `victim_account` | `pl.String` | Yes | Victim's source bank account. |
| `suspect_account` | `pl.String` | No | First-layer beneficiary account reported by victim. |
| `fraud_category` | `pl.String` | Yes | Category string (e.g., `AEPS_CASHOUT`, `INVESTMENT_SCAM`). |
| `reported_timestamp` | `pl.Datetime("us")` | Yes | Time complaint was registered. |
| `disputed_amount` | `pl.Float64` | Yes | Must be $\ge 0.0$. |
| `state` | `pl.String` | Yes | Reporting jurisdiction (e.g., `Bihar`, `Delhi`, `Tamil Nadu`). |

---

## 4. End-to-End Pipeline Architecture

```text
  [Raw CSV / XLSX Files]  (Transactions, Account KYC, NCRP Complaints)
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: Ingestion & Audit Gateway (`src/ingestion/mapper.py`)          │
│  • RapidFuzz token_sort_ratio header mapping (threshold >= 0.80)        │
│  • Vectorized Polars type coercion & priority-ordered row validation    │
│  • Outputs: Conforming pl.DataFrames + IngestionReport (rejected_rows)  │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│ LAYER 2: DuckDB Behavioral Extractor  │   │ LAYER 3: Graph & Money-Trail Engine   │
│ (`src/features/account_features.py`)  │   │ (`src/graph/analyzer.py`)             │
│  • Pass-through retention time (Δt)   │   │  • Directed MultiDiGraph + Weighted G │
│  • Inflow/Outflow drain ratio         │   │  • Bounded k-hop FIFO/Proportional    │
│  • Fan-in / Fan-out counterparty rate │   │    Taint Tracking                     │
│  • Sub-10k structuring (smurfing)     │   │  • Simple Cycle / Wash Loop detector  │
│  • Dormant-to-active awakening jump   │   │  • Louvain clusters & Bridge Centrality│
│  • Shared IMEI / Branch bulk-openings │   │  • Cross-State NCRP ACK correlation   │
└───────────────────┬───────────────────┘   └───────────────────┬───────────────────┘
                    └───────────────────────┬───────────────────┘
                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: Hybrid Explainable Risk Engine (`src/risk/scorer.py`)          │
│  • 70% Transparent Domain Typology Score (configurable thresholds)      │
│  • 30% Unsupervised ML Anomaly Score (IsolationForest + TreeSHAP)       │
│  • Deterministic Evidence Formatter -> Plain-English Investigator Leads │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: Investigator Workbench & Dossier (`src/ui/`, `src/export/`)    │
│  • Streamlit + Cytoscape/agraph interactive node-link canvas            │
│  • Schema mapping review, threshold sliders, timeline playback          │
│  • 1-Click Case Evidence Dossier & Sec 94 BNSS Requisition Draft        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Phase 2 Specifications (Feature Math, Graph & Hybrid XAI)

### 5.1 Behavioral & Metadata Feature Table (`src/features/account_features.py`)
Computed via in-process **DuckDB** SQL over the canonical Polars frames (`ACCOUNT_FEATURE_SCHEMA`), producing one row per distinct account seen in `transactions` or `accounts`:

1. **Rapid Pass-Through Ratio (`pass_through_ratio_15m`):**
   For every inflow event $i$ at $t_{\text{in}}$, sum outflows within $[t_{\text{in}}, t_{\text{in}} + \Delta t]$ (default configurable $\Delta t = 15\text{ min}$):
   $$R_{\text{drain}} = \min\left(1.0, \frac{\sum \text{Outflow}_{[t_{\text{in}}, t_{\text{in}} + \Delta t]}}{\sum \text{Inflow}}\right)$$
   High $R_{\text{drain}} \ge 0.85$ with median retention $\Delta t \le 15\text{ min}$ indicates a transient layering mule.
2. **Fan-In / Fan-Out Asymmetry & Smurfing (`smurf_split_count`, `unique_senders`, `unique_receivers`):**
   - *Distributor Smurfing:* Bulk inflow ($\ge ₹50,000$) followed within 60 minutes by $\ge 8$ distinct outflows where each outflow is in $[₹5,000, ₹9,999.99]$.
   - *Aggregator Mule:* High unique sender count ($\ge 10$) funneled into 1–2 large IMPS/RTGS/NEFT payouts.
3. **Dormant Awakening Velocity (`dormancy_burst_score`):**
   Days between `account_opening_date` (or previous historical transaction) and first high-volume burst. Flag accounts inactive for $\ge 180$ days that suddenly process $> ₹25,000$ in 24 hours.
4. **Shared Metadata & Branch Cohort (`shared_device_account_count`, `branch_cohort_48h_count`):**
   - Number of distinct accounts transacting from the same `device_id` or registered `imei`.
   - Number of accounts opened at the same `branch_code` within a 48-hour window that also appear in suspicious money flows.
5. **Terminal Cash-Out Concentration (`aeps_cashout_inflow_count`, `distinct_complaint_chains_terminated`):**
   Identifies merchant/terminal accounts receiving `AEPS`, `ATM`, or `CASH` exit legs from $\ge 2$ independent upstream chains.

### 5.2 Graph Analytics & Taint Tracking (`src/graph/analyzer.py`)
Built on `networkx.MultiDiGraph` (for exact transaction-edge timelines) and collapsed `networkx.DiGraph` (for centrality and community metrics):

* **Bounded $k$-Hop Money Trail (`trace_money_trail`):**
  Starting from a `disputed_utr` or `victim_account`, performs a temporal forward BFS ($t_{\text{next}} \ge t_{\text{prev}}$, max hops $k \in [1, 6]$). Computes **Proportional Taint** at each hop so investigators see how much of the victim's funds reached each downstream node.
* **Cycle & Wash Detection (`detect_wash_cycles`):**
  Finds closed directed cycles ($A \rightarrow B \rightarrow C \rightarrow A$, length $\le 5$) occurring within a 24-hour window where edge amounts decay monotonically within $[0\%, 15\%]$ per hop (commission deduction).
* **Cross-Jurisdiction Complaint Correlation (`correlate_complaints`):**
  Links multiple `ack_no` records across different `state` values when their forward money trails intersect at a common Layer-2/Layer-3 mule account, shared `device_id`, or terminal merchant.
* **Topology Metrics:**
  Computes `pagerank`, `betweenness_centrality` (bridge nodes linking separate clusters), and `louvain_community_id`.

### 5.3 Hybrid Explainable Risk Engine (`src/risk/scorer.py`)
Produces a `RiskAssessment` table (`RISK_REPORT_SCHEMA`) ranked $0–100$:

$$\text{Final Risk Score} = \text{clip}\left(0.70 \times S_{\text{domain}} + 0.30 \times S_{\text{anomaly}}, 0, 100\right)$$

* **$S_{\text{domain}} \in [0, 100]$:** Weighted sum of transparent, investigator-configurable sub-scores (Pass-Through, Smurfing, Cycle Participation, Shared Device/Branch Ring, Terminal Cash-Out, Direct/Indirect Complaint Link).
* **$S_{\text{anomaly}} \in [0, 100]$:** `sklearn.ensemble.IsolationForest(random_state=42)` fitted on the normalized numeric feature matrix, min-max scaled to $[0, 100]$.
* **Explainability (`top_reasons: list[str]`):**
  Combines exact rule triggers with **TreeSHAP** (`shap.TreeExplainer`) feature attributions on the Isolation Forest, translating the top 3 mathematical contributors into human-readable evidentiary bullets:
  > *Example:* `"[Pass-Through +34 pts] Drained 93.0% of ₹45,000 inflow within 4.0 minutes across 2 UPI hops (UTR: S1-005-1, S1-005-2)."`

---

## 6. Ground-Truth Synthetic Benchmarks (`src/simulator/generator.py`)

To verify detection recall/precision offline without real PII, `SyndicateDataSimulator(seed=42)` generates $5,000–20,000$ legitimate background transactions across $1,000+$ realistic numeric accounts (`29xxxxxxxx` retail, `55xxxxxxxx` merchant) and injects **5 deterministic fraud topologies**:

| Scheme Label | Injected Txns | Topological Signature | Target Detector Verified |
| :--- | :---: | :--- | :--- |
| `rapid_pass_through` | 80 | 20 episodes; dormant PMJDY/Savings hubs drain $\ge 90\%$ of inflow across 3 hops in $\le 15$ min. | Pass-Through $\Delta t$ + Dormancy Awakening |
| `fan_out_smurfing` | 130 | 10 collector accounts receive ₹95k IMPS and split into 12 UPI micro-transfers ($< ₹10,000$). | Fan-Out Smurfing Detector |
| `aeps_terminal_cash_out` | 9 | 3 interstate victim trails hop through mules and cash out via `AEPS` at one terminal (`555500000010`). | Cross-Complaint Trail & Terminal Clustering |
| `branch_collusion_shared_device` | 12 | 6 accounts opened within 48h at branch `BR042` sharing IMEI `351104510123456`. | Shared Metadata & Branch Cohort Ring |
| `circular_wash` | 18 | 6 closed 3-hop loops ($A \rightarrow B \rightarrow C \rightarrow A$) with $3\%$ commission decay per hop. | Directed Cycle / Wash Detector |

---

## 7. Workflow Rules for Developers & AI Coding Agents

1. **Verification Gate before Commit:** Every task must pass all three checks locally before committing:
   ```bash
   uv run pytest -q
   uv run mypy src
   uv run ruff check src tests
   ```
2. **Strict Type Safety:** `mypy --strict` is enforced across `src/`. Never use `# type: ignore` in `src/` (only permitted in negative unit tests testing invalid literal inputs).
3. **Determinism Invariant:** Never call unseeded RNG (`random.*`, `np.random.rand()`), `datetime.now()`, or rely on unordered Python `set` iteration when emitting frames or scores. Two runs on the same input must produce `.equals()` identical Polars DataFrames.
4. **UI Decoupling:** `src/ui/` and `src/export/` are read-only consumers of `src/ingestion/`, `src/features/`, `src/graph/`, and `src/risk/`. Never write data-processing or scoring logic inside Streamlit callbacks.