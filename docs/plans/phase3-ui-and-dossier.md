# Phase 3 — Investigator Workbench & Case Evidence Dossier

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the offline investigator workbench — a deterministic case dossier
generator with a Section 94 BNSS requisition draft, plus a Streamlit UI that
traces money, ranks mule accounts and shows the evidence behind every score.

**Architecture:** `src/export/dossier.py` is pure Python (Polars in, Markdown /
HTML / JSON bytes out) and never imports Streamlit. `src/ui/state.py` holds every
view-model builder as a pure, typed function (mypy `--strict`) so all data
preparation is unit-tested without a server. `src/ui/app.py` only renders those
view models — no scoring, no SQL, no graph maths in callbacks (ARCHITECTURE.md
§7.4). Phase 1/2 layers are consumed read-only.

**Tech Stack:** Python 3.12 · Polars 2.0 · Pydantic 2 · Streamlit 1.66 ·
`st-link-analysis` 0.4 (Cytoscape, frontend bundled locally) · Plotly 7.1 ·
Jinja2 3.1 (available, not required) · pytest 9.1 · mypy 2.4 (`strict`) · ruff.

**Spec:** `ARCHITECTURE.md` §1.5 (legal precision), §4 Layer 5 (workbench and
dossier), §7 (workflow rules); `docs/plans/phase2-risk-and-graph.md` (Phase 2
contracts and corrections). `ARCHITECTURE.md` is the binding blueprint and is
not modified by this phase.

## Global Constraints

- 100% offline at runtime: no CDN, no telemetry, no external API. HTML dossiers
  must contain no `http://` or `https://` reference.
- No new dependencies: `pyproject.toml` already pins every package used here.
- `uv run mypy src` stays clean under `strict = true`; no `# type: ignore` in `src/`.
- Determinism: two runs on the same input produce byte-identical dossier output
  and `.equals()`-identical frames. Never call `datetime.now()` / `date.today()`
  or unseeded RNG; never rely on set iteration for emitted order.
- No ground-truth leakage: `is_fraud`, `fraud_scheme` never reach features,
  scoring, views or the dossier. The workbench loads canonical columns only.
- Verified test gate after every task: `uv run pytest -q`,
  `uv run mypy src`, `uv run ruff check src tests`.
- Preserve user files: do not modify `ARCHITECTURE.md`, `transcript.txt`,
  `README.md`.

## Review Focus

Six failure modes the spec implies but no single task's happy path exercises.
Each has a named test in the owning task.

1. **Markup injection** — account numbers, IFSC codes, bank names and complaint
   text are attacker-influenced data. They must be escaped in both HTML and
   Markdown, and must never break the document structure (Task 5).
2. **Absent optional fields** — no IFSC, no IMEI, no branch, no
   `disputed_utr`, no `suspect_account`, no account master row. Each must
   degrade to an explicit "not recorded" statement, never crash and never
   fabricate a value (Tasks 2, 3, 4).
3. **Empty or unknown selection** — a dossier for an account with no legs, or
   for an ACK that does not exist, must produce a valid document with empty
   sections rather than an exception (Tasks 2, 7).
4. **Unbounded ledgers** — a mule with thousands of legs must stay bounded and
   *say* it is bounded; silent truncation is forbidden (Task 2).
5. **Over-claiming** — the dossier must never present a score, a SHAP
   attribution or a FIFO taint estimate as verified evidence, must cite only
   Section 94 BNSS in the requisition, and must keep Section 106 BNSS /
   CFCFRMS lien-marking as an explicitly separate workflow (Tasks 3, 4, 5).
6. **Rerun recomputation** — Streamlit re-executes the script on every widget
   event. The pipeline (IsolationForest, TreeSHAP, DuckDB, NetworkX) must run
   only when the dataset is loaded and its bundle must be cached in
   `st.session_state`; and a loaded benchmark must carry real `IngestionReport`s
   rather than empty ones (Task 9, tested via `AppTest`).

---

## 0. Verified pre-flight findings

| Question | Verdict |
| :--- | :--- |
| Streamlit / Plotly / link-analysis versions | `streamlit 1.66.0`, `plotly 7.1.0`, `st-link-analysis 0.4.0`, `jinja2 3.1.6` — all in `pyproject.toml`, all in `.venv`. |
| Is the graph canvas offline? | **Yes.** `st_link_analysis` ships `frontend/build/{cytoscape,index,apache-arrow,...}.js` (~900 KB) and declares the component by local path. No remote URL is fetched. |
| Does Streamlit fetch anything at runtime? | Only optional Google theme fonts and usage stats. Both removed by `.streamlit/config.toml` (`[theme] font = "sans serif"`, `[browser] gatherUsageStats = false`). |
| Can the UI be tested without a server? | **Yes.** `streamlit.testing.v1.AppTest.from_file(...).run()` executes the script headlessly; verified `at.exception`, `at.title`, `at.tabs`, `at.get("download_button")` all work on this version. |
| `st_link_analysis` API | `st_link_analysis(elements: dict, layout: str\|dict, node_styles: list[NodeStyle], edge_styles: list[EdgeStyle], height: int, key: str\|None)`. `elements = {"nodes": [{"data": {"id","label",...}}], "edges": [{"data": {"id","source","target","label",...}}]}`. Styling is by node/edge `label`. |
| Domain sub-scores available to the UI? | **Not yet** — `_evaluate()` computes `_pass_through`… `_complaint` but drops them when building `RISK_REPORT_SCHEMA`. Task 1 exposes them on the typed `RiskAssessment` instead of changing the Phase 2 frame. |
| `IngestionReport.rejected_rows` shape | Source columns (pristine) + `rejection_reason` (String). `write_csv()` gives header + rows; empty report gives a schema-only frame. |
| Complaint link evidence | `ComplaintLink(ack_no_a, ack_no_b, state_a, state_b, shared_accounts, shared_device_ids, shared_terminal_accounts, link_basis)`; states include `Bihar` (simulator `COMPLAINT_STATES`). |
| Money trail ordering | `MoneyTrail.legs` arrive in chronological sweep order; each leg carries `hop`, `txn_id`, `payment_mode`, `amount`, `taint_amount`. Sort explicitly in views. |
| Does the benchmark survive an ingester round-trip? | **Yes.** `frame.write_csv()` → `DatasetIngester` reproduces `transactions` / `accounts` / `complaints` exactly (`.equals()` verified), so the demo dataset can carry real `IngestionReport`s. |
| Can malformed audit rows be appended safely? | **Yes.** Appending three hostile CSV legs (bad amount, missing UTR, duplicate UTR) yields exactly `{unparseable_amount: 1, missing:txn_id: 1, duplicate_utr: 1}` and leaves the accepted frame byte-identical to the simulator output. Rejection keys use the `missing:<column>` form. |
| Does the Cytoscape component run under `AppTest`? | **Yes.** `st_link_analysis` with six `hop N` edge styles renders headlessly with no exception (only a `labeled` deprecation warning from the library itself). |
| Simulator bounds | `SyndicateDataSimulator.generate(background_transactions=...)` requires `[5000, 20000]`; the UI must not ask for less. |

---

## 1. File structure

| File | Responsibility |
| :--- | :--- |
| `src/core/contracts.py` *(modify)* | Dossier models, `BNSS_SECTION_94` / `BNSS_SECTION_106`, `DOSSIER_DISCLAIMER`, `REQUISITION_DOCUMENTS`, domain keys/labels; `RiskAssessment.domain_components` + `.top_archetype`. |
| `src/core/coerce.py` *(new)* | `as_str / as_float / as_int / as_datetime / as_date` — strict-safe narrowing of `Mapping[str, object]` rows. |
| `src/risk/scorer.py` *(modify)* | `ARCHETYPE_LABELS`, `score_with_assessments()`, populate the new assessment fields. Additive only; Phase 2 frame unchanged. |
| `src/export/dossier.py` *(new)* | `DossierAccount`, `DossierConfig`, `InvestigationDossierGenerator` (facts / indicators / requisition / 3 renderers), `rejected_rows_csv`, `format_inr`, `bank_name_for`. Pure Python, no Streamlit. |
| `src/ui/state.py` *(new)* | `run_pipeline`, ingestion view, leaderboard, account detail, graph elements, cycle / collusion / terminal / correlation views, dossier assembly for account / cluster / complaint scopes. Pure, no Streamlit. |
| `src/ui/app.py` *(new)* | Streamlit rendering only: control bar + 4 tabs. |
| `.streamlit/config.toml` *(new)* | Offline hardening (no fonts CDN, no usage stats) + theme. |
| `tests/test_phase3.py` *(new)* | All Phase 3 unit tests, including the headless `AppTest` smoke tests. |

Commit sequence (one per task, each green in isolation): `phase3(0)` this plan
document, `phase3(1)` contracts + scorer extension, `phase3(2)` dossier Section A
(+ `core/coerce.py`), `phase3(3)` Section B, `phase3(4)` Section C requisition,
`phase3(5)` renderers, `phase3(6)` pipeline + ingestion + leaderboard views,
`phase3(7)` account detail + dossier assembly, `phase3(8)` graph/cycle/collusion/
terminal/correlation views, `phase3(9)` Streamlit app + offline config + headless
`AppTest`.

---

## 2. Contracts additions (exact shapes)

```python
# --- Phase 3: case dossier ---------------------------------------------------
DossierFactCategory = Literal["ACCOUNT", "TRANSACTION", "COMPLAINT", "SHARED_KYC"]
DossierMethod = Literal["DOMAIN_RULE", "ISOLATION_FOREST_SHAP", "FIFO_TAINT", "TOPOLOGY"]
DossierScope = Literal["ACCOUNT", "CLUSTER", "COMPLAINT"]
DomainKey = Literal["pass_through", "smurfing", "cycle", "shared_metadata",
                    "terminal_cashout", "complaint"]

DOMAIN_LABELS: dict[str, str] = {
    "pass_through": "Rapid pass-through",
    "smurfing": "Structuring / fan-out",
    "cycle": "Circular wash",
    "shared_metadata": "Shared KYC / branch ring",
    "terminal_cashout": "Terminal cash-out",
    "complaint": "Complaint linkage",
    "anomaly": "ML anomaly (IsolationForest)",
}
DOMAIN_KEYS: tuple[str, ...] = (
    "pass_through", "smurfing", "cycle", "shared_metadata",
    "terminal_cashout", "complaint",
)

BNSS_SECTION_94: str = (
    "Section 94 of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)"
)
BNSS_SECTION_106: str = "Section 106 of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)"

DOSSIER_DISCLAIMER: str = (
    "Section A reproduces records supplied by the source systems and is not a "
    "certified copy. Section B is analytical output of an automated model: it "
    "is an investigative lead, not evidence and not a probability of guilt. "
    "Section C is a draft requisition for review and signature by the competent "
    "investigating officer; it is not legal advice and does not by itself "
    "authorise lien marking, freezing or seizure."
)

REQUISITION_DOCUMENTS: tuple[str, ...] = (
    "Certified copy of the account opening form (AOF) and KYC documents",
    "Customer master details, including registered mobile number, e-mail, "
    "PAN (if recorded) and introducer details",
    "Certified account statement covering every transaction reference listed "
    "against this bank",
    "IP address, device identifier/IMEI and login/access logs for the listed "
    "transaction references",
    "Branch and opening-official details for the listed accounts",
    "Current account status (active/frozen/closed) and any lien already marked",
)


class DossierFact(BaseModel):
    """Section A: one verifiable record, tied to its source."""

    category: DossierFactCategory
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: Literal["ledger", "account_master", "ncrp", "generator"]
    account_no: str | None = None
    txn_id: str | None = None
    ack_no: str | None = None
    timestamp: datetime | None = None
    amount: float | None = None


class DossierSubScore(BaseModel):
    domain: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: float = Field(ge=0.0, le=100.0)
    weight: float = Field(ge=0.0)
    points: float = Field(ge=0.0)


class DossierIndicator(BaseModel):
    """Section B: one analytical measurement. Never evidence."""

    indicator: str = Field(min_length=1)
    method: DossierMethod
    value: float
    unit: Literal["score", "ratio", "count", "rupees"] = "score"
    threshold: float | None = None
    points: float = 0.0
    explanation: str = Field(min_length=1)
    txn_ids: list[str] = Field(default_factory=list)


class RequisitionRequest(BaseModel):
    bank_name: str = Field(min_length=1)
    ifsc_prefix: str = Field(min_length=4, max_length=12)
    accounts: list[str] = Field(default_factory=list)
    utrs: list[str] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)


class DossierRequisition(BaseModel):
    statute_reference: str = Field(min_length=1)
    addressee: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    case_reference: str = Field(min_length=1)
    paragraphs: list[str] = Field(default_factory=list)
    bank_requests: list[RequisitionRequest] = Field(default_factory=list)
    legal_note: str = Field(min_length=1)


class DossierRiskSummary(BaseModel):
    account_no: str
    risk_score: float = Field(ge=0.0, le=100.0)
    domain_score: float = Field(ge=0.0, le=100.0)
    anomaly_score: float = Field(ge=0.0, le=100.0)
    risk_band: str
    top_archetype: str
    sub_scores: list[DossierSubScore] = Field(default_factory=list)


class InvestigationDossier(BaseModel):
    case_reference: str = Field(min_length=1)
    scope: DossierScope
    subject_accounts: list[str] = Field(default_factory=list)
    facts: list[DossierFact] = Field(default_factory=list)
    indicators: list[DossierIndicator] = Field(default_factory=list)
    risk_summary: DossierRiskSummary | None = None
    requisition: DossierRequisition
    disclaimer: str = DOSSIER_DISCLAIMER
```

`RiskAssessment` gains two defaulted fields (backward compatible):

```python
    domain_components: dict[str, float] = Field(default_factory=dict)  # 0..100 each
    top_archetype: str = ""
```

---

## 3. Implementation tasks (TDD, in order)

### Task 1: Expose domain sub-scores and archetypes from the scorer

**Files:**
- Modify: `src/core/contracts.py` (RiskAssessment fields + the Phase 3 models above)
- Modify: `src/risk/scorer.py`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Consumes: existing `RiskScorer._evaluate(features, graph) -> tuple[pl.DataFrame, list[RiskAssessment]]`.
- Produces:
  - `RiskScorer.score_with_assessments(features, graph=None) -> tuple[pl.DataFrame, list[RiskAssessment]]`
  - `RiskAssessment.domain_components: dict[str, float]` (keys are `DOMAIN_KEYS`, values 0–100)
  - `RiskAssessment.top_archetype: str`
  - `scorer.ARCHETYPE_LABELS: dict[str, str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase3.py
from __future__ import annotations

from datetime import datetime

import polars as pl

from core.contracts import ACCOUNT_FEATURE_SCHEMA, DOMAIN_KEYS, GRAPH_INDICATOR_SCHEMA
from risk.scorer import RiskScorer

BASE = datetime(2024, 6, 1, 10, 0)


def _feature_row(account_no: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in ACCOUNT_FEATURE_SCHEMA}
    row.update(
        account_no=account_no, txn_count=4, inflow_count=2, outflow_count=2,
        inflow_total=100_000.0, outflow_total=0.0, net_flow=0.0,
        unique_senders=1, unique_receivers=2, fan_in_fan_out_asymmetry=1.0,
        pass_through_ratio_15m=0.95, median_retention_minutes=3.0,
        smurf_bulk_inflow_count=1, smurf_split_count=10, large_payout_count=0,
        burst_24h_max_total=0.0, dormancy_days_before_burst=200.0,
        dormancy_burst_score=1.0, shared_device_account_count=0,
        branch_cohort_48h_count=0, aeps_cashout_inflow_count=0,
        direct_complaint_count=0, pass_through_utrs=[], smurf_utrs=[],
        first_txn_timestamp=BASE, last_txn_timestamp=BASE,
    )
    row.update(overrides)
    return row


def _graph_row(account_no: str) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in GRAPH_INDICATOR_SCHEMA}
    row.update(
        account_no=account_no, pagerank=0.1, betweenness_centrality=0.0,
        louvain_community_id=1, cycle_count=0, cycle_max_amount=0.0,
        cycle_txn_ids=[], complaint_link_count=0, complaint_ack_nos=[],
        trail_taint_max=0.0, trail_hops_max=0, terminal_chain_count=0,
    )
    return row


def test_score_with_assessments_returns_the_same_frame_as_score() -> None:
    features = pl.DataFrame(
        [_feature_row(f"A{i:02d}") for i in range(12)],
        schema=ACCOUNT_FEATURE_SCHEMA,
    )
    graph = pl.DataFrame([_graph_row(f"A{i:02d}") for i in range(12)],
                         schema=GRAPH_INDICATOR_SCHEMA)
    scorer = RiskScorer()
    frame, assessments = scorer.score_with_assessments(features, graph)
    assert frame.equals(scorer.score(features, graph))
    assert len(assessments) == 12


def test_assessment_exposes_six_domain_components_and_an_archetype() -> None:
    features = pl.DataFrame([_feature_row("MULE")], schema=ACCOUNT_FEATURE_SCHEMA)
    graph = pl.DataFrame([_graph_row("MULE")], schema=GRAPH_INDICATOR_SCHEMA)
    _, assessments = RiskScorer().score_with_assessments(features, graph)
    assessment = assessments[0]
    assert set(assessment.domain_components) == set(DOMAIN_KEYS)
    assert all(0.0 <= value <= 100.0 for value in assessment.domain_components.values())
    # 0.95 pass-through ratio with 3-minute retention saturates the rule
    assert assessment.domain_components["pass_through"] == 100.0
    assert assessment.top_archetype == "Rapid pass-through mule"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase3.py -q`
Expected: FAIL — `ImportError: cannot import name 'DOMAIN_KEYS'` (then
`AttributeError: 'RiskScorer' object has no attribute 'score_with_assessments'`).

- [ ] **Step 3: Write minimal implementation**

In `src/core/contracts.py`, add the Task-1 model fields and the Phase-3 block
from §2 (`DOMAIN_KEYS`, `DOMAIN_LABELS`, `BNSS_*`, `DOSSIER_DISCLAIMER`,
`REQUISITION_DOCUMENTS`, `DossierFact`, `DossierSubScore`, `DossierIndicator`,
`RequisitionRequest`, `DossierRequisition`, `DossierRiskSummary`,
`InvestigationDossier`).

In `src/risk/scorer.py`, add the labels next to the existing module constants
(the import line already ends with `from core.contracts import ...` — append
`DOMAIN_KEYS` to it):

```python
from core.contracts import DOMAIN_KEYS  # added to the existing contracts import

ARCHETYPE_LABELS: dict[str, str] = {
    "pass_through": "Rapid pass-through mule",
    "smurfing": "Structuring / fan-out mule",
    "cycle": "Circular wash participant",
    "shared_metadata": "Shared-KYC collusion ring",
    "terminal_cashout": "Terminal cash-out point",
    "complaint": "Complaint-linked mule",
}
ARCHETYPE_NONE: str = "No domain indicator"
```

Inside `_evaluate`, replace the per-row `RiskAssessment(...)` construction (the
loop over `frame.to_dicts()`) with the component computation below. No helper
indirection is needed:

```python
        component_columns = {
            "pass_through": ("_pass_through", cfg.pass_through_weight),
            "smurfing": ("_smurfing", cfg.smurfing_weight),
            "cycle": ("_cycle", cfg.cycle_weight),
            "shared_metadata": ("_shared", cfg.shared_metadata_weight),
            "terminal_cashout": ("_terminal", cfg.terminal_cashout_weight),
            "complaint": ("_complaint", cfg.complaint_weight),
        }
        for index, row in enumerate(rows):
            components = {
                key: round(float(row[column]) * 100.0, 4)
                for key, (column, _) in component_columns.items()
            }
            points = {
                key: component_columns[key][1] * components[key] / 100.0
                for key in DOMAIN_KEYS
            }
            best = max(DOMAIN_KEYS, key=lambda key: (points[key], -DOMAIN_KEYS.index(key)))
            top = ARCHETYPE_LABELS[best] if points[best] > 0 else ARCHETYPE_NONE
            assessments.append(
                RiskAssessment(
                    account_no=str(row["account_no"]),
                    risk_score=float(row["_risk_score"]),
                    domain_score=float(row["_domain_score"]),
                    anomaly_score=float(row["_anomaly_score"]),
                    risk_band=self._band(float(row["_risk_score"])),
                    explanations=explanations,
                    domain_components=components,
                    top_archetype=top,
                )
            )
```

and the public method:

```python
    def score_with_assessments(
        self,
        features: pl.DataFrame,
        graph: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, list[RiskAssessment]]:
        """Risk report and typed assessments from one fitted model."""
        return self._evaluate(features, graph)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_phase3.py -q` → PASS.
Then the full gate: `uv run pytest -q`, `uv run mypy src`,
`uv run ruff check src tests` → all green, Phase 2's 187 tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/core/contracts.py src/risk/scorer.py tests/test_phase3.py
git commit -m "phase3(1): expose domain sub-scores and archetypes for the workbench"
```

---

### Task 2: Dossier Section A — verified facts, bounded ledger, rejected-row CSV

**Files:**
- Create: `src/core/coerce.py`
- Create: `src/export/dossier.py`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Consumes: `core.contracts` models; `core.contracts.IngestionReport`.
- Produces:
  - `core.coerce.as_str(value: object, default: str = "") -> str`
  - `core.coerce.as_float(value: object, default: float = 0.0) -> float`
  - `core.coerce.as_int(value: object, default: int = 0) -> int`
  - `core.coerce.as_datetime(value: object) -> datetime | None`
  - `core.coerce.as_date(value: object) -> date | None`
  - `export.dossier.format_inr(amount: float) -> str`
  - `export.dossier.rejected_rows_csv(report: IngestionReport) -> bytes`
  - `export.dossier.DossierAccount` (frozen dataclass), `DossierConfig`,
    `InvestigationDossierGenerator`

`DossierAccount` (frozen dataclass) and `DossierConfig` (Pydantic model) are
defined in full in Step 3. `DossierAccount` carries `account_no`, optional
`metadata` / `risk` / `feature_row` / `graph_row` mappings, the account's
`transactions` and `complaints` rows, an optional `trail`, `cycles`,
`imei_peers`, `branch_peers`, and `ledger_txn_ids: frozenset[str] | None`
(`None` = every leg).

- [ ] **Step 1: Write the failing test**

```python
# extend the test imports
from datetime import date, timedelta

from core.contracts import (
    ACCOUNT_METADATA_SCHEMA, COMPLAINT_SCHEMA, REQUISITION_DOCUMENTS,
    TRANSACTION_SCHEMA, IngestionReport,
)
from export.dossier import (
    DossierAccount, DossierConfig, InvestigationDossierGenerator,
    format_inr, rejected_rows_csv,
)


def _account_row(account_no: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "account_no": account_no, "ifsc": "UTIB0000123",
        "account_opening_date": date(2024, 5, 1), "branch_code": "BR042",
        "mobile_no": "9876543210", "imei": "351104510123456",
        "kyc_pincode": "800001", "account_type": "SAVINGS",
    }
    row.update(overrides)
    return row


def _txn(txn_id: str, sender: str, receiver: str, amount: float,
         minutes: int, mode: str = "UPI") -> dict[str, object]:
    return {
        "txn_id": txn_id, "sender_account": sender, "sender_ifsc": "UTIB0000123",
        "receiver_account": receiver, "receiver_ifsc": "SBIN0000421",
        "amount": amount, "timestamp": BASE + timedelta(minutes=minutes),
        "payment_mode": mode, "device_id": "DEV-1", "ip_address": "10.0.0.1",
    }


def _generator(account_no: str = "MULE", **kwargs: object) -> InvestigationDossierGenerator:
    account = DossierAccount(
        account_no=account_no,
        metadata=_account_row(account_no),
        transactions=(
            _txn("U1", "VICTIM", account_no, 100.0, 0),
            _txn("U2", account_no, "MULE2", 90.0, 5),
            _txn("U3", account_no, "MULE2", 10.0, 6, mode="AEPS"),
        ),
        complaints=(
            {"ack_no": "ACK1", "victim_account": "VICTIM", "suspect_account": account_no,
             "fraud_category": "UPI_FRAUD", "reported_timestamp": BASE + timedelta(hours=2),
             "disputed_amount": 100.0, "disputed_utr": "U1", "state": "Bihar"},
        ),
        imei_peers=(_account_row("PEER1", branch_code="BR099"),),
        branch_peers=(_account_row("PEER2", imei="359999999999999",
                                   account_opening_date=date(2024, 5, 2)),),
    )
    return InvestigationDossierGenerator([account], DossierConfig(**kwargs))


def test_section_a_lists_ledger_account_ncrp_and_shared_kyc_facts() -> None:
    dossier = _generator().build()
    categories = [fact.category for fact in dossier.facts]
    assert categories == sorted(categories, key=["ACCOUNT", "TRANSACTION", "COMPLAINT", "SHARED_KYC"].index)
    assert {"ACCOUNT", "TRANSACTION", "COMPLAINT", "SHARED_KYC"} <= set(categories)

    utrs = [fact.txn_id for fact in dossier.facts if fact.category == "TRANSACTION"]
    assert utrs == ["U1", "U2", "U3"]                       # chronological
    inflow = next(fact for fact in dossier.facts if fact.txn_id == "U1")
    assert inflow.label == "Inflow from VICTIM (UTIB0000123)"
    assert "₹100.00" in inflow.value and "UTR U1" in inflow.value
    assert inflow.source == "ledger"

    ack = next(fact for fact in dossier.facts if fact.ack_no == "ACK1")
    assert ack.category == "COMPLAINT" and ack.source == "ncrp"
    assert "Bihar" in ack.value and "₹100.00" in ack.value

    imei = next(fact for fact in dossier.facts if fact.label == "Shared IMEI in account master")
    assert "351104510123456" in imei.value and "PEER1" in imei.value
    branch = next(fact for fact in dossier.facts if fact.label.startswith("Same-branch account"))
    assert "PEER2" in branch.value and "BR042" in branch.value


def test_missing_metadata_and_imei_are_stated_not_invented() -> None:
    account = DossierAccount(account_no="GHOST", transactions=())
    dossier = InvestigationDossierGenerator([account]).build()
    record = next(fact for fact in dossier.facts if fact.category == "ACCOUNT")
    assert record.value == "No account master record supplied for this account"
    assert record.source == "generator"
    assert not any(fact.category == "SHARED_KYC" for fact in dossier.facts)


def test_ledger_is_bounded_and_says_so() -> None:
    legs = tuple(_txn(f"T{i:03d}", "VICTIM", "MULE", 1.0, i) for i in range(12))
    account = DossierAccount(account_no="MULE", metadata=_account_row("MULE"), transactions=legs)
    dossier = InvestigationDossierGenerator(
        [account], DossierConfig(max_ledger_rows=5)
    ).build()
    ledger = [fact for fact in dossier.facts if fact.category == "TRANSACTION"]
    assert len(ledger) == 6                       # 5 legs + the truncation notice
    truncated = next(fact for fact in ledger if fact.label == "Ledger listing truncated")
    assert "5 of 12" in truncated.value
    assert truncated.source == "generator"


def test_unknown_account_selection_values_never_invent_optional_fields() -> None:
    account = DossierAccount(
        account_no="MULE",
        metadata=_account_row("MULE", ifsc=None, imei=None, account_opening_date=None),
        transactions=(_txn("U1", "VICTIM", "MULE", 50.0, 0),),
    )
    dossier = InvestigationDossierGenerator([account]).build()
    text = " ".join(fact.value for fact in dossier.facts)
    assert "IFSC not recorded" in text or "IFSC not supplied" in text
    assert "351104510123456" not in text


def test_rejected_rows_csv_is_deterministic_and_keeps_reasons() -> None:
    rejected = pl.DataFrame(
        {"utr": ["A", "B"], "amt": ["x", "10"], "rejection_reason": ["unparseable_amount", "negative_amount"]}
    )
    report = IngestionReport(rows_total=2, rows_accepted=0, rows_rejected=2,
                            rejected_rows=rejected)
    payload = rejected_rows_csv(report)
    assert payload == rejected_rows_csv(report)
    text = payload.decode("utf-8")
    assert text.splitlines()[0] == "utr,amt,rejection_reason"
    assert "unparseable_amount" in text


def test_format_inr_uses_indian_grouping() -> None:
    assert format_inr(0) == "₹0.00"
    assert format_inr(999) == "₹999.00"
    assert format_inr(1_000) == "₹1,000.00"
    assert format_inr(100_000) == "₹1,00,000.00"
    assert format_inr(12_345_678.9) == "₹1,23,45,678.90"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase3.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'export.dossier'`.

- [ ] **Step 3: Write minimal implementation**

`src/core/coerce.py`:

```python
"""Typed narrowing helpers for row mappings (`dict[str, object]`)."""

from __future__ import annotations

from datetime import date, datetime


def as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return None


def as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None
```

`src/export/dossier.py` (Section A part):

```python
"""Deterministic case-evidence dossier and Section 94 BNSS requisition draft."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping, Sequence

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from core.coerce import as_date, as_datetime, as_float, as_str
from core.contracts import (
    DOSSIER_DISCLAIMER, REQUISITION_DOCUMENTS, DossierFact, DossierIndicator,
    DossierRequisition, DossierRiskSummary, InvestigationDossier, IngestionReport,
    MoneyTrail, RiskAssessment, WashCycle,
)

DEFAULT_BANK_NAMES: dict[str, str] = {
    "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank", "PUNB": "Punjab National Bank", "KKBK": "Kotak Mahindra Bank",
    "BARB": "Bank of Baroda", "CNRB": "Canara Bank", "IOBA": "Indian Overseas Bank",
    "UCBA": "UCO Bank", "UBIN": "Union Bank of India", "IDIB": "Indian Bank",
    "PSIB": "Punjab & Sind Bank", "MAHB": "Bank of Maharashtra",
    "CBIN": "Central Bank of India", "BDBL": "Bandhan Bank",
    "FINO": "Fino Payments Bank", "IPPB": "India Post Payments Bank",
    "AIRP": "Airtel Payments Bank", "PYTM": "Paytm Payments Bank",
}

_FACT_ORDER: dict[str, int] = {
    "ACCOUNT": 0, "TRANSACTION": 1, "COMPLAINT": 2, "SHARED_KYC": 3,
}


def format_inr(amount: float) -> str:
    """Indian digit grouping with 2 dp, locale-independent and deterministic."""
    negative = amount < 0
    whole, _, paise = f"{abs(amount):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups) + "," + tail
    return f"{'-' if negative else ''}₹{whole}.{paise}"


def format_timestamp(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else "not recorded"


def bank_name_for(ifsc: str | None, table: Mapping[str, str]) -> str:
    prefix = ifsc[:4].upper() if ifsc else ""
    if not prefix:
        return "Bank not identified (no IFSC in supplied records)"
    return table.get(prefix, f"Bank with IFSC prefix {prefix} (name not in reference table)")


def rejected_rows_csv(report: IngestionReport) -> bytes:
    """Audit CSV of rejected rows, exactly as captured during ingestion."""
    return report.rejected_rows.write_csv().encode("utf-8")


@dataclass(frozen=True)
class DossierAccount:
    account_no: str
    metadata: Mapping[str, object] | None = None
    risk: RiskAssessment | None = None
    feature_row: Mapping[str, object] | None = None
    graph_row: Mapping[str, object] | None = None
    transactions: Sequence[Mapping[str, object]] = ()
    complaints: Sequence[Mapping[str, object]] = ()
    trail: MoneyTrail | None = None
    cycles: Sequence[WashCycle] = ()
    imei_peers: Sequence[Mapping[str, object]] = ()
    branch_peers: Sequence[Mapping[str, object]] = ()
    ledger_txn_ids: frozenset[str] | None = None   # None = every leg


_DEFAULT_DOMAIN_WEIGHTS: dict[str, float] = {
    "pass_through": 25.0,
    "smurfing": 20.0,
    "cycle": 20.0,
    "shared_metadata": 10.0,
    "terminal_cashout": 10.0,
    "complaint": 15.0,
    "anomaly": 30.0,
}


class DossierConfig(BaseModel):
    case_reference: str = "PG-CASE-0001"
    scope: DossierScope = "ACCOUNT"
    generated_on: date | None = None
    max_ledger_rows: int = Field(default=200, ge=1)
    deadline_days: int = Field(default=7, ge=1)
    bank_names: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_BANK_NAMES))
    requested_documents: tuple[str, ...] = REQUISITION_DOCUMENTS
    domain_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_DOMAIN_WEIGHTS)
    )


class InvestigationDossierGenerator:
    def __init__(self, accounts: Sequence[DossierAccount],
                 config: DossierConfig | None = None) -> None:
        self._accounts = tuple(sorted(accounts, key=lambda item: item.account_no))
        self._config = config or DossierConfig()
        self._built: InvestigationDossier | None = None

    # --- Section A --------------------------------------------------------
    def _facts_for(self, account: DossierAccount) -> list[DossierFact]:
        facts: list[DossierFact] = [self._account_fact(account)]
        facts.extend(self._ledger_facts(account))
        facts.extend(self._complaint_facts(account))
        facts.extend(self._shared_kyc_facts(account))
        return facts

    def _account_fact(self, account: DossierAccount) -> DossierFact:
        metadata = account.metadata
        if not metadata:
            return DossierFact(
                category="ACCOUNT", label="Account record",
                value="No account master record supplied for this account",
                source="generator", account_no=account.account_no,
            )
        ifsc = as_str(metadata.get("ifsc")) or "not recorded"
        opened = as_date(metadata.get("account_opening_date"))
        value = (
            f"A/C {account.account_no} · IFSC {ifsc} · "
            f"branch {as_str(metadata.get('branch_code')) or 'not recorded'} · "
            f"{as_str(metadata.get('account_type')) or 'type not recorded'} · "
            f"opened {opened.isoformat() if opened else 'not recorded'}"
        )
        return DossierFact(category="ACCOUNT", label="Account record", value=value,
                           source="account_master", account_no=account.account_no)

    def _ledger_facts(self, account: DossierAccount) -> list[DossierFact]:
        legs = [leg for leg in account.transactions
                if account.ledger_txn_ids is None
                or as_str(leg.get("txn_id")) in account.ledger_txn_ids]
        legs.sort(key=lambda leg: (as_datetime(leg.get("timestamp")) or datetime.min,
                                   as_str(leg.get("txn_id"))))
        limit = self._config.max_ledger_rows
        shown = legs[:limit]
        facts: list[DossierFact] = []
        for leg in shown:
            outbound = as_str(leg.get("sender_account")) == account.account_no
            counterparty = as_str(leg.get("receiver_account") if outbound
                                  else leg.get("sender_account"))
            ifsc_key = "receiver_ifsc" if outbound else "sender_ifsc"
            counterparty_ifsc = as_str(leg.get(ifsc_key)) or "IFSC not recorded"
            amount = as_float(leg.get("amount"))
            timestamp = as_datetime(leg.get("timestamp"))
            txn_id = as_str(leg.get("txn_id"))
            mode = as_str(leg.get("payment_mode")) or "mode not recorded"
            facts.append(DossierFact(
                category="TRANSACTION",
                label=(f"Outflow to {counterparty} ({counterparty_ifsc})" if outbound
                       else f"Inflow from {counterparty} ({counterparty_ifsc})"),
                value=(f"{format_inr(amount)} via {mode} on "
                       f"{format_timestamp(timestamp)} (UTR {txn_id})"),
                source="ledger", account_no=account.account_no, txn_id=txn_id,
                timestamp=timestamp, amount=amount,
            ))
        if len(legs) > limit:
            facts.append(DossierFact(
                category="TRANSACTION", label="Ledger listing truncated",
                value=(f"Showing the first {limit} of {len(legs)} legs for "
                       f"A/C {account.account_no}; the complete statement must be "
                       "obtained under the requisition."),
                source="generator", account_no=account.account_no,
            ))
        return facts
```

`_complaint_facts` and `_shared_kyc_facts` (full bodies; the value strings are
pinned by the tests above):

```python
    def _complaint_facts(self, account: DossierAccount) -> list[DossierFact]:
        rows = sorted(
            account.complaints,
            key=lambda row: as_str(row.get("ack_no")),
        )
        facts: list[DossierFact] = []
        for row in rows:
            ack_no = as_str(row.get("ack_no"))
            reported = as_datetime(row.get("reported_timestamp"))
            amount = as_float(row.get("disputed_amount"))
            facts.append(DossierFact(
                category="COMPLAINT",
                label=(
                    f"NCRP complaint {ack_no} "
                    f"({as_str(row.get('fraud_category')) or 'category not recorded'})"
                ),
                value=(
                    f"Victim {as_str(row.get('victim_account')) or 'not recorded'} · "
                    f"suspect {as_str(row.get('suspect_account')) or 'not named'} · "
                    f"disputed {format_inr(amount)} · "
                    f"UTR {as_str(row.get('disputed_utr')) or 'not supplied'} · "
                    f"reported {format_timestamp(reported)} · "
                    f"state {as_str(row.get('state')) or 'not recorded'}"
                ),
                source="ncrp",
                account_no=account.account_no,
                ack_no=ack_no or None,
                timestamp=reported,
                amount=amount,
            ))
        return facts

    def _shared_kyc_facts(self, account: DossierAccount) -> list[DossierFact]:
        facts: list[DossierFact] = []
        imei = as_str((account.metadata or {}).get("imei"))
        for peer in account.imei_peers:
            peer_no = as_str(peer.get("account_no"))
            if not imei or not peer_no:
                continue
            facts.append(DossierFact(
                category="SHARED_KYC",
                label="Shared IMEI in account master",
                value=(
                    f"IMEI {imei} is also recorded for A/C {peer_no} "
                    f"({as_str(peer.get('ifsc')) or 'IFSC not recorded'})"
                ),
                source="account_master",
                account_no=account.account_no,
            ))
        branch = as_str((account.metadata or {}).get("branch_code"))
        opened = as_date((account.metadata or {}).get("account_opening_date"))
        for peer in account.branch_peers:
            peer_no = as_str(peer.get("account_no"))
            peer_opened = as_date(peer.get("account_opening_date"))
            if not branch or opened is None or peer_opened is None or not peer_no:
                continue
            hours = abs((peer_opened - opened).total_seconds()) / 3600.0
            if hours > 48.0:
                continue
            facts.append(DossierFact(
                category="SHARED_KYC",
                label="Same-branch account opened within 48 h",
                value=(
                    f"A/C {peer_no} was opened at branch {branch} on "
                    f"{peer_opened.isoformat()}, {hours:.1f} h from this account"
                ),
                source="account_master",
                account_no=account.account_no,
            ))
        return facts
```

`build()` (Section A only at this stage; Sections B/C added in Tasks 3–4):

```python
    def build(self) -> InvestigationDossier:
        if self._built is not None:
            return self._built
        facts = [fact for account in self._accounts for fact in self._facts_for(account)]
        facts.sort(key=lambda item: (
            _FACT_ORDER[item.category],
            item.account_no or "",
            item.timestamp or datetime.min,
            item.txn_id or "",
            item.ack_no or "",
            item.label,
        ))
        dossier = InvestigationDossier(
            case_reference=self._config.case_reference,
            scope=self._config.scope,
            subject_accounts=[account.account_no for account in self._accounts],
            facts=facts,
            indicators=[],
            # Minimal valid requisition: Task 4 replaces this inline literal with
            # `requisition=self._requisition()` (bank-grouped, Section 94 BNSS).
            requisition=DossierRequisition(
                statute_reference=BNSS_SECTION_94,
                addressee="The Nodal Officer of each bank listed below",
                subject=(
                    f"Production of records under {BNSS_SECTION_94} — "
                    f"case {self._config.case_reference}"
                ),
                case_reference=self._config.case_reference,
                legal_note="Draft for review by the competent investigating officer.",
            ),
            disclaimer=DOSSIER_DISCLAIMER,
        )
        self._built = dossier
        return dossier
```

- [ ] **Step 4: Run test to verify it passes**

`uv run pytest tests/test_phase3.py -q` → PASS, then the three gates.

- [ ] **Step 5: Commit**

```bash
git add src/core/coerce.py src/export/dossier.py tests/test_phase3.py
git commit -m "phase3(2): build dossier section A from ledger, KYC and NCRP records"
```

---

### Task 3: Dossier Section B — risk summary, rule indicators, SHAP, FIFO taint

**Files:**
- Modify: `src/export/dossier.py`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Consumes: `DossierAccount.risk: RiskAssessment | None` (Task 1 fields),
  `DossierAccount.feature_row`, `.graph_row`, `.trail`, `.cycles`.
- Produces: `InvestigationDossierGenerator._indicators_for(account) -> list[DossierIndicator]`,
  `._risk_summary() -> DossierRiskSummary | None`, populated `dossier.indicators`
  and `dossier.risk_summary`.

- [ ] **Step 1: Write the failing test**

```python
# extend the test imports
from core.contracts import MoneyTrail, MoneyTrailLeg, RiskAssessment


def _scored_assessment(account_no: str = "MULE") -> RiskAssessment:
    """A real assessment from a fitted model (the anomaly line needs >= 10 rows)."""
    features = pl.DataFrame(
        [_feature_row(account_no)]
        + [
            _feature_row(f"B{i:02d}", pass_through_ratio_15m=0.05,
                         median_retention_minutes=600.0, smurf_bulk_inflow_count=0,
                         smurf_split_count=0, dormancy_days_before_burst=2.0)
            for i in range(11)
        ],
        schema=ACCOUNT_FEATURE_SCHEMA,
    )
    graph = pl.DataFrame(
        [_graph_row(account_no)] + [_graph_row(f"B{i:02d}") for i in range(11)],
        schema=GRAPH_INDICATOR_SCHEMA,
    )
    _, assessments = RiskScorer().score_with_assessments(features, graph)
    return next(item for item in assessments if item.account_no == account_no)


def test_section_b_separates_indicators_from_facts() -> None:
    account = DossierAccount(
        account_no="MULE",
        metadata=_account_row("MULE"),
        risk=_scored_assessment("MULE"),
        feature_row=_feature_row("MULE"),
        graph_row=_graph_row("MULE"),
        transactions=(_txn("U1", "VICTIM", "MULE", 100.0, 0),),
        trail=MoneyTrail(
            origin_account="MULE", origin_amount=100.0, origin_txn_id="U1",
            max_hops=3, total_taint=100.0, account_taint={"MULE2": 90.0},
            legs=[MoneyTrailLeg(hop=1, txn_id="U2", sender_account="MULE",
                                receiver_account="MULE2", amount=90.0,
                                timestamp=BASE, payment_mode="UPI", taint_amount=90.0)],
        ),
    )
    generator = InvestigationDossierGenerator([account])
    dossier = generator.build()

    methods = {indicator.method for indicator in dossier.indicators}
    assert "DOMAIN_RULE" in methods and "FIFO_TAINT" in methods
    assert "ISOLATION_FOREST_SHAP" in methods
    assert all(indicator.explanation for indicator in dossier.indicators)
    # nothing analytical leaks into the evidentiary section
    fact_text = " ".join(f"{f.label} {f.value}" for f in dossier.facts).lower()
    assert "shap" not in fact_text
    assert "taint" not in fact_text
    assert "risk" not in fact_text

    assert dossier.risk_summary is not None
    assert len(dossier.risk_summary.sub_scores) == 7          # 6 domains + anomaly
    assert {score.domain for score in dossier.risk_summary.sub_scores} >= set(DOMAIN_KEYS)


def test_fifo_taint_indicator_states_the_assumption() -> None:
    account = DossierAccount(
        account_no="MULE", metadata=_account_row("MULE"),
        trail=MoneyTrail(
            origin_account="MULE", origin_amount=100.0, origin_txn_id="U1",
            max_hops=3, total_taint=90.0, account_taint={"MULE2": 90.0},
            legs=[MoneyTrailLeg(hop=1, txn_id="U2", sender_account="MULE",
                                receiver_account="MULE2", amount=90.0,
                                timestamp=BASE, payment_mode="UPI", taint_amount=90.0)],
        ),
    )
    dossier = InvestigationDossierGenerator([account]).build()
    taint = next(i for i in dossier.indicators if i.method == "FIFO_TAINT")
    assert taint.unit == "rupees"
    assert taint.value == 90.0
    assert "FIFO" in taint.explanation or "first-in-first-out" in taint.explanation
    assert taint.txn_ids == ["U2"]


def test_account_without_risk_still_produces_an_empty_section_b() -> None:
    dossier = InvestigationDossierGenerator([DossierAccount(account_no="GHOST")]).build()
    assert dossier.indicators == []
    assert dossier.risk_summary is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase3.py -q`
Expected: FAIL — `TypeError: InvestigationDossier() got an unexpected keyword ...`
or `assert [] == [...]` on `dossier.indicators`.

- [ ] **Step 3: Write minimal implementation**

```python
# module level, next to `_FACT_ORDER`
from typing import Literal

IndicatorUnit = Literal["score", "ratio", "count", "rupees"]

_INDICATOR_UNITS: dict[str, IndicatorUnit] = {
    "pass_through": "ratio", "smurfing": "count", "cycle_participation": "count",
    "shared_metadata": "count", "terminal_cashout": "count",
    "complaint_relationship": "count", "isolation_forest_anomaly": "score",
}
_TOPOLOGY_FIELDS: tuple[tuple[str, str, IndicatorUnit], ...] = (
    ("pagerank", "PageRank centrality", "score"),
    ("betweenness_centrality", "Betweenness centrality (bridge role)", "score"),
    ("cycle_count", "Closed money cycles involving this account", "count"),
    ("terminal_chain_count", "Independent chains terminating here", "count"),
    ("complaint_link_count", "NCRP cases linked through shared evidence", "count"),
)

# ... then, inside `InvestigationDossierGenerator`:
    def _indicators_for(self, account: DossierAccount) -> list[DossierIndicator]:
        indicators: list[DossierIndicator] = []
        risk = account.risk
        if risk is not None:
            for item in risk.explanations:
                indicators.append(DossierIndicator(
                    indicator=item.indicator,
                    method=("ISOLATION_FOREST_SHAP"
                            if item.indicator == "isolation_forest_anomaly"
                            else "DOMAIN_RULE"),
                    value=item.measured_value if item.measured_value is not None else 0.0,
                    unit=_INDICATOR_UNITS.get(item.indicator, "score"),
                    threshold=item.threshold,
                    points=item.points,
                    explanation=item.detail,
                    txn_ids=list(item.evidence_txn_ids),
                ))
        graph = account.graph_row or {}
        for indicator, label, unit in _TOPOLOGY_FIELDS:
            value = as_float(graph.get(indicator))
            if value <= 0:
                continue
            indicators.append(DossierIndicator(
                indicator=indicator, method="TOPOLOGY", value=value, unit=unit,
                explanation=(f"Graph topology measurement: {label.lower()} = "
                             f"{value:.4f}. Computed from the supplied ledger only."),
            ))
        trail = account.trail
        if trail is not None and trail.legs:
            indicators.append(DossierIndicator(
                indicator="fifo_taint_reached", method="FIFO_TAINT",
                value=trail.total_taint, unit="rupees",
                explanation=(
                    f"{format_inr(trail.total_taint)} of the "
                    f"{format_inr(trail.origin_amount)} traced from "
                    f"A/C {trail.origin_account} left it within {trail.max_hops} "
                    "hops under the first-in-first-out tainted-balance "
                    "assumption. Funds held by the account are not counted as "
                    "having moved."),
                txn_ids=[leg.txn_id for leg in trail.legs],
            ))
        return indicators
```

`_risk_summary()` picks the highest-scoring subject (tie → smallest account_no).
Narrow with a comprehension — `# type: ignore` is forbidden (ARCHITECTURE §7.2):

```python
    def _risk_summary(self) -> DossierRiskSummary | None:
        candidates: list[tuple[DossierAccount, RiskAssessment]] = [
            (account, account.risk)
            for account in self._accounts
            if account.risk is not None
        ]
        if not candidates:
            return None
        account, risk = min(
            candidates, key=lambda pair: (-pair[1].risk_score, pair[0].account_no)
        )
        components = risk.domain_components
        weights = self._config.domain_weights
        sub_scores: list[DossierSubScore] = []
        for key in DOMAIN_KEYS:
            value = max(0.0, min(100.0, components.get(key, 0.0)))
            weight = weights.get(key, 0.0)
            sub_scores.append(DossierSubScore(
                domain=key, label=DOMAIN_LABELS[key], value=value,
                weight=weight, points=weight * value / 100.0,
            ))
        anomaly_weight = weights.get("anomaly", 0.0)
        sub_scores.append(DossierSubScore(
            domain="anomaly", label=DOMAIN_LABELS["anomaly"],
            value=risk.anomaly_score, weight=anomaly_weight,
            points=anomaly_weight * risk.anomaly_score / 100.0,
        ))
        return DossierRiskSummary(
            account_no=risk.account_no,
            risk_score=risk.risk_score,
            domain_score=risk.domain_score,
            anomaly_score=risk.anomaly_score,
            risk_band=risk.risk_band,
            top_archetype=risk.top_archetype,
            sub_scores=sub_scores,
        )
```

`DossierConfig.domain_weights` is a plain `dict[str, float]` (defaults mirror
`RiskConfig`: 25/20/20/10/10/15 and anomaly 30) so the export layer does not
depend on the scorer. Wire both into `build()`: `indicators=[...]`,
`risk_summary=self._risk_summary()`.

- [ ] **Step 4: Run test to verify it passes**

`uv run pytest tests/test_phase3.py -q` → PASS, then the three gates.

- [ ] **Step 5: Commit**

```bash
git add src/export/dossier.py tests/test_phase3.py
git commit -m "phase3(3): add dossier section B risk indicators with taint provenance"
```

---

### Task 4: Dossier Section C — Section 94 BNSS requisition, grouped by bank

**Files:**
- Modify: `src/export/dossier.py`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Consumes: `DossierAccount.metadata["ifsc"]`, `.transactions`, `.complaints`,
  `DossierConfig.bank_names`, `.requested_documents`, `.deadline_days`.
- Produces: `InvestigationDossierGenerator._requisition() -> DossierRequisition`
  with banks grouped by IFSC prefix (`bank_name_for`), sorted by prefix.

- [ ] **Step 1: Write the failing test**

```python
def test_requisition_cites_section_94_and_keeps_lien_marking_separate() -> None:
    dossier = _generator().build()
    req = dossier.requisition
    assert req.statute_reference == "Section 94 of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)"
    assert "Nodal Officer" in req.addressee
    body = " ".join(req.paragraphs)
    assert "Section 94" in body
    assert "Section 106" in body and "CFCFRMS" in body
    assert "separate" in body.lower()
    assert "not legal advice" in req.legal_note.lower()
    assert dossier.disclaimer.startswith("Section A reproduces records")


def test_requisition_groups_accounts_by_bank_and_lists_only_their_utrs() -> None:
    axis = DossierAccount(
        account_no="A1", metadata=_account_row("A1", ifsc="UTIB0000123"),
        transactions=(_txn("U-AXIS", "VICTIM", "A1", 100.0, 0),),
    )
    sbi = DossierAccount(
        account_no="S1", metadata=_account_row("S1", ifsc="SBIN0000421"),
        transactions=(_txn("U-SBI", "A1", "S1", 90.0, 5),),
    )
    dossier = InvestigationDossierGenerator([axis, sbi]).build()
    groups = {group.ifsc_prefix: group for group in dossier.requisition.bank_requests}
    assert groups["UTIB"].bank_name == "Axis Bank"
    assert groups["SBIN"].bank_name == "State Bank of India"
    assert groups["UTIB"].accounts == ["A1"]
    assert groups["SBIN"].accounts == ["S1"]
    # A1 is the sender of U-SBI, so both banks must be asked about that leg
    assert {"U-AXIS", "U-SBI"} <= set(groups["UTIB"].utrs)
    assert "U-AXIS" not in groups["SBIN"].utrs      # S1 never touches the Axis leg
    assert groups["SBIN"].utrs == ["U-SBI"]
    assert groups["SBIN"].documents == list(REQUISITION_DOCUMENTS)


def test_requisition_handles_an_account_without_ifsc() -> None:
    account = DossierAccount(account_no="NOIFSC",
                             metadata=_account_row("NOIFSC", ifsc=None))
    dossier = InvestigationDossierGenerator([account]).build()
    group = dossier.requisition.bank_requests[0]
    assert group.ifsc_prefix == "UNKNOWN"
    assert "not identified" in group.bank_name.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase3.py -q` → FAIL on `bank_requests == []`.

- [ ] **Step 3: Write minimal implementation**

```python
    def _requisition(self) -> DossierRequisition:
        config = self._config
        groups: dict[str, dict[str, set[str]]] = {}
        names: dict[str, str] = {}
        for account in self._accounts:
            ifsc = as_str((account.metadata or {}).get("ifsc"))
            prefix = ifsc[:4].upper() if ifsc else "UNKNOWN"
            names.setdefault(prefix, bank_name_for(ifsc, config.bank_names))
            bucket = groups.setdefault(prefix, {"accounts": set(), "utrs": set()})
            bucket["accounts"].add(account.account_no)
            allowed = account.ledger_txn_ids
            for leg in account.transactions:
                txn_id = as_str(leg.get("txn_id"))
                if not txn_id or (allowed is not None and txn_id not in allowed):
                    continue
                bucket["utrs"].add(txn_id)
            for row in account.complaints:
                txn_id = as_str(row.get("disputed_utr"))
                if txn_id and (allowed is None or txn_id in allowed):
                    bucket["utrs"].add(txn_id)

        bank_requests = [
            RequisitionRequest(
                ifsc_prefix=prefix,
                bank_name=names[prefix],
                accounts=sorted(groups[prefix]["accounts"]),
                utrs=sorted(groups[prefix]["utrs"]),
                documents=list(config.requested_documents),
            )
            for prefix in sorted(groups)
        ]
        case = config.case_reference
        paragraphs = [
            (f"Whereas an investigation is being conducted in case {case} into "
             "offences relating to cheating and dishonest misappropriation "
             "committed through electronic payment channels, and the accounts "
             "and transaction references listed below are reasonably connected "
             "with that investigation;"),
            ("And whereas the records sought below are required for the purpose "
             "of the investigation and are expected to be in your custody or "
             "control in electronic form;"),
            (f"Now therefore, in exercise of the powers under {BNSS_SECTION_94}, "
             "you are hereby directed to cause production of the documents "
             "listed against your bank, in electronic form where available, "
             f"within {config.deadline_days} days of receipt of this "
             "requisition, duly certified by an authorised officer;"),
            ("This requisition is limited to the production of records. Any "
             "requirement of lien marking, freezing or seizure shall be pursued "
             f"separately through the CFCFRMS process and {BNSS_SECTION_106}, "
             "as applicable, and is not covered by this communication;"),
            ("A compliance report identifying the officer who certified the "
             "records, the date of production and any record that could not be "
             "produced (with reasons) shall accompany the response."),
        ]
        return DossierRequisition(
            statute_reference=BNSS_SECTION_94,
            addressee="The Nodal Officer of each bank listed below",
            subject=f"Production of records under {BNSS_SECTION_94} — case {case}",
            case_reference=case,
            paragraphs=paragraphs,
            bank_requests=bank_requests,
            legal_note=(
                "Draft for review by the competent investigating officer. This "
                "document is not legal advice, does not certify the records "
                "listed in Section A, and must be signed by an officer "
                "authorised to issue it."
            ),
        )
```

Then in `build()` replace the inline `DossierRequisition(...)` literal (added in
Task 2) with `requisition=self._requisition()` and delete the literal.

- [ ] **Step 4: Run test to verify it passes**

`uv run pytest tests/test_phase3.py -q` → PASS, then the three gates.

- [ ] **Step 5: Commit**

```bash
git add src/export/dossier.py tests/test_phase3.py
git commit -m "phase3(4): draft Section 94 BNSS requisition grouped by bank"
```

---

### Task 5: Dossier renderers — Markdown, HTML, JSON (offline, escaped, deterministic)

**Files:**
- Modify: `src/export/dossier.py`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Produces:
  - `.render_markdown(dossier: InvestigationDossier | None = None) -> str`
  - `.render_html(dossier: InvestigationDossier | None = None) -> bytes`
  - `.render_json(dossier: InvestigationDossier | None = None) -> bytes`

- [ ] **Step 1: Write the failing test**

```python
# extend the test imports
from core.contracts import InvestigationDossier


def test_renderers_are_deterministic() -> None:
    first = _generator().render_json()
    second = _generator().render_json()
    assert first == second
    assert _generator().render_markdown() == _generator().render_markdown()
    assert _generator().render_html() == _generator().render_html()


def test_json_round_trips_through_the_contract() -> None:
    dossier = _generator().build()
    restored = InvestigationDossier.model_validate_json(_generator().render_json())
    assert restored == dossier


def test_html_is_self_contained_and_print_ready() -> None:
    payload = _generator().render_html().decode("utf-8")
    assert payload.startswith("<!DOCTYPE html>")
    assert "http://" not in payload and "https://" not in payload
    assert "<script" not in payload.lower()
    assert "@media print" in payload
    assert "Section A" in payload and "Section B" in payload and "Section C" in payload


def test_values_are_escaped_in_html_and_markdown() -> None:
    hostile = DossierAccount(
        account_no="<script>alert(1)</script>",
        metadata=_account_row("A1", branch_code="BR|042"),
        transactions=(_txn("U1", "VICTIM", "A1", 10.0, 0),),
    )
    generator = InvestigationDossierGenerator([hostile])
    html_payload = generator.render_html().decode("utf-8")
    assert "<script>alert(1)</script>" not in html_payload
    assert "&lt;script&gt;" in html_payload
    markdown = generator.render_markdown()
    table_rows = [line for line in markdown.splitlines() if line.startswith("|")]
    widths = {line.count("|") - line.count("\\|") for line in table_rows}
    assert len(widths) == 1                 # no value can break the table structure
    assert "BR\\|042" in markdown           # the pipe is escaped, not dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase3.py -q`
Expected: FAIL — `AttributeError: 'InvestigationDossierGenerator' object has no attribute 'render_json'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _html_text(value: object) -> str:
    return html.escape(str(value), quote=True)


_PRINT_CSS: str = (
    "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
    "max-width:960px;margin:2rem auto;padding:0 1rem;color:#111;line-height:1.45}"
    "h1{font-size:1.6rem}h2{margin-top:2rem;border-bottom:2px solid #111;padding-bottom:.25rem}"
    "table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.5rem 0 1rem}"
    "th,td{border:1px solid #999;padding:.35rem .5rem;text-align:left;vertical-align:top}"
    "th{background:#eee}.meta{color:#444;font-size:.85rem}"
    ".note{border-left:4px solid #b00;padding:.5rem .75rem;background:#fdf3f3;font-size:.85rem}"
    "@media print{body{margin:0}h2{page-break-after:avoid}table{page-break-inside:auto}}"
)


class InvestigationDossierGenerator:
    # `__init__`, `_facts_for`, `_ledger_facts`, `_complaint_facts`,
    # `_shared_kyc_facts`, `_indicators_for`, `_risk_summary`, `_requisition`
    # and `build` are defined in Tasks 2–4; add the three renderers to that class.

    def render_markdown(self, dossier: InvestigationDossier | None = None) -> str:
        data = dossier or self.build()
        lines = [
            f"# Case Evidence Dossier — {data.case_reference}", "",
            f"**Scope:** {data.scope}  ",
            f"**Subject accounts:** {', '.join(data.subject_accounts) or 'none'}  ",
            f"**Generated on:** {self._generated_on_text()}  ", "",
            "## Section A — Verified Evidentiary Facts", "",
            "> Records exactly as supplied by the source systems. Not a certified copy.", "",
            "| Category | Record | Detail | Source |", "| :--- | :--- | :--- | :--- |",
        ]
        for fact in data.facts:
            lines.append(
                f"| {_md_cell(fact.category)} | {_md_cell(fact.label)} | "
                f"{_md_cell(fact.value)} | {_md_cell(fact.source)} |"
            )
        lines += ["", "## Section B — Analytical & AI Risk Indicators", "",
                  "> Automated analysis. Investigative leads, not evidence.", ""]
        if data.risk_summary is not None:
            summary = data.risk_summary
            lines += [f"**Risk score:** {summary.risk_score:.2f}/100 "
                      f"({summary.risk_band}) · domain {summary.domain_score:.2f} · "
                      f"anomaly {summary.anomaly_score:.2f} · "
                      f"archetype: {summary.top_archetype}", "",
                      "| Sub-score | Value | Weight | Points |", "| :--- | ---: | ---: | ---: |"]
            for score in summary.sub_scores:
                lines.append(f"| {_md_cell(score.label)} | {score.value:.1f} | "
                             f"{score.weight:.0f} | {score.points:.1f} |")
            lines.append("")
        lines += ["| Indicator | Method | Value | Explanation |",
                  "| :--- | :--- | ---: | :--- |"]
        for indicator in data.indicators:
            lines.append(f"| {_md_cell(indicator.indicator)} | {_md_cell(indicator.method)} | "
                         f"{indicator.value:.4f} | {_md_cell(indicator.explanation)} |")
        lines += ["", "## Section C — Section 94 BNSS Requisition (Draft)", "",
                  f"**{data.requisition.subject}**", "",
                  f"To: {data.requisition.addressee}", ""]
        lines += [f"{paragraph}" for paragraph in data.requisition.paragraphs]
        for group in data.requisition.bank_requests:
            lines += ["", f"### {group.bank_name} (IFSC prefix {group.ifsc_prefix})", "",
                      f"- Accounts: {', '.join(group.accounts) or 'none'}",
                      f"- Transaction references: {', '.join(group.utrs) or 'none'}",
                      "- Documents required:"]
            lines += [f"  - {document}" for document in group.documents]
        lines += ["", f"*{data.requisition.legal_note}*", "",
                  f"**Disclaimer:** {data.disclaimer}", ""]
        return "\n".join(lines)
```

`render_html` builds the same three sections with `_html_text` for every value
and `_html_table(headers, rows)`; `render_json` is
`data.model_dump_json(indent=2).encode("utf-8")`. `_generated_on_text()` returns
`self._config.generated_on.isoformat()` or `"not recorded"` — never `today()`.

- [ ] **Step 4: Run test to verify it passes**

`uv run pytest tests/test_phase3.py -q` → PASS, then the three gates.

- [ ] **Step 5: Commit**

```bash
git add src/export/dossier.py tests/test_phase3.py
git commit -m "phase3(5): render offline markdown, html and json dossiers"
```

---

### Task 6: Pipeline bundle, ingestion audit view and leaderboard

**Files:**
- Create: `src/ui/state.py`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Produces:
  - `state.PipelineBundle` (frozen dataclass: `transactions`, `accounts`,
    `complaints`, `features`, `graph`, `risk`, `assessments`, `analyzer`,
    `wash_cycles`, `complaint_links`, `reports`)
  - `state.run_pipeline(transactions, accounts=None, complaints=None, *,
    feature_config=None, graph_config=None, risk_config=None, reports=None) -> PipelineBundle`
  - `state.build_ingestion_view(reports: Mapping[str, IngestionReport]) -> IngestionView`
  - `state.build_leaderboard(bundle, *, min_score=0.0, band=None, archetype=None) -> LeaderboardView`

- [ ] **Step 1: Write the failing test**

```python
# extend the test imports
from core.contracts import COMPLAINT_SCHEMA, TRANSACTION_SCHEMA
from ingestion.mapper import DatasetIngester
from ui.state import build_ingestion_view, build_leaderboard, run_pipeline


def _pipeline_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    transactions = pl.DataFrame([
        _txn("T0", "V2", "HUB", 50_000.0, -10),
        _txn("T1", "V", "HUB", 100_000.0, 0),
        _txn("T2", "HUB", "M2", 95_000.0, 5),
        _txn("T3", "M2", "TERM", 93_000.0, 10, mode="AEPS"),
        _txn("T4", "OTHER", "SHOP", 500.0, 20),
    ], schema=TRANSACTION_SCHEMA)
    accounts = pl.DataFrame([_account_row("HUB"), _account_row("M2"),
                             _account_row("TERM")], schema=ACCOUNT_METADATA_SCHEMA)
    complaints = pl.DataFrame([
        {"ack_no": "ACK1", "victim_account": "V", "suspect_account": "HUB",
         "fraud_category": "UPI_FRAUD", "reported_timestamp": BASE + timedelta(hours=1),
         "disputed_amount": 100_000.0, "disputed_utr": "T1", "state": "Bihar"},
        {"ack_no": "ACK2", "victim_account": "V2", "suspect_account": "HUB",
         "fraud_category": "UPI_FRAUD", "reported_timestamp": BASE + timedelta(hours=2),
         "disputed_amount": 50_000.0, "disputed_utr": None, "state": "Jharkhand"},
    ], schema=COMPLAINT_SCHEMA)
    return transactions, accounts, complaints


def test_pipeline_is_deterministic_and_reuses_phase2_layers() -> None:
    frames = _pipeline_frames()
    first = run_pipeline(*frames)
    second = run_pipeline(*frames)
    assert first.features.equals(second.features)
    assert first.graph.equals(second.graph)
    assert first.risk.equals(second.risk)
    assert [a.account_no for a in first.assessments] == sorted(a.account_no for a in first.assessments)
    assert first.wash_cycles == ()
    assert first.complaint_links == ()


def test_leaderboard_ranks_by_score_then_account() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    view = build_leaderboard(bundle)
    scores = [row.risk_score for row in view.rows]
    assert scores == sorted(scores, reverse=True)
    assert view.rows[0].account_no == "HUB"
    assert view.rows[0].risk_band
    assert view.rows[0].top_archetype == "Rapid pass-through mule"
    filtered = build_leaderboard(bundle, min_score=100.0)
    assert filtered.rows == ()
    assert filtered.total_accounts == len(bundle.assessments)


def test_ingestion_view_summarises_failures_and_mappings() -> None:
    csv_bytes = (
        "UTR,From Account,To Account,Amount,Txn Date,Mode\n"
        "T1,V,HUB,100,2024-06-01 10:00:00,upi\n"
        "T2,V,HUB,not-a-number,2024-06-01 10:05:00,upi\n"
        ",V,HUB,50,2024-06-01 10:06:00,upi\n"
    ).encode()
    _, report = DatasetIngester().ingest_transactions(csv_bytes)
    view = build_ingestion_view({"transactions": report})
    table = view.tables[0]
    assert table.table == "transactions"
    assert (table.rows_total, table.rows_accepted, table.rows_rejected) == (3, 1, 2)
    failures = dict(table.failure_counts)
    assert failures["unparseable_amount"] == 1
    assert failures["missing:txn_id"] == 1        # note the `missing:<column>` form
    mapped = {row.raw_name: row for row in table.mappings}
    assert mapped["UTR"].canonical_name == "txn_id"
    assert mapped["UTR"].matched_synonym == "utr"
    assert mapped["UTR"].confidence == 1.0
    assert mapped["Txn Date"].canonical_name == "timestamp"
    assert view.total_rejected == 2 and view.total_rows == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase3.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.state'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Pure view-model builders for the investigator workbench.

No Streamlit import may appear in this module: every function is a deterministic
transformation of Phase 1/2 artefacts into render-ready structures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from core.coerce import as_date, as_datetime, as_float, as_int, as_str
from core.contracts import (
    ACCOUNT_METADATA_SCHEMA, COMPLAINT_SCHEMA, ComplaintLink, DossierFact,
    DossierIndicator, FeatureConfig, GraphConfig, IngestionReport,
    InvestigationDossier, MoneyTrail, RiskAssessment, RiskConfig, WashCycle,
)
from export.dossier import (
    DossierAccount, DossierConfig, InvestigationDossierGenerator,
)
from features.account_features import extract_account_features
from graph.analyzer import GraphAnalyzer
from risk.scorer import RiskScorer


@dataclass(frozen=True)
class PipelineBundle:
    transactions: pl.DataFrame
    accounts: pl.DataFrame
    complaints: pl.DataFrame
    features: pl.DataFrame
    graph: pl.DataFrame
    risk: pl.DataFrame
    assessments: tuple[RiskAssessment, ...]
    analyzer: GraphAnalyzer
    wash_cycles: tuple[WashCycle, ...]
    complaint_links: tuple[ComplaintLink, ...]
    reports: Mapping[str, IngestionReport]


def run_pipeline(
    transactions: pl.DataFrame,
    accounts: pl.DataFrame | None = None,
    complaints: pl.DataFrame | None = None,
    *,
    feature_config: FeatureConfig | None = None,
    graph_config: GraphConfig | None = None,
    risk_config: RiskConfig | None = None,
    reports: Mapping[str, IngestionReport] | None = None,
) -> PipelineBundle:
    features = extract_account_features(transactions, accounts, complaints, feature_config)
    analyzer = GraphAnalyzer(transactions, accounts, complaints, graph_config)
    graph = analyzer.topology_metrics()
    scorer = RiskScorer(risk_config)
    risk, assessments = scorer.score_with_assessments(features, graph)
    return PipelineBundle(
        transactions=transactions,
        accounts=accounts if accounts is not None else pl.DataFrame(schema=ACCOUNT_METADATA_SCHEMA),
        complaints=complaints if complaints is not None else pl.DataFrame(schema=COMPLAINT_SCHEMA),
        features=features,
        graph=graph,
        risk=risk,
        assessments=tuple(assessments),
        analyzer=analyzer,
        wash_cycles=tuple(analyzer.detect_wash_cycles()),
        complaint_links=tuple(analyzer.correlate_complaints()),
        reports=dict(reports or {}),
    )


@dataclass(frozen=True)
class MappingRow:
    raw_name: str
    canonical_name: str | None
    matched_synonym: str | None
    confidence: float


@dataclass(frozen=True)
class IngestionTableView:
    table: str
    source: str
    rows_total: int
    rows_accepted: int
    rows_rejected: int
    failure_counts: tuple[tuple[str, int], ...]
    mappings: tuple[MappingRow, ...]
    rejected_preview: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class IngestionView:
    tables: tuple[IngestionTableView, ...]
    total_rows: int
    total_accepted: int
    total_rejected: int


def build_ingestion_view(reports: Mapping[str, IngestionReport]) -> IngestionView:
    tables: list[IngestionTableView] = []
    for table in sorted(reports):
        report = reports[table]
        mappings = tuple(
            MappingRow(raw_name=item.raw_name, canonical_name=item.canonical_name,
                       matched_synonym=item.matched_synonym, confidence=item.confidence)
            for _, item in sorted(report.column_mapping.items())
        )
        preview_frame = report.rejected_rows.head(20)
        rejected_preview = tuple(
            tuple(str(value) for value in row)
            for row in preview_frame.iter_rows()
        )
        tables.append(IngestionTableView(
            table=table, source=report.source or "not recorded",
            rows_total=report.rows_total, rows_accepted=report.rows_accepted,
            rows_rejected=report.rows_rejected,
            failure_counts=tuple(sorted(report.failure_counts.items())),
            mappings=mappings, rejected_preview=rejected_preview,
        ))
    return IngestionView(
        tables=tuple(tables),
        total_rows=sum(item.rows_total for item in tables),
        total_accepted=sum(item.rows_accepted for item in tables),
        total_rejected=sum(item.rows_rejected for item in tables),
    )


@dataclass(frozen=True)
class LeaderboardRow:
    account_no: str
    risk_score: float
    domain_score: float
    anomaly_score: float
    risk_band: str
    top_archetype: str
    complaint_count: int
    top_reason: str


@dataclass(frozen=True)
class LeaderboardView:
    rows: tuple[LeaderboardRow, ...]
    total_accounts: int
    flagged_accounts: int


_FLAGGED_BANDS = frozenset({"HIGH", "CRITICAL"})


def build_leaderboard(
    bundle: PipelineBundle,
    *,
    min_score: float = 0.0,
    band: str | None = None,
    archetype: str | None = None,
) -> LeaderboardView:
    reasons = {
        str(row["account_no"]): row.get("top_reasons") or []
        for row in bundle.risk.to_dicts()
    }
    complaints = {
        str(row["account_no"]): as_int(row.get("direct_complaint_count"))
        for row in bundle.features.select("account_no", "direct_complaint_count").to_dicts()
    }
    rows: list[LeaderboardRow] = []
    for assessment in bundle.assessments:
        if assessment.risk_score < min_score:
            continue
        if band is not None and assessment.risk_band != band:
            continue
        if archetype is not None and assessment.top_archetype != archetype:
            continue
        top = reasons.get(assessment.account_no) or []
        rows.append(LeaderboardRow(
            account_no=assessment.account_no,
            risk_score=round(assessment.risk_score, 2),
            domain_score=round(assessment.domain_score, 2),
            anomaly_score=round(assessment.anomaly_score, 2),
            risk_band=assessment.risk_band,
            top_archetype=assessment.top_archetype,
            complaint_count=complaints.get(assessment.account_no, 0),
            top_reason=str(top[0]) if top else "",
        ))
    rows.sort(key=lambda row: (-row.risk_score, row.account_no))
    return LeaderboardView(
        rows=tuple(rows),
        total_accounts=len(bundle.assessments),
        flagged_accounts=sum(1 for a in bundle.assessments if a.risk_band in _FLAGGED_BANDS),
    )
```

Note `iter_rows()` on the *rejected preview* is permitted: it is a presentation
sample of ≤20 rows, not a data pipeline. (The no-`iter_rows` rule in
ARCHITECTURE §5.1 applies to feature computation.) If the reviewer prefers,
use `preview_frame.to_dicts()` and join the values — either is acceptable; the
test only pins order and content.

- [ ] **Step 4: Run test to verify it passes**

`uv run pytest tests/test_phase3.py -q` → PASS, then the three gates.

- [ ] **Step 5: Commit**

```bash
git add src/ui/state.py tests/test_phase3.py
git commit -m "phase3(6): add pipeline bundle, ingestion audit and leaderboard views"
```

---

### Task 7: Account detail and dossier assembly (account / cluster / complaint scopes)

**Files:**
- Modify: `src/ui/state.py`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Produces:
  - `state.build_dossier_account(bundle, account_no) -> DossierAccount`
  - `state.build_account_dossier(bundle, account_no) -> InvestigationDossier`
  - `state.build_cluster_dossier(bundle, account_nos: Sequence[str]) -> InvestigationDossier`
  - `state.build_complaint_dossier(bundle, ack_no: str, *, max_hops: int = 3) -> InvestigationDossier`
  - `state.SubScoreRow`, `state.TimelineRow`, `state.AccountDetailView`,
    `state.build_account_detail(bundle, account_no) -> AccountDetailView`

- [ ] **Step 1: Write the failing test**

```python
# extend the test imports
from ui.state import (
    build_account_detail, build_account_dossier, build_cluster_dossier,
    build_complaint_dossier,
)


def test_account_dossier_facts_and_indicators_are_separated() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    dossier = build_account_dossier(bundle, "HUB")
    assert dossier.scope == "ACCOUNT"
    assert dossier.subject_accounts == ["HUB"]
    assert any(fact.txn_id == "T1" for fact in dossier.facts)
    assert any(indicator.method == "FIFO_TAINT" for indicator in dossier.indicators)
    assert dossier.risk_summary is not None
    assert dossier.risk_summary.account_no == "HUB"
    assert all(fact.source != "ledger" or fact.txn_id for fact in dossier.facts)


def test_account_detail_exposes_seven_sub_scores_and_a_chronological_timeline() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    detail = build_account_detail(bundle, "HUB")
    assert [score.domain for score in detail.sub_scores][:6] == list(DOMAIN_KEYS)
    assert detail.sub_scores[-1].domain == "anomaly"
    assert [row.txn_id for row in detail.timeline] == ["T0", "T1", "T2"]
    assert [row.direction for row in detail.timeline] == ["IN", "IN", "OUT"]
    assert detail.metadata["ifsc"] == "UTIB0000123"


def test_complaint_dossier_does_not_list_unrelated_victim_spending() -> None:
    transactions, accounts, complaints = _pipeline_frames()
    transactions = pl.concat([
        transactions,
        pl.DataFrame([_txn("V-LEGIT", "V", "SHOP2", 700.0, 30)],
                     schema=TRANSACTION_SCHEMA),
    ])
    bundle = run_pipeline(transactions, accounts, complaints)
    dossier = build_complaint_dossier(bundle, "ACK1")
    assert dossier.scope == "COMPLAINT"
    assert "V-LEGIT" not in [fact.txn_id for fact in dossier.facts]
    assert "T1" in [fact.txn_id for fact in dossier.facts]
    assert "ACK1" in [fact.ack_no for fact in dossier.facts]


def test_complaint_dossier_without_utr_uses_the_suspect_leg() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    dossier = build_complaint_dossier(bundle, "ACK2")
    # ACK2 has no disputed_utr: the inferred origin is V2's leg to the named suspect
    assert "T0" in [fact.txn_id for fact in dossier.facts]
    assert "ACK2" in [fact.ack_no for fact in dossier.facts]
    assert dossier.indicators          # trail-based indicators exist


def test_unknown_account_and_ack_produce_valid_empty_documents() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    ghost = build_account_dossier(bundle, "NO-SUCH-ACCOUNT")
    assert ghost.subject_accounts == ["NO-SUCH-ACCOUNT"]
    assert ghost.facts and ghost.facts[0].source == "generator"
    missing = build_complaint_dossier(bundle, "ACK-NOT-THERE")
    assert missing.scope == "COMPLAINT" and missing.facts == []


def test_cluster_dossier_covers_every_selected_account() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    dossier = build_cluster_dossier(bundle, ["M2", "HUB"])
    assert dossier.scope == "CLUSTER"
    assert dossier.subject_accounts == ["HUB", "M2"]
    account_facts = [fact for fact in dossier.facts if fact.category == "ACCOUNT"]
    assert [fact.account_no for fact in account_facts] == ["HUB", "M2"]
    assert dossier.risk_summary is not None
    assert dossier.risk_summary.account_no == "HUB"      # highest score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase3.py -q` → FAIL on `build_account_dossier` not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def _row_for(frame: pl.DataFrame, account_no: str) -> Mapping[str, object] | None:
    if frame.height == 0:
        return None
    matched = frame.filter(pl.col("account_no") == account_no)
    return matched.to_dicts()[0] if matched.height else None


def _legs_for(transactions: pl.DataFrame, account_no: str) -> tuple[dict[str, object], ...]:
    legs = transactions.filter(
        (pl.col("sender_account") == account_no) | (pl.col("receiver_account") == account_no)
    ).sort(["timestamp", "txn_id"])
    return tuple(legs.to_dicts())


def _complaints_for(complaints: pl.DataFrame, account_no: str) -> tuple[dict[str, object], ...]:
    matched = complaints.filter(
        (pl.col("victim_account") == account_no) | (pl.col("suspect_account") == account_no)
    ).sort(["ack_no"])
    return tuple(matched.to_dicts())


def _imei_peers(accounts: pl.DataFrame, master: Mapping[str, object] | None) -> tuple[dict[str, object], ...]:
    imei = as_str((master or {}).get("imei"))
    if not imei or accounts.height == 0:
        return ()
    peers = accounts.filter(
        (pl.col("imei") == imei) & (pl.col("account_no") != as_str((master or {}).get("account_no")))
    ).sort("account_no")
    return tuple(peers.to_dicts())


def _branch_peers(accounts: pl.DataFrame, master: Mapping[str, object] | None) -> tuple[dict[str, object], ...]:
    branch = as_str((master or {}).get("branch_code"))
    opened = as_date((master or {}).get("account_opening_date"))
    if not branch or opened is None or accounts.height == 0:
        return ()
    peers: list[tuple[float, dict[str, object]]] = []
    for row in accounts.filter(pl.col("branch_code") == branch).to_dicts():
        if as_str(row.get("account_no")) == as_str(master.get("account_no")):
            continue
        peer_opened = as_date(row.get("account_opening_date"))
        if peer_opened is None:
            continue
        hours = abs((peer_opened - opened).total_seconds()) / 3600.0
        if hours <= 48.0:
            peers.append((hours, row))
    peers.sort(key=lambda item: (item[0], as_str(item[1].get("account_no"))))
    return tuple(row for _, row in peers)


def build_dossier_account(bundle: PipelineBundle, account_no: str,
                          *, ledger_txn_ids: frozenset[str] | None = None,
                          max_hops: int | None = None) -> DossierAccount:
    master = _row_for(bundle.accounts, account_no)
    assessment = next((a for a in bundle.assessments if a.account_no == account_no), None)
    trail = (
        bundle.analyzer.trace_money_trail(account_no, max_hops=max_hops)
        if account_no in set(bundle.features["account_no"].to_list())
        else None
    )
    return DossierAccount(
        account_no=account_no, metadata=master, risk=assessment,
        feature_row=_row_for(bundle.features, account_no),
        graph_row=_row_for(bundle.graph, account_no),
        transactions=_legs_for(bundle.transactions, account_no),
        complaints=_complaints_for(bundle.complaints, account_no),
        trail=trail,
        cycles=tuple(c for c in bundle.wash_cycles if account_no in c.cycle_accounts),
        imei_peers=_imei_peers(bundle.accounts, master),
        branch_peers=_branch_peers(bundle.accounts, master),
        ledger_txn_ids=ledger_txn_ids,
    )


def _dossier_config(scope: str, case_reference: str) -> DossierConfig:
    return DossierConfig(scope=scope, case_reference=case_reference)


def build_account_dossier(bundle: PipelineBundle, account_no: str) -> InvestigationDossier:
    return InvestigationDossierGenerator(
        [build_dossier_account(bundle, account_no)],
        _dossier_config("ACCOUNT", f"PG-AC-{account_no}"),
    ).build()


def build_cluster_dossier(bundle: PipelineBundle, account_nos: Sequence[str]) -> InvestigationDossier:
    accounts = [build_dossier_account(bundle, number) for number in sorted(set(account_nos))]
    return InvestigationDossierGenerator(
        accounts, _dossier_config("CLUSTER", "PG-CL-" + "-".join(sorted(set(account_nos))[:3])),
    ).build()


def build_complaint_dossier(bundle: PipelineBundle, ack_no: str, *, max_hops: int = 3) -> InvestigationDossier:
    trail = bundle.analyzer.trace_complaint_trail(ack_no, max_hops=max_hops)
    involved: set[str] = set()
    txn_ids: set[str] = {leg.txn_id for leg in trail.legs}
    if trail.origin_account:
        involved.add(trail.origin_account)
    for leg in trail.legs:
        involved.add(leg.sender_account)
        involved.add(leg.receiver_account)
    rows = bundle.complaints.filter(pl.col("ack_no") == ack_no).to_dicts()
    if rows:
        row = rows[0]
        for key in ("victim_account", "suspect_account"):
            value = as_str(row.get(key))
            if value:
                involved.add(value)
        disputed = as_str(row.get("disputed_utr"))
        if disputed:
            txn_ids.add(disputed)
    else:
        return InvestigationDossierGenerator(
            [], _dossier_config("COMPLAINT", f"PG-ACK-{ack_no}")
        ).build()
    scoped = frozenset(txn_ids)
    accounts = [build_dossier_account(bundle, number, ledger_txn_ids=scoped)
                for number in sorted(involved)]
    return InvestigationDossierGenerator(
        accounts, _dossier_config("COMPLAINT", f"PG-ACK-{ack_no}")
    ).build()
```

`AccountDetailView` (SubScoreRow/TimelineRow included) and
`build_account_detail` project the account dossier:

```python
@dataclass(frozen=True)
class SubScoreRow:
    domain: str
    label: str
    value: float
    weight: float
    points: float


@dataclass(frozen=True)
class TimelineRow:
    timestamp: datetime
    direction: str          # "IN" | "OUT"
    counterparty: str
    amount: float
    txn_id: str
    payment_mode: str


@dataclass(frozen=True)
class AccountDetailView:
    account_no: str
    metadata: Mapping[str, str]
    risk: RiskAssessment | None
    sub_scores: tuple[SubScoreRow, ...]
    timeline: tuple[TimelineRow, ...]
    facts: tuple[DossierFact, ...]
    indicators: tuple[DossierIndicator, ...]
    dossier: InvestigationDossier


def build_account_detail(bundle: PipelineBundle, account_no: str) -> AccountDetailView:
    dossier = build_account_dossier(bundle, account_no)
    account = build_dossier_account(bundle, account_no)
    sub_scores: tuple[SubScoreRow, ...] = ()
    if dossier.risk_summary is not None:
        sub_scores = tuple(
            SubScoreRow(domain=score.domain, label=score.label, value=score.value,
                        weight=score.weight, points=score.points)
            for score in dossier.risk_summary.sub_scores
        )
    timeline: list[TimelineRow] = []
    for leg in account.transactions:
        outbound = as_str(leg.get("sender_account")) == account_no
        moment = as_datetime(leg.get("timestamp"))
        if moment is None:
            continue
        timeline.append(TimelineRow(
            timestamp=moment,
            direction="OUT" if outbound else "IN",
            counterparty=as_str(leg.get("receiver_account") if outbound else leg.get("sender_account")),
            amount=as_float(leg.get("amount")), txn_id=as_str(leg.get("txn_id")),
            payment_mode=as_str(leg.get("payment_mode")),
        ))
    timeline.sort(key=lambda row: (row.timestamp, row.txn_id))
    metadata = {
        key: as_str(value)
        for key, value in (account.metadata or {}).items()
        if isinstance(value, str)
    }
    return AccountDetailView(
        account_no=account_no, metadata=metadata, risk=account.risk,
        sub_scores=sub_scores, timeline=tuple(timeline),
        facts=tuple(dossier.facts), indicators=tuple(dossier.indicators),
        dossier=dossier,
    )
```

- [ ] **Step 4: Run test to verify it passes**

`uv run pytest tests/test_phase3.py -q` → PASS, then the three gates.

- [ ] **Step 5: Commit**

```bash
git add src/ui/state.py tests/test_phase3.py
git commit -m "phase3(7): assemble account, cluster and complaint dossier scopes"
```

---

### Task 8: Graph canvas elements, wash loops, collusion, terminals, correlation

**Files:**
- Modify: `src/ui/state.py`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Produces:
  - `state.GraphNode`, `state.GraphEdge`, `state.GraphView`
  - `state.build_trail_graph(bundle, *, start: str, max_hops: int = 3) -> GraphView`
  - `state.CycleRow` / `state.CycleView` / `build_cycle_view(bundle)`
  - `state.CollusionGroup` / `state.CollusionView` / `build_collusion_view(bundle)`
  - `state.TerminalRow` / `state.TerminalView` / `build_terminal_view(bundle)`
  - `state.CorrelationRow` / `state.CorrelationHotspot` / `state.CorrelationView` /
    `build_correlation_view(bundle)`

- [ ] **Step 1: Write the failing test**

```python
# extend the test imports
from simulator.generator import SyndicateDataSimulator
from ui.state import (
    build_collusion_view, build_correlation_view, build_cycle_view,
    build_terminal_view, build_trail_graph,
)


def test_trail_graph_is_traceable_by_account_utr_and_complaint() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    by_account = build_trail_graph(bundle, start="HUB", max_hops=2)
    assert by_account.edges and all(edge.taint_amount > 0 for edge in by_account.edges)
    assert {node.id for node in by_account.nodes} >= {"HUB", "M2", "TERM"}

    by_utr = build_trail_graph(bundle, start="T1", max_hops=2)
    assert by_utr.origin_label.endswith("T1")
    assert all(edge.txn_id != "T1" for edge in by_utr.edges)   # origin leg, not a trail leg

    by_ack = build_trail_graph(bundle, start="ACK1", max_hops=2)
    assert any(edge.txn_id == "T1" for edge in by_ack.edges)  # complaint-scoped seed leg


def test_trail_graph_elements_are_renderable_and_deterministic() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    view = build_trail_graph(bundle, start="HUB", max_hops=3)
    assert view.elements == build_trail_graph(bundle, start="HUB", max_hops=3).elements
    node_ids = {node["data"]["id"] for node in view.elements["nodes"]}
    assert all(edge["data"]["source"] in node_ids for edge in view.elements["edges"])
    assert all(edge["data"]["target"] in node_ids for edge in view.elements["edges"])
    terminal = next(node for node in view.nodes if node.id == "TERM")
    assert terminal.kind == "TERMINAL"
    assert terminal.risk_band


def test_unknown_trace_start_yields_an_empty_canvas() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    view = build_trail_graph(bundle, start="NO-SUCH-ID", max_hops=3)
    assert view.nodes == () and view.edges == ()
    assert view.elements == {"nodes": [], "edges": []}


def test_collusion_and_terminal_and_correlation_views() -> None:
    bundle = run_pipeline(*_pipeline_frames())
    collusion = build_collusion_view(bundle)
    imei_groups = [group for group in collusion.groups if group.basis == "IMEI"]
    assert imei_groups and all(len(group.accounts) >= 2 for group in collusion.groups)

    terminals = build_terminal_view(bundle)
    assert [row.account_no for row in terminals.rows] == ["TERM"]
    assert terminals.rows[0].terminal_inflow == 93_000.0

    correlation = build_correlation_view(bundle)
    assert correlation.rows == ()           # no shared evidence in this ledger
    assert correlation.hotspots == ()


def test_cycle_view_lists_a_closed_wash_loop() -> None:
    # every hop loses <= 15 %, so Phase 2 reports this as a wash cycle
    transactions = pl.DataFrame([
        _txn("C1", "A", "B", 100_000.0, 0),
        _txn("C2", "B", "C", 97_000.0, 10),
        _txn("C3", "C", "A", 95_000.0, 20),
    ], schema=TRANSACTION_SCHEMA)
    view = build_cycle_view(run_pipeline(transactions))
    assert len(view.rows) == 1
    row = view.rows[0]
    assert row.accounts == ("A", "B", "C")
    assert row.txn_ids == ("C1", "C2", "C3")
    assert row.total_amount == 292_000.0      # Phase 2 sums the legs
    assert 0.0 < row.max_decay_ratio <= 0.15
```

Complaint correlation test (separate, uses the simulator so real links exist):

```python
def test_correlation_view_flags_interstate_links() -> None:
    sim = SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)
    bundle = run_pipeline(sim.transactions, sim.accounts, sim.complaints)
    view = build_correlation_view(bundle)
    # the seeded scenario routes three state-distinct complaints through one
    # AEPS terminal, so links must exist (verified: 3 links, 1 shared terminal)
    assert view.rows
    assert any(row.interstate for row in view.rows)
    assert any(row.shared_terminals for row in view.rows)
    hotspot = view.hotspots[0]
    assert len(hotspot.ack_nos) >= 2
    assert all(item.ack_nos == tuple(sorted(item.ack_nos)) for item in view.hotspots)
    assert view.hotspots == tuple(
        sorted(view.hotspots, key=lambda item: (-len(item.ack_nos), item.key))
    )
    assert view.rows == tuple(sorted(view.rows, key=lambda item: (item.ack_a, item.ack_b)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase3.py -q` → FAIL on `build_trail_graph` not defined.

- [ ] **Step 3: Write minimal implementation**

```python
_TERMINAL_MODES = frozenset({"AEPS", "ATM", "CASH"})


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    kind: str                 # "ACCOUNT" | "TERMINAL"
    risk_score: float
    risk_band: str
    community: int
    caption: str


@dataclass(frozen=True)
class GraphEdge:
    id: str
    source: str
    target: str
    txn_id: str
    amount: float
    taint_amount: float
    hop: int
    timestamp: datetime
    payment_mode: str


@dataclass(frozen=True)
class GraphView:
    origin_label: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    elements: dict[str, list[dict[str, dict[str, str]]]]


def _node_lookup(bundle: PipelineBundle) -> dict[str, tuple[float, str, int]]:
    risk = {str(row["account_no"]): row for row in bundle.risk.to_dicts()}
    graph = {str(row["account_no"]): row for row in bundle.graph.to_dicts()}
    return {
        account: (
            as_float(risk.get(account, {}).get("risk_score")),
            as_str(risk.get(account, {}).get("risk_band")) or "UNSCORED",
            as_int(graph.get(account, {}).get("louvain_community_id")),
        )
        for account in set(risk) | set(graph)
    }


def _resolve_trail(bundle: PipelineBundle, start: str, max_hops: int) -> MoneyTrail:
    if start.startswith("ACK"):
        return bundle.analyzer.trace_complaint_trail(start, max_hops=max_hops)
    return bundle.analyzer.trace_money_trail(start, max_hops=max_hops)


def build_trail_graph(bundle: PipelineBundle, *, start: str, max_hops: int = 3) -> GraphView:
    trail = _resolve_trail(bundle, start, max_hops)
    lookup = _node_lookup(bundle)
    legs = sorted(trail.legs, key=lambda leg: (leg.hop, leg.timestamp, leg.txn_id))
    terminal_nodes = {
        leg.receiver_account for leg in legs if leg.payment_mode in _TERMINAL_MODES
    }
    order: list[str] = []
    if trail.origin_account and legs:
        order.append(trail.origin_account)
    for leg in legs:
        for account in (leg.sender_account, leg.receiver_account):
            if account not in order:
                order.append(account)
    nodes = tuple(
        GraphNode(
            id=account,
            label=account,
            kind="TERMINAL" if account in terminal_nodes else "ACCOUNT",
            risk_score=lookup.get(account, (0.0, "UNSCORED", 0))[0],
            risk_band=lookup.get(account, (0.0, "UNSCORED", 0))[1],
            community=lookup.get(account, (0.0, "UNSCORED", 0))[2],
            caption=f"{account} · {lookup.get(account, (0.0, 'UNSCORED', 0))[1]}",
        )
        for account in order
    )
    edges = tuple(
        GraphEdge(
            id=leg.txn_id, source=leg.sender_account, target=leg.receiver_account,
            txn_id=leg.txn_id, amount=leg.amount, taint_amount=leg.taint_amount,
            hop=leg.hop, timestamp=leg.timestamp, payment_mode=leg.payment_mode,
        )
        for leg in legs
    )
    elements: dict[str, list[dict[str, dict[str, str]]]] = {
        "nodes": [
            {"data": {"id": node.id, "label": node.kind.lower(), "caption": node.caption,
                      "band": node.risk_band}}
            for node in nodes
        ],
        "edges": [
            {"data": {"id": edge.id, "source": edge.source, "target": edge.target,
                      "label": f"hop {edge.hop}",
                      "caption": (f"{edge.txn_id} · ₹{edge.amount:.2f} · "
                                  f"tainted ₹{edge.taint_amount:.2f}")}}
            for edge in edges
        ],
    }
    origin_label = (
        f"UTR {trail.origin_txn_id}" if trail.origin_txn_id else f"A/C {trail.origin_account}"
    )
    return GraphView(origin_label=origin_label, nodes=nodes, edges=edges, elements=elements)
```

The four sub-view models (all frozen dataclasses; fields shown are exactly what
the app renders):

```python
@dataclass(frozen=True)
class CycleRow:
    accounts: tuple[str, ...]        # cycle order, starting at the smallest id
    txn_ids: tuple[str, ...]
    total_amount: float
    max_decay_ratio: float
    start_timestamp: datetime


@dataclass(frozen=True)
class CycleView:
    rows: tuple[CycleRow, ...]       # sorted by (-max_decay_ratio, accounts)


@dataclass(frozen=True)
class CollusionGroup:
    basis: str                       # "IMEI" | "BRANCH"
    key: str                         # the IMEI, or the branch code
    accounts: tuple[str, ...]        # sorted
    opening_dates: tuple[str, ...]   # ISO dates or "not recorded"


@dataclass(frozen=True)
class CollusionView:
    groups: tuple[CollusionGroup, ...]   # sorted by (basis, key)


@dataclass(frozen=True)
class TerminalRow:
    account_no: str
    terminal_inflow: float
    cashout_count: int
    upstream_accounts: tuple[str, ...]
    terminal_chain_count: int


@dataclass(frozen=True)
class TerminalView:
    rows: tuple[TerminalRow, ...]    # sorted by (-terminal_inflow, account_no)


@dataclass(frozen=True)
class CorrelationRow:
    ack_a: str
    state_a: str
    ack_b: str
    state_b: str
    link_basis: str
    shared_accounts: tuple[str, ...]
    shared_devices: tuple[str, ...]
    shared_terminals: tuple[str, ...]
    interstate: bool


@dataclass(frozen=True)
class CorrelationHotspot:
    key: str
    ack_nos: tuple[str, ...]         # sorted


@dataclass(frozen=True)
class CorrelationView:
    rows: tuple[CorrelationRow, ...]      # sorted by (ack_a, ack_b)
    hotspots: tuple[CorrelationHotspot, ...]  # sorted by (-len(ack_nos), key)
```

`build_cycle_view` maps `bundle.wash_cycles` (already canonicalised by Phase 2).
`build_collusion_view` groups the account master frame by `imei` (non-null) and
by `branch_code` + opening dates within 48 h (both dates known), keeping groups
of ≥2. `build_terminal_view` sums terminal-mode inflows per receiver from
`bundle.transactions`, joins `aeps_cashout_inflow_count` /
`terminal_chain_count` from features/graph, and only includes accounts that
received at least one terminal-mode leg. `build_correlation_view` maps
`ComplaintLink` rows (`interstate = state_a != state_b`) and groups the linked
ACKs on each `shared_accounts` / `shared_terminal_accounts` entry.

- [ ] **Step 4: Run test to verify it passes**

`uv run pytest tests/test_phase3.py -q` → PASS, then the three gates. Note the
simulator-backed correlation test may return early on an empty link set; the
scenario assertion is what matters, and Phase 2's
`TestComplaintCorrelation` already pins link production.

- [ ] **Step 5: Commit**

```bash
git add src/ui/state.py tests/test_phase3.py
git commit -m "phase3(8): add graph canvas, cycle, collusion, terminal and correlation views"
```

---

### Task 9: Streamlit app, offline config, headless UI tests

**Files:**
- Create: `src/ui/app.py`
- Create: `.streamlit/config.toml`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Consumes every Task 6–8 builder, plus `state.dossier_markdown/html/json`
  (thin formatters added to `ui/state.py` in this task).
- Produces `ui.app.main() -> None`;
  `ui.app.load_benchmark(background: int = 5_000) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, IngestionReport]]`
  (the benchmark is round-tripped through `DatasetIngester` so Tab 1 has real
  mapping confidences, accepted/rejected counts and a downloadable rejected-rows
  CSV — verified: canonical frames are `.equals()` after the round trip).
- The pipeline bundle is cached in `st.session_state["pg_bundle"]` and only
  rebuilt when "Load seeded benchmark" or "Build from uploads" is clicked;
  widget reruns never re-fit IsolationForest/TreeSHAP or rebuild the graph.

- [ ] **Step 1: Write the failing test**

```python
# extend the test imports
from pathlib import Path

from ui import state


def test_dossier_format_wrappers_delegate() -> None:
    dossier = _generator().build()
    assert state.dossier_markdown(dossier) == _generator().render_markdown(dossier)
    assert state.dossier_html(dossier) == _generator().render_html(dossier)
    assert state.dossier_json(dossier) == _generator().render_json(dossier)


def test_app_renders_empty_state_without_a_dataset() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("src/ui/app.py", default_timeout=120).run()
    assert not app.exception, [str(item) for item in app.exception]
    titles = [title.value for title in app.title]
    assert any("PraharGraph" in title for title in titles)
    labels = [tab.label for tab in app.tabs]
    assert labels == [
        "1 · Ingestion & Data Hygiene",
        "2 · Mule Risk Ranking & XAI",
        "3 · Money Trail & Syndicates",
        "4 · Complaint Correlation & Dossier",
    ]


def test_offline_config_disables_telemetry_and_remote_fonts() -> None:
    text = Path(".streamlit/config.toml").read_text()
    assert "gatherUsageStats = false" in text
    assert 'font = "sans serif"' in text


def test_app_benchmark_load_populates_ingestion_metrics() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("src/ui/app.py", default_timeout=300)
    app.run()
    assert not app.exception, [str(item) for item in app.exception]
    app.button(key="load_benchmark").click().run()
    assert not app.exception, [str(item) for item in app.exception]

    metrics = {
        item.label: int(str(item.value).replace(",", ""))
        for item in app.metric
    }
    # the seeded benchmark is ingested, so Tab 1 shows real numbers
    assert metrics["Rows total"] > 0
    assert metrics["Accepted"] > 0
    assert metrics["Rejected"] >= 3
    assert metrics["Rows total"] == metrics["Accepted"] + metrics["Rejected"]


def test_benchmark_load_returns_ingestion_reports() -> None:
    from ui.app import load_benchmark

    transactions, accounts, complaints, reports = load_benchmark()
    assert transactions.height > 0 and accounts.height > 0 and complaints.height > 0
    assert set(reports) == {"transactions", "accounts", "complaints"}
    txn_report = reports["transactions"]
    assert txn_report.rows_rejected == 3
    assert set(txn_report.failure_counts) == {
        "unparseable_amount", "missing:txn_id", "duplicate_utr",
    }
    assert txn_report.rows_accepted == transactions.height
    assert txn_report.rows_total == transactions.height + 3
    assert "rejection_reason" in txn_report.rejected_rows.columns
    assert reports["accounts"].rows_rejected == 0
    assert reports["complaints"].rows_rejected == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase3.py -q`
Expected: FAIL — `FileNotFoundError: src/ui/app.py`.

- [ ] **Step 3: Write minimal implementation**

`.streamlit/config.toml`:

```toml
[server]
headless = true

[browser]
gatherUsageStats = false

[theme]
base = "light"
font = "sans serif"
```

`src/ui/app.py` (structure; every rendering block is a small `_render_*`
function taking view models, so no analysis happens in Streamlit callbacks):

```python
"""PraharGraph investigator workbench (offline Streamlit front end)."""

from __future__ import annotations

from typing import Any

import polars as pl
import plotly.graph_objects as go
import streamlit as st
from st_link_analysis import EdgeStyle, NodeStyle, st_link_analysis

from core.contracts import (
    ACCOUNT_METADATA_SCHEMA, COMPLAINT_SCHEMA, TRANSACTION_SCHEMA, RISK_DISCLAIMER,
    IngestionReport,
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
    result = SyndicateDataSimulator(seed=42).generate(background_transactions=background)
    transactions = result.transactions.select(list(TRANSACTION_SCHEMA))
    accounts = result.accounts.select(list(ACCOUNT_METADATA_SCHEMA))
    complaints = result.complaints.select(list(COMPLAINT_SCHEMA))

    first_utr = str(transactions["txn_id"][0])
    ingester = DatasetIngester()
    transactions, txn_report = ingester.ingest_transactions(
        (transactions.write_csv() + _audit_rows(first_utr)).encode("utf-8")
    )
    accounts, acc_report = ingester.ingest_accounts(accounts.write_csv().encode("utf-8"))
    complaints, cmp_report = ingester.ingest_complaints(
        complaints.write_csv().encode("utf-8")
    )
    return transactions, accounts, complaints, {
        "transactions": txn_report,
        "accounts": acc_report,
        "complaints": cmp_report,
    }


def _load_uploaded(file: Any, table: str) -> tuple[pl.DataFrame, IngestionReport]:
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
    uploads = {
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
        frames: dict[str, pl.DataFrame] = {}
        reports: dict[str, IngestionReport] = {}
        for table, file in uploads.items():
            if file is None:
                continue
            frame, report = _load_uploaded(file, table)
            frames[table] = frame
            reports[table] = report          # exactly one report per table
        if "transactions" in frames:
            st.session_state["pg_bundle"] = state.run_pipeline(
                frames["transactions"],
                frames.get("accounts", pl.DataFrame(schema=ACCOUNT_METADATA_SCHEMA)),
                frames.get("complaints", pl.DataFrame(schema=COMPLAINT_SCHEMA)),
                reports=reports,
            )
        else:
            st.sidebar.error("A transactions file is required.")

    bundle: state.PipelineBundle | None = st.session_state.get("pg_bundle")
    if bundle is None:
        st.info("Load the seeded benchmark or upload CSV/XLSX files to begin.")
    return bundle


def main() -> None:
    st.set_page_config(page_title="PraharGraph — Investigator Workbench",
                       page_icon="🛡", layout="wide")
    st.title("PraharGraph — Investigator Workbench")
    st.caption("Offline mule-account investigation for Bihar Police · " + RISK_DISCLAIMER)
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
```

Rendering blocks (each takes view models; `bundle` may be `None`):

```python
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
        st.dataframe([{"header": row.raw_name, "canonical": row.canonical_name,
                       "synonym": row.matched_synonym, "confidence": round(row.confidence, 3)}
                      for row in table.mappings], use_container_width=True, hide_index=True)
        st.dataframe([{"reason": reason, "rows": count}
                      for reason, count in table.failure_counts],
                     use_container_width=True, hide_index=True)
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
    selected = st.selectbox("Inspect account", [row.account_no for row in leaderboard.rows])
    detail = state.build_account_detail(bundle, selected)
    facts_column, indicators_column = st.columns(2)
    with facts_column:
        st.markdown("### Section A — Verified facts")
        st.dataframe([{"Category": fact.category, "Record": fact.label,
                       "Detail": fact.value, "Source": fact.source}
                      for fact in detail.facts], use_container_width=True, hide_index=True)
    with indicators_column:
        st.markdown("### Section B — Analytical indicators")
        st.dataframe([{"Indicator": item.indicator, "Method": item.method,
                       "Value": item.value, "Points": item.points,
                       "Explanation": item.explanation}
                      for item in detail.indicators],
                     use_container_width=True, hide_index=True)
        if detail.sub_scores:
            st.plotly_chart(
                go.Figure(go.Scatterpolar(
                    r=[score.value for score in detail.sub_scores]
                        + [detail.sub_scores[0].value],
                    theta=[score.label for score in detail.sub_scores]
                        + [detail.sub_scores[0].label],
                    fill="toself", name="Sub-scores",
                )).update_layout(polar={"range": [0, 100]}, showlegend=False),
                use_container_width=True,
            )
    if detail.timeline:
        figure = go.Figure()
        for row in detail.timeline:
            figure.add_trace(go.Scatter(
                x=[row.timestamp], y=[row.amount],
                mode="markers+text", text=[row.txn_id], textposition="top center",
                marker={"symbol": "triangle-down" if row.direction == "OUT" else "circle",
                        "color": "#b00020" if row.direction == "OUT" else "#1a7f37",
                        "size": 12},
                name=row.direction, showlegend=False,
                hovertemplate=("%{text}<br>" + row.direction + " " + row.counterparty
                               + "<br>₹%{y:,.2f}<extra></extra>"),
            ))
        st.plotly_chart(figure.update_layout(title="Inflow / outflow timeline",
                                             yaxis_title="Amount (₹)"),
                        use_container_width=True)


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
        st.dataframe([{"Accounts": " → ".join(row.accounts), "UTRs": ", ".join(row.txn_ids),
                       "Total": row.total_amount, "Max decay": row.max_decay_ratio}
                      for row in state.build_cycle_view(bundle).rows],
                     use_container_width=True, hide_index=True)
    with collusion:
        st.dataframe([{"Basis": group.basis, "Key": group.key,
                       "Accounts": ", ".join(group.accounts)}
                      for group in state.build_collusion_view(bundle).groups],
                     use_container_width=True, hide_index=True)
    with terminals:
        st.dataframe([{"Terminal": row.account_no, "Terminal inflow": row.terminal_inflow,
                       "Cash-out legs": row.cashout_count,
                       "Upstream chains": row.terminal_chain_count}
                      for row in state.build_terminal_view(bundle).rows],
                     use_container_width=True, hide_index=True)


def _render_case(bundle: state.PipelineBundle | None) -> None:
    if bundle is None:
        return
    correlation = state.build_correlation_view(bundle)
    st.subheader("Cross-state NCRP correlation")
    st.dataframe([{"ACK A": row.ack_a, "State A": row.state_a,
                   "ACK B": row.ack_b, "State B": row.state_b,
                   "Basis": row.link_basis, "Interstate": row.interstate}
                  for row in correlation.rows], use_container_width=True, hide_index=True)
    st.dataframe([{"Shared account/terminal": hotspot.key, "ACKs": ", ".join(hotspot.ack_nos)}
                  for hotspot in correlation.hotspots],
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Case evidence dossier")
    scope = st.radio("Dossier scope", ["Account", "Cluster", "Complaint ACK"], horizontal=True)
    dossier = None
    if scope == "Account":
        account_nos = [row.account_no for row in state.build_leaderboard(bundle).rows[:200]]
        if account_nos:
            dossier = state.build_account_dossier(
                bundle, st.selectbox("Account", account_nos, key="dossier_account"))
    elif scope == "Cluster":
        account_nos = [row.account_no for row in state.build_leaderboard(bundle).rows[:20]]
        chosen = st.multiselect("Cluster accounts", account_nos, key="dossier_cluster")
        if chosen:
            dossier = state.build_cluster_dossier(bundle, chosen)
    else:
        ack_nos = sorted(set(bundle.complaints["ack_no"].to_list()))
        if ack_nos:
            dossier = state.build_complaint_dossier(
                bundle, st.selectbox("Complaint ACK", ack_nos, key="dossier_ack"))
    if dossier is None:
        st.info("Select a subject to preview the dossier.")
        return
    st.markdown(state.dossier_markdown(dossier))
    st.download_button("Download .md", data=state.dossier_markdown(dossier).encode(),
                       file_name=f"{dossier.case_reference}.md", mime="text/markdown")
    st.download_button("Download .html", data=state.dossier_html(dossier),
                       file_name=f"{dossier.case_reference}.html", mime="text/html")
    st.download_button("Download .json", data=state.dossier_json(dossier),
                       file_name=f"{dossier.case_reference}.json", mime="application/json")
```

Two small module-level wrappers keep rendering free of generator plumbing (add
them to `ui/state.py`, unit-testable):

```python
def dossier_markdown(dossier: InvestigationDossier) -> str:
    return InvestigationDossierGenerator([]).render_markdown(dossier)


def dossier_html(dossier: InvestigationDossier) -> bytes:
    return InvestigationDossierGenerator([]).render_html(dossier)


def dossier_json(dossier: InvestigationDossier) -> bytes:
    return InvestigationDossierGenerator([]).render_json(dossier)
```

(`render_*` accept a dossier argument, so an empty generator is just a
formatter — no state, no config, no side effects.)

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_phase3.py -q
uv run pytest -q && uv run mypy src && uv run ruff check src tests
PYTHONPATH=src uv run streamlit run src/ui/app.py --server.headless true --server.port 8501
```

Manual acceptance for the last command (browser at `http://localhost:8501`,
local only): (a) load the benchmark, (b) Tab 1 shows accepted/rejected counts,
(c) Tab 2 ranks HUB-like accounts and shows separated facts/indicators,
(d) Tab 3 traces an ACK and hovers an edge showing UTR + taint + hop,
(e) Tab 4 downloads all three dossier formats.

- [ ] **Step 5: Commit**

```bash
git add src/ui/app.py .streamlit/config.toml tests/test_phase3.py
git commit -m "phase3(9): add offline streamlit workbench with four investigation tabs"
```

---

## 4. Acceptance checklist

- [ ] Dossier builds for an account, a cluster and a complaint chain; all three
      formats download and are byte-identical across runs.
- [ ] Section A contains only ledger / account-master / NCRP / generator rows;
      Section B contains every score, sub-score, SHAP and taint statement.
- [ ] Requisition cites Section 94 BNSS, groups accounts by IFSC prefix, lists
      each bank's own accounts and UTRs, and keeps Section 106 / CFCFRMS
      explicitly separate.
- [ ] HTML has no external reference (`http`, `https`, `<script`).
- [ ] Every value is escaped in HTML and Markdown.
- [ ] Rejected-rows CSV downloads with `rejection_reason` intact.
- [ ] Leaderboard, timeline, canvas elements and views are deterministic and
      unit-tested; the app renders headlessly with `AppTest`.
- [ ] Clicking "Load seeded Bihar benchmark dataset" ingests the frames, so
      Tab 1 reports non-zero accepted/rejected rows, real column mappings and a
      downloadable rejected-rows CSV (`rows_total == accepted + rejected`).
- [ ] `state.run_pipeline` runs only on an explicit load click: the finished
      `PipelineBundle` lives in `st.session_state["pg_bundle"]`, so slider and
      selectbox reruns never re-fit IsolationForest/TreeSHAP or rebuild DuckDB
      and NetworkX state.
- [ ] `uv run pytest -q`, `uv run mypy src`, `uv run ruff check src tests` green.
- [ ] No Streamlit import in `src/ui/state.py`, no analysis in `src/ui/app.py`
      callbacks.

## 5. Out of scope

Threshold sliders that change feature computation (config is fixed per run for
reproducibility), authentication, persistence, live feeds, PDF generation,
`datetime.now()` stamps, or any change to Phase 1/2 scoring behaviour beyond the
additive fields in Task 1.
