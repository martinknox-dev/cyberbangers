# PraharGraph Phase 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish PraharGraph's Phase 1 core: Pydantic + Polars data contracts, a fuzzy-schema CSV/XLSX ingester with rejection reporting, and a deterministic ground-truth synthetic fraud simulator — all TDD-built, offline, and `mypy` strict-clean.

**Architecture:** Ingestion is multi-table: it reads messy CSVs/XLSX workbooks into all-`String` Polars frames for three judge-supplied table kinds — transactions, NCRP complaints, and account metadata — claiming columns via RapidFuzz `token_sort_ratio` against a curated synonym dictionary per table, normalizing dates/amounts/modes with vectorized `Expr` chains (zero data-row loops), and splitting rows into accepted/rejected via a priority-ordered first-match-wins reason column plus keep-first primary-key dedupe. The simulator produces a legitimate transaction background plus five hand-shaped fraud topologies carrying boolean ground-truth labels — every scheme sender/receiver resolves to a numeric, attribute-rich account-master row (no orphaned or letter-coded accounts) — plus a complaint chain whose disputed UTRs join back to real inflow transactions, all from a seeded NumPy generator so two runs are frame-identical.

**Tech Stack:** Python 3.12 · Polars 2.0.0 · Pydantic 2.14.1 · RapidFuzz 3.14.6 · NumPy (transitive via scikit-learn) · openpyxl 3.1.5 (Polars XLSX engine) · pytest 9.1.1 · mypy 2.4.0 (strict) · ruff 0.17.

**Spec:** `docs/plans/phase1-foundation.md` — this document is the spec; Task 1 commits it verbatim to that path (it currently lives at `~/.opencode/plan/phase1-foundation.md` while this is plan mode later promoted to that path). Every later task argues from this file.

## Global Constraints

- Pure offline execution. No external network calls anywhere in `src/` or `tests/` (environment/dependency resolution at setup time is not program-issued I/O; the stack is already locked in `uv.lock`).
- Use Polars idioms throughout ingestion; no `iter_rows()/iter_slices()`, no row-wise Python loops over dataframes, no Pandas conversions. All dataframe transforms are expression-based. Python list-comprehensions are allowed only for building *config-scale* materialization (≤1,000 build-time rows, e.g. account master records) inside the simulator module entry points.
- DuckDB is deliberately deferred to Phase 2 (graph/bikram/risk consume it); this phase is Polars-only. Do not add runtime dependencies.
- Strict typing: `uv run mypy src` exits clean (0 errors) with `strict = true` + the single openpyxl override. No `# type: ignore`.
- Reproducibility: same seed + same parameters ⇒ identical frames (`.equals()`), across processes and machines. No wall-clock reads, no OS-RNG, no dict/set iteration luck, no local-timezone dependence (the base epoch is computed from a fixed UTC datetime literal pattern).
- Identifier-like values (account numbers, IFSC, IMEI, pincodes, mobiles) are always `pl.String`/`str` to preserve leading zeros; only `amount` is Float64 and timestamps are `pl.Datetime("us")` (naive, ledger-style).
- Stack floors in `pyproject.toml` are floors, not targets — do not bump/rewrite dependencies in this phase.
- Commit after every task, message format `phase1(N): <what>`.
- `[project.scripts] prahar-graph = "prahar_graph:main"` in `pyproject.toml` is an inert leftover (module does not exist) — leave untouched this phase; the CLI lands later.

## Locked Design Decisions (reviewed; do not relitigate)

1. **Fuzzy matching** = `rapidfuzz.fuzz.token_sort_ratio` (spec-mandated) over normalized headers (`lowercase`, each non-alphanumeric run → one space). Claim threshold: confidence ≥ 0.80 (inclusive; `80.0/100 ≥ 0.80 - 1e-9` so `"imei_device"`→0.80 counts).
2. **Date precedence:** DD/MM/YYYY before MM/DD/YYYY (Indian banking context). Ambiguous `05/06/2024` ⇒ 5 June 2024; `12/31/2024` falls through to MM/DD ⇒ Dec 31. ISO and epoch detection first.
3. **Mandatory fields** (row rejected `missing:<field>` if absent/empty), per table: transactions = `txn_id`, `sender_account`, `receiver_account`, `amount`, `timestamp`, `payment_mode`; complaints = `ack_no`, `victim_account`, `fraud_category`, `reported_timestamp`, `disputed_amount`, `state`; accounts = `account_no`, `ifsc`, `branch_code`, `account_type`. IFSC/device/IP/suspect-account/disputed-UTR optional; IFSC present-but-malformed ⇒ `invalid_ifsc` (transactions *and* accounts tables).
4. **Rejection policy:** first-match-wins over fixed priority (missing → amount → timestamp → ifsc → mode → duplicate). Each rejected row carries exactly one machine-readable reason.
5. **Duplicate UTRs:** dedupe only rows that survived field rules; keep first occurrence in raw file order; later ones rejected `duplicate_utr`.
6. **Zero amounts:** accepted (spec says *non-negative*; only `< 0` rejects with `negative_amount`).
7. **`rejected_rows` columns** are the *original source headers* (not canonical) + `rejection_reason` — the audit preserves source fidelity.
8. **Ground truth:** transaction-level `is_fraud: bool` + nullable `fraud_scheme: str|null` appended next to canonical columns (`LABELED_TRANSACTION_SCHEMA`).
9. **CSV ragged rows** tolerated (`truncate_ragged_lines=True`): extra trailing cells are dropped by the reader and the (clean) row proceeds to normal validation; the ingest must never crash on shape errors.
10. **XLSX** = first worksheet only (`pl.read_excel(..., engine="openpyxl")`); every cell is cast to String before normalization so numeric/datetime native cells share the text pipeline (Excel-native datetimes serialize as `2024-01-02 00:00:00.000000`, which the `%.f` timestamp format catches).
11. **Seven payment modes:** `UPI, IMPS, NEFT, RTGS, AEPS, ATM, CASH` (real Indian bank dumps carry RTGS transfers and ATM withdrawals). Alias table `MODE_CANONICAL` additionally maps `BHIM UPI`→UPI, `UPI P2P`/`UPI/P2P`/`UPI/P2M`→UPI, `AEPS CASH OUT`→AEPS, `MICRO ATM`→AEPS (micro-ATM rails are AEPS), `CASH DEPOSIT`/`CASHOUT`→CASH, `ATM WITHDRAWAL`→ATM. Unknown modes (e.g. `CHEQUE`) still reject.
12. **Multi-table ingestion:** one `DatasetIngester`, three entries — `ingest_transactions()` (+ `.ingest()`-with/`ingest_csv()`/`ingest_xlsx()` default), `ingest_complaints()`, `ingest_accounts()`. Per-table synonym tables (`CANONICAL_SYNONYMS`, `COMPLAINT_SYNONYMS`, `ACCOUNT_SYNONYMS` behind a `TABLE_SYNONYMS` registry keyed by `TableKind = Literal["transactions","complaints","accounts"]`), per-table canonical/schema/mandatory config, and per-table dedupe: transactions keep first by `txn_id` (`duplicate_utr`), complaints by `ack_no` (`duplicate_ack_no`), accounts by `account_no` (`duplicate_account_no`).
13. **No orphaned or named scheme accounts:** every sender/receiver of every emitted transaction — background *and* all five schemes — must resolve to a row in `accounts`. Scheme participants get realistic numeric account numbers from the reserved suffix block `≥ 90_000_000` (`"29" + 8 digits`, disjoint from background retail `00000000..00000899` and merchant `5500..…`) via one allocator `_scheme_account(block, seq)`; no letter-coded ids like `HUB_001`/`LOOP_0_A`. Master rows carry attribute truth for Phase-2 signals (e.g. rapid hubs opened ≥ 180 days before base with PMJDY/SAVINGS type; colluders share branch/IMEI).
14. **Complaint ground truth joins:** 16 NCRP rows (8 rapid pass-through, 5 fan-out, 3 AEPS episodes), each with a non-null `disputed_utr` that is a real transaction id — the initial victim→mule inflow leg — where `victim_account` == that transaction's `sender_account` and `suspect_account` == its `receiver_account` (post-join-verifiable). States rotate across Bihar, Delhi, Maharashtra, Tamil Nadu, Jharkhand.

## Review Focus

The five failure modes the spec implies that no single core test covered naturally — each is pinned by the named test:

1. **Ambiguous slash-dates** (`05/06/2024`): must parse as 5 June (DD/MM precedence). → Task 4 `test_dd_mm_precedes_mm_dd`.
2. **Zero-amount rows**: accepted, counted in rows_accepted. → Task 5 `test_zero_amount_accepted`.
3. **Header-only/empty CSV**: empty conforming frame + zeroed report, never raises. → Task 5 `test_header_only_csv_returns_empty_report`.
4. **Ragged CSV row** (extra comma): file still ingests; truncated extras do not crash and do not corrupt the row. → Task 5 `test_ragged_csv_rows_do_not_crash`.
5. **Canonical mapping collision**: higher confidence wins; exact tie keeps the first-seen raw column; the loser keeps its score with `canonical_name=None`. → Task 3 `test_canonical_collision_*` pair.

---

## Interfaces (single source of truth for every task)

From `src/core/contracts.py`:

```python
PaymentMode = Literal["UPI", "IMPS", "NEFT", "RTGS", "AEPS", "ATM", "CASH"]
AccountType = Literal["SAVINGS", "PMJDY", "CURRENT"]
TableKind = Literal["transactions", "complaints", "accounts"]

IFSC_PATTERN: re.Pattern[str]           # compiled ^[A-Z]{4}0[0-9A-Z]{6}$
MANDATORY_COLUMNS: tuple[str, ...]      # transactions' 6 fields, see Decision 3
TRANSACTION_SCHEMA: dict[str, pl.DataType]     # ordered; String ids, Float64 amount, Datetime("us")
ACCOUNT_METADATA_SCHEMA: dict[str, pl.DataType]
COMPLAINT_SCHEMA: dict[str, pl.DataType]       # incl. disputed_utr (link to transaction)
TRANSACTION_COLUMNS: tuple[str, ...]    # canonical column order
COMPLAINT_COLUMNS: tuple[str, ...]      # derived tuple(COMPLAINT_SCHEMA)
ACCOUNT_COLUMNS: tuple[str, ...]        # derived tuple(ACCOUNT_METADATA_SCHEMA)
MANDATORY_BY_TABLE: dict[TableKind, tuple[str, ...]]   # Decision 3 per-table sets

class Transaction(BaseModel): ...       # see Task 2 block
class AccountMetadata(BaseModel): ...
class Complaint(BaseModel): ...         # + disputed_utr: str | None
class ColumnMapping(BaseModel): ...     # raw_name, canonical_name?, matched_synonym?, confidence∈[0,1]
class IngestionReport(BaseModel): ...   # source?, rows_total, rows_accepted, rows_rejected,
                                        # failure_counts, column_mapping, rejected_rows: pl.DataFrame
```

From `src/ingestion/mapper.py`:

```python
DEFAULT_CONFIDENCE_THRESHOLD: float     # 0.80
CANONICAL_SYNONYMS: dict[str, tuple[str, ...]]    # transactions
COMPLAINT_SYNONYMS: dict[str, tuple[str, ...]]    # complaints
ACCOUNT_SYNONYMS: dict[str, tuple[str, ...]]      # accounts
TABLE_SYNONYMS: dict[TableKind, dict[str, tuple[str, ...]]]
TS_FORMATS: tuple[str, ...]
MODE_CANONICAL: dict[str, str]          # 7 canonical modes + aliases (Decision 11)

def normalize_header(name: str) -> str
def map_headers(
    raw_columns: Sequence[str],
    confidence_threshold: float = 0.80,
    table: TableKind = "transactions",
) -> dict[str, ColumnMapping]

class DatasetIngester:
    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None
    def ingest(self, source: str | Path | bytes,
               *, table: TableKind = "transactions") -> tuple[pl.DataFrame, IngestionReport]
    def ingest_csv(self, source: str | Path | bytes,
                   *, table: TableKind = "transactions") -> tuple[pl.DataFrame, IngestionReport]  # explicit reader
    def ingest_xlsx(self, source: str | Path | bytes,
                    *, table: TableKind = "transactions") -> tuple[pl.DataFrame, IngestionReport]
    def ingest_transactions(self, source: str | Path | bytes) -> tuple[pl.DataFrame, IngestionReport]
    def ingest_complaints(self, source: str | Path | bytes) -> tuple[pl.DataFrame, IngestionReport]
    def ingest_accounts(self, source: str | Path | bytes) -> tuple[pl.DataFrame, IngestionReport]
```

`ingest_transactions` is thin sugar over `ingest(source, table="transactions")` (which is also `.ingest()`'s default — the alias the judges' transaction dumps hit); `ingest_complaints`/`ingest_accounts` likewise.

From `src/simulator/generator.py`:

```python
SCHEME_RAPID_PASS_THROUGH/FAN_OUT_SMURFING/AEPS_TERMINAL_CASH_OUT/BRANCH_COLLUSION/CIRCULAR_WASH: str  # labels
SCHEME_LABELS: tuple[str, ...]
SCHEME_TXN_COUNTS: dict[str, int]           # {scheme_label: exact txn count emitted}
LABELED_TRANSACTION_SCHEMA: dict[str, pl.DataType]  # TRANSACTION_SCHEMA + is_fraud/fraud_scheme

class SimulationResult(BaseModel):          # transactions / accounts / complaints frames
class SyndicateDataSimulator:
    def __init__(self, seed: int = 42) -> None
    def generate(self, background_transactions: int = 12_000) -> SimulationResult  # 5_000..20_000 validated
```

Import style: top-level absolute (`from core.contracts import ...`). The repo's editable install (`.pth` → `src/`) makes `src` importable under `uv run`; pytest additionally sets `pythonpath = ["src"]` (Task 1).

### Verification baseline (all confirmed live in this environment before writing this plan)

- Polars 2.0.0 / DuckDB 1.5.6 / Pydantic 2.14.1 / RapidFuzz 3.14.6 / networkx 3.7 / openpyxl 3.1.5 / mypy 2.4.0 / pytest 9.1.1.
- `pl.read_csv(..., infer_schema_length=0)` ⇒ every column String, from path or `BytesIO`; `truncate_ragged_lines` parameter exists.
- `pl.read_excel(BytesIO, engine="openpyxl")` works (calamine/fastexcel is NOT installed — never use the default engine); openpyxl engine + floating/numeric/datetime cells all coerce through our String-cast.
- `Expr.str.strptime(pl.Datetime, fmt, strict=False)` chains under `pl.coalesce` parse mixed formats and yield null on garbage; `%.f` and `%I/%p` variants parse; `05/06/2024` under `%d/%m/%Y` ⇒ 2024-06-05.
- `pl.from_epoch(col_int64, time_unit="ms"/"s")` works inside `pl.when`; threshold split at `10**12` (ms) vs `10**9..10**12` (s).
- `Expr.replace_strict(dict, default=None)` maps unknowns to null; `Expr.is_first_distinct()` exists.
- `mypy --strict` passes every used library — the only error source in a probe was `openpyxl` missing stubs, fixed by the `[[tool.mypy.overrides]]` block (verified: 0 issues).
- Fuzzy scores for all five test header styles (three transaction styles + the complaint and account styles) were computed with this exact algorithm (probe run; probe mirrors `normalize_header` — non-alphanumeric runs → single space): complete coverage at ≥ 0.80 with the specific confidences baked into Task 3 tests, plus verified noise/distractor rejections (`Narration` 52.2, `SenderIFSC` 75.0, `ipaddress` 73.7, …). Complaint-style worst margins: `Complaint Ack No`→ack_no 0.8966, `Disputed UTR Ref`→disputed_utr 0.8571; account-style worst margins: `Home Branch Code`→branch_code 0.8148, `Date of Opening`→account_opening_date 0.8889 — all verified with runner-up collisions far below.
- Multi-table/date/pattern expression probes: reuse of the `TS_FORMATS` coalesce chain + `.cast(pl.Date, strict=False)` parses `15/01/2023` (DD/MM precedence), `2023-01-15`, openpyxl-native `2023-01-15 00:00:00.000000`, and the `12/25/2023` MM/DD fallback to the correct dates with null for garbage; `_pattern_expr` (strip → `str.contains(pattern)` → keep-or-null) validates mobile `^\d{10,12}$`/imei `^\d{15}$`/pincode `^\d{6}$` incl. whitespace-tolerant pass and letter-suffix fail; `ACCOUNT_TYPE_CANONICAL` (`SB`, `SB ACCOUNT`, `SAVINGS ACCOUNT`→SAVINGS; `CA`, `CURRENT ACCOUNT`→CURRENT) maps via `replace_strict`, unknowns (`JOINT`) → null.
- Reduced-scale expression-graph probes reproduced the exact accepted/rejected splits, priority-wins rule, keep-first dedupe, comma-numeric amounts (`"1,20,000.50"` → 120000.5), mode aliasing (`BHIM UPI` → UPI), epoch stamp parsing, and zero-amount acceptance described in the Locked Decisions.
- Generator mechanics probe: seeded `np.random.default_rng`, polars-native `concat_str`/`pad_start`/`int_range` id formatting, `pl.from_epoch` timestamp column construction, schema-cast via `df.cast(mapping)`, `pl.concat` + sort determinism, and `DataFrame.equals` reproducibility across instances all verified.

---

### Task 1: Plan doc, pytest + mypy project config

**Files:**
- Create: `docs/plans/phase1-foundation.md` (this plan, verbatim)
- Modify: `pyproject.toml` (append tool tables only — no dependency changes, `uv.lock` untouched)

**Interfaces:**
- Consumes: nothing.
- Produces: pytest `pythonpath = ["src"]` config (every later test writes `from core…` imports), mypy strict config verified clean; the spec doc at its repo path.

- [ ] **Step 1: Create `docs/plans/phase1-foundation.md`** — copy the reviewed plan file verbatim:

```bash
mkdir -p docs/plans
cp ~/.opencode/plan/phase1-foundation.md docs/plans/phase1-foundation.md
```

- [ ] **Step 2: Append tool config to `pyproject.toml`** (append at end of file):

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.mypy]
python_version = "3.12"
strict = true
mypy_path = "src"

[[tool.mypy.overrides]]
module = "openpyxl.*"
ignore_missing_imports = true
```

- [ ] **Step 3: Verify tooling smoke-checks (read-only runs)**

Run: `uv run pytest --collect-only -q` → expect `no tests ran` (exit code 5; config parsed, correct testpaths).
Run: `uv run mypy src` → expect `Success: no issues found in 8 source files`.
Run: `uv run python -c "import core; print('ok')"` → expect `ok`.

- [ ] **Step 4: Commit**

```bash
git add docs/plans/phase1-foundation.md pyproject.toml
git commit -m "phase1(1): commit TDD plan doc and pytest/mypy strict config"
```

---

### Task 2: `core/contracts.py` — Pydantic models + Polars schemas

**Files:**
- Create: `src/core/contracts.py`
- Test: `tests/test_phase1.py` (first section: `TestContracts`)

**Interfaces:**
- Consumes: nothing.
- Produces (exact names later tasks import): `PaymentMode`, `AccountType`, `TableKind`, `IFSC_PATTERN`, `MANDATORY_COLUMNS`, `MANDATORY_BY_TABLE`, `TRANSACTION_COLUMNS`, `COMPLAINT_COLUMNS`, `ACCOUNT_COLUMNS`, `TRANSACTION_SCHEMA`, `ACCOUNT_METADATA_SCHEMA`, `COMPLAINT_SCHEMA`, `Transaction`, `AccountMetadata`, `Complaint`, `ColumnMapping`, `IngestionReport`.

- [ ] **Step 1: Write failing tests** — create `tests/test_phase1.py`:

```python
"""Phase 1 acceptance tests: contracts, fuzzy ingestion, deterministic simulator."""
import json
from datetime import date, datetime

import polars as pl
import pytest
from pydantic import ValidationError

from core.contracts import (
    ACCOUNT_COLUMNS,
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_COLUMNS,
    COMPLAINT_SCHEMA,
    IFSC_PATTERN,
    MANDATORY_BY_TABLE,
    MANDATORY_COLUMNS,
    TRANSACTION_COLUMNS,
    TRANSACTION_SCHEMA,
    AccountMetadata,
    ColumnMapping,
    Complaint,
    IngestionReport,
    Transaction,
)

VALID_TXN_ROW: dict[str, object] = {
    "txn_id": "UTR2024060100001",
    "sender_account": "2900000001",
    "sender_ifsc": "SBIN0001234",
    "receiver_account": "2900000002",
    "receiver_ifsc": "SBIN0009876",
    "amount": 15000.0,
    "timestamp": datetime(2024, 6, 1, 10, 30, 0),
    "payment_mode": "UPI",
    "device_id": "D000123",
    "ip_address": "10.20.30.40",
}


class TestContracts:
    def test_valid_transaction_row_validates(self) -> None:
        txn = Transaction.model_validate(VALID_TXN_ROW)
        assert txn.amount == 15000.0
        assert txn.payment_mode == "UPI"

    def test_transaction_rejects_negative_amount(self) -> None:
        with pytest.raises(ValidationError):
            Transaction.model_validate({**VALID_TXN_ROW, "amount": -1.0})

    def test_transaction_rejects_bad_payment_mode(self) -> None:
        with pytest.raises(ValidationError):
            Transaction.model_validate({**VALID_TXN_ROW, "payment_mode": "VISA-KEEP"})

    def test_transaction_rejects_empty_txn_id(self) -> None:
        with pytest.raises(ValidationError):
            Transaction.model_validate({**VALID_TXN_ROW, "txn_id": ""})

    def test_account_metadata_constraints(self) -> None:
        ok = AccountMetadata(
            account_no="2900000001",
            ifsc="SBIN0001234",
            account_opening_date=date(2023, 1, 15),
            branch_code="BR001",
            mobile_no="9178451234",
            imei="351104510123456",
            kyc_pincode="800013",
            account_type="PMJDY",
        )
        assert ok.account_type == "PMJDY"
        with pytest.raises(ValidationError):
            AccountMetadata(
                account_no="1", ifsc="BAD", branch_code="B", account_type="JOINT"  # type: ignore[arg-type]
            )

    def test_complaint_roundtrip_json(self) -> None:
        c = Complaint(
            ack_no="ACK20240001",
            victim_account="2900000007",
            suspect_account="5500000001",
            fraud_category="AEPS_CASHOUT",
            reported_timestamp=datetime(2024, 6, 2, 9, 0, 0),
            disputed_amount=32000.0,
            disputed_utr="UTR2024060100001",
            state="Bihar",
        )
        dumped = json.loads(c.model_dump_json())
        assert dumped["disputed_amount"] == 32000.0
        assert dumped["disputed_utr"] == "UTR2024060100001"

    def test_complaint_disputed_utr_optional(self) -> None:
        with_utr = Complaint(
            ack_no="ACK20240002",
            victim_account="2900000007",
            fraud_category="UPI_FRAUD",
            reported_timestamp=datetime(2024, 6, 2, 9, 0, 0),
            disputed_amount=1500.0,
            state="Bihar",
        )
        assert with_utr.disputed_utr is None   # nullable, links only when known

    def test_report_defaults_hold_dataframe(self) -> None:
        report = IngestionReport()
        assert report.rows_total == 0 and report.rows_accepted == 0
        assert report.rejected_rows.is_empty()

    def test_ifsc_pattern_is_strict(self) -> None:
        assert IFSC_PATTERN.fullmatch("SBIN0001234") is not None
        assert IFSC_PATTERN.fullmatch("sbin0001234") is None
        assert IFSC_PATTERN.fullmatch("SBIN00123") is None

    def test_transaction_schema_is_string_first(self) -> None:
        assert list(TRANSACTION_COLUMNS) == [
            "txn_id", "sender_account", "sender_ifsc", "receiver_account",
            "receiver_ifsc", "amount", "timestamp", "payment_mode",
            "device_id", "ip_address",
        ]
        assert TRANSACTION_SCHEMA["amount"] == pl.Float64
        assert TRANSACTION_SCHEMA["timestamp"] == pl.Datetime("us")
        for c in TRANSACTION_SCHEMA:
            if c not in ("amount", "timestamp"):
                assert TRANSACTION_SCHEMA[c] == pl.String, c

    def test_account_and_complaint_schemas(self) -> None:
        assert [ACCOUNT_METADATA_SCHEMA[c] for c in ("account_opening_date", "account_type")] == [
            pl.Date, pl.String,
        ]
        assert list(COMPLAINT_SCHEMA) == [
            "ack_no", "victim_account", "suspect_account", "fraud_category",
            "reported_timestamp", "disputed_amount", "disputed_utr", "state",
        ]
        assert list(COMPLAINT_COLUMNS) == list(COMPLAINT_SCHEMA)
        assert list(ACCOUNT_COLUMNS) == list(ACCOUNT_METADATA_SCHEMA)
        assert [COMPLAINT_SCHEMA[c] for c in ("reported_timestamp", "disputed_amount")] == [
            pl.Datetime("us"), pl.Float64,
        ]

    def test_mandatory_columns_subset_of_schema(self) -> None:
        assert MANDATORY_COLUMNS == (
            "txn_id", "sender_account", "receiver_account",
            "amount", "timestamp", "payment_mode",
        )
        assert set(MANDATORY_COLUMNS) <= set(TRANSACTION_SCHEMA)
        assert MANDATORY_BY_TABLE == {
            "transactions": MANDATORY_COLUMNS,
            "complaints": (
                "ack_no", "victim_account", "fraud_category",
                "reported_timestamp", "disputed_amount", "state",
            ),
            "accounts": ("account_no", "ifsc", "branch_code", "account_type"),
        }
        assert set(MANDATORY_BY_TABLE["complaints"]) <= set(COMPLAINT_SCHEMA)
        assert set(MANDATORY_BY_TABLE["accounts"]) <= set(ACCOUNT_METADATA_SCHEMA)

    def test_column_mapping_confidence_bounds(self) -> None:
        m = ColumnMapping(raw_name="Payee", canonical_name="receiver_account",
                          matched_synonym="payee", confidence=1.0)
        assert m.confidence == 1.0
        with pytest.raises(ValidationError):
            ColumnMapping(raw_name="x", canonical_name=None, confidence=1.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase1.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'core.contracts'`.

- [ ] **Step 3: Write `src/core/contracts.py`**

```python
"""Phase 1 data contracts: Pydantic models and canonical Polars schemas.

Single source of truth for every entity flowing through PraharGraph.
Identifier-like values (accounts, IFSC, IMEI, pins, mobiles) are strings to
preserve leading zeros; only amount is Float64 and timestamps are
Datetime("us") (naive, ledger-style).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

PaymentMode = Literal["UPI", "IMPS", "NEFT", "RTGS", "AEPS", "ATM", "CASH"]
AccountType = Literal["SAVINGS", "PMJDY", "CURRENT"]
TableKind = Literal["transactions", "complaints", "accounts"]

IFSC_PATTERN = re.compile(r"^[A-Z]{4}0[0-9A-Z]{6}$")

MANDATORY_COLUMNS: tuple[str, ...] = (
    "txn_id",
    "sender_account",
    "receiver_account",
    "amount",
    "timestamp",
    "payment_mode",
)

TRANSACTION_SCHEMA: dict[str, pl.DataType] = {
    "txn_id": pl.String,
    "sender_account": pl.String,
    "sender_ifsc": pl.String,
    "receiver_account": pl.String,
    "receiver_ifsc": pl.String,
    "amount": pl.Float64,
    "timestamp": pl.Datetime("us"),
    "payment_mode": pl.String,
    "device_id": pl.String,
    "ip_address": pl.String,
}

TRANSACTION_COLUMNS: tuple[str, ...] = tuple(TRANSACTION_SCHEMA)

ACCOUNT_METADATA_SCHEMA: dict[str, pl.DataType] = {
    "account_no": pl.String,
    "ifsc": pl.String,
    "account_opening_date": pl.Date,
    "branch_code": pl.String,
    "mobile_no": pl.String,
    "imei": pl.String,
    "kyc_pincode": pl.String,
    "account_type": pl.String,
}

COMPLAINT_SCHEMA: dict[str, pl.DataType] = {
    "ack_no": pl.String,
    "victim_account": pl.String,
    "suspect_account": pl.String,
    "fraud_category": pl.String,
    "reported_timestamp": pl.Datetime("us"),
    "disputed_amount": pl.Float64,
    "disputed_utr": pl.String,
    "state": pl.String,
}

COMPLAINT_COLUMNS: tuple[str, ...] = tuple(COMPLAINT_SCHEMA)
ACCOUNT_COLUMNS: tuple[str, ...] = tuple(ACCOUNT_METADATA_SCHEMA)

MANDATORY_BY_TABLE: dict[TableKind, tuple[str, ...]] = {
    "transactions": MANDATORY_COLUMNS,
    "complaints": (
        "ack_no",
        "victim_account",
        "fraud_category",
        "reported_timestamp",
        "disputed_amount",
        "state",
    ),
    "accounts": ("account_no", "ifsc", "branch_code", "account_type"),
}


class Transaction(BaseModel):
    """One financial transaction (ledger-level ingest record)."""

    txn_id: str = Field(min_length=1)  # unique UTR/reference
    sender_account: str = Field(min_length=1)
    sender_ifsc: str | None = Field(default=None, pattern=IFSC_PATTERN.pattern)
    receiver_account: str = Field(min_length=1)
    receiver_ifsc: str | None = Field(default=None, pattern=IFSC_PATTERN.pattern)
    amount: float = Field(ge=0)
    timestamp: datetime
    payment_mode: PaymentMode
    device_id: str | None = None
    ip_address: str | None = None


class AccountMetadata(BaseModel):
    """Customer/master-level account attributes (KYC-style)."""

    account_no: str = Field(min_length=1)
    ifsc: str = Field(pattern=IFSC_PATTERN.pattern)
    account_opening_date: date | None = None
    branch_code: str = Field(min_length=1)
    mobile_no: str | None = Field(default=None, pattern=r"^\d{10,12}$")
    imei: str | None = Field(default=None, pattern=r"^\d{15}$")
    kyc_pincode: str | None = Field(default=None, pattern=r"^\d{6}$")
    account_type: AccountType


class Complaint(BaseModel):
    """Cyber-crime acknowledgment (NCRP-lite) linking victim to suspect."""

    ack_no: str = Field(min_length=1)
    victim_account: str = Field(min_length=1)
    suspect_account: str | None = None
    fraud_category: str = Field(min_length=1)
    reported_timestamp: datetime
    disputed_amount: float = Field(ge=0)
    disputed_utr: str | None = None   # transaction reference binding the dispute to the ledger
    state: str = Field(min_length=1)


class ColumnMapping(BaseModel):
    """Fuzzy header-to-canonical mapping result for one source column."""

    raw_name: str
    canonical_name: str | None = None
    matched_synonym: str | None = None
    confidence: float = Field(ge=0, le=1)


class IngestionReport(BaseModel):
    """Machine-readable audit of one `DatasetIngester` run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str | None = None
    rows_total: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0
    failure_counts: dict[str, int] = Field(default_factory=dict)
    column_mapping: dict[str, ColumnMapping] = Field(default_factory=dict)
    rejected_rows: pl.DataFrame = pl.DataFrame()
```

Two deliberate details (do not "fix"):
- `rejected_rows: pl.DataFrame = pl.DataFrame()` — pydantic needs the default because the frame is optional in a fresh report; `ConfigDict(arbitrary_types_allowed=True)` permits the Polars type.
- `account_type="JOINT"` in the negative test needs `# type: ignore[arg-type]` — a Literal-typed field can't accept an invalid value without losing mypy strictness in the test. Test files stay strict-clean.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase1.py -q` → all PASS (`TestContracts` is the only section yet).
Run: `uv run mypy src` → `Success: no issues found in 9 source files`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_phase1.py src/core/contracts.py
git commit -m "phase1(2): pydantic contracts and canonical polars schemas"
```

---

### Task 3: `ingestion/mapper.py` — synonym table + fuzzy header mapping

**Files:**
- Create: `src/ingestion/mapper.py` (mapping half only; readers, normalization and `DatasetIngester` arrive in Task 4)
- Test: `tests/test_phase1.py` (add `TestHeaderMapping`)

**Interfaces:**
- Consumes: `ColumnMapping` from `core.contracts`.
- Produces (Task 4/5 rely on exact names): `CANONICAL_SYNONYMS` (transactions), `COMPLAINT_SYNONYMS`, `ACCOUNT_SYNONYMS`, `TABLE_SYNONYMS`, `DEFAULT_CONFIDENCE_THRESHOLD`, `normalize_header(name: str) -> str`, `map_headers(raw_columns: Sequence[str], confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD, table: TableKind = "transactions") -> dict[str, ColumnMapping]` — keyed by the reader's original raw column names; a losing claimant stays present with `canonical_name=None` and its own confidence.

Every expected mapping/confidence in the tests (transaction styles *and* the complaint/account styles, plus the cross-table silo checks) was *computed* with this exact algorithm before being written here (probe in Verification baseline). Do not edit any synonym table without re-running the probe.

- [ ] **Step 1: Write failing tests** — append to `tests/test_phase1.py`:

```python
from ingestion.mapper import map_headers, normalize_header

STYLE_A_HEADERS: list[str] = [   # core-banking CSV export (bank style, underscored codes)
    "UTR", "Debit_Acc", "Fr_IFSC", "PAYEE", "To_IFSC", "Amount_INR",
    "Txn_Time", "Mode", "Device", "IP",
]
STYLE_B_HEADERS: list[str] = [   # branch XLSX sheet (spaced title-case)
    "Transaction Ref", "Originator Acct", "Remitter IFSC", "To Account",
    "Beneficiary Branch IFSC", "Txn Amount", "Txn DateTime", "Pay Method",
    "Device No", "Customer IP",
]
STYLE_C_HEADERS: list[str] = [   # complaint-portal dump (mixed snake + CamelCase)
    "txn ref id", "sender account", "Sender IFSC Code", "beneficiary_account",
    "Receiver IFSC Code", "amount", "datetime_of_txn", "channel",
    "imei_device", "IP",
]

STYLE_A_EXPECTED: dict[str, str] = {
    "UTR": "txn_id", "Debit_Acc": "sender_account", "Fr_IFSC": "sender_ifsc",
    "PAYEE": "receiver_account", "To_IFSC": "receiver_ifsc",
    "Amount_INR": "amount", "Txn_Time": "timestamp", "Mode": "payment_mode",
    "Device": "device_id", "IP": "ip_address",
}
STYLE_B_EXPECTED: dict[str, str] = {
    "Transaction Ref": "txn_id", "Originator Acct": "sender_account",
    "Remitter IFSC": "sender_ifsc", "To Account": "receiver_account",
    "Beneficiary Branch IFSC": "receiver_ifsc", "Txn Amount": "amount",
    "Txn DateTime": "timestamp", "Pay Method": "payment_mode",
    "Device No": "device_id", "Customer IP": "ip_address",
}
STYLE_C_EXPECTED: dict[str, str] = {
    "txn ref id": "txn_id", "sender account": "sender_account",
    "Sender IFSC Code": "sender_ifsc", "beneficiary_account": "receiver_account",
    "Receiver IFSC Code": "receiver_ifsc", "amount": "amount",
    "datetime_of_txn": "timestamp", "channel": "payment_mode",
    "imei_device": "device_id", "IP": "ip_address",
}

COMPLAINT_HEADERS: list[str] = [   # NCRP-portal style dump (report-officer wording)
    "Complaint Ack No", "Victim Acc No", "Suspect Acc No", "Fraud Category",
    "Reported On", "Cheated Amount", "Disputed UTR Ref", "State",
]
ACCOUNT_HEADERS: list[str] = [    # core-banking customer master sheet
    "Account Number", "IFSC Code", "Date of Opening", "Home Branch Code",
    "Mobile Number", "Device IMEI", "KYC Pincode", "Acct Type",
]
COMPLAINT_EXPECTED: dict[str, str] = {
    "Complaint Ack No": "ack_no", "Victim Acc No": "victim_account",
    "Suspect Acc No": "suspect_account", "Fraud Category": "fraud_category",
    "Reported On": "reported_timestamp", "Cheated Amount": "disputed_amount",
    "Disputed UTR Ref": "disputed_utr", "State": "state",
}
ACCOUNT_EXPECTED: dict[str, str] = {
    "Account Number": "account_no", "IFSC Code": "ifsc",
    "Date of Opening": "account_opening_date", "Home Branch Code": "branch_code",
    "Mobile Number": "mobile_no", "Device IMEI": "imei",
    "KYC Pincode": "kyc_pincode", "Acct Type": "account_type",
}
MULTI_TABLE_EXACT_CONFS: dict[tuple[str, str], float] = {
    # (table, raw header): probe-computed confidence
    ("complaints", "Complaint Ack No"): 0.8966,
    ("complaints", "Disputed UTR Ref"): 0.8571,
    ("accounts", "Date of Opening"): 0.8889,
    ("accounts", "Home Branch Code"): 0.8148,
}


class TestHeaderMapping:
    def test_normalize_header(self) -> None:
        assert normalize_header("Beneficiary-Acc  No#2") == "beneficiary acc no"
        assert normalize_header("  UTR ") == "utr"

    def test_style_a_full_coverage(self) -> None:
        mapping = map_headers(STYLE_A_HEADERS)
        for raw, canonical in STYLE_A_EXPECTED.items():
            assert mapping[raw].canonical_name == canonical, raw
            assert mapping[raw].confidence >= 0.80, (raw, mapping[raw].confidence)

    def test_style_b_full_coverage(self) -> None:
        mapping = map_headers(STYLE_B_HEADERS)
        for raw, canonical in STYLE_B_EXPECTED.items():
            assert mapping[raw].canonical_name == canonical, raw
            assert mapping[raw].confidence >= 0.80, raw

    def test_style_c_full_coverage_with_exact_confidences(self) -> None:
        mapping = map_headers(STYLE_C_HEADERS)
        exact_confs: dict[str, float] = {
            "txn ref id": 0.824, "sender account": 1.0, "Sender IFSC Code": 0.815,
            "beneficiary_account": 1.0, "Receiver IFSC Code": 0.839,
            "amount": 1.0, "datetime_of_txn": 0.889, "channel": 1.0,
            "imei_device": 0.80, "IP": 1.0,
        }
        for raw, canonical in STYLE_C_EXPECTED.items():
            assert mapping[raw].canonical_name == canonical, raw
            assert abs(mapping[raw].confidence - exact_confs[raw]) <= 1e-3, raw

    def test_complaint_and_account_style_coverage_with_exact_confs(self) -> None:
        m_c = map_headers(COMPLAINT_HEADERS, table="complaints")
        for raw, canonical in COMPLAINT_EXPECTED.items():
            assert m_c[raw].canonical_name == canonical, raw
            assert m_c[raw].confidence >= 0.80, raw
        m_a = map_headers(ACCOUNT_HEADERS, table="accounts")
        for raw, canonical in ACCOUNT_EXPECTED.items():
            assert m_a[raw].canonical_name == canonical, raw
            assert m_a[raw].confidence >= 0.80, raw
        # exact-confidence checks (the marginal headers per table):
        for raw in ("Complaint Ack No", "Disputed UTR Ref"):
            assert abs(m_c[raw].confidence - MULTI_TABLE_EXACT_CONFS[("complaints", raw)]) <= 1e-3, raw
        for raw in ("Date of Opening", "Home Branch Code"):
            assert abs(m_a[raw].confidence - MULTI_TABLE_EXACT_CONFS[("accounts", raw)]) <= 1e-3, raw

    def test_default_call_maps_transactions_only(self) -> None:
        # Without `table=`, complaint/account-specific headers must NOT claim
        # transaction canonicals, and transaction headers must not claim
        # complaint canonicals (tables are siloed by the synonym registry).
        tx_style = map_headers(["Cheated Amount", "Victim Acc No"])
        assert tx_style["Cheated Amount"].canonical_name is None
        assert tx_style["Victim Acc No"].canonical_name is None
        cx_style = map_headers(["Amount_INR", "Txn_Time"], table="complaints")
        assert cx_style["Amount_INR"].canonical_name is None
        assert cx_style["Txn_Time"].canonical_name is None

    def test_noise_headers_unmapped(self) -> None:
        mapping = map_headers(
            ["Narration", "Remarks", "Sl No", "Balance", "SenderIFSC", "ReceiverBranchIFSC"]
        )
        for raw, entry in mapping.items():
            assert entry.canonical_name is None, raw
            assert entry.confidence < 0.80, raw

    def test_below_threshold_is_dropped(self) -> None:
        mapping = map_headers(["Narration"], confidence_threshold=0.55)
        assert mapping["Narration"].canonical_name is None  # best score 52.2 < 55

    def test_canonical_collision_exact_tie_keeps_first(self) -> None:
        mapping = map_headers(["To Account", "PAYEE"])  # both hit receiver_account @ 100
        assert mapping["To Account"].canonical_name == "receiver_account"
        assert mapping["PAYEE"].canonical_name is None     # tie -> first-seen wins
        assert mapping["PAYEE"].confidence == 1.0          # loser keeps its own score

    def test_canonical_collision_higher_confidence_wins(self) -> None:
        # "Credit_Acc_No" scores ~0.741 against credit_account; "To Account" 1.0.
        mapping = map_headers(["Credit_Acc_No", "To Account"])
        assert mapping["To Account"].canonical_name == "receiver_account"
        assert mapping["Credit_Acc_No"].canonical_name is None
        assert mapping["Credit_Acc_No"].confidence < 0.80
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase1.py::TestHeaderMapping -q`
Expected: `ModuleNotFoundError: No module named 'ingestion.mapper'`.

- [ ] **Step 3: Write mapping half of `src/ingestion/mapper.py`**

```python
"""Fuzzy schema ingestion gateway.

Header mapping follows the spec's RapidFuzz token-sort-ratio approach;
readers + normalization (Task 4) and validation + reporting (Task 5)
complete the pipeline in this same module. Everything runs as Polars
expression chains: no row-level Python loops over dataframes, no Pandas.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from rapidfuzz import fuzz

from core.contracts import ColumnMapping, TableKind

# --- header mapping -------------------------------------------------------

DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.80

CANONICAL_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "txn_id": (
        "txn_id", "utr", "utr_no", "transaction_id", "txn_ref",
        "transaction_reference", "ref_no", "rrn", "utrid",
    ),
    "sender_account": (
        "sender_account", "sender_acc_no", "remitter", "payer", "from_account",
        "debit_account", "debitor", "originator_account", "sender",
        "sender_account_no",
    ),
    "sender_ifsc": (
        "sender_ifsc", "from_ifsc", "fr_ifsc", "sender_branch_ifsc",
        "debit_ifsc", "remitter_ifsc",
    ),
    "receiver_account": (
        "receiver_account", "receiver_acc_no", "beneficiary_acc",
        "beneficiary_account", "payee", "to_account", "credit_account",
        "beneficiary", "payee_account",
    ),
    "receiver_ifsc": (
        "receiver_ifsc", "to_ifsc", "beneficiary_ifsc", "credit_ifsc",
        "beneficiary_branch_ifsc",
    ),
    "amount": (
        "amount", "txn_amount", "transaction_amount", "amount_inr", "amt",
        "value",
    ),
    "timestamp": (
        "timestamp", "txn_timestamp", "txn_time", "txn_datetime",
        "transaction_datetime", "date_time", "txn_date",
    ),
    "payment_mode": (
        "payment_mode", "mode_of_payment", "payment_channel", "txn_mode",
        "payment_method", "channel", "mode",
    ),
    "device_id": ("device_id", "device", "imei", "device_no", "dev_id"),
    "ip_address": ("ip_address", "ip", "ip_addr", "customer_ip", "client_ip"),
}

COMPLAINT_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "ack_no": (
        "ack_no", "acknowledgement_no", "acknowledgment_no", "complaint_ack",
        "complaint_no", "ncrp_ack",
    ),
    "victim_account": (
        "victim_account", "victim_acc_no", "complainant_account",
        "complainant_acc", "defrauded_account",
    ),
    "suspect_account": (
        "suspect_account", "suspect_acc_no", "accused_account",
        "culprit_account", "fraudster_account",
    ),
    "fraud_category": (
        "fraud_category", "category_of_fraud", "complaint_category", "fraud_type",
    ),
    "reported_timestamp": (
        "reported_timestamp", "report_datetime", "complaint_datetime",
        "reported_on", "report_time",
    ),
    "disputed_amount": (
        "disputed_amount", "cheated_amount", "fraud_amount", "loss_amount",
        "complaint_amount",
    ),
    "disputed_utr": (
        "disputed_utr", "disputed_utr_no", "disputed_transaction_id", "utr",
        "transaction_reference",
    ),
    "state": ("state", "states", "jurisdiction", "in_state"),
}

ACCOUNT_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "account_no": (
        "account_no", "account_number", "acct_no", "account", "cust_account_no",
    ),
    "ifsc": ("ifsc", "ifsc_code", "branch_ifsc"),
    "account_opening_date": (
        "account_opening_date", "opening_date", "account_opened_on", "opened_date",
    ),
    "branch_code": ("branch_code", "branch", "home_branch", "sol_id"),
    "mobile_no": (
        "mobile_no", "mobile_number", "phone_no", "contact_number", "cust_mobile",
    ),
    "imei": ("imei", "imei_number", "device_imei"),
    "kyc_pincode": ("kyc_pincode", "pin_code", "pincode", "zip", "postal_code"),
    "account_type": ("account_type", "acct_type", "account_category"),
}

TABLE_SYNONYMS: Final[dict[TableKind, dict[str, tuple[str, ...]]]] = {
    "transactions": CANONICAL_SYNONYMS,
    "complaints": COMPLAINT_SYNONYMS,
    "accounts": ACCOUNT_SYNONYMS,
}

_HEADER_NORM_RE: Final = re.compile(r"[^a-z0-9]+")


def normalize_header(name: str) -> str:
    """Lowercase and collapse every non-alphanumeric run to single spaces."""
    return _HEADER_NORM_RE.sub(" ", name.lower()).strip()


def _best_synonym(
    norm_raw: str, synonyms: dict[str, tuple[str, ...]]
) -> tuple[float, str, str] | None:
    """Highest synonym score across all canonicals, or None when nothing matched."""
    best_score: float = 0.0
    best_canonical: str = ""
    best_synonym: str = ""
    for canonical, candidates in synonyms.items():
        for synonym in candidates:
            score = fuzz.token_sort_ratio(norm_raw, normalize_header(synonym))
            if score > best_score:
                best_score, best_canonical, best_synonym = score, canonical, synonym
    if best_score == 0.0:
        return None
    return (best_score, best_canonical, best_synonym)


def map_headers(
    raw_columns: Sequence[str],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    table: TableKind = "transactions",
) -> dict[str, ColumnMapping]:
    """Map raw header names to canonical columns with fuzzy confidence.

    One raw header maps to at most one canonical; one canonical accepts at
    most one raw header. Highest score wins; exact ties keep the first-seen
    raw column. A losing claimant stays in the result with
    ``canonical_name=None`` and its own confidence score (audit trail).
    The synonym registry is per table (`TABLE_SYNONYMS` keyed on `TableKind`):
    transaction, complaint and account dumps never cross-claim headers.
    """
    cutoff = confidence_threshold - 1e-9
    synonyms = TABLE_SYNONYMS[table]
    claims: dict[str, tuple[float, int]] = {}
    best_per_raw: dict[int, tuple[float, str, str] | None] = {}
    for index, raw in enumerate(raw_columns):
        scored = _best_synonym(normalize_header(raw), synonyms)
        eligible = (
            scored if scored is not None and scored[0] / 100.0 >= cutoff else None
        )
        best_per_raw[index] = eligible
        if eligible is not None:
            score, canonical, _synonym = eligible
            current = claims.get(canonical)
            if current is None or score > current[0]:
                claims[canonical] = (score, index)

    out: dict[str, ColumnMapping] = {}
    for index, raw in enumerate(raw_columns):
        scored = best_per_raw[index]
        if scored is None:
            out[raw] = ColumnMapping(
                raw_name=raw, canonical_name=None,
                matched_synonym=None, confidence=0.0,
            )
            continue
        score, canonical, synonym = scored
        if claims.get(canonical) == (score, index):
            out[raw] = ColumnMapping(
                raw_name=raw, canonical_name=canonical,
                matched_synonym=synonym, confidence=round(score / 100.0, 4),
            )
        else:
            out[raw] = ColumnMapping(
                raw_name=raw, canonical_name=None,
                matched_synonym=None, confidence=round(score / 100.0, 4),
            )
    return out
```

(Task 3 ends here — `ColumnMapping` is the only contract needed until Task 4 adds the readers/normalization imports and Task 5 adds `MANDATORY_COLUMNS` and `IngestionReport`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase1.py::TestHeaderMapping -q` → all PASS.
Run: `uv run mypy src` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/mapper.py tests/test_phase1.py
git commit -m "phase1(3): rapidfuzz header mapping with verified synonym coverage"
```

---

### Task 4: `DatasetIngester` — readers + normalization (happy path)

**Files:**
- Modify: `src/ingestion/mapper.py` (append readers, normalization, ingester methods; initial validation-less `_ingest_frame` returns all-accepted rows + report scaffolding)
- Test: `tests/test_phase1.py` (add `TestIngestHappyPath`)

**Interfaces:**
- Consumes: Task 3 `map_headers`/`DEFAULT_CONFIDENCE_THRESHOLD`/`TABLE_SYNONYMS`; contracts `TRANSACTION_COLUMNS`, `COMPLAINT_COLUMNS`, `ACCOUNT_COLUMNS`, `TRANSACTION_SCHEMA`, `COMPLAINT_SCHEMA`, `ACCOUNT_METADATA_SCHEMA`, `TableKind`, `MANDATORY_COLUMNS`, `IngestionReport`.
- Produces (Task 5 relies on these names): `TS_FORMATS`, `MODE_CANONICAL` (7 modes + alias table), `ACCOUNT_TYPE_CANONICAL`, `TABLE_COLUMNS`, `TABLE_SCHEMA`, helpers `_looks_like_xlsx(source) -> bool`, `_read_csv(source) -> pl.DataFrame` (all-String), `_read_xlsx(source) -> pl.DataFrame` (all-String), `_canonical_raw(mapping, canonical) -> str | None`, `_epoch_expr(src) / _timestamp_expr(src) / _amount_expr(src) / _mode_expr(src) / _ifsc_expr(src) / _date_expr(src) / _pattern_expr(src, pattern) / _account_type_expr(src) -> pl.Expr`, `_normalize(df, mapping, table) -> pl.DataFrame` (emits `_raw_<canonical>` audit columns + one typed column per canonical for that table's own column set; `_raw_*` columns are cleaned via strip+empty-→-null so validation may use `.is_null()` directly), and `DatasetIngester._ingest_frame(raw, source_label, table)`; public methods `ingest/ingest_csv/ingest_xlsx(source, *, table="transactions")` plus sugar `ingest_transactions/ingest_complaints/ingest_accounts`.

No canonical column may be absent from `_normalize` output in Task 4 — the `_raw_*` columns always exist for every canonical of the selected table (unmapped canonicals become all-null via `pl.lit(None, pl.String)`), and canonical expression helpers handle null/null-of-None inputs gracefully; downstream code must never key on "column missing".

- [ ] **Step 1: Write failing tests** — append to `tests/test_phase1.py`:

```python
import io
import openpyxl

from ingestion.mapper import DatasetIngester


def csv_bytes(rows: list[str]) -> bytes:
    header = ",".join(STYLE_A_HEADERS) + "\n"
    body = "\n".join(rows)
    return (header + body + ("\n" if body else "")).encode("utf-8")


def style_a_xlsx(header: list[str], rows: list[list[object]]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    for c, name in enumerate(header, start=1):
        ws.cell(column=c, row=1, value=name)
    for r, row in enumerate(rows, start=2):
        for c, value in enumerate(row, start=1):
            ws.cell(column=c, row=r, value=value)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestIngestHappyPath:
    def test_csv_path_all_three_header_styles_accepted(self, tmp_path) -> None:
        style_rows: dict[str, list[str]] = {
            "a": [
                "UTR_A1,2900000001,SBIN0001234,2900000002,SBIN0009876,15000.50,2024-06-01 10:30:00,UPI,D000123,10.20.30.40",
            ],
            "b": [
                "UTR_B1,2900000001,SBIN0001234,2900000002,SBIN0009876,2499.00,01/06/2024 11:45:30,IMPS,D000124,10.20.30.41",
            ],
            "c": [
                "UTR_C1,2900000001,SBIN0001234,2900000002,SBIN0009876,777.75,2024-06-01T12:00:00,NEFT,D000125,10.20.30.42",
            ],
        }
        headers_by: dict[str, list[str]] = {
            "a": STYLE_A_HEADERS, "b": STYLE_B_HEADERS, "c": STYLE_C_HEADERS,
        }
        for key, rows in style_rows.items():
            path = tmp_path / f"style_{key}.csv"
            path.write_bytes(
                (",".join(headers_by[key]) + "\n" + rows[0] + "\n").encode("utf-8")
            )
            df, report = DatasetIngester().ingest(str(path))
            assert df.height == 1
            assert report.rows_total == 1
            assert report.rows_accepted == 1 and report.rows_rejected == 0
            assert list(df.columns) == list(TRANSACTION_COLUMNS)
            assert df.dtypes == [TRANSACTION_SCHEMA[c] for c in TRANSACTION_COLUMNS]
            assert df["txn_id"].item() == f"UTR_{key.upper()}1"

    def test_dd_mm_precedes_mm_dd(self, tmp_path) -> None:
        rows = [
            "UTR_D1,2900000001,SBIN0001234,2900000002,SBIN0009876,500.00,05/06/2024 09:15:00,UPI,,",
            "UTR_D2,2900000001,SBIN0001234,2900000002,SBIN0009876,500.00,12/31/2024 09:15:00,UPI,,",
            "UTR_D3,2900000001,SBIN0001234,2900000002,SBIN0009876,500.00,05/06/2024 09:15:52 PM,UPI,,",
        ]
        path = tmp_path / "dd_mm.csv"
        path.write_bytes((",".join(STYLE_A_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest(str(path))
        assert report.rows_rejected == 0
        stamps = {r["txn_id"]: r["timestamp"] for r in df.to_dicts()}
        assert stamps["UTR_D1"] == datetime(2024, 6, 5, 9, 15, 0)
        assert stamps["UTR_D2"] == datetime(2024, 12, 31, 9, 15, 0)
        assert stamps["UTR_D3"] == datetime(2024, 6, 5, 21, 15, 52)

    def test_unix_timestamp_seconds_and_ms(self, tmp_path) -> None:
        rows = [
            "UTR_E1,1,SBIN0000001,2,SBIN0000002,10,1717200000,UPI,,",
            "UTR_E2,1,SBIN0000001,2,SBIN0000002,10,1717200002000,IMPS,,",
        ]
        path = tmp_path / "epoch.csv"
        path.write_bytes((",".join(STYLE_A_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest(str(path))
        assert report.rows_rejected == 0
        assert sorted(df["timestamp"].to_list()) == [
            datetime(2024, 6, 1, 0, 0, 0), datetime(2024, 6, 1, 0, 0, 2),
        ]

    def test_xlsx_bytes_with_native_cells(self) -> None:
        payload = style_a_xlsx(
            STYLE_A_HEADERS,
            [[
                "UTR_X1", "2900000001", "sbin0001234", "2900000002",
                "SBIN0009876", 100.5, datetime(2024, 6, 1, 3, 4, 5), "BHIM UPI",
                "D000001", "10.0.0.1",
            ]],
        )
        df, report = DatasetIngester().ingest(payload)   # bytes -> xlsx via PK magic sniff
        assert report.rows_accepted == 1
        assert df["amount"].item() == pytest.approx(100.5)
        assert df["timestamp"].item() == datetime(2024, 6, 1, 3, 4, 5)
        assert df["payment_mode"].item() == "UPI"
        assert df["sender_ifsc"].item() == "SBIN0001234"   # lowercased input upcased + validated

    def test_bytes_csv_via_sniff(self) -> None:
        payload = csv_bytes([
            "UTR_F1,1,SBIN0000001,2,SBIN0000002,25.00,2024-06-01 10:00:00,CASH,dev1,1.2.3.4",
        ])
        df, report = DatasetIngester().ingest(payload)
        assert report.rows_accepted == 1
        assert df["payment_mode"].item() == "CASH"

    def test_explicit_reader_override_on_suffixless_path(self, tmp_path) -> None:
        path = tmp_path / "only"
        path.write_bytes(csv_bytes([
            "UTR_G1,1,SBIN0000001,2,SBIN0000002,25.00,2024-06-01 10:00:00,CASH,dev2,1.2.3.5",
        ]))
        as_csv, report = DatasetIngester().ingest_csv(str(path))
        assert as_csv.height == 1 and report.rows_accepted == 1
        import zipfile

        with pytest.raises(openpyxl.utils.exceptions.InvalidFileException):
            DatasetIngester().ingest_xlsx(str(path))

    def test_mapping_report_records_confidence(self, tmp_path) -> None:
        path = tmp_path / "m.csv"
        path.write_bytes(csv_bytes([
            "UTR_H1,1,SBIN0000001,2,SBIN0000002,25.00,2024-06-01 10:00:00,CASH,dev3,1.2.3.6",
        ]))
        _df, report = DatasetIngester().ingest(str(path))
        m = report.column_mapping["Debit_Acc"]
        assert m.canonical_name == "sender_account"
        assert m.matched_synonym == "debit_account"
        assert m.confidence >= 0.80
        assert report.column_mapping["UTR"].canonical_name == "txn_id"


def complaint_csv_bytes(rows: list[str]) -> bytes:
    header = ",".join(COMPLAINT_HEADERS) + "\n"
    body = "\n".join(rows)
    return (header + body + ("\n" if body else "")).encode("utf-8")


def account_xlsx_bytes(rows: list[list[object]]) -> bytes:
    return style_a_xlsx(ACCOUNT_HEADERS, rows)


class TestIngestMultiTableHappy:
    def test_complaints_csv_happy_path(self, tmp_path) -> None:
        path = tmp_path / "ncrp.csv"
        path.write_bytes(complaint_csv_bytes([
            'ACK20240001,2900000007,2990100000,AEPS_CASHOUT,02/06/2024 09:15:00,"32,000.50",UTR2024060100001,Bihar',
        ]))
        df, report = DatasetIngester().ingest_complaints(str(path))
        assert df.height == 1
        assert list(df.columns) == list(COMPLAINT_COLUMNS)
        assert df["ack_no"].item() == "ACK20240001"
        assert df["disputed_utr"].item() == "UTR2024060100001"
        # DD/MM precedence + comma-junk amount survive the shared pipeline:
        assert df["reported_timestamp"].item() == datetime(2024, 6, 2, 9, 15, 0)
        assert df["disputed_amount"].item() == pytest.approx(32000.5)
        assert df["suspect_account"].item() == "2990100000"
        assert report.rows_accepted == 1 and report.rows_rejected == 0

    def test_complaints_headers_do_not_claim_transaction_columns(self, tmp_path) -> None:
        path = tmp_path / "ncrp.csv"
        path.write_bytes(complaint_csv_bytes([
            "ACK20240002,2900000007,2990100001,UPI_FRAUD,02/06/2024 09:15:00,1500,UTR2024060100002,Delhi",
        ]))
        df, report = DatasetIngester().ingest_complaints(str(path))
        assert df.height == 1 and report.rows_accepted == 1

    def test_accounts_xlsx_native_date_cell(self) -> None:
        payload = account_xlsx_bytes([[
            "2900000007", "sbin0001234", date(2023, 1, 15), "BR001",
            9178451234, "351104510123456", "800013", "savings",
        ]])
        df, report = DatasetIngester().ingest_accounts(payload)   # PK magic sniff
        assert df.height == 1
        assert list(df.columns) == list(ACCOUNT_COLUMNS)
        # openpyxl-native datetime cell -> String "2023-01-15 00:00:00.000000" -> Date:
        assert df["account_opening_date"].item() == date(2023, 1, 15)
        assert df["account_type"].item() == "SAVINGS"     # lowercase alias upcased
        assert df["ifsc"].item() == "SBIN0001234"         # lowercase input upcased
        assert df["mobile_no"].item() == "9178451234"     # numeric cell cast to String
        assert df["imei"].item() == "351104510123456"
        assert report.rows_accepted == 1

    def test_payment_mode_alias_expansion(self, tmp_path) -> None:
        rows = [
            "UTR_N1,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:00:00,RTGS,,",
            "UTR_N2,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:05:00,ATM WITHDRAWAL,,",
            "UTR_N3,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:10:00,MICRO ATM,,",
            "UTR_N4,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:15:00,UPI/P2M,,",
            "UTR_N5,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:20:00,UPI/P2P,,",
        ]
        path = tmp_path / "modes.csv"
        path.write_bytes((",".join(STYLE_A_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest(str(path))
        assert report.rows_rejected == 0
        assert df["payment_mode"].to_list() == ["RTGS", "ATM", "AEPS", "UPI", "UPI"]
```

Pinned behavior note for `test_explicit_reader_override_on_suffixless_path`: an explicit reader choice means caller intent, and a non-XLSX payload through `ingest_xlsx` raises openpyxl's `InvalidFileException` (polars/openpyxl refuses the non-spreadsheet payload) rather than silently misreading it. The *behavior* (no silent misread) is the invariant, not the error taxonomy; the exact type was pinned during execution (Task 4 Step 4).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase1.py::TestIngestHappyPath -q`
Expected: FAIL — `ImportError: cannot import name 'DatasetIngester' from 'ingestion.mapper'` (Task 3 has no class shell).

- [ ] **Step 3: Implement readers + normalization + ingester methods (mapper.py continue)**

Place after the Task-3 code. Import list becomes:

```python
import io
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import polars as pl
from rapidfuzz import fuzz

from core.contracts import (
    ACCOUNT_COLUMNS,
    ACCOUNT_METADATA_SCHEMA,
    COMPLAINT_COLUMNS,
    COMPLAINT_SCHEMA,
    IFSC_PATTERN,
    TRANSACTION_COLUMNS,
    TRANSACTION_SCHEMA,
    ColumnMapping,
    IngestionReport,
    TableKind,
)
```
(`MANDATORY_COLUMNS` and nothing else lands in Task 5's validation step; `IFSC_PATTERN` is needed here by `_ifsc_expr`; `TableKind` keys the per-table registries.)

```python
# --- canonical value expressions -------------------------------------------

TS_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%Y-%m-%d",
)

MODE_CANONICAL: Final[dict[str, str]] = {
    "UPI": "UPI",
    "BHIM UPI": "UPI",
    "UPI P2P": "UPI",
    "UPI/P2P": "UPI",
    "UPI/P2M": "UPI",
    "IMPS": "IMPS",
    "NEFT": "NEFT",
    "RTGS": "RTGS",
    "AEPS": "AEPS",
    "AEPS CASH OUT": "AEPS",
    "MICRO ATM": "AEPS",
    "ATM": "ATM",
    "ATM WITHDRAWAL": "ATM",
    "CASH": "CASH",
    "CASH DEPOSIT": "CASH",
    "CASHOUT": "CASH",
}

# Account-type aliases ( indian passbook vocabulary -> canonical AccountType ).
ACCOUNT_TYPE_CANONICAL: Final[dict[str, str]] = {
    "SAVINGS": "SAVINGS",
    "SAVINGS ACCOUNT": "SAVINGS",
    "SB": "SAVINGS",
    "SB ACCOUNT": "SAVINGS",
    "PMJDY": "PMJDY",
    "CURRENT": "CURRENT",
    "CURRENT ACCOUNT": "CURRENT",
    "CA": "CURRENT",
}

# Per-table canonical column set + typed schema (single source: contracts).
TABLE_COLUMNS: Final[dict[TableKind, tuple[str, ...]]] = {
    "transactions": TRANSACTION_COLUMNS,
    "complaints": COMPLAINT_COLUMNS,
    "accounts": ACCOUNT_COLUMNS,
}
TABLE_SCHEMA: Final[dict[TableKind, dict[str, pl.DataType]]] = {
    "transactions": TRANSACTION_SCHEMA,
    "complaints": COMPLAINT_SCHEMA,
    "accounts": ACCOUNT_METADATA_SCHEMA,
}

_XLSX_MAGIC: Final[bytes] = b"PK\x03\x04"
_NUMERIC_RE: Final = re.compile(r"^\d+$")
_CURRENCY_JUNK_RE: Final = re.compile(r"[,\s\u20b9]")


def _looks_like_xlsx(source: str | Path | bytes) -> bool:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source[:4]) == _XLSX_MAGIC
    return Path(source).suffix.lower() in {".xlsx", ".xls"}


def _source_label(source: str | Path | bytes) -> str | None:
    if isinstance(source, (bytes, bytearray)):
        return "bytes"
    return str(source)


def _read_csv(source: str | Path | bytes) -> pl.DataFrame:
    """All-String frame: every normalization decision happens downstream."""
    if isinstance(source, (bytes, bytearray)):
        data = io.BytesIO(bytes(source))
    else:
        data = str(source)
    return pl.read_csv(
        data,
        infer_schema_length=0,
        truncate_ragged_lines=True,
        quote_char='"',
    )


def _read_xlsx(source: str | Path | bytes) -> pl.DataFrame:
    """First worksheet; every cell coerced to String for the unified pipeline."""
    if isinstance(source, (bytes, bytearray)):
        df = pl.read_excel(io.BytesIO(bytes(source)), engine="openpyxl")
    else:
        df = pl.read_excel(str(source), engine="openpyxl")
    return df.select(
        [pl.col(name).cast(pl.String, strict=False).alias(name) for name in df.columns]
    )


def _canonical_raw(mapping: dict[str, ColumnMapping], canonical: str) -> str | None:
    """Original source column claimed by the fuzzy mapper for one canonical."""
    for raw_name, entry in mapping.items():
        if entry.canonical_name == canonical:
            return raw_name
    return None


def _epoch_expr(src: pl.Expr) -> pl.Expr:
    """UNIX seconds/milliseconds detection for purely numeric timestamps."""
    digits = pl.when(src.str.contains(_NUMERIC_RE.pattern)).then(src).otherwise(None)
    as_int = digits.cast(pl.Int64, strict=False)
    return (
        pl.when(as_int >= 10**12)
        .then(pl.from_epoch(as_int, time_unit="ms"))
        .when(as_int >= 10**9)
        .then(pl.from_epoch(as_int, time_unit="s"))
        .otherwise(None)
    )


def _timestamp_expr(src: pl.Expr) -> pl.Expr:
    """Epoch first, then DD/MM-before-MM/DD format fallbacks (task Decision 2)."""
    fallback = pl.coalesce(
        *[
            src.str.strptime(pl.Datetime, fmt, strict=False)
            for fmt in TS_FORMATS
        ]
    )
    return pl.coalesce(_epoch_expr(src), fallback)


def _amount_expr(src: pl.Expr) -> pl.Expr:
    """Strip commas/currency/whitespace, quiet cast; garbage -> null."""
    return src.str.replace_all(_CURRENCY_JUNK_RE.pattern, "").cast(
        pl.Float64, strict=False
    )


def _mode_expr(src: pl.Expr) -> pl.Expr:
    return (
        src.str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(r"\s+", " ")
        .replace_strict(MODE_CANONICAL, default=None)
    )


def _ifsc_expr(src: pl.Expr) -> pl.Expr:
    uppercase = src.str.strip_chars().str.to_uppercase()
    return pl.when(
        uppercase.is_not_null() & uppercase.str.contains(IFSC_PATTERN.pattern)
    ).then(uppercase).otherwise(None)


def _date_expr(src: pl.Expr) -> pl.Expr:
    """Calendar date via the shared timestamp chain (DD/MM precedence inherited),
    then truncated to Date — covers ISO dates and openpyxl-native datetime strings."""
    as_dt = _timestamp_expr(src)
    return as_dt.cast(pl.Date, strict=False)


def _pattern_expr(src: pl.Expr, pattern: str) -> pl.Expr:
    """Keep the cleaned value when it matches `pattern`, else null (audit-grade)."""
    cleaned = src.str.strip_chars()
    return pl.when(
        cleaned.is_not_null() & cleaned.str.contains(pattern)
    ).then(cleaned).otherwise(None)


def _account_type_expr(src: pl.Expr) -> pl.Expr:
    return (
        src.str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(r"\s+", " ")
        .replace_strict(ACCOUNT_TYPE_CANONICAL, default=None)
    )


def _normalize(
    df: pl.DataFrame, mapping: dict[str, ColumnMapping], table: TableKind
) -> pl.DataFrame:
    """Compose raw-text audit columns + typed canonical columns.

    Guarantees: for every canonical in TABLE_COLUMNS[table], both a cleaned
    `_raw_<canonical>` (String, "" -> null) and the canonical column itself
    exist afterwards, exactly once, regardless of the source's raw names.
    """
    columns = TABLE_COLUMNS[table]
    raw_steps: list[pl.Expr] = []
    for canonical in columns:
        source = _canonical_raw(mapping, canonical)
        expr = (
            pl.col(source).cast(pl.String, strict=False).str.strip_chars().replace("", None)
            if source is not None
            else pl.lit(None, dtype=pl.String)
        )
        raw_steps.append(expr.alias(f"_raw_{canonical}"))
    work = df.with_columns(raw_steps)

    canonical_steps: list[pl.Expr] = []
    for canonical in columns:
        src = pl.col(f"_raw_{canonical}")
        expr: pl.Expr
        if table == "transactions":
            if canonical == "amount":
                expr = _amount_expr(src)
            elif canonical == "timestamp":
                expr = _timestamp_expr(src)
            elif canonical == "payment_mode":
                expr = _mode_expr(src)
            elif canonical in ("sender_ifsc", "receiver_ifsc"):
                expr = _ifsc_expr(src)
            else:
                expr = src
        elif table == "complaints":
            if canonical == "disputed_amount":
                expr = _amount_expr(src)
            elif canonical == "reported_timestamp":
                expr = _timestamp_expr(src)
            else:
                expr = src
        else:
            if canonical == "account_opening_date":
                expr = _date_expr(src)
            elif canonical == "ifsc":
                expr = _ifsc_expr(src)
            elif canonical == "account_type":
                expr = _account_type_expr(src)
            elif canonical == "mobile_no":
                expr = _pattern_expr(src, r"^\d{10,12}$")
            elif canonical == "imei":
                expr = _pattern_expr(src, r"^\d{15}$")
            elif canonical == "kyc_pincode":
                expr = _pattern_expr(src, r"^\d{6}$")
            else:
                expr = src
        canonical_steps.append(expr.alias(canonical))
    return work.with_columns(canonical_steps)


class DatasetIngester:
    """CSV/XLSX ingester with fuzzy headers, validation and audit reporting.

    One class, three table kinds: transactions (default), NCRP complaints,
    and account metadata — each with its own synonym registry, canonical
    set and validation rules (Decision 12).
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = confidence_threshold

    def ingest(
        self, source: str | Path | bytes,
        *, table: TableKind = "transactions",
    ) -> tuple[pl.DataFrame, IngestionReport]:
        """Dispatch CSV vs XLSX by PK-zip magic (bytes) or suffix (paths)."""
        if _looks_like_xlsx(source):
            return self.ingest_xlsx(source, table=table)
        return self.ingest_csv(source, table=table)

    def ingest_csv(
        self, source: str | Path | bytes,
        *, table: TableKind = "transactions",
    ) -> tuple[pl.DataFrame, IngestionReport]:
        return self._ingest_frame(_read_csv(source), _source_label(source), table)

    def ingest_xlsx(
        self, source: str | Path | bytes,
        *, table: TableKind = "transactions",
    ) -> tuple[pl.DataFrame, IngestionReport]:
        return self._ingest_frame(_read_xlsx(source), _source_label(source), table)

    def ingest_transactions(self, source: str | Path | bytes) -> tuple[pl.DataFrame, IngestionReport]:
        return self.ingest(source, table="transactions")

    def ingest_complaints(self, source: str | Path | bytes) -> tuple[pl.DataFrame, IngestionReport]:
        return self.ingest(source, table="complaints")

    def ingest_accounts(self, source: str | Path | bytes) -> tuple[pl.DataFrame, IngestionReport]:
        return self.ingest(source, table="accounts")

    def _ingest_frame(
        self, raw: pl.DataFrame, source_label: str | None, table: TableKind
    ) -> tuple[pl.DataFrame, IngestionReport]:
        """Happy-path: normalize + emit every row as accepted (Task 5 wires rejections)."""
        mapping = map_headers(raw.columns, self.confidence_threshold, table=table)
        work = _normalize(raw, mapping, table)
        accepted = work.select(
            [
                pl.col(c).cast(TABLE_SCHEMA[table][c], strict=False)
                for c in TABLE_COLUMNS[table]
            ]
        )
        rejected = pl.DataFrame(
            schema={
                name: pl.String
                for name in (*[str(c) for c in raw.columns], "rejection_reason")
            }
        )
        report = IngestionReport(
            source=source_label,
            rows_total=raw.height,
            rows_accepted=accepted.height,
            rows_rejected=0,
            failure_counts={},
            column_mapping=mapping,
            rejected_rows=rejected,
        )
        return accepted, report
```

Note on `rejected` scaffolding schema: raw CSV columns are all `pl.String` (all-String reader) and XLSX is cast to String by `_read_xlsx` — so typing every rejected column as String in the head-0 scaffold is faithful and becomes exact in Task 5.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase1.py::TestIngestHappyPath -q` → all PASS.
Run: `uv run pytest tests/test_phase1.py -q` → Tasks 2–3 sections still PASS.
Run: `uv run mypy src` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/mapper.py tests/test_phase1.py
git commit -m "phase1(4): csv/xlsx readers and expression-based normalization"
```

---

### Task 5: Validation pipeline + `IngestionReport` rejections

**Files:**
- Modify: `src/ingestion/mapper.py` (add `TABLE_DEDUPE` + `_validation_reason(work, table) -> pl.Expr`; rewrite `_ingest_frame` to split accepted/rejected with keep-first per-table dedupe and full report metrics)
- Test: `tests/test_phase1.py` (add `TestIngestRejections`)

**Interfaces:**
- Consumes: `_raw_*` + canonical columns guaranteed by `_normalize` (Task 4), `MANDATORY_COLUMNS` + `MANDATORY_BY_TABLE` + `TableKind` from contracts.
- Produces: the final `ingest*` behavior for all three tables: accepted frame conforms exactly to `TABLE_COLUMNS[table]`/`TABLE_SCHEMA[table]` dtypes; `report.rows_total/rows_accepted/rows_rejected` consistently satisfy `rows_total == rows_accepted + rows_rejected`; `failure_counts` reason→count; `rejected_rows` = raw source columns (original header names, String) + `rejection_reason`. Dedupe key per table (Decision 12): `_txn_id`→`duplicate_utr`, `_ack_no`→`duplicate_ack_no`, `_account_no`→`duplicate_account_no` — keep first occurrence in raw file order.

- [ ] **Step 1: Write failing tests** — append (note `report.rejected_rows` uses RAW source column names, e.g. `"UTR"`):

```python
class TestIngestRejections:
    ROW = (
        "UTR_R1,2900000001,SBIN0001234,2900000002,SBIN0009876,100.00,"
        "2024-06-01 10:30:00,UPI,D000001,10.0.0.9"
    )

    def _ingest_rows(self, tmp_path, rows: list[str]):
        path = tmp_path / "reject.csv"
        path.write_bytes(csv_bytes(rows))
        return DatasetIngester().ingest(str(path))

    def test_negative_amount_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            self.ROW,
            "UTR_R2,2900000001,SBIN0001234,2900000002,SBIN0009876,-5.00,2024-06-01 10:31:00,UPI,,",
        ])
        assert report.rows_total == 2
        assert report.rows_rejected == 1 and df.height == 1
        assert report.failure_counts == {"negative_amount": 1}
        assert report.rejected_rows["UTR"].item() == "UTR_R2"
        assert report.rejected_rows["rejection_reason"].item() == "negative_amount"

    def test_unparseable_timestamp_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R3,1,SBIN0000001,2,SBIN0000002,10.00,NOT-A-DATE,UPI,,",
        ])
        assert report.failure_counts == {"unparseable_timestamp": 1}
        assert df.height == 0

    def test_unparseable_amount_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R4,1,SBIN0000001,2,SBIN0000002,ten,2024-06-01 10:30:00,UPI,,",
        ])
        assert report.failure_counts == {"unparseable_amount": 1}
        assert df.height == 0

    def test_missing_mandatory_fields(self, tmp_path) -> None:
        rows = [
            ",1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:30:00,UPI,,",              # missing:txn_id
            "UTR_M2,,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:30:00,UPI,,",         # missing:sender_account
            "UTR_M3,1,SBIN0000001,,SBIN0000002,10,2024-06-01 10:30:00,UPI,,",         # missing:receiver_account
            "UTR_M4,1,SBIN0000001,2,SBIN0000002,,2024-06-01 10:30:00,UPI,,",          # missing:amount
            "UTR_M5,1,SBIN0000001,2,SBIN0000002,10,,UPI,,",                            # missing:timestamp
            "UTR_M6,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:30:00,,,1.1.1.1",     # missing:payment_mode
        ]
        df, report = self._ingest_rows(tmp_path, rows)
        assert df.height == 0
        assert report.rows_rejected == 6
        assert report.failure_counts == {
            "missing:txn_id": 1, "missing:sender_account": 1,
            "missing:receiver_account": 1, "missing:amount": 1,
            "missing:timestamp": 1, "missing:payment_mode": 1,
        }

    def test_duplicate_utr_keep_first(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            self.ROW,
            self.ROW.replace("100.00", "200.00"),
        ])
        assert report.failure_counts == {"duplicate_utr": 1}
        assert df.height == 1 and df["amount"].item() == 100.0
        assert report.rejected_rows["UTR"].item() == "UTR_R1"

    def test_invalid_ifsc_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R5,1,SBIN1234,2,SBIN0000002,10,2024-06-01 10:30:00,UPI,,",
        ])
        assert report.failure_counts == {"invalid_ifsc": 1}
        assert df.height == 0

    def test_unsupported_payment_mode_rejected(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R6,1,SBIN0000001,2,SBIN0000002,10,2024-06-01 10:30:00,CHEQUE,,",
        ])
        assert report.failure_counts == {"unsupported_payment_mode": 1}

    def test_priority_missing_beats_all(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R7,,SBIN0000001,2,SBIN0000002,-10,NOT-A-DATE,ZZZ,,",
        ])
        assert report.failure_counts == {"missing:sender_account": 1}

    def test_mixed_reasons_aggregate(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_R8,1,SBIN0000001,2,SBIN0000002,-1,2024-06-01 10:30:00,UPI,,",          # negative_amount
            "UTR_R9,1,SBIN0000001,2,SBIN0000002,10,NOPE,UPI,,",                          # unparseable_timestamp
        ])
        assert report.failure_counts == {"negative_amount": 1, "unparseable_timestamp": 1}
        assert report.rows_accepted == 0 and df.height == 0
        assert set(report.rejected_rows.columns) == set(STYLE_A_HEADERS) | {"rejection_reason"}

    def test_zero_amount_accepted(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            "UTR_Z1,1,SBIN0000001,2,SBIN0000002,0,2024-06-01 10:30:00,UPI,,",
        ])
        assert report.rows_rejected == 0 and df.height == 1
        assert df["amount"].item() == 0.0

    def test_header_only_csv_returns_empty_report(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [])
        assert df.height == 0 and df.width == len(TRANSACTION_COLUMNS)
        assert report.rows_total == 0 and report.rows_accepted == 0
        assert report.rows_rejected == 0 and report.failure_counts == {}
        assert report.column_mapping["UTR"].canonical_name == "txn_id"
        assert report.rejected_rows.is_empty()

    def test_ragged_csv_rows_do_not_crash(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            self.ROW + ",EXTRA,CELLS",          # reader truncates extras; row parses
            self.ROW.replace("UTR_R1", "UTR_R10"),
        ])
        assert report.rows_total == 2
        assert report.rows_rejected == 0
        assert df.height == 2

    def test_quoted_timestamp_cell_still_parses(self, tmp_path) -> None:
        df, report = self._ingest_rows(tmp_path, [
            'UTR_CX1,1,SBIN0000001,2,SBIN0000002,10,"2024-06-01 10:30:00",UPI,,',
        ])
        assert report.rows_rejected == 0
        assert df["timestamp"].item() == datetime(2024, 6, 1, 10, 30, 0)


class TestIngestMultiTableRejections:
    def test_complaint_missing_ack_duplicate_ack(self, tmp_path) -> None:
        rows = [
            # empty ack -> missing:ack_no (also: suspect + disputed UTR optional, fine)
            ",2900000007,2990100000,AEPS_CASHOUT,02/06/2024 09:15:00,32000,UTR2024060100001,Bihar",
            "ACK20240005,2900000008,2990100001,UPI_FRAUD,02/06/2024 10:15:00,1500,UTR2024060100002,Delhi",
            "ACK20240005,2900000008,2990100001,UPI_FRAUD,02/06/2024 10:20:00,1500,UTR2024060100002,Delhi",  # dup ack
        ]
        path = tmp_path / "ncrp_reject.csv"
        path.write_bytes(complaint_csv_bytes(rows))
        df, report = DatasetIngester().ingest_complaints(str(path))
        assert report.rows_total == 3
        assert df.height == 1
        assert df["ack_no"].item() == "ACK20240005"
        assert report.failure_counts == {"duplicate_ack_no": 1, "missing:ack_no": 1}
        assert report.rejected_rows["rejection_reason"].to_list()[0] == "missing:ack_no"

    def test_complaint_missing_mandatory_field(self, tmp_path) -> None:
        rows = [
            # empty state -> missing:state; amount/timestamp fine
            "ACK20240006,2900000007,2990100002,AEPS_CASHOUT,02/06/2024 09:15:00,9000,UTR2024060100003,",
        ]
        path = tmp_path / "ncrp_state.csv"
        path.write_bytes(complaint_csv_bytes(rows))
        df, report = DatasetIngester().ingest_complaints(str(path))
        assert report.failure_counts == {"missing:state": 1}
        assert df.height == 0

    def test_accounts_rejections(self, tmp_path) -> None:
        rows = [
            # ifsc malformed -> invalid_ifsc
            "2994000000,SBIN00,15/03/2024,BR042,9812000000,351104510900001,800013,CURRENT",
            # unsupported account type -> unsupported_account_type
            "2994000001,SBIN0001234,15/03/2024,BR042,9812000001,351104510900002,800013,JOINT",
            # pincode not digits -> invalid_kyc_pincode
            "2994000002,SBIN0001234,15/03/2024,BR042,9812000002,351104510900003,12AB56,CURRENT",
            # mobile too short -> invalid_mobile_no (5 digits)
            "2994000003,SBIN0001234,15/03/2024,BR042,98123,351104510900004,800013,CURRENT",
            # clean row -> accepted, plus duplicate account_no below
            "2994000004,SBIN0001234,2024-03-15,BR042,9812000004,351104510900005,800013,SAVINGS",
            "2994000004,SBIN0001234,2024-03-20,BR042,9812000005,351104510900006,800013,SAVINGS",  # dup account_no
        ]
        path = tmp_path / "accounts_reject.csv"
        path.write_bytes((",".join(ACCOUNT_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest_accounts(str(path))
        assert report.rows_total == 6
        assert report.rows_accepted == 1 and df.height == 1
        assert df["account_no"].item() == "2994000004"
        assert df["account_opening_date"].item() == date(2024, 3, 15)   # keep-first
        assert report.failure_counts == {
            "duplicate_account_no": 1, "invalid_ifsc": 1, "invalid_kyc_pincode": 1,
            "invalid_mobile_no": 1, "unsupported_account_type": 1,
        }

    def test_accounts_missing_mandatory(self, tmp_path) -> None:
        rows = [
            # empty branch code -> missing:branch_code
            "2994000005,SBIN0001234,16/03/2024,,9812000006,351104510900007,800013,CURRENT",
        ]
        path = tmp_path / "accounts_missing.csv"
        path.write_bytes((",".join(ACCOUNT_HEADERS) + "\n" + "\n".join(rows) + "\n").encode())
        df, report = DatasetIngester().ingest_accounts(str(path))
        assert report.failure_counts == {"missing:branch_code": 1}
        assert df.height == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase1.py::TestIngestRejections -q`
Expected: FAIL — negative amounts/duplicates currently accepted (Task 4's `_ingest_frame` has no validation), so e.g. `test_negative_amount_rejected` fails on `rows_rejected == 1` (actual 0).

- [ ] **Step 3: Implement validation + rejection in `src/ingestion/mapper.py`**

Add `MANDATORY_BY_TABLE` to the contracts import list in this file (Task 4 already added `TableKind`), and place `TABLE_DEDUPE` + `_validation_reason` above `DatasetIngester`:

```python
TABLE_DEDUPE: Final[dict[TableKind, tuple[str, str]]] = {
    "transactions": ("_raw_txn_id", "duplicate_utr"),
    "complaints": ("_raw_ack_no", "duplicate_ack_no"),
    "accounts": ("_raw_account_no", "duplicate_account_no"),
}


def _validation_reason(work: pl.DataFrame, table: TableKind) -> pl.Expr:
    """First-match-wins rejection reason over a fixed priority order
    (Decision 4): missing fields first, then table-specific value rules.
    Primary-key duplicates are handled separately after field rules
    (Decisions 5 and 12), keyed per table by TABLE_DEDUPE.
    """
    missing = [
        pl.when(pl.col(f"_raw_{canonical}").is_null())
        .then(pl.lit(f"missing:{canonical}"))
        for canonical in MANDATORY_BY_TABLE[table]
    ]
    rules: list[pl.Expr] = []
    if table == "transactions":
        rules.append(
            pl.when(pl.col("_raw_amount").is_not_null() & pl.col("amount").is_null())
            .then(pl.lit("unparseable_amount"))
        )
        rules.append(
            pl.when(pl.col("amount").is_not_null() & (pl.col("amount") < 0))
            .then(pl.lit("negative_amount"))
        )
        rules.append(
            pl.when(pl.col("_raw_timestamp").is_not_null() & pl.col("timestamp").is_null())
            .then(pl.lit("unparseable_timestamp"))
        )
        rules.append(
            pl.when(
                (pl.col("_raw_sender_ifsc").is_not_null() & pl.col("sender_ifsc").is_null())
                | (pl.col("_raw_receiver_ifsc").is_not_null() & pl.col("receiver_ifsc").is_null())
            ).then(pl.lit("invalid_ifsc"))
        )
        rules.append(
            pl.when(pl.col("_raw_payment_mode").is_not_null() & pl.col("payment_mode").is_null())
            .then(pl.lit("unsupported_payment_mode"))
        )
    elif table == "complaints":
        rules.append(
            pl.when(
                pl.col("_raw_disputed_amount").is_not_null()
                & pl.col("disputed_amount").is_null()
            ).then(pl.lit("unparseable_amount"))
        )
        rules.append(
            pl.when(
                pl.col("disputed_amount").is_not_null() & (pl.col("disputed_amount") < 0)
            ).then(pl.lit("negative_amount"))
        )
        rules.append(
            pl.when(
                pl.col("_raw_reported_timestamp").is_not_null()
                & pl.col("reported_timestamp").is_null()
            ).then(pl.lit("unparseable_timestamp"))
        )
    else:
        rules.append(
            pl.when(pl.col("_raw_ifsc").is_not_null() & pl.col("ifsc").is_null())
            .then(pl.lit("invalid_ifsc"))
        )
        rules.append(
            pl.when(
                pl.col("_raw_account_type").is_not_null() & pl.col("account_type").is_null()
            ).then(pl.lit("unsupported_account_type"))
        )
        rules.append(
            pl.when(pl.col("_raw_mobile_no").is_not_null() & pl.col("mobile_no").is_null())
            .then(pl.lit("invalid_mobile_no"))
        )
        rules.append(
            pl.when(pl.col("_raw_imei").is_not_null() & pl.col("imei").is_null())
            .then(pl.lit("invalid_imei"))
        )
        rules.append(
            pl.when(pl.col("_raw_kyc_pincode").is_not_null() & pl.col("kyc_pincode").is_null())
            .then(pl.lit("invalid_kyc_pincode"))
        )
    return pl.coalesce(*missing, *rules)
```

Then replace the body of `_ingest_frame` (Task 4 version) with the validated pipeline:

```python
    def _ingest_frame(
        self, raw: pl.DataFrame, source_label: str | None, table: TableKind
    ) -> tuple[pl.DataFrame, IngestionReport]:
        mapping = map_headers(raw.columns, self.confidence_threshold, table=table)
        work = _normalize(raw, mapping, table)
        work = work.with_columns(
            _validation_reason(work, table).alias("rejection_reason")
        )

        audit_col, duplicate_reason = TABLE_DEDUPE[table]
        bad = work.filter(pl.col("rejection_reason").is_not_null())
        surviving = work.filter(pl.col("rejection_reason").is_null())
        surviving = surviving.with_columns(
            pl.col(audit_col).is_first_distinct().alias("_first_key")
        )
        accepted = (
            surviving.filter(pl.col("_first_key"))
            .select(
                [
                    pl.col(c).cast(TABLE_SCHEMA[table][c], strict=False)
                    for c in TABLE_COLUMNS[table]
                ]
            )
        )
        raw_cols = [pl.col(name) for name in raw.columns]
        duplicates = surviving.filter(~pl.col("_first_key")).select(
            raw_cols + [pl.lit(duplicate_reason).alias("rejection_reason")]
        )
        rejected = pl.concat(
            [
                bad.select(raw_cols + [pl.col("rejection_reason")]),
                duplicates,
            ],
            how="vertical",
        )
        counts = rejected.group_by("rejection_reason").len().sort("rejection_reason")
        failure_counts = {
            str(row["rejection_reason"]): int(row["len"])
            for row in counts.to_dicts()
        }
        report = IngestionReport(
            source=source_label,
            rows_total=raw.height,
            rows_accepted=accepted.height,
            rows_rejected=rejected.height,
            failure_counts=failure_counts,
            column_mapping=mapping,
            rejected_rows=rejected,
        )
        return accepted, report
```

Why the invariant `rows_total == rows_accepted + rows_rejected` holds: `bad ∪ duplicates-of-surviving` partitions the raw frame by construction (`surviving` = work minus bad, `_first_key` splits surviving in two), preserving raw order semantics. The partition holds identically for all three tables — only the reason names and dedupe key differ (Decision 12).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase1.py -q` → whole suite PASS incl. Tasks 2–4 regression.
Run: `uv run mypy src` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/mapper.py tests/test_phase1.py
git commit -m "phase1(5): priority-ordered validation, keep-first dedupe, full report"
```

---

### Task 6: `simulator/generator.py` — background layer

**Files:**
- Create: `src/simulator/generator.py` (constants, `SimulationResult`, `_txn_frame`, `_accounts`, `_background`, `SyndicateDataSimulator` with `generate()` wired; `_scheme_frames()` stub returns `[]` until Task 7 fills it)
- Test: `tests/test_phase1.py` (add `TestSimulatorBackground`)

**Interfaces:**
- Consumes: `ACCOUNT_METADATA_SCHEMA`, `COMPLAINT_SCHEMA`, `TRANSACTION_SCHEMA` from contracts.
- Produces (Task 7 relies on): `SCHEME_*` label constants, `SCHEME_TXN_COUNTS`, `LABELED_TRANSACTION_SCHEMA`, `_BASE_EPOCH_S`, `_RETAIL_ACCOUNTS = 900`, `_MERCHANT_ACCOUNTS = 100`, `_txn_frame(cols: dict[str, object]) -> pl.DataFrame`, `SyndicateDataSimulator.generate(background_transactions: int = 12_000) -> SimulationResult` — with `accounts` of height 1000 in Task 6 (grows to 1274 when Task 7 appends the 273 scheme participants + 1 terminal merchant), background ids `BG#########` (unique, sorted by timestamp).

- [ ] **Step 1: Write failing tests** — append to `tests/test_phase1.py`:

```python
from simulator.generator import LABELED_TRANSACTION_SCHEMA, SyndicateDataSimulator


class TestSimulatorBackground:
    def test_default_shape_and_validation(self) -> None:
        result = SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)
        t = result.transactions
        # height = background count + emitted labeled rows (0 in Task 6; 249 after Task 7)
        assert t.height == 5_000 + int(t["is_fraud"].sum())
        with pytest.raises(ValueError):
            SyndicateDataSimulator().generate(background_transactions=4_999)
        with pytest.raises(ValueError):
            SyndicateDataSimulator().generate(background_transactions=20_001)

    def test_schema_conformance_and_defaults(self) -> None:
        result = SyndicateDataSimulator(seed=42).generate()
        assert result.transactions.columns == list(LABELED_TRANSACTION_SCHEMA)
        assert result.accounts.columns == list(ACCOUNT_METADATA_SCHEMA)
        assert result.accounts.height == 1000
        assert result.transactions.dtypes == [
            LABELED_TRANSACTION_SCHEMA[c] for c in result.transactions.columns
        ]

    def test_background_all_unlabeled(self) -> None:
        result = SyndicateDataSimulator().generate(background_transactions=6_000)
        bg = result.transactions.filter(~pl.col("is_fraud"))
        assert bg.height == 6_000
        assert bg["fraud_scheme"].null_count() == bg.height

    def test_txn_ids_unique(self) -> None:
        result = SyndicateDataSimulator().generate()
        ids = result.transactions["txn_id"]
        assert ids.n_unique() == ids.len()

    def test_background_ids_bg_prefixed(self) -> None:
        result = SyndicateDataSimulator().generate()
        bg_ids = result.transactions.filter(~pl.col("is_fraud"))["txn_id"]
        assert bg_ids.str.starts_with("BG").all()

    def test_all_accounts_referenceable(self) -> None:
        result = SyndicateDataSimulator().generate()
        accounts = set(result.accounts["account_no"].to_list())
        used = (
            set(result.transactions["sender_account"].to_list())
            | set(result.transactions["receiver_account"].to_list())
        )
        assert used <= accounts

    def test_transactions_sorted(self) -> None:
        result = SyndicateDataSimulator().generate()
        expected = result.transactions.sort(["timestamp", "txn_id"])
        assert result.transactions.equals(expected)

    def test_deterministic_same_seed(self) -> None:
        a = SyndicateDataSimulator(seed=42).generate()
        b = SyndicateDataSimulator(seed=42).generate()
        assert a.transactions.equals(b.transactions)
        assert a.accounts.equals(b.accounts)

    def test_different_seed_changes_stream(self) -> None:
        a = SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)
        b = SyndicateDataSimulator(seed=7).generate(background_transactions=5_000)
        assert not a.transactions.equals(b.transactions)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase1.py::TestSimulatorBackground -q`
Expected: `ModuleNotFoundError: No module named 'simulator.generator'`.

- [ ] **Step 3: Write `src/simulator/generator.py` (background layer)**

```python
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
    "is_fraud": pl.Boolean,
    "fraud_scheme": pl.String,
}

# 2024-06-01T00:00:00Z epoch seconds, fixed at import (no wall clock/tz table).
_BASE_EPOCH_S: Final[int] = 1717200000
_RETAIL_ACCOUNTS: Final[int] = 900
_MERCHANT_ACCOUNTS: Final[int] = 100
_SCHEME_IFSC: Final[str] = "UTIB0042000"


def _txn_frame(cols: dict[str, Any]) -> pl.DataFrame:
    """Round-trip raw column payload into the labeled canonical frame.

    `cols["timestamp"]` holds epoch **seconds** (int64); ids are Strings,
    amounts floats, flags booleans. Scheme rows are already confluent with
    the anti-fraud label convention.
    """
    df = pl.DataFrame(cols)
    df = df.with_columns(pl.from_epoch("timestamp", time_unit="s"))
    return df.select(list(LABELED_TRANSACTION_SCHEMA)).cast(LABELED_TRANSACTION_SCHEMA)


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
        transactions = (
            pl.concat([background, *schemes], how="vertical")
            .select(list(LABELED_TRANSACTION_SCHEMA))
            .cast(LABELED_TRANSACTION_SCHEMA)
            .sort(["timestamp", "txn_id"])
        )
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
        return base.with_columns(
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
                pl.lit(None, dtype=pl.String).alias("fraud_scheme"),
            ]
        ).select(list(LABELED_TRANSACTION_SCHEMA)).cast(LABELED_TRANSACTION_SCHEMA)

    # === fraud schemes (Task 7) ===========================================

    def _scheme_frames(self) -> list[pl.DataFrame]:
        """Returns [] in Task 6; the five builders are appended here by Task 7."""
        return []

    def _complaints(self) -> pl.DataFrame:
        """Task 7 wires the ground truth: 16 rows (episodes of schemes 1/2/3)."""
        rows: list[dict[str, object]] = []
        return pl.DataFrame(rows, schema=COMPLAINT_SCHEMA)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase1.py::TestSimulatorBackground -q` → all PASS.
Run: `uv run pytest tests/test_phase1.py -q` → whole suite still PASS.
Run: `uv run mypy src` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/simulator/generator.py tests/test_phase1.py
git commit -m "phase1(6): seeded background accounts and legitimate transaction layer"
```

---

### Task 7: Five fraud topologies + ground-truth labels + complaints

**Files:**
- Modify: `src/simulator/generator.py` (add scheme constants + `_scheme_account` allocator + `_scheme_entries`; replace the `_row`, builders, `_accounts` tail and complaints stub)
- Test: `tests/test_phase1.py` (add `TestSchemes` + `TestComplaints`)

**Interfaces:**
- Consumes: `_txn_frame`, `SCHEME_*`, `LABELED_TRANSACTION_SCHEMA`, `_BASE_EPOCH_S`, `_SCHEME_IFSC`, `_accounts` — Task 6 names; `COMPLAINT_SCHEMA` from contracts.
- Produces: `SHARED_BRANCH_IMEI`, `TERMINAL_MERCHANT_ACCOUNT`, `COMPLAINT_STATES`, `_SCHEME_IFSC_BY_BLOCK`, `_scheme_account(block, seq)`, `_scheme_entries()`. `SyndicateDataSimulator.generate` now emits the five schemes in exact counts (`SCHEME_TXN_COUNTS`: 80 + 130 + 9 + 12 + 18 = 249 labeled rows); `accounts` height 1274 (base 1000 + 274 appended rows: 273 scheme participants + the terminal merchant); `complaints` = 16 rows conforming to `COMPLAINT_SCHEMA`, each carrying a `disputed_utr` that resolves to a real transaction whose sender is the victim and receiver the suspect (Decision 14). **No orphaned or letter-coded accounts** (Decision 13): every transaction sender/receiver — background and schemes — resolves to an `accounts` row.

**Scheme account-block allocation (Decision 13).** `_scheme_account(block, seq)` returns `"29" + f"{90_000_000 + block * 10_000 + seq:08d}"` — numeric strings whose suffix space (`≥ 90_000_000`) is disjoint from background retail suffixes (`0..899`) and all `"55…"` merchant numbers:

| block | role | count | attribute truth |
|-------|------|-------|-----------------|
| 0 | rapid origins (victims/payers) | 20 | SAVINGS, opened Oct 2021+ (legit, long-wallets) |
| 1 | rapid hubs | 20 | PMJDY (even e) / SAVINGS, opened 2023-05 → ≥180d before base |
| 2/3/4 | rapid hop relays | 20+20+20 | CURRENT, opened May 2024 (fresh) |
| 5 | fan-out payers | 10 | SAVINGS, opened Jan 2024 |
| 6 | fan-out collectors | 10 | CURRENT, opened late Apr 2024 |
| 7 | micro wallets | 120 | PMJDY, opened May 2024 |
| 8 | AEPS victims | 3 | SAVINGS, opened Feb 2023 |
| 9/10 | AEPS mule tiers SM1/SM2 | 3+3 | CURRENT, opened Apr 2024 |
| 11 | branch colluders | 6 | CURRENT, `SBIN0000421`/`BR042`, `9162…` mobiles, shared IMEI, openings within 48h |
| 12/13/14 | circular-wash A/B/C nodes | 6+6+6 | SAVINGS, opened Nov 2023 |
| — | terminal merchant | 1 | pre-existing `TERMINAL_MERCHANT_ACCOUNT` row (unchanged) |

Scheme *transaction legs* keep `_SCHEME_IFSC` (as in Task 6) — Phase-2 graph joins key on `account_no`, which always resolves; per-role account-master IFSCs live in `_SCHEME_IFSC_BY_BLOCK`.

- [ ] **Step 1: Write failing tests** — append to `tests/test_phase1.py`:

```python
import re
from datetime import timedelta
from simulator.generator import (
    SCHEME_AEPS_TERMINAL_CASH_OUT,
    SCHEME_BRANCH_COLLUSION,
    SCHEME_CIRCULAR_WASH,
    SCHEME_FAN_OUT_SMURFING,
    SCHEME_RAPID_PASS_THROUGH,
    SCHEME_TXN_COUNTS,
    SHARED_BRANCH_IMEI,
    TERMINAL_MERCHANT_ACCOUNT,
    SimulationResult,
)

_TERMINAL_ACCOUNT = TERMINAL_MERCHANT_ACCOUNT   # single source of truth lives in the generator
_SHARED_IMEI = SHARED_BRANCH_IMEI


def accounts_set(sim: SimulationResult) -> set[str]:
    return set(sim.accounts["account_no"].to_list())


class TestSchemes:
    @pytest.fixture()
    def sim(self) -> SimulationResult:
        return SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)

    def test_labeled_counts_exact(self, sim) -> None:
        got = {
            row["fraud_scheme"]: row["len"]
            for row in sim.transactions.group_by("fraud_scheme").len().to_dicts()
        }
        labeled = {k: v for k, v in got.items() if k is not None}
        assert labeled == SCHEME_TXN_COUNTS

    def test_total_rows_match_background_plus_schemes(self, sim) -> None:
        assert sim.transactions.height == 5_000 + sum(SCHEME_TXN_COUNTS.values())

    def test_labels_integrity(self, sim) -> None:
        t = sim.transactions
        matches = t.select(
            (pl.col("is_fraud") == pl.col("fraud_scheme").is_not_null()).all()
        ).item()
        assert bool(matches)

    def test_no_orphaned_or_letter_coded_accounts(self, sim) -> None:
        # Decision 13: every emitted sender/receiver — background and all five
        # schemes — resolves to a master row; ids are numeric only.
        t = sim.transactions
        used = set(t["sender_account"].to_list()) | set(t["receiver_account"].to_list())
        assert used <= accounts_set(sim)
        assert not any(re.search(r"[A-Za-z]", account) for account in used)
        assert sim.accounts.height == 1_274          # 1000 base + 273 scheme + 1 merchant

    def test_rapid_pass_through_drain_window_hub_age(self, sim) -> None:
        t = sim.transactions.filter(pl.col("fraud_scheme") == SCHEME_RAPID_PASS_THROUGH)
        assert t.height == 80
        for e in range(20):
            rows = t.filter(
                pl.col("txn_id").str.starts_with(f"S1-{e:03d}-")
            ).sort("txn_id")
            inflow = rows.filter(pl.col("txn_id").str.ends_with("-0"))["amount"].item()
            hops = rows.filter(~pl.col("txn_id").str.ends_with("-0"))
            drain = hops["amount"].sum() / inflow
            assert drain >= 0.90, (e, drain)
            window = hops["timestamp"].max() - rows["timestamp"].min()
            assert window <= timedelta(minutes=15)
            assert hops.height == 3
            # roles derived structurally: the inflow leg's receiver IS the hub
            hub = t.filter(pl.col("txn_id") == f"S1-{e:03d}-0")["receiver_account"].item()
            hub_sent = rows.filter(pl.col("sender_account") == hub)
            assert hub_sent["amount"].sum() <= inflow  # hub can't spend its inflow twice
            # attribute truth for Phase 2 (Decision 13): hub account is aged, low-tier
            hub_row = sim.accounts.filter(pl.col("account_no") == hub)
            assert hub_row.height == 1
            opened = hub_row["account_opening_date"].item()
            assert opened <= date(2024, 6, 1) - timedelta(days=180)
            assert hub_row["account_type"].item() in ("PMJDY", "SAVINGS")

    def test_fan_out_smurfing_micro_amounts(self, sim) -> None:
        t = sim.transactions.filter(pl.col("fraud_scheme") == SCHEME_FAN_OUT_SMURFING)
        for e in range(10):
            episode = t.filter(pl.col("txn_id").str.starts_with(f"S2-{e:03d}-"))
            inflow = episode.filter(pl.col("txn_id").str.ends_with("-0"))
            assert inflow.height == 1
            coll = inflow["receiver_account"].item()          # structural collector id
            micro = episode.filter(~pl.col("txn_id").str.ends_with("-0"))
            assert micro.height == 12
            assert (micro["sender_account"] == coll).all()    # collector emits all micro-sends
            assert micro["receiver_account"].n_unique() == 12
            assert (micro["amount"] < 10_000).all()

    def test_aeps_terminal_cash_out_common_point(self, sim) -> None:
        t = sim.transactions.filter(pl.col("fraud_scheme") == SCHEME_AEPS_TERMINAL_CASH_OUT)
        inbound = t.filter(pl.col("receiver_account") == _TERMINAL_ACCOUNT)
        assert inbound.height == 3
        assert (inbound["payment_mode"] == "AEPS").all()
        assert (inbound["device_id"] == "AEPSDEV000001").all()
        for trail in range(3):
            # role derivation: leg 0 is victim->tier1, leg 1 tier1->tier2, leg 2 tier2->terminal
            episode = t.filter(
                pl.col("txn_id").str.starts_with(f"S3-{trail:02d}-")
            ).sort("txn_id")
            senders = episode["sender_account"].to_list()
            receivers = episode["receiver_account"].to_list()
            victim, tier1, tier2 = senders[0], receivers[0], senders[2]
            assert receivers[1] == tier2
            assert tier1 not in {victim, tier2}
            actors = {victim, tier1, tier2, _TERMINAL_ACCOUNT}
            assert actors <= accounts_set(sim)

    def test_branch_collusion_accounts_and_device(self, sim) -> None:
        colluders = sim.accounts.filter(pl.col("imei") == _SHARED_IMEI)
        assert colluders.height == 6
        assert (colluders["ifsc"] == "SBIN0000421").all()
        assert (colluders["branch_code"] == "BR042").all()
        opens = colluders["account_opening_date"].to_list()
        assert max(opens) - min(opens) <= timedelta(days=2)
        for mob in colluders["mobile_no"].to_list():
            assert mob.startswith("9162")
        t = sim.transactions.filter(pl.col("fraud_scheme") == SCHEME_BRANCH_COLLUSION)
        assert t.height == 12
        assert (t["device_id"] == _SHARED_IMEI).all()
        colluder_set = set(colluders["account_no"].to_list())
        for row in t.to_dicts():
            assert row["sender_account"] in colluder_set
            assert row["receiver_account"] in colluder_set

    def test_circular_wash_closed_loop(self, sim) -> None:
        t = sim.transactions.filter(pl.col("fraud_scheme") == SCHEME_CIRCULAR_WASH)
        assert t.height == 18
        for loop in range(6):
            episode = t.filter(
                pl.col("txn_id").str.starts_with(f"S5-{loop:02d}-")   # builder pads 2 digits
            ).sort("txn_id")
            edges = episode.to_dicts()      # legs commute in txn_id order: 0,1,2
            assert edges[1]["sender_account"] == edges[0]["receiver_account"]   # A->B, B->C
            assert edges[2]["sender_account"] == edges[1]["receiver_account"]   # then C->...
            assert edges[0]["sender_account"] == edges[2]["receiver_account"]   # ...->A (closed)
            amounts = [edge["amount"] for edge in edges]
            assert amounts[0] > amounts[1] > amounts[2]      # 3% commission decay
            actors = {
                edge["sender_account"] for edge in edges
            } | {
                edge["receiver_account"] for edge in edges
            }
            assert actors <= accounts_set(sim)
            assert len(actors) == 3

    def test_invalid_background_transaction_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            SyndicateDataSimulator().generate(background_transactions=1)


class TestComplaints:
    @pytest.fixture()
    def sim(self) -> SimulationResult:
        return SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)

    def test_complaint_ground_truth_joins(self, sim) -> None:
        c = sim.complaints
        assert c.height == 16
        assert c.columns == list(COMPLAINT_SCHEMA)
        assert c["ack_no"].n_unique() == 16
        assert c["disputed_utr"].null_count() == 0
        named_states = set(c["state"].to_list())
        assert named_states == {"Bihar", "Delhi", "Maharashtra", "Tamil Nadu", "Jharkhand"}
        joined = c.join(
            sim.transactions.select(["txn_id", "sender_account", "receiver_account"]),
            left_on="disputed_utr", right_on="txn_id",
            how="left",
        )
        # polars drops the right join key (`txn_id`) in this join shape, so
        # matched-ness is asserted via the joined data columns:
        assert joined["sender_account"].null_count() == 0
        assert joined["receiver_account"].null_count() == 0
        # every complaint cites a real transaction, and victim/suspect are
        # exactly that disputable leg's sender/receiver (Decision 14):
        assert (joined["victim_account"] == joined["sender_account"]).all()
        assert (joined["suspect_account"] == joined["receiver_account"]).all()

    def test_complaints_per_scheme(self, sim) -> None:
        c = sim.complaints
        joined = c.join(
            sim.transactions.select(["txn_id", "fraud_scheme"]),
            left_on="disputed_utr", right_on="txn_id",
            how="left",
        )
        scheme_counts = {
            str(row["fraud_scheme"]): row["len"]
            for row in joined.group_by("fraud_scheme").len().to_dicts()
        }
        assert scheme_counts == {
            SCHEME_RAPID_PASS_THROUGH: 8,
            SCHEME_FAN_OUT_SMURFING: 5,
            SCHEME_AEPS_TERMINAL_CASH_OUT: 3,
        }
```

(otherwise the test-file header only grows by `re` and `timedelta` — Task 7 owns that edit; `date`/`datetime` are already in the Task-2 header)

Implementer notes:
- Add `import re` and extend `timedelta` into the test file's `from datetime import ...` header (Task 7 owns that edit; both are used by the orphan-id check, the rapid-window and hub-age assertions).
- `test_invalid_background_transaction_count_rejected` intentionally duplicates Task 6's bounds check — harmless documentation of the invariant under scheme presence; it stays.
- Task 7 also fixes one Task 6 expectation: `test_schema_conformance_and_defaults` changes to `assert result.accounts.height == 1274` (base 1000 + 273 scheme participants + 1 terminal merchant appended by Task 7).
- Task 7 also broadens Task 6's `test_background_accounts_referenceable`: rename it to `test_all_accounts_referenceable` and drop the `~pl.col("is_fraud")` filter — with schemes present, *every* sender/receiver (background and scheme alike) must resolve to an `accounts` row (Decision 13). The renamed test then runs for both Task-6 (background-only) and Task-7 (full) states.
- Scheme txn-id padding is fixed and shared by builders/tests: `S1`/`S2` pad episodes to **3** digits (`f"S1-{e:03d}-0"`); `S3`/`S4`/`S5` pad to **2** digits (`f"S5-{loop:02d}-0"`), so the matching test filters use identical formats. (Task 6's other count/label/id tests are written label-count-aware above, so they hold unchanged across the Task 6→7 transition.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase1.py::TestSchemes -q`
Expected: FAIL — `labeled` counts come back `{}` (schemes not wired yet) or count assertion mismatch (background unlabeled only).

- [ ] **Step 3: Implement schemes + complaints (generator.py fill)**

First add module-level scheme constants, the account allocator and the participant master-row builder (right after `_SCHEME_IFSC`, so Task 6's content above stays untouched):

```python
SHARED_BRANCH_IMEI: Final[str] = "351104510123456"
TERMINAL_MERCHANT_ACCOUNT: Final[str] = "555500000010"

COMPLAINT_STATES: Final[tuple[str, ...]] = (
    "Bihar", "Delhi", "Maharashtra", "Tamil Nadu", "Jharkhand",
)

# Account-master IFSCs by scheme block (account rows only; txn legs keep
# _SCHEME_IFSC, graph joins key on account_no).
_SCHEME_IFSC_BY_BLOCK: Final[dict[int, str]] = {
    0: "UTIB0003331",   # rapid origins (legitimate long-standing payers)
    1: "SBIN0009001",   # rapid hubs (aged, low-tier)
    2: "UTIB0004401",   # rapid hop relay 1
    3: "UTIB0004402",   # rapid hop relay 2
    4: "UTIB0004403",   # rapid hop relay 3
    5: "UTIB0003332",   # fan-out payers
    6: "UTIB0003302",   # fan-out collectors
    7: "SBIN0007701",   # micro wallets
    8: "UTIB0002222",   # AEPS victims (long-standing customers)
    9: "UTIB0004409",   # AEPS mule tier 1
    10: "UTIB0004410",  # AEPS mule tier 2
    11: "SBIN0000421",  # branch colluders (BR042)
    12: "UTIB0005512",  # circular-wash node A
    13: "UTIB0005513",  # circular-wash node B
    14: "UTIB0005514",  # circular-wash node C
}


def _scheme_account(block: int, seq: int) -> str:
    """Reserved scheme-account number: "29" + suffix >= 90_000_000.

    Numeric-only (Decision 13), disjoint from background retail suffixes
    (0..899 -> "2900000000..2900000899") and merchant numbers ("55…").
    """
    return f"29{90_000_000 + block * 10_000 + seq:08d}"


def _scheme_entries() -> list[dict[str, object]]:
    """Master rows for every scheme participant (273 rows + merchant, appended
    by `_accounts`). Attributes are ground truth for Phase-2 signals:
    rapid hubs are aged >= 180 days before the base epoch with low-tier types;
    micro wallets/fresh collectors are recently opened; colluders share one
    branch, a shared device IMEI, and phone-prefix kinship opened within 48h.
    """
    entries: list[dict[str, object]] = []

    def entry(block: int, seq: int, *, opened: date, atype: str, branch: str) -> None:
        entries.append({
            "account_no": _scheme_account(block, seq),
            "ifsc": _SCHEME_IFSC_BY_BLOCK[block],
            "account_opening_date": opened,
            "branch_code": branch,
            "mobile_no": f"98{block:02d}{seq:06d}",          # 10 digits, entry-unique
            "imei": f"35{block:02d}{seq:06d}00000",          # 15 digits
            "kyc_pincode": "800013" if block % 2 == 0 else "110001",
            "account_type": atype,
        })

    for e in range(20):
        entry(_BLOCK_RAPID_ORIGIN, e, opened=date(2021, 10, 1) + timedelta(days=e),
              atype="SAVINGS", branch=f"BR{40 + e // 5:03d}")
        entry(_BLOCK_RAPID_HUB, e, opened=date(2023, 5, 1) + timedelta(days=e // 2),
              atype="PMJDY" if e % 2 == 0 else "SAVINGS", branch=f"BR9{e // 2:02d}")
        for hop_block in (_BLOCK_RAPID_HOP1, _BLOCK_RAPID_HOP2, _BLOCK_RAPID_HOP3):
            entry(hop_block, e, opened=date(2024, 5, 20) + timedelta(days=e % 11),
                  atype="CURRENT", branch=f"BR{60 + e:03d}")
    for e in range(10):
        entry(_BLOCK_FAN_PAYER, e, opened=date(2024, 1, 10) + timedelta(days=e),
              atype="SAVINGS", branch=f"BR{50 + e:03d}")
        entry(_BLOCK_FAN_COLL, e, opened=date(2024, 4, 28) + timedelta(days=e),
              atype="CURRENT", branch=f"BR{80 + e:03d}")
        for i in range(12):
            entry(_BLOCK_FAN_MICRO, e * 100 + i, opened=date(2024, 5, 25) + timedelta(days=i),
                  atype="PMJDY", branch=f"BR{90 + i:03d}")
    for t in range(3):
        entry(_BLOCK_AEPS_VICT, t, opened=date(2023, 2, 15) + timedelta(days=t),
              atype="SAVINGS", branch="BR102")
        entry(_BLOCK_AEPS_SM1, t, opened=date(2024, 4, 2) + timedelta(days=t),
              atype="CURRENT", branch="BR330")
        entry(_BLOCK_AEPS_SM2, t, opened=date(2024, 4, 2) + timedelta(days=t),
              atype="CURRENT", branch="BR331")
    for i in range(6):
        opened_at = datetime(2024, 5, 20, 11, 0) + timedelta(hours=7 * i)
        colluder = _scheme_account(_BLOCK_BRANCH_COLLUDER, i)
        entries.append({
            "account_no": colluder,
            "ifsc": _SCHEME_IFSC_BY_BLOCK[_BLOCK_BRANCH_COLLUDER],       # SBIN0000421
            "account_opening_date": opened_at.date(),                    # spans <= 48h
            "branch_code": "BR042",
            "mobile_no": f"9162{901000 + i * 731}",                      # shared prefix
            "imei": SHARED_BRANCH_IMEI,                                  # shared device
            "kyc_pincode": "800013",
            "account_type": "CURRENT",
        })
    for loop in range(6):
        for block in (_BLOCK_LOOP_A, _BLOCK_LOOP_B, _BLOCK_LOOP_C):
            entry(block, loop, opened=date(2023, 11, 20) + timedelta(days=loop),
                  atype="SAVINGS", branch=f"BR{20 + loop:03d}")
    assert len(entries) == 273
    return entries
```

with the block-name constants defined just above `_SCHEME_IFSC_BY_BLOCK` — a single role↔block mapping shared by the builders, `_scheme_entries` and `_complaints`:

```python
_BLOCK_RAPID_ORIGIN: Final[int] = 0
_BLOCK_RAPID_HUB: Final[int] = 1
_BLOCK_RAPID_HOP1: Final[int] = 2
_BLOCK_RAPID_HOP2: Final[int] = 3
_BLOCK_RAPID_HOP3: Final[int] = 4
_BLOCK_FAN_PAYER: Final[int] = 5
_BLOCK_FAN_COLL: Final[int] = 6
_BLOCK_FAN_MICRO: Final[int] = 7
_BLOCK_AEPS_VICT: Final[int] = 8
_BLOCK_AEPS_SM1: Final[int] = 9
_BLOCK_AEPS_SM2: Final[int] = 10
_BLOCK_BRANCH_COLLUDER: Final[int] = 11
_BLOCK_LOOP_A: Final[int] = 12
_BLOCK_LOOP_B: Final[int] = 13
_BLOCK_LOOP_C: Final[int] = 14
```

Then replace the `# === fraud schemes (Task 7)` body block wholesale (from the seam comment through the circular-wash builder):

```python
    # === fraud schemes (Task 7) ===========================================

    def _scheme_frames(self) -> list[pl.DataFrame]:
        return [
            self._scheme_rapid_pass_through(),
            self._scheme_fan_out_smurfing(),
            self._scheme_aeps_terminal_cash_out(),
            self._scheme_branch_collusion(),
            self._scheme_circular_wash(),
        ]

    def _row(
        self,
        *,
        txn_id: str,
        sender: str,
        receiver: str,
        amount: float,
        epoch_s: int,
        mode: str,
        device: str,
        ip: str = "10.10.0.1",
        is_fraud: bool = True,
        scheme: str | None = None,
    ) -> dict[str, object]:
        return {
            "txn_id": txn_id,
            "sender_account": sender,
            "sender_ifsc": _SCHEME_IFSC,
            "receiver_account": receiver,
            "receiver_ifsc": _SCHEME_IFSC,
            "amount": round(amount, 2),
            "timestamp": epoch_s,
            "payment_mode": mode,
            "device_id": device,
            "ip_address": ip,
            "is_fraud": is_fraud,
            "fraud_scheme": scheme,
        }
```

This `_row` helper is the single scheme-row shape; every builder below composes `_txn_frame({key: [row[key] for row in rows] for key in rows[0]})` from the row dicts it emits (`_txn_frame` then casts epoch-second integers to `Datetime("us")` and enforces `LABELED_TRANSACTION_SCHEMA`). All participants come from `_scheme_account(...)` — numeric ids guaranteed present in `accounts` (Decision 13).

Scheme builders (each pure arithmetic — no rng):

```python
    def _scheme_rapid_pass_through(self) -> pl.DataFrame:
        """20 episodes: hub inflow drained >=90% across hops within 15 min."""
        rows: list[dict[str, object]] = []
        for e in range(20):
            t0 = _BASE_EPOCH_S + 120 + e * 37 * 60
            hub = _scheme_account(_BLOCK_RAPID_HUB, e)
            inflow = 40_000.0 + e * 1_000.0
            rows.append(self._row(   # victim pays the mule hub
                txn_id=f"S1-{e:03d}-0",
                sender=_scheme_account(_BLOCK_RAPID_ORIGIN, e), receiver=hub,
                amount=inflow, epoch_s=t0, mode="UPI", device=f"S1DEV{e:03d}",
                scheme=SCHEME_RAPID_PASS_THROUGH,
            ))
            rows.append(self._row(   # hop 1: 48% of inflow
                txn_id=f"S1-{e:03d}-1", sender=hub,
                receiver=_scheme_account(_BLOCK_RAPID_HOP1, e),
                amount=inflow * 0.48, epoch_s=t0 + 120, mode="UPI",
                device=f"S1DEV{e:03d}", scheme=SCHEME_RAPID_PASS_THROUGH,
            ))
            rows.append(self._row(   # hop 2: 45% of inflow (hub total 0.93)
                txn_id=f"S1-{e:03d}-2", sender=hub,
                receiver=_scheme_account(_BLOCK_RAPID_HOP2, e),
                amount=inflow * 0.45, epoch_s=t0 + 240, mode="UPI",
                device=f"S1DEV{e:03d}", scheme=SCHEME_RAPID_PASS_THROUGH,
            ))
            rows.append(self._row(   # relay from hop 1's address
                txn_id=f"S1-{e:03d}-3", sender=_scheme_account(_BLOCK_RAPID_HOP1, e),
                receiver=_scheme_account(_BLOCK_RAPID_HOP3, e),
                amount=inflow * 0.46, epoch_s=t0 + 420, mode="UPI",
                device=f"S1DEV{e:03d}", scheme=SCHEME_RAPID_PASS_THROUGH,
            ))
        return _txn_frame({
            key: [row[key] for row in rows] for key in rows[0]
        })

    def _scheme_fan_out_smurfing(self) -> pl.DataFrame:
        """10 collectors distributing to 12 micro wallets under the 10k cap."""
        rows: list[dict[str, object]] = []
        for e in range(10):
            t0 = _BASE_EPOCH_S + 1_800 + e * 53 * 60
            coll = _scheme_account(_BLOCK_FAN_COLL, e)
            rows.append(self._row(
                txn_id=f"S2-{e:03d}-0", sender=_scheme_account(_BLOCK_FAN_PAYER, e),
                receiver=coll, amount=95_000.0, epoch_s=t0, mode="IMPS",
                device=f"S2DEV{e:03d}", scheme=SCHEME_FAN_OUT_SMURFING,
            ))
            for i in range(12):
                amount = 6_000.0 + ((i * 911) % 3_000) + 0.37   # always < 10_000
                rows.append(self._row(
                    txn_id=f"S2-{e:03d}-{i + 1:02d}", sender=coll,
                    receiver=_scheme_account(_BLOCK_FAN_MICRO, e * 100 + i),
                    amount=amount, epoch_s=t0 + 180 + i * 120, mode="UPI",
                    device=f"S2DEV{e:03d}", scheme=SCHEME_FAN_OUT_SMURFING,
                ))
        return _txn_frame({
            key: [row[key] for row in rows] for key in rows[0]
        })

    def _scheme_aeps_terminal_cash_out(self) -> pl.DataFrame:
        """3 distinct complaint trails all cashing out at one AEPS terminal."""
        rows: list[dict[str, object]] = []
        for trail in range(3):
            t0 = _BASE_EPOCH_S + (2 + trail * 3) * 86_400   # June 3/6/9
            victim = _scheme_account(_BLOCK_AEPS_VICT, trail)
            tier1 = _scheme_account(_BLOCK_AEPS_SM1, trail)
            tier2 = _scheme_account(_BLOCK_AEPS_SM2, trail)
            rows.append(self._row(   # victim -> mule tier 1 (disputed leg; Decision 14)
                txn_id=f"S3-{trail:02d}-0", sender=victim, receiver=tier1,
                amount=40_000.0, epoch_s=t0, mode="IMPS", device=f"SMDEV{trail:02d}",
                scheme=SCHEME_AEPS_TERMINAL_CASH_OUT,
            ))
            rows.append(self._row(
                txn_id=f"S3-{trail:02d}-1", sender=tier1, receiver=tier2,
                amount=36_000.0, epoch_s=t0 + 180, mode="IMPS",
                device=f"SMDEV{trail:02d}", scheme=SCHEME_AEPS_TERMINAL_CASH_OUT,
            ))
            rows.append(self._row(
                txn_id=f"S3-{trail:02d}-2", sender=tier2,
                receiver=TERMINAL_MERCHANT_ACCOUNT, amount=25_000.0,
                epoch_s=t0 + 360, mode="AEPS", device="AEPSDEV000001",
                scheme=SCHEME_AEPS_TERMINAL_CASH_OUT,
            ))
        return _txn_frame({
            key: [row[key] for row in rows] for key in rows[0]
        })

    def _scheme_branch_collusion(self) -> pl.DataFrame:
        """Same-branch, opened-within-48h, shared IMEI; 12 transfers among them."""
        rows: list[dict[str, object]] = []
        base = _BASE_EPOCH_S + 4 * 86_400   # one shared demo day
        for i in range(6):
            sender = _scheme_account(_BLOCK_BRANCH_COLLUDER, i)
            peer = _scheme_account(_BLOCK_BRANCH_COLLUDER, (i + 1) % 6)
            rows.append(self._row(
                txn_id=f"S4-{i:02d}-F", sender=sender, receiver=peer,
                amount=2_000.0 + i * 317.25, epoch_s=base + i * 30,
                mode="UPI", device=SHARED_BRANCH_IMEI,
                scheme=SCHEME_BRANCH_COLLUSION,
            ))
            rows.append(self._row(
                txn_id=f"S4-{i:02d}-B", sender=peer, receiver=sender,
                amount=1_800.0 + i * 210.5, epoch_s=base + i * 45,
                mode="UPI", device=SHARED_BRANCH_IMEI,
                scheme=SCHEME_BRANCH_COLLUSION,
            ))
        return _txn_frame({key: [row[key] for row in rows] for key in rows[0]})

    def _scheme_circular_wash(self) -> pl.DataFrame:
        """6 closed loops A->B->C->A with 3% commission decay per hop."""
        rows: list[dict[str, object]] = []
        for loop in range(6):
            base = _BASE_EPOCH_S + 30_000 + loop * 11 * 60
            a_node = _scheme_account(_BLOCK_LOOP_A, loop)
            b_node = _scheme_account(_BLOCK_LOOP_B, loop)
            c_node = _scheme_account(_BLOCK_LOOP_C, loop)
            x0 = 120_000.0 + loop * 5_000.0
            hop1 = x0 * 0.97
            hop2 = hop1 * 0.97
            rows.append(self._row(
                txn_id=f"S5-{loop:02d}-0", sender=a_node, receiver=b_node,
                amount=x0, epoch_s=base, mode="UPI", device=f"S5DEV{loop:02d}",
                scheme=SCHEME_CIRCULAR_WASH,
            ))
            rows.append(self._row(
                txn_id=f"S5-{loop:02d}-1", sender=b_node, receiver=c_node,
                amount=hop1, epoch_s=base + 60, mode="UPI", device=f"S5DEV{loop:02d}",
                scheme=SCHEME_CIRCULAR_WASH,
            ))
            rows.append(self._row(
                txn_id=f"S5-{loop:02d}-2", sender=c_node, receiver=a_node,
                amount=hop2, epoch_s=base + 120, mode="UPI", device=f"S5DEV{loop:02d}",
                scheme=SCHEME_CIRCULAR_WASH,
            ))
        return _txn_frame({key: [row[key] for row in rows] for key in rows[0]})
```

(`hop2` closes the loop — the amounts strictly decrease `x0 > hop1 > hop2`, matching `test_circular_wash_closed_loop`'s decay assertions. There is no `hop3` variable: the third leg IS the final decay step. The loop variable is named `loop`, never `l`, to stay lint-clean under ruff's E741.)

Branch-collusion and every other scheme's membership: the builders above reference account numbers that must exist in the accounts table — append the participant master rows in `_accounts` before its return, plus the terminal merchant. Replace the old colluder/merchant insert with (right BEFORE `return pl.DataFrame(cols, schema=ACCOUNT_METADATA_SCHEMA)`):

```python
        for participant in _scheme_entries():
            for field, value in participant.items():
                cols[field].append(value)
        cols["account_no"].append(TERMINAL_MERCHANT_ACCOUNT)
        cols["ifsc"].append("UTIB0000777")
        cols["account_opening_date"].append(date(2022, 3, 10))
        cols["branch_code"].append("BR077")
        cols["mobile_no"].append("9170001111")
        cols["imei"].append("351104510998871")
        cols["kyc_pincode"].append("800013")
        cols["account_type"].append("CURRENT")
```

(Checks: appended rows = 273 via `_scheme_entries`' internal assertion + 1 merchant ⇒ height 1000 + 274 = **1274** ✓; ids disjoint from background retail/merchant ranges ✓ numeric-only ✓; colluders share one exact IMEI so the imei-filter test identifies exactly them — the merchant uses a DIFFERENT imei, keeping the count at 6; colluder opening dates span 2024-05-20 → 2024-05-21 (7h×5 = 35h) ⇒ within 48h ✓; rapid hubs (block 1) open 2023-05-01..2023-05-10, all > 180 days before the 2024-06-01 base ✓ with PMJDY/SAVINGS types ✓; colluder mobiles all share the `9162` prefix, 10 digits ✓; every emitted merchant/mobile/imei value still matches `AccountMetadata`'s patterns ✓.)

Complaints (replace the Task-6 `rows` empty stub wholesale with this helper — Decision 14: 16 rows, every `disputed_utr` is the initial victim→mule inflow leg of its episode, states rotate across all five):

```python
    def _complaints(self) -> pl.DataFrame:
        """Ground truth: 8 rapid + 5 fan-out + 3 AEPS NCRP complaints.

        Every row's disputed_utr cites a real transaction (the initial
        victim->mule inflow leg); victim_account is that leg's payer and
        suspect_account its receiver, so a join against transactions
        verifies both (TestComplaints pins this).
        """
        rows: list[dict[str, object]] = []

        def add(
            victim: str, suspect: str, disputed_utr: str,
            disputed_amount: float, fraud_category: str,
            reported_at: datetime,
        ) -> None:
            index = len(rows)
            rows.append({
                "ack_no": f"ACK2024{index + 1:06d}",
                "victim_account": victim,
                "suspect_account": suspect,
                "fraud_category": fraud_category,
                "reported_timestamp": reported_at,
                "disputed_amount": disputed_amount,
                "disputed_utr": disputed_utr,
                "state": COMPLAINT_STATES[index % len(COMPLAINT_STATES)],
            })

        # rapid pass-through: the origin account disputes its UPI inflow to the hub
        for e in (0, 2, 4, 6, 8, 10, 12, 14):
            add(
                victim=_scheme_account(_BLOCK_RAPID_ORIGIN, e),
                suspect=_scheme_account(_BLOCK_RAPID_HUB, e),
                disputed_utr=f"S1-{e:03d}-0",
                disputed_amount=40_000.0 + e * 1_000.0,
                fraud_category="RAPID_PASS_THROUGH",
                reported_at=datetime(2024, 6, 1, 14, 0) + timedelta(hours=e),
            )
        # fan-out smurfing: the payer disputes the bulk transfer to its collector
        for e in (0, 2, 4, 6, 8):
            add(
                victim=_scheme_account(_BLOCK_FAN_PAYER, e),
                suspect=_scheme_account(_BLOCK_FAN_COLL, e),
                disputed_utr=f"S2-{e:03d}-0",
                disputed_amount=95_000.0,
                fraud_category="SMURFING_FAN_OUT",
                reported_at=datetime(2024, 6, 2, 9, 0) + timedelta(hours=e),
            )
        # AEPS cash-out ring: the victim disputes the inflow sent to mule tier 1
        for t in range(3):
            add(
                victim=_scheme_account(_BLOCK_AEPS_VICT, t),
                suspect=_scheme_account(_BLOCK_AEPS_SM1, t),
                disputed_utr=f"S3-{t:02d}-0",
                disputed_amount=40_000.0,
                fraud_category="AEPS_CASHOUT",
                reported_at=datetime(2024, 6, 3 + t * 3, 20, 0),
            )
        return pl.DataFrame(rows, schema=COMPLAINT_SCHEMA)
```

Ack numbers `ACK2024000001..ACK2024000016` are unique ✓ (`ACK2024` + 6 digits each); reported timestamps are naive `datetime` values (schema casts to `Datetime("us")` ✓); disputed amounts match the disputed leg's actual amount (rapid `40_000 + e*1_000`, fan `95_000`, AEPS `40_000`) so downstream reconciliation reconciles exactly; `index % 5` rotation puts all five states in the table (16 rows ⇒ each state appears ≥ 3 times) ✓; every disputed UTR is a scheme inflow transaction id emitted by the builders above (`S1-EEE-0`, `S2-EEE-0`, `S3-0t-0`) ✓ — and the join test in `TestComplaints` proves victim == sender and suspect == receiver for all 16 rows.


- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase1.py -q` → whole suite PASS (with Task-6 height fix 1274 in `test_schema_conformance_and_defaults` and the renamed `test_all_accounts_referenceable`).
Run: `uv run mypy src` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/simulator/generator.py tests/test_phase1.py
git commit -m "phase1(7): five labeled fraud topologies with deterministic ground truth"
```

---

### Task 8: Whole-phase verification

**Files:** none (verification only).

- [ ] **Step 1:** `uv run pytest -q` → all green.
- [ ] **Step 2:** `uv run mypy src` → `Success: no issues found`.
- [ ] **Step 3:** `uv run ruff check src tests` → clean (fix trivial lint if flagged, then re-run).
- [ ] **Step 4:** Offline smoke check (ingest + simulator determinism):

```bash
uv run python - <<'EOF'
from core.contracts import COMPLAINT_COLUMNS
from ingestion.mapper import DatasetIngester
from simulator.generator import SyndicateDataSimulator

header = "UTR,Debit_Acc,Fr_IFSC,PAYEE,To_IFSC,Amount_INR,Txn_Time,Mode,Device,IP\n"
row = "UTR_R1,2900000001,SBIN0001234,2900000002,SBIN0009876,100.00,2024-06-01 10:30:00,UPI,D1,1.2.3.4\n"
df, report = DatasetIngester().ingest((header + row).encode())
assert report.rows_accepted == 1 and report.rows_rejected == 0
print("ingest smoke ok")

ch = "Complaint Ack No,Victim Acc No,Suspect Acc No,Fraud Category,Reported On,Cheated Amount,Disputed UTR Ref,State\n"
crow = "ACK2024999999,2900000001,2900000002,UPI_FRAUD,02/06/2024 09:00:00,5000,UTR_RX,Bihar\n"
cf, creport = DatasetIngester().ingest_complaints((ch + crow).encode())
assert creport.rows_accepted == 1 and list(cf.columns) == list(COMPLAINT_COLUMNS)
print("complaints ingest smoke ok")

a = SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)
b = SyndicateDataSimulator(seed=42).generate(background_transactions=5_000)
assert a.transactions.equals(b.transactions)
assert a.accounts.height == 1274
assert a.complaints["disputed_utr"].is_in(a.transactions["txn_id"]).all()
print("simulator smoke ok")
EOF
```

Expected: both `... ok` lines; no network calls; completes in a few seconds (5,000-row background twice).

- [ ] **Step 5 (only if fixes were needed):** fix, re-run steps 1–3, commit `phase1(8): phase verification green`.

---

## Self-Review (author-checked before submission, incl. post-review amendments)

- **Spec coverage:** deliverable 1 → Task 2; deliverable 2 → Tasks 3–5 (now covering transactions, NCRP complaints and account metadata); deliverable 3 → Tasks 6–7; spec bullet groups as labeled test classes (`TestHeaderMapping` covers 3 transaction styles + the complaint/account styles + cross-table silo checks; `TestIngestRejections` covers negative/corrupt-date/duplicate-UTR; `TestSimulatorBackground` + `TestSchemes`/`TestComplaints` cover reproducibility and ground truth). `mypy src` cleanliness → Task 1 config + per-task verification. "Polars and DuckDB idioms, no iterrows" → Global Constraints; DuckDB deferred. "Pure offline" → Global Constraints + generator determinism choices.
- **Placeholder scan:** every implementation step contains complete code; two deliberate staged "stubs" (`_scheme_frames` → `[]` in Task 6, complaints `rows=[]` in Task 6) are replaced wholesale by Task 7 code blocks above, with their final implementations spelled out.
- **Type consistency:** `ColumnMapping` fields consistent across Tasks 2/3/4/5; report field names consistent in Tasks 2/4/5; scheme label constants don't drift; `SCHEME_TXN_COUNTS` arithmetic (80/130/9/12/18) matches every builder's loop construction; `test_xlsx_bytes_with_native_cells` + `test_dd_mm_precedes_mm_dd` share `STYLE_A_HEADERS` with Task 3's verified mappings; `Task 6 → 1274 accounts` height adjustment (273 scheme participants + merchant) tracked in Task 7's edit list and asserted by `test_no_orphaned_or_letter_coded_accounts` + the Task-8 smoke.
- **Review amendments applied (6):** seed test uses legal `5_000` background count; `PaymentMode` grown to 7 modes with RTGS/ATM + alias table (Decision 11) incl. micro-ATM→AEPS; multi-table ingestion via `ingest_transactions/ingest_complaints/ingest_accounts` (+ `.ingest()` default, Decision 12) with probe-verified per-table synonym sets; `disputed_utr` added to `COMPLAINT_SCHEMA`/`Complaint`/builder; no orphaned or letter-coded accounts (Decision 13: `_scheme_account` numeric allocator, 273 attribute-rich master rows, structural role derivation in tests); complaints ground truth now 16 interstate rows with every `disputed_utr` join-verified to a real victim→mule leg (Decision 14).
- **Review Focus coverage:** ambiguous dd/mm (Task 4 test), zero-amount (Task 5), header-only CSV (Task 5), ragged rows (Task 5), canonical collision (Task 3). All pinned.
- **Fuzzy confidence claims** (Task 3 exact-confs blocks, both the transaction styles and `MULTI_TABLE_EXACT_CONFS`, plus the silo-check scores `Amount_INR` 72.7 / `Txn_Time` 52.2 / `Complaint Ack No` 61.5 / `Cheated Amount` 66.7) were verified numerically with the real `normalize_header`; the higher-confidence-wins test uses a verified real pair (`Credit_Acc_No` 0.741 losing to `To Account` 1.0 on receiver_account). If ANY synonym table is edited, re-run the probe — the exact-confidence values are the only plan elements coupled to synonym-table contents.
