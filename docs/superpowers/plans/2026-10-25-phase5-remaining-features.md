# Phase 5 Remaining Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 high-impact features from Phase 5 backlog: SAR/STR Export, Batch Processing CLI, Schema Validation Report, and Temporal Pattern Mining — all extending existing infrastructure.

**Architecture:** Each feature extends existing modules (dossier, CLI, ingestion, features) rather than creating new subsystems. Shared interfaces: `PipelineBundle`, `RiskAssessment`, `IngestionReport`, `FeatureConfig`.

**Tech Stack:** Python 3.12, Polars, Pydantic, Click (CLI), existing `prahar-verify` entrypoint, existing `export.dossier` module.

**Spec:** This plan implements items 2-7 from Phase 5 backlog, prioritized by impact and reuse of existing code.

## Global Constraints

- No new external dependencies in `pyproject.toml` (stdlib + existing only)
- Strict TDD: test first, watch fail, minimal code, verify pass
- `mypy --strict` and `ruff check` must pass on all changes
- All new public functions need type hints and docstrings
- Backward compatibility: no breaking changes to existing `src/ui/app.py` signatures
- Default values on all new dataclass/ Pydantic fields

## Review Focus

Five input classes/failure modes most likely to bite users:

1. **Malformed CSV/XLSX with mixed schemas** → Ingestion should reject cleanly, not crash
2. **Empty or single-row datasets** → All pipelines must handle gracefully (already fixed)
3. **Non-UTC timestamps with ambiguous formats** → Must parse or reject with clear reason
4. **Duplicate UTRs across different files** → Must deduplicate consistently
5. **Massive files (100k+ rows)** → Memory/time limits must be respected via tier config

---

### Task 1: SAR/STR Export Module

**Files:**
- Create: `src/export/regulatory.py`
- Modify: `src/export/__init__.py` (export new module)
- Modify: `src/export/dossier.py` (reuse `InvestigationDossierGenerator`)
- Test: `tests/test_phase5_regulatory.py`

**Interfaces:**
- Consumes: `InvestigationDossier` (from `build_account_dossier`), `RiskAssessment`, `PipelineBundle`
- Produces: SAR/STR JSON/CSV per RBI/FIU-IND format spec

- [ ] **Step 1: Write failing test for SAR export**

```python
def test_sar_export_produces_valid_json():
    from ui import state
    from export.regulatory import SARExporter
    from core.contracts import RiskConfig

    txns = pl.DataFrame({
        "txn_id": ["T1"], "sender_account": ["A"], "receiver_account": ["B"],
        "amount": [500000.0], "timestamp": ["2024-06-01 10:00:00"],
        "payment_mode": ["UPI"],
    }).with_columns(pl.col("timestamp").str.to_datetime())

    bundle = state.run_pipeline(txns, config=RiskConfig())
    dossier = state.build_account_dossier(bundle, "A")
    exporter = SARExporter()
    sar_json = exporter.export_sar(dossier, reporting_entity="BANK001")
    
    data = json.loads(sar_json)
    assert data["report_type"] == "SAR"
    assert data["reporting_entity"] == "BANK001"
    assert "subject_accounts" in data
    assert "suspicious_activity" in data
    assert "grounds_for_suspicion" in data
```

- [ ] **Step 2: Run test to verify it fails**
```bash
pytest tests/test_phase5_regulatory.py::test_sar_export_produces_valid_json -v
```
Expected: ModuleNotFoundError / function not defined

- [ ] **Step 3: Write minimal SARExporter class**
```python
# src/export/regulatory.py
from export.dossier import InvestigationDossier
from core.contracts import RiskAssessment
import json

class SARExporter:
    def export_sar(self, dossier: InvestigationDossier, reporting_entity: str) -> str:
        risk = dossier.risk_summary
        return json.dumps({
            "report_type": "SAR",
            "reporting_entity": reporting_entity,
            "subject_accounts": [a.account_no for a in dossier.accounts],
            "suspicious_activity": {
                "risk_score": risk.risk_score if risk else 0,
                "risk_band": risk.risk_band if risk else "LOW",
                "operational_role": risk.operational_role if risk else "STANDARD_ACCOUNT",
            },
            "grounds_for_suspicion": [e.detail for a in dossier.accounts for e in a.indicators],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**
```bash
pytest tests/test_phase5_regulatory.py::test_sar_export_produces_valid_json -v
```
Expected: PASS

- [ ] **Step 5: Add STR export test and implementation**
```python
def test_str_export_produces_valid_csv():
    # Similar pattern for STR CSV format
    pass
```

- [ ] **Step 6: Commit**
```bash
git add src/export/regulatory.py src/export/__init__.py tests/test_phase5_regulatory.py
git commit -m "feat: add SAR/STR regulatory export module"
```

---

### Task 2: Batch Processing CLI

**Files:**
- Modify: `src/simulator/benchmark.py` (extend `main()` for batch mode)
- Create: `src/cli/batch.py`
- Modify: `src/__init__.py` or `pyproject.toml` (new entrypoint `prahar-batch`)
- Test: `tests/test_phase5_batch_cli.py`

**Interfaces:**
- Consumes: Directory of CSV/XLSX files, `FeatureConfig`/`RiskConfig`
- Produces: Per-file risk reports + aggregate summary CSV

- [ ] **Step 1: Write failing test for batch CLI**

```python
def test_batch_cli_processes_directory(tmp_path):
    import subprocess
    import json
    import polars as pl

    # Create test files
    for i in range(3):
        df = pl.DataFrame({
            "txn_id": [f"T{i}_1"], "sender_account": [f"A{i}"],
            "receiver_account": [f"B{i}"], "amount": [10000.0],
            "timestamp": ["2024-06-01 10:00:00"], "payment_mode": ["UPI"],
        })
        df.write_csv(tmp_path / f"file_{i}.csv")

    # Run batch CLI
    result = subprocess.run([
        "prahar-batch", "--input-dir", str(tmp_path),
        "--output", str(tmp_path / "summary.csv"),
        "--config-tier", "LITE"
    ], capture_output=True, text=True, cwd="/home/martinknox/MysticKoder/Projects/prahar-graph")

    assert result.returncode == 0
    summary = pl.read_csv(tmp_path / "summary.csv")
    assert summary.height == 3
    assert "file_name" in summary.columns
    assert "accounts_processed" in summary.columns
    assert "high_critical_count" in summary.columns
```

- [ ] **Step 2: Run test to verify it fails**
```bash
pytest tests/test_phase5_batch_cli.py::test_batch_cli_processes_directory -v
```
Expected: Command not found / ModuleNotFoundError

- [ ] **Step 3: Implement batch.py with Click CLI**

```python
# src/cli/batch.py
import click
import polars as pl
from pathlib import Path
from ingestion.mapper import DatasetIngester
from features.account_features import extract_account_features
from graph.analyzer import GraphAnalyzer
from risk.scorer import RiskScorer
from core.contracts import FeatureConfig, GraphConfig, RiskConfig

@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--output", type=click.Path(), required=True)
@click.option("--config-tier", type=click.Choice(["LITE", "STANDARD", "WORKSTATION"]), default="LITE")
def batch(input_dir, output, config_tier):
    # ... implementation ...
```

- [ ] **Step 4: Run test to verify it passes**
```bash
pytest tests/test_phase5_batch_cli.py::test_batch_cli_processes_directory -v
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/cli/batch.py src/__init__.py tests/test_phase5_batch_cli.py
git commit -m "feat: add prahar-batch CLI for batch processing"
```

---

### Task 3: Schema Validation Report

**Files:**
- Create: `src/validation/schema_report.py`
- Modify: `src/ingestion/mapper.py` (add validation hooks)
- Test: `tests/test_phase5_schema_validation.py`

**Interfaces:**
- Consumes: Raw DataFrame from ingestion, expected schema (`TRANSACTION_SCHEMA`, etc.)
- Produces: ValidationReport with column stats, null rates, outliers, type mismatches

- [ ] **Step 1: Write failing test**

```python
def test_schema_validation_report_detects_anomalies():
    from validation.schema_report import SchemaValidator
    from core.contracts import TRANSACTION_SCHEMA

    # Data with issues: null sender, outlier amount, wrong type
    df = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3", "T4"],
        "sender_account": ["A", None, "C", "D"],
        "receiver_account": ["B", "B", "C", "D"],
        "amount": [100.0, 1e12, 50.0, "not-a-number"],
        "timestamp": ["2024-06-01", "2024-06-01", "2024-06-01", "2024-06-01"],
        "payment_mode": ["UPI", "UPI", "UPI", "UNKNOWN"],
    })

    validator = SchemaValidator(TRANSACTION_SCHEMA)
    report = validator.validate(df)

    assert report.null_counts["sender_account"] == 1
    assert report.outliers["amount"] >= 1
    assert report.type_mismatches["amount"] >= 1
    assert report.column_stats["amount"]["max"] > 1e6
```

- [ ] **Step 2: Run test to verify it fails**
```bash
pytest tests/test_phase5_schema_validation.py::test_schema_validation_report_detects_anomalies -v
```
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement SchemaValidator**

```python
# src/validation/schema_report.py
from dataclasses import dataclass, field
from typing import Dict, Any
import polars as pl

@dataclass
class ValidationReport:
    total_rows: int
    null_counts: Dict[str, int]
    type_mismatches: Dict[str, int]
    outliers: Dict[str, int]
    column_stats: Dict[str, Dict[str, Any]]
    passed: bool

class SchemaValidator:
    def __init__(self, expected_schema: Dict[str, pl.DataType]):
        self.schema = expected_schema

    def validate(self, df: pl.DataFrame) -> ValidationReport:
        null_counts = {col: df[col].null_count() for col in df.columns}
        
        type_mismatches = {}
        for col, expected in self.schema.items():
            if col in df.columns:
                try:
                    df[col].cast(expected, strict=True)
                    type_mismatches[col] = 0
                except:
                    type_mismatches[col] = df.height
        
        outliers = {}
        for col in df.columns:
            if df[col].dtype in (pl.Float64, pl.Int64):
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outlier_count = df.filter(
                    (pl.col(col) < q1 - 1.5 * iqr) | (pl.col(col) > q3 + 1.5 * iqr)
                ).height
                outliers[col] = outlier_count
        
        column_stats = {}
        for col in df.columns:
            if df[col].dtype in (pl.Float64, pl.Int64):
                column_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
        
        passed = all(v == 0 for v in null_counts.values()) and \
                 all(v == 0 for v in type_mismatches.values())
        
        return ValidationReport(
            total_rows=df.height,
            null_counts=null_counts,
            type_mismatches=type_mismatches,
            outliers=outliers,
            column_stats=column_stats,
            passed=passed,
        )
```

- [ ] **Step 4: Run test to verify it passes**
```bash
pytest tests/test_phase5_schema_validation.py::test_schema_validation_report_detects_anomalies -v
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/validation/schema_report.py tests/test_phase5_schema_validation.py
git commit -m "feat: add schema validation report for data quality"
```

---

### Task 4: Temporal Pattern Mining

**Files:**
- Create: `src/features/temporal.py`
- Modify: `src/features/account_features.py` (integrate temporal features)
- Modify: `src/core/contracts.py` (add temporal columns to `ACCOUNT_FEATURE_SCHEMA`)
- Test: `tests/test_phase5_temporal.py`

**Interfaces:**
- Consumes: Transaction DataFrame with timestamps
- Produces: Per-account temporal features (hour-of-day burst, day-of-week burst, velocity)

- [ ] **Step 1: Write failing test**

```python
def test_temporal_features_detect_hourly_burst():
    from features.temporal import compute_temporal_features
    import polars as pl

    # Account A bursts at 2 AM (suspicious)
    rows = []
    for i in range(20):
        rows.append({
            "txn_id": f"T{i}", "sender_account": "A", "receiver_account": f"B{i}",
            "amount": 1000.0,
            "timestamp": f"2024-06-01 02:{i%60:02d}:00",
            "payment_mode": "UPI",
        })
    txns = pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_datetime())

    features = compute_temporal_features(txns)
    row = features.filter(pl.col("account_no") == "A").to_dicts()[0]

    # Should detect night-time burst
    assert row["night_txn_ratio"] > 0.8
    assert row["peak_hour"] == 2
    assert row["velocity_txns_per_hour"] > 10
```

- [ ] **Step 2: Run test to verify it fails**
```bash
pytest tests/test_phase5_temporal.py::test_temporal_features_detect_hourly_burst -v
```
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement temporal features**

```python
# src/features/temporal.py
import polars as pl
import numpy as np

def compute_temporal_features(transactions: pl.DataFrame) -> pl.DataFrame:
    """Compute per-account temporal behavioral features."""
    
    # Ensure timestamp is datetime
    txns = transactions.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us"))
    )
    
    # Extract temporal components
    txns = txns.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("weekday"),  # 1=Mon
        pl.col("timestamp").dt.date().alias("date"),
    ])
    
    # Per-account aggregations
    result = txns.group_by("sender_account").agg([
        pl.len().alias("total_txns"),
        pl.col("hour").mean().alias("avg_hour"),
        pl.col("hour").std().alias("hour_std"),
        # Night ratio (22:00-06:00)
        (pl.col("hour").is_in(range(22, 24)) | pl.col("hour").is_in(range(0, 6)))
        .sum().cast(pl.Float64) / pl.len().alias("night_txn_ratio"),
        # Peak hour
        pl.col("hour").mode().first().alias("peak_hour"),
        # Velocity: txns per active hour
        (pl.len().cast(pl.Float64) / 
         pl.col("hour").n_unique().cast(pl.Float64)).alias("velocity_txns_per_hour"),
        # Day-of-week concentration
        (pl.col("weekday").value_counts().get_column("count").max() / pl.len())
        .alias("dow_concentration"),
        # Active days
        pl.col("date").n_unique().alias("active_days"),
    ])
    
    return result.rename({"sender_account": "account_no"})
```

- [ ] **Step 4: Add columns to ACCOUNT_FEATURE_SCHEMA**

```python
# src/core/contracts.py - add to ACCOUNT_FEATURE_SCHEMA
"night_txn_ratio": pl.Float64(),
"peak_hour": pl.Int64(),
"velocity_txns_per_hour": pl.Float64(),
"dow_concentration": pl.Float64(),
"active_days": pl.Int64(),
```

- [ ] **Step 5: Integrate in account_features.py extract_account_features()**

```python
# In extract_account_features, after base features:
temporal = compute_temporal_features(transactions)
frame = frame.join(temporal, on="account_no", how="left")
```

- [ ] **Step 6: Run test to verify it passes**
```bash
pytest tests/test_phase5_temporal.py::test_temporal_features_detect_hourly_burst -v
```
Expected: PASS

- [ ] **Step 7: Commit**
```bash
git add src/features/temporal.py src/features/account_features.py src/core/contracts.py tests/test_phase5_temporal.py
git commit -m "feat: add temporal pattern mining for burst detection"
```

---

### Task 5: Integration & Full Test Suite

**Files:**
- All modified files above
- Test: Run full suite

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -q
uv run mypy src
uv run ruff check src tests
uv run prahar-verify
```

- [ ] **Step 2: Verify all 4 features work together**

```bash
# Test SAR export on benchmark
# Test batch CLI on sample dir
# Test schema validation on sample files
# Test temporal features on benchmark
```

- [ ] **Step 3: Final commit & push**

```bash
git add -A
git commit -m "phase5(3): SAR/STR export, batch CLI, schema validation, temporal patterns"
git push origin main
```

---

## Execution Order

| Priority | Feature | Est. Time | Dependencies |
|----------|---------|-----------|--------------|
| 1 | SAR/STR Export | 45 min | Existing dossier |
| 2 | Batch CLI | 45 min | Existing `prahar-verify` |
| 3 | Schema Validation | 45 min | Existing ingestion |
| 4 | Temporal Patterns | 60 min | Existing features |

**Total: ~3.5 hours** — feasible for tonight if focused.

---

## Self-Review Checklist

- [ ] Spec coverage: All 4 features from backlog have tasks
- [ ] Placeholder scan: No "TBD", "implement later", vague steps
- [ ] Type consistency: All interfaces use existing types (`PipelineBundle`, `RiskAssessment`, `Polars DataFrame`)
- [ ] Review focus: Each of the 5 failure modes has test in owning task

---

**Ready for execution.** The plan uses existing infrastructure, adds no new dependencies, and each task produces independently testable deliverables. Recommended execution: **subagent-driven** for independent verification per task.