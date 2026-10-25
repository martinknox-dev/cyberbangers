"""Phase 5 batch CLI tests."""

import subprocess

import polars as pl


def test_batch_cli_processes_directory(tmp_path):
    """Batch CLI should process multiple CSV files and produce summary."""
    # Create test transaction files
    for i in range(3):
        df = pl.DataFrame({
            "txn_id": [f"T{i}_1"],
            "sender_account": [f"A{i}"],
            "receiver_account": [f"B{i}"],
            "amount": [10000.0 + i * 1000],
            "timestamp": ["2024-06-01 10:00:00"],
            "payment_mode": ["UPI"],
        })
        df.write_csv(tmp_path / f"file_{i}.csv")

    # Run batch CLI
    result = subprocess.run(
        [
            "prahar-batch", "--input-dir", str(tmp_path),
            "--output", str(tmp_path / "summary.csv"),
            "--config-tier", "LITE"
        ],
        capture_output=True, text=True, cwd="/home/martinknox/MysticKoder/Projects/prahar-graph", check=False,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    summary = pl.read_csv(tmp_path / "summary.csv")
    assert summary.height == 3
    assert "file_name" in summary.columns
    assert "accounts_processed" in summary.columns
    assert "high_critical_count" in summary.columns
    assert "total_risk_score" in summary.columns


def test_batch_cli_handles_empty_directory(tmp_path):
    """Batch CLI should handle empty input directory gracefully."""
    result = subprocess.run(
        [
            "prahar-batch", "--input-dir", str(tmp_path),
            "--output", str(tmp_path / "summary.csv"),
            "--config-tier", "LITE"
        ],
        capture_output=True, text=True, cwd="/home/martinknox/MysticKoder/Projects/prahar-graph", check=False,
    )

    assert result.returncode == 0
    summary = pl.read_csv(tmp_path / "summary.csv")
    assert summary.height == 0


def test_batch_cli_respects_config_tier(tmp_path):
    """Batch CLI should respect the config tier setting."""
    df = pl.DataFrame({
        "txn_id": ["T1"], "sender_account": ["A"], "receiver_account": ["B"],
        "amount": [500000.0], "timestamp": ["2024-06-01 10:00:00"],
        "payment_mode": ["UPI"],
    })
    df.write_csv(tmp_path / "file.csv")

    # Run with WORKSTATION tier
    result = subprocess.run(
        [
            "prahar-batch", "--input-dir", str(tmp_path),
            "--output", str(tmp_path / "summary.csv"),
            "--config-tier", "WORKSTATION"
        ],
        capture_output=True, text=True, cwd="/home/martinknox/MysticKoder/Projects/prahar-graph", check=False,
    )

    assert result.returncode == 0
    summary = pl.read_csv(tmp_path / "summary.csv")
    assert summary.height == 1
    # WORKSTATION tier should use more estimators, detect more
    assert "high_critical_count" in summary.columns