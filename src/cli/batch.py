"""Batch processing CLI for PraharGraph.

Process multiple transaction files and produce aggregate risk summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import click
import polars as pl

from core.contracts import FeatureConfig, GraphConfig, RiskConfig
from core.hardware import ComputeTier
from features.account_features import extract_account_features
from graph.analyzer import GraphAnalyzer
from risk.scorer import RiskScorer


def _process_file(file_path: Path, feature_config: FeatureConfig, graph_config: GraphConfig, risk_config: RiskConfig) -> dict[str, Any]:
    """Process a single transaction file and return summary statistics."""
    # Read the CSV file
    try:
        transactions = pl.read_csv(file_path)
    except (OSError, pl.exceptions.NoDataError):
        # If we can't read it, return empty summary
        return {
            "file_name": file_path.name,
            "accounts_processed": 0,
            "high_critical_count": 0,
            "total_risk_score": 0.0,
            "max_risk_score": 0.0,
            "error": "Failed to read file",
        }

    # Ensure timestamp is datetime
    if "timestamp" in transactions.columns:
        transactions = transactions.with_columns(pl.col("timestamp").str.to_datetime())

    # Extract features
    features = extract_account_features(transactions, config=feature_config)

    # Analyze graph
    analyzer = GraphAnalyzer(transactions, config=graph_config)
    graph = analyzer.topology_metrics()

    # Score risk
    scorer = RiskScorer(risk_config)
    risk_report, _ = scorer.score_with_assessments(features, graph)

    # Compute summary statistics
    accounts_processed = risk_report.height
    high_critical = risk_report.filter(pl.col("risk_band").is_in(["HIGH", "CRITICAL"]))
    high_critical_count = high_critical.height
    total_risk_score_val: float = cast(float, risk_report["risk_score"].sum()) if accounts_processed > 0 else 0.0
    max_risk_score_val: float = cast(float, risk_report["risk_score"].max()) if accounts_processed > 0 else 0.0

    return {
        "file_name": file_path.name,
        "accounts_processed": accounts_processed,
        "high_critical_count": high_critical_count,
        "total_risk_score": round(total_risk_score_val, 2),
        "max_risk_score": round(max_risk_score_val, 2),
        "error": "",
    }


@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, help="Directory containing transaction CSV files")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Output summary CSV file")
@click.option("--config-tier", type=click.Choice(["LITE", "STANDARD", "WORKSTATION"]), default="LITE", help="Compute tier configuration")
@click.option("--pattern", default="*.csv", help="File pattern to match (default: *.csv)")
def batch(input_dir: Path, output: Path, config_tier: str, pattern: str) -> None:
    """Process multiple transaction files and produce aggregate risk summary."""
    input_path = Path(input_dir)
    tier: ComputeTier = config_tier  # type: ignore[assignment]

    # Resolve configs based on tier
    feature_config = FeatureConfig(compute_tier=tier)
    graph_config = GraphConfig(compute_tier=tier)
    risk_config = RiskConfig(compute_tier=tier)

    # Find matching files
    files = sorted(input_path.glob(pattern))
    if not files:
        # Write empty summary
        empty_df = pl.DataFrame({
            "file_name": [],
            "accounts_processed": [],
            "high_critical_count": [],
            "total_risk_score": [],
            "max_risk_score": [],
            "error": [],
        })
        empty_df.write_csv(output)
        click.echo(f"No files found matching {pattern} in {input_dir}")
        return

    # Process each file
    results = []
    with click.progressbar(files, label="Processing files") as bar:
        for file_path in bar:
            result = _process_file(file_path, feature_config, graph_config, risk_config)
            results.append(result)

    # Write summary CSV
    summary_df = pl.DataFrame(results)
    summary_df.write_csv(output)
    click.echo(f"Processed {len(results)} files. Summary written to {output}")


if __name__ == "__main__":
    batch()