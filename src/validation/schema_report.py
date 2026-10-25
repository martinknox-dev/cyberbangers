"""Schema validation report for data quality assessment.

Provides statistical analysis of DataFrame columns against expected schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import polars as pl


@dataclass
class ValidationReport:
    """Results of schema validation."""
    total_rows: int
    null_counts: dict[str, int]
    type_mismatches: dict[str, int]
    outliers: dict[str, int]
    column_stats: dict[str, dict[str, Any]]
    passed: bool


class SchemaValidator:
    """Validate DataFrame against expected schema and compute quality metrics."""

    def __init__(self, expected_schema: dict[str, pl.DataType]) -> None:
        """Initialize with expected schema.

        Args:
            expected_schema: Dictionary mapping column names to expected Polars types.
        """
        self.schema = expected_schema

    def validate(self, df: pl.DataFrame) -> ValidationReport:
        """Validate DataFrame against schema and compute quality metrics.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationReport with validation results and statistics.
        """
        null_counts = {col: df[col].null_count() for col in df.columns}

        # Check type compatibility - only for columns present in the DataFrame
        type_mismatches: dict[str, int] = {}
        for col in df.columns:
            if col in self.schema:
                expected_type = self.schema[col]
                try:
                    df[col].cast(expected_type, strict=True)
                    type_mismatches[col] = 0
                except (TypeError, ValueError, OverflowError, pl.exceptions.InvalidOperationError):
                    type_mismatches[col] = df.height

        # Detect outliers using IQR method for numeric columns
        outliers: dict[str, int] = {}
        column_stats: dict[str, dict[str, Any]] = {}

        for col in df.columns:
            if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8):
                # Compute IQR-based outliers
                q1 = df[col].quantile(0.25) or 0.0
                q3 = df[col].quantile(0.75) or 0.0
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_count = df.filter(
                    (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
                ).height
                outliers[col] = outlier_count

                # Compute column statistics
                column_stats[col] = {
                    "mean": cast(float, df[col].mean()),
                    "std": cast(float, df[col].std()),
                    "min": cast(float, df[col].min()),
                    "max": cast(float, df[col].max()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                }
            elif df[col].dtype == pl.String:
                column_stats[col] = {
                    "unique_count": df[col].n_unique(),
                    "null_count": df[col].null_count(),
                }
            else:
                column_stats[col] = {
                    "null_count": df[col].null_count(),
                }

        # Overall pass/fail: no nulls in required columns, no type mismatches
        # Only check columns that are actually in the DataFrame
        required_columns_in_df = set(self.schema.keys()) & set(df.columns)
        has_required_nulls = any(null_counts.get(col, 0) > 0 for col in required_columns_in_df)
        has_type_mismatches = any(v > 0 for v in type_mismatches.values())

        passed = not has_required_nulls and not has_type_mismatches

        return ValidationReport(
            total_rows=df.height,
            null_counts=null_counts,
            type_mismatches=type_mismatches,
            outliers=outliers,
            column_stats=column_stats,
            passed=passed,
        )


__all__ = ["SchemaValidator", "ValidationReport"]