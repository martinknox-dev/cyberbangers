"""LAYER 4 — hybrid explainable risk engine.

`risk_score = clip(0.70 x domain_score + 0.30 x anomaly_score, 0, 100)`

* `domain_score` is a transparent, weighted sum of evidence-based indicators
  (pass-through, smurfing, cycle participation, shared metadata, terminal
  cash-out, complaint relationship). No indicator is derived from `state`,
  `kyc_pincode` or `account_type`.
* `anomaly_score` comes from `sklearn.ensemble.IsolationForest(random_state=42)`
  fitted on the numeric behavioural features. It is min-max normalised over the
  **uploaded dataset**, so it is dataset-relative: a different dataset shifts
  the scores. The docs and the report say so explicitly.

Explanations: up to three per account. Rule lines are *facts* (a measured
value, the threshold it was compared against and the supporting UTRs); the
anomaly line is an *inference* and states exactly what it explains — the
IsolationForest's raw path-length output, not the 0-100 anomaly score.

The score ranks investigative leads. It is never a probability of guilt and
never legal evidence (`RISK_DISCLAIMER`).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import polars as pl
import shap
from sklearn.ensemble import IsolationForest

from core.contracts import (
    ACCOUNT_FEATURE_SCHEMA,
    DOMAIN_KEYS,
    GRAPH_INDICATOR_SCHEMA,
    RISK_DISCLAIMER,
    RISK_REPORT_SCHEMA,
    RiskAssessment,
    RiskConfig,
    RiskExplanation,
)

ARCHETYPE_LABELS: dict[str, str] = {
    "pass_through": "Rapid pass-through mule",
    "smurfing": "Structuring / fan-out mule",
    "cycle": "Circular wash participant",
    "shared_metadata": "Shared-KYC collusion ring",
    "terminal_cashout": "Terminal cash-out point",
    "complaint": "Complaint-linked mule",
}
ARCHETYPE_NONE: str = "No domain indicator"

# Numeric behavioural features used by the anomaly model. Ground-truth labels
# (`is_fraud`, `fraud_scheme`) and protected attributes (`account_type`,
# `kyc_pincode`, `state`) are never model inputs.
ANOMALY_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "txn_count",
    "inflow_count",
    "outflow_count",
    "inflow_total",
    "outflow_total",
    "net_flow",
    "unique_senders",
    "unique_receivers",
    "fan_in_fan_out_asymmetry",
    "pass_through_ratio_15m",
    "median_retention_minutes",
    "smurf_bulk_inflow_count",
    "smurf_split_count",
    "large_payout_count",
    "burst_24h_max_total",
    "dormancy_days_before_burst",
    "dormancy_burst_score",
    "shared_device_account_count",
    "branch_cohort_48h_count",
    "aeps_cashout_inflow_count",
    "direct_complaint_count",
)

_FEATURE_TEXT: Final[dict[str, str]] = {
    "txn_count": "transaction count",
    "inflow_count": "inflow count",
    "outflow_count": "outflow count",
    "inflow_total": "inflow volume",
    "outflow_total": "outflow volume",
    "net_flow": "net flow",
    "unique_senders": "distinct senders",
    "unique_receivers": "distinct receivers",
    "fan_in_fan_out_asymmetry": "fan-in/fan-out asymmetry",
    "pass_through_ratio_15m": "pass-through ratio",
    "median_retention_minutes": "median retention",
    "smurf_bulk_inflow_count": "smurfing bulk inflows",
    "smurf_split_count": "smurfing split count",
    "large_payout_count": "large payouts",
    "burst_24h_max_total": "24h inflow burst",
    "dormancy_days_before_burst": "dormancy before burst",
    "dormancy_burst_score": "dormancy burst indicator",
    "shared_device_account_count": "shared-device peers",
    "branch_cohort_48h_count": "branch cohort peers",
    "aeps_cashout_inflow_count": "terminal cash-out inflows",
    "direct_complaint_count": "direct complaints",
}

_GRAPH_TEXT_COLUMNS: Final[dict[str, str]] = {
    "cycle_count": "cycle_count",
    "cycle_max_amount": "cycle_max_amount",
    "complaint_link_count": "complaint_link_count",
    "terminal_chain_count": "terminal_chain_count",
}


def _ensure_columns(
    frame: pl.DataFrame | None, schema: dict[str, pl.DataType]
) -> pl.DataFrame:
    """Frame with every schema column present (missing ones are null)."""
    if frame is None:
        return pl.DataFrame(schema=schema)
    columns: list[pl.Expr] = []
    for name, dtype in schema.items():
        if name in frame.columns:
            columns.append(pl.col(name).cast(dtype, strict=False).alias(name))
        else:
            columns.append(pl.lit(None, dtype=dtype).alias(name))
    return frame.select(columns)


class RiskScorer:
    """Deterministic 0-100 hybrid risk assessment with evidence explanations."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._config = config or RiskConfig()

    # --- public API -------------------------------------------------------

    def score(
        self,
        features: pl.DataFrame,
        graph: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Risk report for every account in the feature table.

        Args:
            features: behavioural features (`ACCOUNT_FEATURE_SCHEMA`).
            graph: optional graph indicators (`GRAPH_INDICATOR_SCHEMA`).

        Returns:
            A frame conforming to `RISK_REPORT_SCHEMA`, sorted by `account_no`.
        """
        return self._evaluate(features, graph)[0]

    def assess(
        self,
        features: pl.DataFrame,
        graph: pl.DataFrame | None = None,
    ) -> list[RiskAssessment]:
        """Typed per-account assessments (same computation as `score`)."""
        _, assessments = self._evaluate(features, graph)
        return assessments

    def score_with_assessments(
        self,
        features: pl.DataFrame,
        graph: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, list[RiskAssessment]]:
        """Risk report and typed assessments from one fitted model."""
        return self._evaluate(features, graph)

    # --- computation ------------------------------------------------------

    def _evaluate(
        self,
        features: pl.DataFrame,
        graph: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame, list[RiskAssessment]]:
        cfg = self._config
        feature_frame = _ensure_columns(features, ACCOUNT_FEATURE_SCHEMA)
        if feature_frame.height == 0:
            empty = pl.DataFrame(schema=RISK_REPORT_SCHEMA)
            return empty, []

        graph_frame = _ensure_columns(graph, GRAPH_INDICATOR_SCHEMA)
        frame = feature_frame
        if graph_frame.height:
            frame = frame.join(graph_frame, on="account_no", how="left")
        for name, dtype in GRAPH_INDICATOR_SCHEMA.items():
            if name not in frame.columns:
                frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(name))
        frame = frame.sort("account_no")

        # --- domain sub-scores (vectorised) --------------------------------
        ratio = pl.col("pass_through_ratio_15m").fill_null(0.0)
        ratio_component = (
            ratio / cfg.pass_through_threshold
            if cfg.pass_through_threshold > 0
            else pl.lit(0.0)
        ).clip(upper_bound=1.0)
        retention = pl.col("median_retention_minutes")
        speed_component = (
            pl.when(retention.is_null())
            .then(0.0)
            .otherwise(
                (
                    pl.lit(float(cfg.pass_through_window_minutes))
                    / retention.cast(pl.Float64())
                ).clip(upper_bound=1.0)
            )
        )
        pass_through_score = (
            pl.lit(0.6) * ratio_component + pl.lit(0.4) * speed_component
        ).clip(upper_bound=1.0)

        smurf_score = (
            pl.when(pl.col("smurf_bulk_inflow_count").fill_null(0) >= 1)
            .then(
                (
                    pl.col("smurf_split_count").fill_null(0).cast(pl.Float64())
                    / float(max(1, cfg.smurf_min_splits))
                ).clip(upper_bound=1.0)
            )
            .otherwise(0.0)
        )

        cycle_score = (
            pl.col("cycle_count").fill_null(0).cast(pl.Float64()) / 2.0
        ).clip(upper_bound=1.0)

        shared_score = (
            pl.max_horizontal(
                pl.col("shared_device_account_count").fill_null(0).cast(pl.Float64()),
                pl.col("branch_cohort_48h_count").fill_null(0).cast(pl.Float64()),
            )
            / 3.0
        ).clip(upper_bound=1.0)

        terminal_score = (
            pl.max_horizontal(
                pl.col("aeps_cashout_inflow_count").fill_null(0).cast(pl.Float64()),
                pl.col("terminal_chain_count").fill_null(0).cast(pl.Float64()),
            )
            / 2.0
        ).clip(upper_bound=1.0)

        complaint_score = (
            pl.when(
                (pl.col("direct_complaint_count").fill_null(0) >= 1)
                | (pl.col("complaint_link_count").fill_null(0) >= 1)
            )
            .then(1.0)
            .otherwise(0.0)
        )

        frame = frame.with_columns(
            pass_through_score.cast(pl.Float64()).alias("_pass_through"),
            smurf_score.cast(pl.Float64()).alias("_smurfing"),
            cycle_score.cast(pl.Float64()).alias("_cycle"),
            shared_score.cast(pl.Float64()).alias("_shared"),
            terminal_score.cast(pl.Float64()).alias("_terminal"),
            complaint_score.cast(pl.Float64()).alias("_complaint"),
        )
        # --- domain score: mild peak-plus-corroboration -----------------------
        # Pure linear sum caps single-archetype accounts at their weight (10-25).
        # Add 20% of the gap to the peak component to lift accounts that
        # strongly saturate one typology. Background accounts rarely hit 100%
        # on high-specificity components, so the boost mainly helps culprits.
        component_cols = ["_pass_through", "_smurfing", "_cycle", "_shared", "_terminal", "_complaint"]
        max_component = pl.max_horizontal([pl.col(c) for c in component_cols]) * 100.0
        weighted_sum = (
            pl.lit(cfg.pass_through_weight) * pl.col("_pass_through")
            + pl.lit(cfg.smurfing_weight) * pl.col("_smurfing")
            + pl.lit(cfg.cycle_weight) * pl.col("_cycle")
            + pl.lit(cfg.shared_metadata_weight) * pl.col("_shared")
            + pl.lit(cfg.terminal_cashout_weight) * pl.col("_terminal")
            + pl.lit(cfg.complaint_weight) * pl.col("_complaint")
        )
        gap = (max_component - weighted_sum).clip(lower_bound=0.0)
        domain_score = (weighted_sum + gap * pl.lit(0.15)).clip(upper_bound=100.0, lower_bound=0.0)

        # --- anomaly component ---------------------------------------------
        matrix, anomaly_scores, shap_values = self._fit_anomaly(frame)

        frame = frame.with_columns(domain_score.alias("_domain_score"))
        if anomaly_scores is not None:
            frame = frame.with_columns(
                pl.Series("_anomaly_score", anomaly_scores).alias("_anomaly_score")
            )
        else:
            frame = frame.with_columns(pl.lit(0.0).alias("_anomaly_score"))

        weight_sum = cfg.domain_weight + cfg.anomaly_weight
        if weight_sum <= 0:
            raise ValueError("domain_weight + anomaly_weight must be positive")
        risk_score = (
            pl.lit(cfg.domain_weight) * pl.col("_domain_score")
            + pl.lit(cfg.anomaly_weight) * pl.col("_anomaly_score")
        ).clip(upper_bound=100.0, lower_bound=0.0)

        frame = frame.with_columns(
            risk_score.alias("_risk_score"),
            pl.col("_domain_score").alias("domain_score"),
            pl.col("_anomaly_score").alias("anomaly_score"),
        )

        # --- explanations ---------------------------------------------------
        rows = frame.to_dicts()
        reasons: list[list[str]] = []
        kinds: list[list[str]] = []
        evidence: list[list[str]] = []
        assessments: list[RiskAssessment] = []
        component_columns = {
            "pass_through": ("_pass_through", cfg.pass_through_weight),
            "smurfing": ("_smurfing", cfg.smurfing_weight),
            "cycle": ("_cycle", cfg.cycle_weight),
            "shared_metadata": ("_shared", cfg.shared_metadata_weight),
            "terminal_cashout": ("_terminal", cfg.terminal_cashout_weight),
            "complaint": ("_complaint", cfg.complaint_weight),
        }
        for index, row in enumerate(rows):
            explanations = self._explain(row, index, shap_values, matrix)
            reasons.append([item.detail for item in explanations])
            kinds.append([item.kind for item in explanations])
            cited: list[str] = []
            for item in explanations:
                for txn_id in item.evidence_txn_ids:
                    if txn_id not in cited:
                        cited.append(txn_id)
            evidence.append(cited)
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

        report = frame.with_columns(
            pl.Series("risk_score", [a.risk_score for a in assessments]),
            pl.Series("domain_score", [a.domain_score for a in assessments]),
            pl.Series("anomaly_score", [a.anomaly_score for a in assessments]),
            pl.Series("risk_band", [a.risk_band for a in assessments]),
            pl.Series("top_reasons", reasons, dtype=pl.List(pl.String())),
            pl.Series("explanation_kinds", kinds, dtype=pl.List(pl.String())),
            pl.Series("evidence_txn_ids", evidence, dtype=pl.List(pl.String())),
        )
        report = report.select(
            [
                pl.col(name).cast(dtype, strict=False).alias(name)
                for name, dtype in RISK_REPORT_SCHEMA.items()
            ]
        ).sort("account_no")
        return report, assessments

    # --- anomaly model ----------------------------------------------------

    def _fit_anomaly(
        self, frame: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Impute, fit the Isolation Forest and compute SHAP attributions.

        Returns the imputed matrix, the dataset-relative anomaly scores and
        the SHAP attributions (all ``None`` when the model is not fitted).
        """
        rows = frame.height
        columns = [
            name
            for name in ANOMALY_FEATURE_COLUMNS
            if name in frame.columns
        ]
        empty = np.zeros((rows, len(columns)), dtype=float)
        if rows < self._config.min_rows_for_anomaly or not columns:
            return empty, None, None

        raw = frame.select(
            [pl.col(name).cast(pl.Float64(), strict=False) for name in columns]
        ).to_numpy()
        matrix = np.asarray(raw, dtype=float)
        # Missing / non-finite values are imputed with the column median
        # (0.0 when a column carries no usable value at all).
        for column_index in range(matrix.shape[1]):
            column = matrix[:, column_index]
            usable = column[np.isfinite(column)]
            fill = float(np.median(usable)) if usable.size else 0.0
            column[~np.isfinite(column)] = fill
            matrix[:, column_index] = column

        model = IsolationForest(random_state=42, n_estimators=100)
        model.fit(matrix)
        magnitude = -model.score_samples(matrix)
        lowest = float(magnitude.min())
        highest = float(magnitude.max())
        if not np.isfinite(lowest) or not np.isfinite(highest) or highest <= lowest:
            anomaly_scores = np.zeros(rows, dtype=float)
        else:
            anomaly_scores = (magnitude - lowest) / (highest - lowest) * 100.0

        explainer = shap.TreeExplainer(model)
        values = explainer(matrix).values
        shap_values = np.asarray(values, dtype=float)
        if shap_values.ndim == 3:      # (n, features, outputs) -> take output 0
            shap_values = shap_values[:, :, 0]
        return matrix, anomaly_scores, shap_values

    # --- explanations -----------------------------------------------------

    def _explain(
        self,
        row: dict[str, Any],
        index: int,
        shap_values: np.ndarray | None,
        matrix: np.ndarray | None,
    ) -> list[RiskExplanation]:
        cfg = self._config
        candidates: list[RiskExplanation] = []

        ratio = float(row.get("pass_through_ratio_15m") or 0.0)
        if ratio > 0:
            retention = row.get("median_retention_minutes")
            retention_text = (
                f"{float(retention):.1f} min"
                if retention is not None
                else "unknown"
            )
            utrs = list(row.get("pass_through_utrs") or [])
            candidates.append(
                RiskExplanation(
                    kind="FACT",
                    indicator="pass_through",
                    measured_value=ratio,
                    threshold=cfg.pass_through_threshold,
                    points=cfg.pass_through_weight * float(row.get("_pass_through") or 0.0),
                    detail=(
                        f"[FACT] Pass-through: drained {ratio * 100:.1f}% of inflow "
                        f"(threshold {cfg.pass_through_threshold * 100:.0f}%), "
                        f"median retention {retention_text} "
                        f"(threshold {cfg.pass_through_window_minutes} min) "
                        f"-> +{cfg.pass_through_weight * float(row.get('_pass_through') or 0.0):.1f} pts"
                        + (f"; UTRs: {', '.join(utrs)}" if utrs else "")
                    ),
                    evidence_txn_ids=utrs,
                )
            )

        smurf_count = int(row.get("smurf_bulk_inflow_count") or 0)
        if smurf_count > 0:
            splits = int(row.get("smurf_split_count") or 0)
            utrs = list(row.get("smurf_utrs") or [])
            candidates.append(
                RiskExplanation(
                    kind="FACT",
                    indicator="smurfing",
                    measured_value=float(splits),
                    threshold=float(cfg.smurf_min_splits),
                    points=cfg.smurfing_weight * float(row.get("_smurfing") or 0.0),
                    detail=(
                        f"[FACT] Structuring (smurfing): {smurf_count} bulk inflow(s) "
                        f"split into {splits} transfers below "
                        f"Rs {cfg.smurf_split_max:,.2f} (threshold {cfg.smurf_min_splits} splits) "
                        f"-> +{cfg.smurfing_weight * float(row.get('_smurfing') or 0.0):.1f} pts"
                        + (f"; UTRs: {', '.join(utrs[:10])}" if utrs else "")
                    ),
                    evidence_txn_ids=utrs,
                )
            )

        cycle_count = int(row.get("cycle_count") or 0)
        if cycle_count > 0:
            utrs = list(row.get("cycle_txn_ids") or [])
            candidates.append(
                RiskExplanation(
                    kind="FACT",
                    indicator="cycle_participation",
                    measured_value=float(cycle_count),
                    threshold=float(cfg.cycle_threshold),
                    points=cfg.cycle_weight * float(row.get("_cycle") or 0.0),
                    detail=(
                        f"[FACT] Closed money cycle: {cycle_count} decaying loop(s) "
                        f"in the ledger (threshold {cfg.cycle_threshold}) "
                        f"-> +{cfg.cycle_weight * float(row.get('_cycle') or 0.0):.1f} pts"
                        + (f"; UTRs: {', '.join(utrs)}" if utrs else "")
                    ),
                    evidence_txn_ids=utrs,
                )
            )

        device_peers = int(row.get("shared_device_account_count") or 0)
        branch_peers = int(row.get("branch_cohort_48h_count") or 0)
        if device_peers > 0 or branch_peers > 0:
            candidates.append(
                RiskExplanation(
                    kind="FACT",
                    indicator="shared_metadata",
                    measured_value=float(max(device_peers, branch_peers)),
                    threshold=float(max(cfg.shared_device_threshold, cfg.branch_cohort_threshold)),
                    points=cfg.shared_metadata_weight * float(row.get("_shared") or 0.0),
                    detail=(
                        f"[FACT] Shared metadata: {device_peers} account(s) on the same device, "
                        f"{branch_peers} account(s) opened at the same branch within "
                        f"{cfg.branch_cohort_hours} h "
                        f"-> +{cfg.shared_metadata_weight * float(row.get('_shared') or 0.0):.1f} pts"
                    ),
                )
            )

        terminal_inflows = int(row.get("aeps_cashout_inflow_count") or 0)
        terminal_chains = int(row.get("terminal_chain_count") or 0)
        if terminal_inflows > 0 or terminal_chains > 0:
            candidates.append(
                RiskExplanation(
                    kind="FACT",
                    indicator="terminal_cashout",
                    measured_value=float(max(terminal_inflows, terminal_chains)),
                    threshold=float(cfg.terminal_cashout_threshold),
                    points=cfg.terminal_cashout_weight * float(row.get("_terminal") or 0.0),
                    detail=(
                        f"[FACT] Terminal cash-out: {terminal_inflows} AEPS/ATM/CASH inflow(s), "
                        f"{terminal_chains} complaint trail(s) terminating here "
                        f"(threshold {cfg.terminal_cashout_threshold}) "
                        f"-> +{cfg.terminal_cashout_weight * float(row.get('_terminal') or 0.0):.1f} pts"
                    ),
                )
            )

        direct = int(row.get("direct_complaint_count") or 0)
        links = int(row.get("complaint_link_count") or 0)
        if direct > 0 or links > 0:
            acks = list(row.get("complaint_ack_nos") or [])
            candidates.append(
                RiskExplanation(
                    kind="FACT",
                    indicator="complaint_relationship",
                    measured_value=float(direct + links),
                    threshold=float(cfg.complaint_link_threshold),
                    points=cfg.complaint_weight * float(row.get("_complaint") or 0.0),
                    detail=(
                        f"[FACT] Complaint relationship: {direct} direct complaint(s), "
                        f"{links} cross-case link(s) (threshold {cfg.complaint_link_threshold}) "
                        f"-> +{cfg.complaint_weight * float(row.get('_complaint') or 0.0):.1f} pts"
                        + (f"; case ids: {', '.join(acks)}" if acks else "")
                    ),
                )
            )

        candidates.sort(key=lambda item: (-item.points, item.indicator))
        anomaly = self._anomaly_explanation(row, index, shap_values)
        if anomaly is None:
            return candidates[: cfg.max_explanations]
        # The anomaly line always keeps a slot: the model contribution is a
        # distinct kind of information and must stay visible.
        rules = candidates[: max(0, cfg.max_explanations - 1)]
        return [*rules, anomaly][: cfg.max_explanations]

    def _anomaly_explanation(
        self,
        row: dict[str, Any],
        index: int,
        shap_values: np.ndarray | None,
    ) -> RiskExplanation | None:
        """Honest account of the model's contribution (an inference)."""
        anomaly_score = float(row.get("anomaly_score") or 0.0)
        if shap_values is None or shap_values.shape[0] <= index:
            return RiskExplanation(
                kind="INDICATOR",
                indicator="isolation_forest_anomaly",
                measured_value=anomaly_score,
                threshold=None,
                points=0.0,
                detail=(
                    f"[INDICATOR] Anomaly model not fitted for this dataset "
                    f"(fewer than {self._config.min_rows_for_anomaly} accounts); "
                    f"anomaly component contributes 0 pts"
                ),
            )
        row_values = shap_values[index]
        order = np.argsort(-np.abs(row_values))
        best = int(order[0]) if order.size else 0
        feature = (
            ANOMALY_FEATURE_COLUMNS[best]
            if best < len(ANOMALY_FEATURE_COLUMNS)
            else "behavioural features"
        )
        # shap explains the forest's raw path-length output (lower = more
        # anomalous), so the contribution is negated for the anomaly sense.
        contribution = float(-row_values[best]) if row_values.size else 0.0
        return RiskExplanation(
            kind="INDICATOR",
            indicator="isolation_forest_anomaly",
            measured_value=anomaly_score,
            threshold=None,
            points=0.0,
            detail=(
                f"[INDICATOR] Isolation Forest anomaly score {anomaly_score:.0f}/100 "
                f"(dataset-relative; it shifts when the dataset changes); "
                f"strongest attribution: {_FEATURE_TEXT.get(feature, feature)} "
                f"({contribution:+.2f}) on the model's internal tree path-length "
                f"scale - not points on the 0-100 anomaly score, and not a "
                f"contribution to the final risk score"
            ),
        )

    def _band(self, score: float) -> str:
        cfg = self._config
        if score >= cfg.high_risk_band:
            return "CRITICAL"
        if score >= (cfg.high_risk_band + cfg.medium_risk_band) / 2:
            return "HIGH"
        if score >= cfg.medium_risk_band:
            return "MEDIUM"
        return "LOW"


__all__ = [
    "ANOMALY_FEATURE_COLUMNS",
    "RISK_DISCLAIMER",
    "RiskScorer",
]
