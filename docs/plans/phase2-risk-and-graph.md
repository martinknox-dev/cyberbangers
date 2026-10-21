# Phase 2 — Behavioral Features, Graph Analytics & Explainable Risk

**Status:** implemented (this document is the binding Phase-2 spec, written during
implementation; `ARCHITECTURE.md` remains the system blueprint).

**Scope:** LAYER 2 (DuckDB behavioral extractor), LAYER 3 (NetworkX graph & money
trails), LAYER 4 (hybrid 70/30 risk engine with SHAP). No UI, no dossier, no
network, no service dependencies.

---

## 0. Pre-flight findings (verified before implementation)

| Question | Verdict |
| :--- | :--- |
| `docs/plans/phase2-risk-and-graph.md` existed? | **No** — this file is created by Phase 2; the architecture listed it as expected. |
| Can `shap.TreeExplainer` explain `sklearn.ensemble.IsolationForest`? | **Yes** — shap 0.54.1 + scikit-learn 1.9.1; `ex(X).values` is `(n, features)`, deterministic, no warnings. |
| What does SHAP actually explain for an IsolationForest? | The **raw path-length ensemble output**, *not* `-score_samples`. Lower raw output = more anomalous. Verified: `sum(shap) + base_value` ≠ `-score_samples`; it is the tree path-length total. |
| `networkx.louvain_communities(G, seed=…)` available? | Yes (networkx 3.7), deterministic under a fixed seed. |
| DuckDB `QUALIFY` / interval window frames? | Yes (duckdb 1.5.6). |

**Consequence for Stage C:** SHAP attributions are reported as *direction-aware
contributions to the tree path-length model* (`contribution = -shap_value`,
positive = pushes toward "more anomalous"). They are **not** presented as
contributions to the 0–100 anomaly score, which is a separate dataset-relative
min–max normalisation of `-score_samples`. Both are labelled distinctly in the
explanations.

---

## 1. Contracts (`src/core/contracts.py`)

New typed objects (all `pl.DataType` **instances**, matching the Phase-1 mypy
ruling):

* `ACCOUNT_FEATURE_SCHEMA` — one row per distinct account (behavioural only).
* `GRAPH_INDICATOR_SCHEMA` — one row per account (topology + complaint links).
* `RISK_REPORT_SCHEMA` — one row per account (0–100 score, explanations).

New pydantic models: `FeatureConfig`, `GraphConfig`, `RiskConfig`,
`MoneyTrailLeg`, `MoneyTrail`, `WashCycle`, `ComplaintLink`, `RiskExplanation`,
`RiskAssessment`.

**Explicitly excluded from every model input matrix:** `is_fraud`,
`fraud_scheme` (simulator ground truth), `account_type`, `kyc_pincode`,
`state`, `account_opening_date` (raw attribute) — architecture rule §1.4 (no
geographic/demographic prejudice). `account_opening_date` is used only to
compute the *dormancy* indicator, never as a raw additive penalty.

---

## 2. Stage A — `src/features/account_features.py`

`extract_account_features(transactions, accounts=None, complaints=None, config=None) -> pl.DataFrame`

DuckDB (in-process) over registered Polars frames; Polars for assembly.
No `iter_rows`, no Pandas, no Python row loops.

### 2.1 Configurable defaults (`FeatureConfig`)

| Parameter | Default | Meaning |
| :--- | :---: | :--- |
| `pass_through_window_minutes` | 15 | rapid pass-through window |
| `fan_out_window_minutes` | 60 | smurfing split window |
| `dormancy_threshold_days` | 180 | dormancy flag threshold |
| `burst_amount_threshold` | 25 000 | 24h burst volume that counts as "awakening" |
| `smurf_bulk_threshold` | 50 000 | bulk inflow that may start a smurfing episode |
| `smurf_min_splits` | 8 | distinct split outflows required |
| `smurf_split_min` / `smurf_split_max` | 5 000 / 9 999.99 | structuring band |
| `aggregator_min_senders` | 10 | aggregator-mule sender threshold |
| `branch_cohort_hours` | 48 | branch bulk-opening cohort |
| `terminal_modes` | AEPS, ATM, CASH | terminal cash-out leg modes |

### 2.2 Feature semantics (what each column actually measures)

* `txn_count`, `inflow_count`, `outflow_count` — leg counts (a leg is one
  transaction seen from one side).
* `inflow_total`, `outflow_total`, `net_flow` — sums of leg amounts.
* `unique_senders` / `unique_receivers` — distinct counterparties **into** /
  **out of** the account.
* `fan_in_fan_out_asymmetry` — `(unique_receivers − unique_senders) /
  max(1, unique_receivers + unique_senders)`, in `[-1, 1]`. Negative = aggregator
  (many senders, few receivers), positive = distributor.
* `pass_through_ratio_15m` — `min(1, attributed_outflow / total_inflow)` where
  each **outflow leg** is attributed to the **earliest preceding inflow of the
  same account** inside the window (FIFO, tie-broken by `txn_id`), so no outflow
  is counted twice and only the account's *own* legs participate. This is the
  key anti-contamination rule: a downstream account's transfers are never part
  of another account's outflow.
* `median_retention_minutes` — median `outflow_ts − inflow_ts` over attributed
  pairs; `null` when the account has no measured pair (unknown, not zero).
* `smurf_bulk_inflow_count` — bulk inflows (≥ ₹50 000) followed within 60 min
  by **≥ 8 distinct** structuring-band outflows.
* `smurf_split_count` — the structuring-band outflows attached to those
  qualifying bulk inflows.
* `large_payout_count` — outflows ≥ ₹50 000 in IMPS/RTGS/NEFT (aggregator leg).
* `burst_24h_max_total` — maximum **inflow** volume in any 24-hour window
  (money received; in+out would double-count a pass-through).
* `dormancy_days_before_burst` — days from `account_opening_date` to the burst
  window; `null` when the opening date is unknown. **Dormancy is never inferred
  from the absence of transactions** — an incomplete dataset is not evidence of
  inactivity.
* `dormancy_burst_score` — `1.0` when dormancy ≥ 180 days **and** burst volume
  > ₹25 000; `0.0` otherwise; `null` when dormancy is unknown.
* `shared_device_account_count` — number of **other** accounts seen on any
  shared `device_id` (ledger) or registered `imei` (KYC). A device belongs to
  the **originating (sending) account** only: the receiving side of a leg does
  not "use" the sender's handset. Peer counts are the maximum over the
  account's shared identifiers, minus the account itself.
* `branch_cohort_48h_count` — number of other accounts opened at the same
  `branch_code` within ±48 h of this account's opening date.
* `aeps_cashout_inflow_count` — inbound legs in AEPS/ATM/CASH.
* `direct_complaint_count` — distinct complaints naming the account as victim
  or as the receiver of the disputed UTR (ledger-grounded; deeper chain
  correlation is a graph-layer concern, see §3.4).
* `first_txn_timestamp`, `last_txn_timestamp` — observed activity bounds.

---

## 3. Stage B — `src/graph/analyzer.py`

`GraphAnalyzer(transactions, accounts=None, complaints=None, config=None)`

### 3.1 Graph model

`networkx.MultiDiGraph` keeps every transaction as a distinct edge keyed by
`txn_id` with `amount`, `timestamp`, `payment_mode` attributes; a collapsed
`networkx.DiGraph` is derived for centrality/community metrics. Parallel
transactions stay distinct; self-loops are allowed and handled.

### 3.2 Bounded money trail (`trace_money_trail`)

Temporal forward BFS from a disputed UTR or a victim account, `max_hops ∈ [1, 6]`,
legs ordered by `(timestamp, txn_id)`, and only legs with
`timestamp_next ≥ timestamp_prev` are traversed.

**Taint accounting — FIFO draw-down on a tainted balance** (corrected after
the Phase-2 correctness review; the earlier proportional split was replaced
because it could attribute more tainted money to a transaction than that
transaction moved). Tainted funds are *held* by an account as a tainted
balance and are drawn down first-in-first-out as the account's outgoing
transactions are processed in chronological order:

* a disputed UTR seeds the tainted balance of that transaction's receiver; an
  account seeds it with the declared amount (by default the volume that left
  the account), spendable from that account's first outgoing transfer;
* when an account sends a transaction of amount `X`, it carries
  `min(X, tainted balance held at that moment)` — never more than the
  transaction moved, never more than the account actually held;
* tainted funds that arrive later are credited at their arrival time, so a
  transfer can never carry funds that had not arrived yet, and a later arrival
  at an already-visited account is still processed (no node-once discard);
* taint that is not propagated stays with the account that held it: it is
  never invented into a downstream transfer, and no leg is attributed twice;
* the leg's hop is the tainted-route depth of the funds that funded it, and
  propagation stops at `max_hops`.

`MoneyTrail.total_taint` is the tainted amount that left the origin (the sum
of the first-hop legs). `MoneyTrail.taint_conserved` checks the origin cap,
the per-transaction cap (`taint_amount ≤ amount`) and the per-account cap;
`MoneyTrailLeg` rejects a leg whose taint exceeds its own amount.

**What cannot be concluded.** The ledger does not say *which* rupees left an
account, so the FIFO draw-down is an explicit assumption (tainted funds are
spent first). Funds that entered an account from outside the traced path are
not counted as tainted, and balances cannot be established for accounts whose
history is missing from the frame. The tainted amount is therefore an upper
bound on the disputed money that can be shown to have moved along a leg.

### 3.3 Wash-cycle detection (`detect_wash_cycles`)

Directed cycles of ≤ 5 edges whose edge timestamps span ≤ 24 h and whose
amounts decay monotonically with per-hop decay in `[0 %, 15 %]`
(`0.85 × amount_i ≤ amount_{i+1} ≤ amount_i`). Two-edge reciprocal loops
(`A → B → A`) are cycles too: a ping-pong between two mules is a wash
pattern, and the seeded branch ring produces exactly that. Cycles are
de-duplicated by canonical rotation of the node sequence and returned in
deterministic order.

The search is banded: from an edge of amount `a` only edges in
`[0.85 a, a]` can continue the cycle, so the candidate set is found by
bisecting each node's amount-ordered edge list (this keeps a 100 000-row
ledger inside a few seconds).

### 3.4 Complaint correlation (`correlate_complaints`)

Two complaints are linked only through evidence present in the data: a shared
account in their forward money trails (layer-2/3 intersection), a shared
`device_id`, or a shared terminal merchant. `ack_no` is preserved as the case
identifier; the link records the shared account/device/terminal and both
states. No state-based similarity is ever used as a link.

### 3.5 Topology metrics (`topology_metrics`)

`pagerank`, `betweenness_centrality` (collapsed, unweighted graph),
`louvain_community_id` (`louvain_communities(seed=42)`, communities ordered by
their smallest member so IDs are stable), plus cycle participation, complaint
links and maximum trail taint per account. Output conforms to
`GRAPH_INDICATOR_SCHEMA` and is sorted by `account_no`.

Exact betweenness is `O(V·E)` (≈2.6 s at 1 274 nodes / 20 000 edges), so
`GraphConfig.betweenness_k` switches to networkx's **seeded** sampled
approximation (`k` source nodes, `seed=` passed explicitly — no global RNG
state is touched).

---

## 4. Stage C — `src/risk/scorer.py`

`RiskScorer(config=None).score(features, graph=None) -> pl.DataFrame`

### 4.1 Domain score (70 %)

Weighted, evidence-based sub-scores, each in `[0, 1]`, scaled by its weight
(weights sum to 100):

| Sub-score | Weight | Trigger (graded) |
| :--- | ---: | :--- |
| Pass-through | 25 | `0.6 × min(1, ratio/0.85) + 0.4 × min(1, 15 min / median retention)`; unknown retention earns no speed credit |
| Smurfing | 20 | `min(1, split_count / 8)` when at least one bulk inflow was split |
| Cycle participation | 20 | `min(1, cycle_count / 2)` |
| Shared metadata | 10 | `min(1, max(device peers, branch peers) / 3)` |
| Terminal cash-out | 10 | `min(1, max(AEPS/ATM/CASH inflows, complaint chains terminated) / 2)` |
| Complaint relationship | 15 | `1.0` when the account is named in a complaint **or** is a cross-case link, else `0` |

**No penalty is derived from `state`, `kyc_pincode` or `account_type`.**

### 4.2 Anomaly score (30 %)

`IsolationForest(random_state=42)` fitted on the numeric **behavioural**
feature matrix. Missing / non-finite values are imputed with the column median
(0.0 when a column is entirely missing); a column of all-constant values is
harmless. Datasets smaller than `min_rows_for_anomaly` (10) skip the anomaly
component (documented). The raw anomaly magnitude `-score_samples` is
min–max normalised to `[0, 100]` **over the uploaded dataset** — scores are
therefore dataset-relative and change when the dataset changes (documented in
the API docstring and the report header).

### 4.3 Combination

`risk_score = clip(0.70 × domain_score + 0.30 × anomaly_score, 0, 100)`

### 4.4 Explanations

Up to **three** per account. Each carries: the triggered indicator, the
measured value, the applicable threshold, and the supporting UTRs where
available. Rule-level explanations are deterministic and derived from
measured facts; the SHAP-derived anomaly contribution is labelled as an
inference and states that it explains the tree path-length model, not the
0–100 score. `explanation_kinds` marks each reason `FACT` or `INDICATOR`.

The score is a **lead-ranking aid, never a probability of guilt**; the constant
`RISK_DISCLAIMER` is attached to every `RiskAssessment`.

---

## 5. Verification map

| Requirement | Test |
| :--- | :--- |
| pass-through uses own legs only, no double counting | `test_pass_through_uses_own_legs_only` |
| smurfing episode math | `test_smurfing_episode_counts` |
| dormancy unknown ≠ zero | `test_dormancy_unknown_when_no_opening_date` |
| shared device / branch cohort | `test_shared_device_and_branch_cohort` |
| terminal cash-out | `test_terminal_cashout_indicator` |
| money trail hops + chronological order | `test_trail_respects_hop_limit_and_time` |
| taint conservation | `test_taint_never_exceeds_origin` |
| cycle detection with decay | `test_wash_cycle_detected` |
| complaint correlation only via evidence | `test_complaints_link_only_through_evidence` |
| louvain stability | `test_community_ids_stable` |
| score bounds + determinism | `test_scores_bounded_and_deterministic` |
| SHAP on a real fitted IsolationForest | `test_shap_tree_explainer_works_on_isolation_forest` |
| small/empty dataset | `test_empty_and_small_datasets` |
| constant features | `test_constant_feature_columns` |
| synthetic recall (labels, isolated) | `test_synthetic_topology_recall` (labels used **only** here) |

---

## 5.4 Post-review corrections (Phase 2 correctness pass)

| Finding | Correction |
| :--- | :--- |
| Taint could exceed a transaction's own amount | Taint is drawn down FIFO from a per-account tainted balance and capped by the leg amount (§3.2); `MoneyTrailLeg` validates the cap. |
| The earliest-arrival rule attributed later funds to an earlier transfer | The ledger is swept chronologically; tainted arrivals are credited at their arrival time, so a transfer can only carry funds that had already arrived. |
| Expanding each node once discarded later, larger arrivals | Arrivals are credited even at an already-visited account; only the *edge* is consumed once, never the account. |
| Duplicate `account_no` in metadata produced extra feature rows | Account metadata is deduplicated keep-first (Phase 1 policy) before the join. |
| The graph layer raised raw Polars `ColumnNotFoundError` | Required transaction columns are checked up front and raise `GraphInputError` (a public `ValueError` carrying `missing_columns`); optional columns default to null. |
| The SHAP line could be read as points on the anomaly score | The line now states the internal tree path-length scale explicitly, and a regression test pins the attribution direction against the fitted Isolation Forest. |

---

## 6. Deviations from `ARCHITECTURE.md`

1. **The Phase-2 plan file did not exist**; this document replaces it.
2. `distinct_complaint_chains_terminated` is a **graph-layer** quantity
   (§3.4), not a feature-table column: the feature table carries the
   ledger-grounded `direct_complaint_count` only.
3. `burst_24h_max_total` measures **inflow** volume in the window; measuring
   in+out would double-count a pass-through (money in, money out).
4. `pass_through_ratio_15m` attributes each outflow to the earliest
   preceding inflow (FIFO) rather than re-summing per inflow window, which
   would count one outflow several times.
5. The SHAP contribution is reported against the IsolationForest's **raw
   path-length output**, not the 0–100 anomaly score (verified in §0).
6. **Monetary aggregates are rounded to paise** and derived ratios to 12
   decimals. DuckDB's parallel summation order is not guaranteed, and a
   last-bit difference in a float sum would break the determinism invariant
   (two runs on identical input must return identical frames).
7. A **device belongs to the sending account** only (§2.2); the receiving
   side of a leg does not inherit the sender's handset.
8. **Two-edge reciprocal loops are wash cycles** (§3.3).
9. `distinct_complaint_chains_terminated` is reported as
   `terminal_chain_count` in the graph indicators; the feature table keeps
   `aeps_cashout_inflow_count` and `direct_complaint_count`.
10. **Taint is proportional no more.** `ARCHITECTURE.md` §5.2 asks for
   "proportional taint"; the corrected implementation uses an explicit FIFO
   draw-down (§3.2), because a proportional split can attribute more tainted
   money to a transaction than that transaction moved. The tainted amount is
   still per-transaction and still conserved at the account level.
11. SHAP is applied to the IsolationForest's internal tree path-length output,
   which is *not* on the anomaly-score scale; the explanation says so (§4.4).
