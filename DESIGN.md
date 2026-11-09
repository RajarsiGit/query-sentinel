# Query Sentinel — Design

**Author:** Rajarsi · **Status:** v1 · **Last updated:** 2026-11

## 1. Problem

Most PostgreSQL incidents on a busy fleet are not outages. They are *regressions*: a query that ran in 4 ms last week now runs in 400 ms because a table grew past the point where the planner switches from an index scan to a hash join, because autovacuum fell behind, because statistics went stale after a bulk load, or because someone dropped an "unused" index.

Threshold alerts on p95 latency fire *after* users feel it. Query Sentinel tries to fire *before*: given the current state of a query and the tables it touches, predict whether that query's mean latency will regress by more than `REGRESSION_RATIO` (default 1.5×) within the next horizon `H` (default 3 snapshots).

## 2. Goals and non-goals

**Goals**
- A production-shaped MLOps loop on SageMaker AI: versioned data → pipeline → quality gate → model registry → approval-driven deployment → scoring → drift monitoring.
- Features that a DBA would recognise as causal, not just correlated.
- Run the entire data side for free on a laptop (Docker Postgres 16), so AWS spend is only training and inference.
- One-command deploy and teardown.

**Non-goals (v1)**
- Root-cause explanation beyond feature attributions.
- Automatic remediation (creating indexes, running `ANALYZE`). Sentinel recommends; humans act.
- Multi-tenant fleet scoring. v1 scores one database; the design notes how to scale out in §9.

## 3. Architecture

```
 ┌──────────── data plane (local or ECS) ────────────┐        ┌──────────────────── SageMaker AI ─────────────────────┐
 │                                                    │        │                                                       │
 │  Postgres 16 ◄── workload generator + chaos events │        │  Pipeline: Preprocess ─► Train (XGBoost, Spot)        │
 │      │                                             │  S3    │            ─► Evaluate ─► Condition(PR-AUC ≥ min)     │
 │      ▼                                             │ ─────► │                 ├─ yes ─► RegisterModel (Pending)      │
 │  collector: pg_stat_statements deltas,             │dataset │                 └─ no  ─► Fail                         │
 │  EXPLAIN (GENERIC_PLAN), pg_stat_user_tables       │        │                                                       │
 │      │                                             │        │  Model Registry ── Approved ──► EventBridge            │
 │      ▼                                             │        │                                     │                 │
 │  sentinel.features (shared code) ─► labelled CSV   │        │                         deployer Lambda               │
 │                                                    │        │                                     ▼                 │
 │  scorer ◄────────── invoke ─────────────────────── │ ◄───── │                   Serverless Inference endpoint       │
 │    │  writes sentinel.scores (Grafana)             │        └───────────────────────────────────────────────────────┘
 │    ├─► SNS alert when p(regression) ≥ threshold    │
 │    └─► S3 feature log ─► PSI drift job ─► SNS      │
 └────────────────────────────────────────────────────┘
```

## 4. Data

### 4.1 Source signals (per snapshot, every `SNAPSHOT_INTERVAL` seconds)

| Source | What we take | Why |
|---|---|---|
| `pg_stat_statements` | calls, total/mean/stddev exec time, rows, shared blks hit/read, temp blks written | Cumulative counters → we diff consecutive snapshots to get per-window behaviour |
| `EXPLAIN (FORMAT JSON, GENERIC_PLAN)` | total cost, plan rows, node-type counts, relations touched | PG16's `GENERIC_PLAN` lets us explain the *normalised* `$1` text from pg_stat_statements without guessing parameters |
| `pg_stat_user_tables` + `pg_class` | live/dead tuples, seconds since last (auto)analyze, relation size, indexes | Table growth and stats staleness are the classic regression causes |

### 4.2 Labelling

For query `q` at snapshot `t`:

- `window_mean(t)` = Δtotal_exec_time / Δcalls between `t-1` and `t` (skip windows with Δcalls < `MIN_CALLS`).
- `baseline(t)` = median of `window_mean` over the trailing `BASELINE_WINDOWS` snapshots.
- `label(t) = 1` if `max(window_mean(t+1 … t+H)) ≥ REGRESSION_RATIO × baseline(t)`, else 0.

The label looks only forward; features look only backward. Rows near the end of the series with an incomplete horizon are dropped, not labelled 0.

**We predict onset, not state.** Rows where the query is *already* regressed at `t` (`ratio_to_baseline ≥ REGRESSION_RATIO`) are excluded from training. Otherwise the model learns "slow now → slow soon", scores well, and is useless. The scorer reports already-regressed queries by rule with status `regressed`.

### 4.3 Features (`sentinel/features.py` — the single source of truth)

Latency shape: `window_mean_ms`, `baseline_ms`, `ratio_to_baseline`, `stddev_ms`, `trend_slope` (last 3 windows).
Work per call: `calls_per_sec`, `rows_per_call`, `shared_hit_ratio`, `blks_read_per_call`, `temp_blks_per_call`.
Plan: `plan_total_cost`, `plan_rows`, `n_seq_scan`, `n_index_scan`, `n_nested_loop`, `n_hash_join`, `n_sort`, `cost_change_ratio`, `plan_changed`.
Tables (summed over relations in the plan): `live_tuples`, `tuple_growth_rate`, `dead_tuple_ratio`, `max_secs_since_analyze`, `mod_since_analyze_ratio`, `rel_size_mb`.

**Training/serving skew guard:** both the dataset builder and the scorer import the same `build_feature_rows()` function. There is no second implementation.

### 4.4 Splits

Time-based, never random: first 70% of snapshots → train, next 15% → validation, last 15% → test. A random split leaks future table sizes into training and inflates PR-AUC badly.

## 5. Model

XGBoost (SageMaker built-in container, `binary:logistic`, `eval_metric=aucpr`), `scale_pos_weight` set from the training class ratio. Reasons: tabular data, few thousand to few hundred thousand rows, needs to be cheap to train on Spot and fast on a serverless endpoint, and tree models give usable per-feature attributions.

Primary metric is **PR-AUC**, not ROC-AUC, because positives are rare (typically 5–15%). The gate also reports precision at the alerting threshold, because an alert with 20% precision gets muted within a week.

Baseline to beat: a heuristic rule (`ratio_to_baseline > 1.2 AND tuple_growth_rate > 0`). The evaluation step reports both. If XGBoost doesn't beat the heuristic, that's a finding worth writing up, not hiding.

## 6. SageMaker pipeline

Defined as raw pipeline-definition JSON in `pipeline/definition.py` and upserted with boto3. That is deliberate: the SDK moved from v2 to v3 with breaking changes; the JSON schema did not. Every step's arguments map 1:1 to the corresponding `Create*Job` API, which is easier to review and keeps the pipeline independent of SDK churn.

| Step | Type | Detail |
|---|---|---|
| `Preprocess` | Processing (sklearn container, ml.m5.large) | Time split, writes headerless label-first CSVs + `baseline_stats.json` for drift |
| `Train` | Training (built-in XGBoost, **managed Spot**) | Checkpointing to S3 so Spot interruptions resume |
| `Evaluate` | Processing (XGBoost container) | PR-AUC, ROC-AUC, precision/recall at threshold, heuristic baseline → `evaluation.json` |
| `QualityGate` | Condition | `pr_auc ≥ MinPrAuc` **and** `pr_auc ≥ heuristic_pr_auc` |
| `Register` | RegisterModel | `PendingManualApproval`, metrics attached to the model package |
| `Rejected` | Fail | Pipeline fails loudly with the metric in the message |

Parameters: `InputDataUri`, `MinPrAuc`, `TrainInstanceType`, `MaxDepth`, `Eta`, `NumRound`.

## 7. Deployment and inference

- A human approves the model package in the registry (console or `make approve`).
- EventBridge rule on *SageMaker Model Package State Change* with `ModelApprovalStatus = Approved` invokes the **deployer Lambda**, which creates a Model, a new EndpointConfig, and creates or updates the endpoint. Endpoint updates are blue/green by default, so traffic shifts only after the new variant is healthy.
- Endpoint is **Serverless Inference** (2048 MB, max concurrency 2). Scoring traffic is a burst every few minutes; paying for an always-on instance would be ~20× the cost.
- The **scorer** (CLI; runs locally, or as a scheduled ECS task in production) takes two snapshots, builds features with the shared code, invokes the endpoint, writes rows to `sentinel.scores`, and publishes to SNS above `ALERT_THRESHOLD`.

## 8. Monitoring

Serverless endpoints don't support SageMaker data capture, so SageMaker Model Monitor's managed data-quality jobs aren't available on them. Sentinel does it explicitly instead: the scorer appends every scored feature row to S3, and `sentinel drift` computes the Population Stability Index of each feature against `baseline_stats.json` produced by the pipeline. PSI > 0.2 on any top-10 feature publishes an SNS alert and is the trigger to re-run the pipeline.

The Grafana dashboard (`monitoring/grafana-dashboard.json`) reads `sentinel.scores`: top at-risk queries, score history per query, and alert volume.

## 9. Cost (ap-south-1, estimated — verify against the current price list)

| Item | Usage | Rough monthly |
|---|---|---|
| Processing (2 × ml.m5.large, ~5 min each) | 8 pipeline runs | < $1 |
| Training (ml.m5.xlarge Spot, ~5 min) | 8 runs | < $1 |
| Serverless Inference (2 GB) | 288 invocations/day × ~100 ms | < $2 |
| S3, Lambda, EventBridge, SNS | negligible | < $1 |
| **Total** | | **≈ $5/month** |

The data side (Postgres, workload, collector) runs on Docker locally and costs nothing.

## 10. Risks and open questions

- **Synthetic data.** The workload is designed to reproduce real failure modes, but a model trained on it is a demo, not a product. The README says so. The pipeline is the transferable part.
- **Label noise.** Latency spikes from noisy neighbours look like regressions. `MIN_CALLS` and the median baseline reduce this; per-host load features would help further.
- **`GENERIC_PLAN` ≠ custom plan.** Postgres may choose a custom plan for specific parameters. Generic plans are still the right signal for "what does the planner fall back to", but they can disagree with what actually ran.
- **Scaling to a fleet.** Score each database in its own ECS task, write features to S3 partitioned by `cluster_id`, and add `cluster_id`-level features (instance class, IOPS headroom). The model and pipeline are unchanged.
