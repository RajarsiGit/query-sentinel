# Query Sentinel

**Predict PostgreSQL query regressions before users feel them, with a production-shaped MLOps loop on Amazon SageMaker AI.**

A query that ran in 4 ms last week runs in 400 ms today because a table grew past the point where the planner changes strategy, autovacuum fell behind, statistics went stale after a bulk load, or someone dropped an "unused" index. Latency alerts fire *after* that happens. Query Sentinel scores every query every few minutes and flags the ones likely to regress within the next few snapshots (default horizon: 3 × 30 s, configurable), using signals a DBA would recognise: plan shape and cost, table growth, dead tuples, and time since the last ANALYZE.

> The model is trained on a synthetic workload that reproduces real failure modes. Treat the model as a demo; the pipeline, the feature contract and the deployment loop are the transferable parts. See [DESIGN.md](DESIGN.md).

## How it works

```
Postgres 16 ──► collector ──► shared features ──► labelled CSV ──► S3
 (workload +    pg_stat_statements, EXPLAIN (GENERIC_PLAN),               │
  chaos)        pg_stat_user_tables                                       ▼
                                                    SageMaker Pipeline: Preprocess → Train (Spot) → Evaluate
                                                                        → QualityGate → Register (pending)
                                                                                           │ approve
                                                    EventBridge → deployer Lambda → Serverless endpoint
                                                                                           │
scorer (every 30s) ── same feature code ── invoke ◄────────────────────────────────────────┘
   ├── sentinel.scores → Grafana
   ├── SNS alert when p ≥ 0.7
   └── S3 feature log → PSI drift check → SNS → retrain
```

Choices worth noticing:

- **One feature implementation** (`sentinel/features.py`) used by both training and scoring, so there's no training/serving skew.
- **Onset labels, not state labels.** A query that's already slow isn't a prediction problem; those rows are excluded from training and reported by rule.
- **Time-based splits.** Random splits leak future table sizes and inflate PR-AUC.
- **The quality gate compares the model against a heuristic**, not only an absolute threshold. If XGBoost can't beat a two-line rule, the model doesn't ship.
- **Raw pipeline-definition JSON** instead of SDK classes, because SageMaker Python SDK v3 replaced the v2 APIs while the pipeline schema stayed stable.
- **Managed Spot training + Serverless Inference**: the AWS side costs about $5/month at the default cadence.
- **Custom PSI drift monitoring**, because serverless endpoints don't support data capture for Model Monitor.

## Quickstart

### 1. Local data plane (free)

```bash
make up install            # Postgres 16 + Grafana in Docker; pip install -e ".[dev]"
make data MINUTES=90       # seed, run workload + chaos, collect snapshots every 30s
make dataset               # -> data/dataset.csv
python scripts/local_pipeline.py   # preprocess -> train -> evaluate locally, same scripts as SageMaker
```

### 2. AWS (SageMaker AI)

```bash
make infra EMAIL=you@example.com   # bucket, roles, model registry, deployer, alerts (confirm the SNS email)
eval "$(make -s outputs)"          # exports SENTINEL_BUCKET, SENTINEL_PIPELINE_ROLE_ARN, ...
make pipeline                      # create/update the SageMaker pipeline
make train                         # upload dataset, start an execution
make status                        # watch steps
make approve                       # approve the model -> Lambda deploys the serverless endpoint
make score                         # score live queries every 30s; Grafana at http://localhost:3000
```

Tear everything down with `make destroy`.

## Repository layout

```
sentinel/            collector, shared features, labelling, workload + chaos, scorer, drift, CLI
pipeline/            pipeline definition (JSON), deploy/start/approve CLI, step scripts
infra/               CDK app: bucket, roles, model package group, deployer Lambda, EventBridge, SNS
monitoring/          Grafana dashboard + provisioning
scripts/             local pipeline runner, CI smoke test
tests/               unit tests for features, labels, pipeline references, drift
DESIGN.md            problem, data, labels, model, ops, cost, risks
```

## Results

Fill this in from your own run:

| | PR-AUC | Precision @ 0.7 | Recall @ 0.7 |
|---|---|---|---|
| Heuristic baseline | | | |
| XGBoost | | | |

Top features by gain:

## Roadmap

- Fleet mode: one scorer per cluster on ECS, `cluster_id` features, a shared model
- SHAP attributions in the alert ("likely cause: stats stale on `events`")
- Hyperparameter tuning step in the pipeline
- Shadow-test new models against the live endpoint before approval
