"""Score live queries: snapshot -> shared features -> model -> scores table / SNS / S3 log.

State (the last few snapshots) lives in a small JSON-lines file so the scorer can run
as a one-shot scheduled task (ECS/cron) as well as a long-running loop.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import psycopg

from .collector import take_snapshot
from .config import Settings
from .features import FEATURE_COLUMNS, build_feature_rows
from .labeling import is_already_regressed

log = logging.getLogger("sentinel.scorer")


class Predictor(Protocol):
    version: str

    def predict(self, rows: list[list[float]]) -> list[float]: ...


def _to_csv(rows: list[list[float]]) -> str:
    buf = io.StringIO()
    for r in rows:
        buf.write(",".join(repr(float(v)) for v in r) + "\n")
    return buf.getvalue()


class EndpointPredictor:
    """Calls the SageMaker Serverless endpoint (built-in XGBoost, text/csv in and out)."""

    def __init__(self, endpoint_name: str, region: str):
        import boto3
        self._rt = boto3.client("sagemaker-runtime", region_name=region)
        sm = boto3.client("sagemaker", region_name=region)
        self.endpoint_name = endpoint_name
        self.version = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointConfigName"]

    def predict(self, rows: list[list[float]]) -> list[float]:
        resp = self._rt.invoke_endpoint(EndpointName=self.endpoint_name, ContentType="text/csv",
                                        Accept="text/csv", Body=_to_csv(rows))
        text = resp["Body"].read().decode().strip()
        return [float(x) for x in text.replace("\n", ",").split(",") if x]


class LocalPredictor:
    """Loads the `xgboost-model` file from a pipeline's model.tar.gz — for dev and demos."""

    def __init__(self, model_file: Path):
        import xgboost as xgb
        self._xgb = xgb
        self._booster = xgb.Booster()
        self._booster.load_model(str(model_file))
        self.version = f"local:{model_file.name}"

    def predict(self, rows: list[list[float]]) -> list[float]:
        return [float(p) for p in self._booster.predict(self._xgb.DMatrix(rows))]


def _load_state(path: Path, keep: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()][-keep:]


def _save_state(path: Path, snaps: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for s in snaps:
            fh.write(json.dumps(s) + "\n")


def score_once(cfg: Settings, predictor: Predictor, state_file: Path, *, publish: bool = True) -> list[dict]:
    keep = cfg.baseline_windows + 3
    snaps = _load_state(state_file, keep)
    with psycopg.connect(cfg.dsn, application_name="sentinel-scorer") as conn:
        snaps.append(take_snapshot(conn))
        snaps = snaps[-keep:]
        _save_state(state_file, snaps)

        latest = len(snaps) - 1
        rows = [r for r in build_feature_rows(snaps, cfg.min_calls, cfg.baseline_windows)
                if r["snap_idx"] == latest]
        if not rows:
            log.info("not enough history yet (%d snapshots); nothing scored", len(snaps))
            return []

        candidates = [r for r in rows if not is_already_regressed(r, cfg.regression_ratio)]
        probs = predictor.predict([[r[c] for c in FEATURE_COLUMNS] for r in candidates]) if candidates else []
        results = []
        for r, p in zip(candidates, probs, strict=True):
            results.append({**r, "probability": p, "status": "at_risk" if p >= cfg.alert_threshold else "ok"})
        for r in rows:
            if is_already_regressed(r, cfg.regression_ratio):
                results.append({**r, "probability": 1.0, "status": "regressed"})

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO sentinel.scores (scored_at, queryid, query, probability, window_mean_ms, "
                "baseline_ms, model_version) VALUES (to_timestamp(%s), %s, %s, %s, %s, %s, %s)",
                [(r["ts"], int(r["queryid"]), r["query"], r["probability"], r["window_mean_ms"],
                  r["baseline_ms"], predictor.version) for r in results])
        conn.commit()

    flagged = [r for r in results if r["status"] != "ok"]
    if publish:
        _publish(cfg, flagged)
        _log_features(cfg, candidates)
    return results


def _publish(cfg: Settings, flagged: list[dict]) -> None:
    if not flagged or not cfg.sns_topic_arn:
        return
    import boto3
    lines = [f"[{r['status']}] p={r['probability']:.2f} now={r['window_mean_ms']:.1f}ms "
             f"baseline={r['baseline_ms']:.1f}ms :: {r['query'][:160]}"
             for r in sorted(flagged, key=lambda r: -r["probability"])]
    boto3.client("sns", region_name=cfg.region).publish(
        TopicArn=cfg.sns_topic_arn, Subject=f"Query Sentinel: {len(flagged)} queries flagged"[:100],
        Message="\n".join(lines))


def _log_features(cfg: Settings, rows: list[dict]) -> None:
    """Append scored feature rows to S3 for the drift job (serverless endpoints have no data capture)."""
    if not rows or not cfg.bucket:
        return
    import boto3
    now = dt.datetime.now(dt.UTC)
    key = f"scored-features/dt={now:%Y-%m-%d}/{now:%H%M%S%f}.jsonl"
    body = "\n".join(json.dumps({c: r[c] for c in FEATURE_COLUMNS}) for r in rows)
    boto3.client("s3", region_name=cfg.region).put_object(Bucket=cfg.bucket, Key=key, Body=body.encode())
