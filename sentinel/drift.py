"""Population Stability Index of live (scored) features vs. the training baseline.

`baseline_stats.json` is produced by the pipeline's Preprocess step:
    {"features": {"<name>": {"edges": [...], "expected": [...]}, ...}, "n_train": 1234}
"""
from __future__ import annotations

import bisect
import json
import math
from typing import Any

PSI_ALERT = 0.2
_FLOOR = 1e-4


def _proportions(values: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[bisect.bisect_right(edges, v)] += 1
    n = max(len(values), 1)
    return [c / n for c in counts]


def psi(expected: list[float], actual: list[float]) -> float:
    total = 0.0
    for e, a in zip(expected, actual, strict=True):
        e, a = max(e, _FLOOR), max(a, _FLOOR)
        total += (a - e) * math.log(a / e)
    return total


def drift_report(baseline: dict[str, Any], rows: list[dict[str, float]]) -> dict[str, float]:
    report = {}
    for name, spec in baseline["features"].items():
        values = [float(r[name]) for r in rows if name in r]
        if values:
            report[name] = psi(spec["expected"], _proportions(values, spec["edges"]))
    return dict(sorted(report.items(), key=lambda kv: -kv[1]))


def load_rows_from_s3(bucket: str, prefix: str, region: str, max_objects: int = 2000) -> list[dict]:
    import boto3
    s3 = boto3.client("s3", region_name=region)
    rows: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    seen = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode()
            rows.extend(json.loads(line) for line in body.splitlines() if line.strip())
            seen += 1
            if seen >= max_objects:
                return rows
    return rows
