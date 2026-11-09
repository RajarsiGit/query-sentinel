"""Forward-looking labels for regression *onset*.

A row at snapshot t is labelled 1 when, within the next `horizon` snapshots, the
query's window mean reaches `ratio` x its baseline at t.

Rows where the query is *already* regressed at t (ratio_to_baseline >= ratio) are
excluded: predicting "slow now -> slow soon" is trivial and would let the model
score well while learning nothing. The scorer reports already-regressed queries
by rule instead (see scorer.py).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def add_labels(rows: list[dict[str, Any]], horizon: int = 3, ratio: float = 1.5) -> list[dict[str, Any]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_query[r["queryid"]].append(r)

    labelled: list[dict[str, Any]] = []
    for series in by_query.values():
        series.sort(key=lambda r: r["snap_idx"])
        for i, row in enumerate(series):
            if row["ratio_to_baseline"] >= ratio:
                continue
            t = row["snap_idx"]
            future = [f["window_mean_ms"] for f in series[i + 1:] if t < f["snap_idx"] <= t + horizon]
            if not future:
                continue  # incomplete horizon: unknown, not negative
            labelled.append({**row, "label": int(max(future) >= ratio * row["baseline_ms"])})
    return labelled


def is_already_regressed(row: dict[str, Any], ratio: float = 1.5) -> bool:
    return row["ratio_to_baseline"] >= ratio
