"""Feature engineering — the ONLY implementation, shared by training and scoring.

`build_feature_rows(snapshots)` turns a time-ordered list of snapshots into one
row per (query, snapshot) where that query ran at least MIN_CALLS times in the
window ending at that snapshot. Every feature uses only the current and past
snapshots, never the future; labels are added separately in labeling.py.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

FEATURE_COLUMNS: list[str] = [
    # latency shape
    "window_mean_ms", "baseline_ms", "ratio_to_baseline", "window_stddev_ms", "trend_slope",
    # work per call
    "calls_per_sec", "rows_per_call", "shared_hit_ratio", "blks_read_per_call", "temp_blks_per_call",
    # plan
    "plan_total_cost", "plan_rows", "n_seq_scan", "n_index_scan", "n_nested_loop", "n_hash_join",
    "n_sort", "cost_change_ratio", "plan_changed",
    # tables touched by the plan
    "live_tuples", "tuple_growth_rate", "dead_tuple_ratio", "max_secs_since_analyze",
    "mod_since_analyze_ratio", "rel_size_mb",
]

_EPS = 1e-9


def _window(prev: dict[str, float], cur: dict[str, float]) -> dict[str, float] | None:
    """Diff two cumulative pg_stat_statements rows into one window's behaviour."""
    d_calls = cur["calls"] - prev["calls"]
    d_time = cur["total_exec_time"] - prev["total_exec_time"]
    if d_calls <= 0 or d_time < 0:  # no traffic, or counters were reset / entry evicted
        return None
    # pg_stat_statements keeps a population stddev; recover each side's sum of squares
    # so the window's own stddev is exact rather than the lifetime one.
    def sumsq(r: dict[str, float]) -> float:
        mean = r["total_exec_time"] / r["calls"] if r["calls"] else 0.0
        return r["calls"] * (r["stddev_exec_time"] ** 2 + mean ** 2)

    w_mean = d_time / d_calls
    w_var = max(0.0, (sumsq(cur) - sumsq(prev)) / d_calls - w_mean ** 2)
    d_hit = cur["shared_blks_hit"] - prev["shared_blks_hit"]
    d_read = cur["shared_blks_read"] - prev["shared_blks_read"]
    return {
        "calls": d_calls,
        "mean_ms": w_mean,
        "stddev_ms": math.sqrt(w_var),
        "rows_per_call": (cur["rows"] - prev["rows"]) / d_calls,
        "shared_hit_ratio": d_hit / (d_hit + d_read) if (d_hit + d_read) > 0 else 1.0,
        "blks_read_per_call": d_read / d_calls,
        "temp_blks_per_call": (cur["temp_blks_written"] - prev["temp_blks_written"]) / d_calls,
    }


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xbar = (n - 1) / 2
    ybar = sum(values) / n
    num = sum((i - xbar) * (v - ybar) for i, v in enumerate(values))
    den = sum((i - xbar) ** 2 for i in range(n))
    return num / den if den else 0.0


def _table_features(rels: Iterable[str], tables: dict[str, dict[str, float]],
                    prev_tables: dict[str, dict[str, float]], dt: float) -> dict[str, float]:
    rels = [r for r in rels if r in tables]
    live = sum(tables[r]["n_live_tup"] for r in rels)
    prev_live = sum(prev_tables.get(r, tables[r])["n_live_tup"] for r in rels)
    dead = sum(tables[r]["n_dead_tup"] for r in rels)
    mods = sum(tables[r]["n_mod_since_analyze"] for r in rels)
    return {
        "live_tuples": live,
        # relative growth per minute, so it's comparable across snapshot intervals
        "tuple_growth_rate": ((live - prev_live) / max(prev_live, 1.0)) * (60.0 / max(dt, _EPS)),
        "dead_tuple_ratio": dead / max(live + dead, 1.0),
        "max_secs_since_analyze": max((tables[r]["secs_since_analyze"] for r in rels), default=0.0),
        "mod_since_analyze_ratio": mods / max(live, 1.0),
        "rel_size_mb": sum(tables[r]["size_bytes"] for r in rels) / 1_048_576,
    }


def build_feature_rows(snapshots: list[dict[str, Any]], min_calls: int = 5,
                       baseline_windows: int = 5) -> list[dict[str, Any]]:
    """Return feature rows. Each row also carries `queryid`, `query`, `snap_idx`, `ts`.

    The first row for a query appears once it has at least 2 valid windows of history,
    so that a baseline exists.
    """
    rows: list[dict[str, Any]] = []
    history: dict[str, list[float]] = {}           # queryid -> past valid window means
    last_plan: dict[str, dict[str, Any]] = {}       # queryid -> previous plan summary

    for idx in range(1, len(snapshots)):
        prev, cur = snapshots[idx - 1], snapshots[idx]
        dt = cur["ts"] - prev["ts"]
        prev_stmts = {s["queryid"]: s for s in prev["statements"]}
        for stmt in cur["statements"]:
            qid = stmt["queryid"]
            plan = cur["plans"].get(qid)
            if qid not in prev_stmts:
                if plan:
                    last_plan[qid] = plan
                continue
            w = _window(prev_stmts[qid], stmt)
            if w is None or w["calls"] < min_calls:
                if plan:
                    last_plan[qid] = plan
                continue

            past = history.setdefault(qid, [])
            if len(past) >= 1 and plan is not None:
                recent = past[-baseline_windows:]
                baseline = statistics.median(recent)
                old_plan = last_plan.get(qid, plan)
                row = {
                    "queryid": qid,
                    "query": stmt["query"],
                    "snap_idx": idx,
                    "ts": cur["ts"],
                    "window_mean_ms": w["mean_ms"],
                    "baseline_ms": baseline,
                    "ratio_to_baseline": w["mean_ms"] / max(baseline, _EPS),
                    "window_stddev_ms": w["stddev_ms"],
                    "trend_slope": _slope([*past[-2:], w["mean_ms"]]) / max(baseline, _EPS),
                    "calls_per_sec": w["calls"] / max(dt, _EPS),
                    "rows_per_call": w["rows_per_call"],
                    "shared_hit_ratio": w["shared_hit_ratio"],
                    "blks_read_per_call": w["blks_read_per_call"],
                    "temp_blks_per_call": w["temp_blks_per_call"],
                    "plan_total_cost": plan["total_cost"],
                    "plan_rows": plan["plan_rows"],
                    "n_seq_scan": plan["nodes"].get("Seq Scan", 0),
                    "n_index_scan": plan["nodes"].get("Index Scan", 0) + plan["nodes"].get("Index Only Scan", 0)
                                    + plan["nodes"].get("Bitmap Heap Scan", 0),
                    "n_nested_loop": plan["nodes"].get("Nested Loop", 0),
                    "n_hash_join": plan["nodes"].get("Hash Join", 0),
                    "n_sort": plan["nodes"].get("Sort", 0),
                    "cost_change_ratio": plan["total_cost"] / max(old_plan["total_cost"], _EPS),
                    "plan_changed": int(plan["shape"] != old_plan["shape"]),
                    **_table_features(plan["relations"], cur["tables"], prev["tables"], dt),
                }
                rows.append(row)
            past.append(w["mean_ms"])
            if plan:
                last_plan[qid] = plan
    return rows
