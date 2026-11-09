import math

from conftest import make_snapshot
from sentinel.collector import normalise_for_explain, summarise_plan
from sentinel.features import FEATURE_COLUMNS, _window, build_feature_rows


def series(means, calls_per_window=10, stddev=0.0):
    """Build cumulative snapshots whose per-window means are `means`."""
    snaps, calls, total = [make_snapshot(0, {"q": (0, 0, 0)})], 0, 0.0
    for i, m in enumerate(means, start=1):
        calls += calls_per_window
        total += m * calls_per_window
        snaps.append(make_snapshot(i * 30, {"q": (calls, total, stddev)}))
    return snaps


def test_window_mean_is_diff_of_cumulative_counters():
    prev = {"calls": 10, "total_exec_time": 100, "stddev_exec_time": 0, "rows": 10,
            "shared_blks_hit": 50, "shared_blks_read": 50, "temp_blks_written": 0}
    cur = {**prev, "calls": 30, "total_exec_time": 500, "shared_blks_hit": 150, "shared_blks_read": 50}
    w = _window(prev, cur)
    assert w["calls"] == 20 and w["mean_ms"] == 20.0
    assert w["shared_hit_ratio"] == 1.0  # only hits in this window


def test_window_stddev_is_exact_for_the_window():
    # 10 calls of 1ms, then 10 calls of 3ms: lifetime stddev 1.0, window stddev 0.
    prev = {"calls": 10, "total_exec_time": 10, "stddev_exec_time": 0.0, "rows": 0,
            "shared_blks_hit": 0, "shared_blks_read": 0, "temp_blks_written": 0}
    cur = {**prev, "calls": 20, "total_exec_time": 40, "stddev_exec_time": 1.0}
    w = _window(prev, cur)
    assert w["mean_ms"] == 3.0
    assert math.isclose(w["stddev_ms"], 0.0, abs_tol=1e-9)


def test_counter_reset_is_skipped():
    prev = {"calls": 100, "total_exec_time": 1000, "stddev_exec_time": 0, "rows": 0,
            "shared_blks_hit": 0, "shared_blks_read": 0, "temp_blks_written": 0}
    cur = {**prev, "calls": 5, "total_exec_time": 50}
    assert _window(prev, cur) is None


def test_rows_have_all_features_and_no_future_leakage():
    snaps = series([1, 1, 1, 1, 10, 10])
    rows = build_feature_rows(snaps, min_calls=5)
    assert rows, "expected feature rows"
    for r in rows:
        assert set(FEATURE_COLUMNS) <= r.keys()
    # the row at snapshot 4 (mean 1) must not see the spike that starts at snapshot 5
    r4 = next(r for r in rows if r["snap_idx"] == 4)
    assert r4["window_mean_ms"] == 1 and r4["ratio_to_baseline"] == 1


def test_baseline_is_median_of_trailing_windows():
    rows = build_feature_rows(series([1, 1, 50, 1, 1, 2]), min_calls=5, baseline_windows=5)
    last = rows[-1]
    assert last["baseline_ms"] == 1  # median ignores the single 50ms outlier
    assert last["ratio_to_baseline"] == 2


def test_low_traffic_windows_are_ignored():
    rows = build_feature_rows(series([1, 1, 1, 1], calls_per_window=2), min_calls=5)
    assert rows == []


def test_plan_change_detected():
    snaps = series([1, 1, 1, 1])
    snaps[-1]["plans"]["q"] = {**snaps[-1]["plans"]["q"], "shape": "xyz", "total_cost": 40.0}
    last = build_feature_rows(snaps)[-1]
    assert last["plan_changed"] == 1 and last["cost_change_ratio"] == 4.0


def test_summarise_plan_counts_nodes_and_relations():
    plan = [{"Plan": {"Node Type": "Hash Join", "Total Cost": 120.5, "Plan Rows": 40, "Plans": [
        {"Node Type": "Seq Scan", "Relation Name": "orders"},
        {"Node Type": "Hash", "Plans": [{"Node Type": "Index Scan", "Relation Name": "customers"}]},
    ]}}]
    s = summarise_plan(plan)
    assert s["total_cost"] == 120.5 and s["relations"] == ["customers", "orders"]
    assert s["nodes"]["Hash Join"] == 1 and s["nodes"]["Seq Scan"] == 1 and s["nodes"]["Index Scan"] == 1


def test_typed_literal_params_become_casts():
    q = "SELECT 1 FROM t WHERE ts > now() - interval $2 AND d = date $3 AND x = $1"
    assert normalise_for_explain(q) == "SELECT 1 FROM t WHERE ts > now() - $2::interval AND d = $3::date AND x = $1"
