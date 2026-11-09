import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_snapshot(ts, stmts, tables=None, plans=None):
    """stmts: {queryid: (calls, total_ms, stddev_ms)} cumulative."""
    tables = tables or {"orders": {"n_live_tup": 1000.0, "n_dead_tup": 0.0, "n_mod_since_analyze": 0.0,
                                   "secs_since_analyze": 60.0, "size_bytes": 8_192_000.0}}
    statements = [{"queryid": q, "query": f"SELECT {q}", "calls": float(c), "total_exec_time": float(t),
                   "stddev_exec_time": float(sd), "rows": float(c), "shared_blks_hit": float(c * 10),
                   "shared_blks_read": 0.0, "temp_blks_written": 0.0}
                  for q, (c, t, sd) in stmts.items()]
    default_plan = {"total_cost": 10.0, "plan_rows": 1.0, "nodes": {"Index Scan": 1},
                    "relations": ["orders"], "shape": "abc"}
    plans = plans or {q: default_plan for q in stmts}
    return {"ts": float(ts), "statements": statements, "plans": plans, "tables": tables}
