"""Take point-in-time snapshots of query and table statistics.

A snapshot is a plain dict so it can be written as one JSON line and replayed
later by the dataset builder or the scorer:

    {
      "ts": 1731170000.0,
      "statements": [{queryid, query, calls, total_exec_time, ...}, ...],
      "plans":      {"<queryid>": {total_cost, plan_rows, nodes: {...}, relations: [...], shape}},
      "tables":     {"<relname>": {n_live_tup, n_dead_tup, n_mod_since_analyze, secs_since_analyze, size_bytes}}
    }

pg_stat_statements counters are cumulative; features.py diffs consecutive snapshots.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

# Only app queries: DML/SELECT that don't touch catalogs, and nothing Sentinel itself runs.
STATEMENTS_SQL = """
SELECT s.queryid::text            AS queryid,
       s.query,
       s.calls,
       s.total_exec_time,
       s.stddev_exec_time,
       s.rows,
       s.shared_blks_hit,
       s.shared_blks_read,
       s.temp_blks_written
FROM pg_stat_statements s
JOIN pg_database d ON d.oid = s.dbid
WHERE d.datname = current_database()
  AND s.toplevel
  AND s.query ~* '^\\s*(select|insert|update|delete|with)\\s'
  AND s.query !~* '(pg_|information_schema|sentinel\\.)'
"""

TABLES_SQL = """
SELECT t.relname,
       t.n_live_tup,
       t.n_dead_tup,
       t.n_mod_since_analyze,
       EXTRACT(EPOCH FROM now() - GREATEST(t.last_analyze, t.last_autoanalyze, 'epoch'::timestamptz))
           AS secs_since_analyze,
       pg_total_relation_size(t.relid) AS size_bytes
FROM pg_stat_user_tables t
WHERE t.schemaname = 'public'
"""

TRACKED_NODES = ("Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
                 "Nested Loop", "Hash Join", "Merge Join", "Sort", "Aggregate")


def _walk(plan: dict[str, Any], nodes: Counter, relations: set[str], shape: list[str]) -> None:
    node = plan.get("Node Type", "")
    nodes[node] += 1
    shape.append(node)
    if "Relation Name" in plan:
        relations.add(plan["Relation Name"])
    for child in plan.get("Plans", []):
        _walk(child, nodes, relations, shape)


def summarise_plan(explain_json: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce EXPLAIN (FORMAT JSON) output to the fields the model uses."""
    root = explain_json[0]["Plan"]
    nodes: Counter = Counter()
    relations: set[str] = set()
    shape: list[str] = []
    _walk(root, nodes, relations, shape)
    return {
        "total_cost": float(root.get("Total Cost", 0.0)),
        "plan_rows": float(root.get("Plan Rows", 0.0)),
        "nodes": {k: nodes.get(k, 0) for k in TRACKED_NODES},
        "relations": sorted(relations),
        # A stable fingerprint of the plan's structure: detects plan flips.
        "shape": hashlib.sha1("/".join(shape).encode()).hexdigest()[:12],
    }


# pg_stat_statements turns typed literals like `interval '1 day'` into `interval $2`, which is
# not valid SQL. Rewrite them as casts (`$2::interval`) so EXPLAIN can parse the text.
_TYPED_PARAM = re.compile(
    r"\b(interval|timestamptz|timestamp|date|time|numeric|text|uuid|jsonb|json)\s+(\$\d+)", re.IGNORECASE)


def normalise_for_explain(query: str) -> str:
    return _TYPED_PARAM.sub(r"\2::\1", query)


def explain(conn: psycopg.Connection, query: str) -> dict[str, Any] | None:
    """EXPLAIN the normalised ($1, $2…) query text with PG16's GENERIC_PLAN."""
    try:
        with conn.cursor() as cur:
            # text comes from pg_stat_statements, not user input
            cur.execute("EXPLAIN (FORMAT JSON, GENERIC_PLAN) " + normalise_for_explain(query))
            return summarise_plan(cur.fetchone()[0])
    except psycopg.Error:
        conn.rollback()
        return None


def take_snapshot(conn: psycopg.Connection) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(STATEMENTS_SQL)
        statements = [dict(r) for r in cur.fetchall()]
        cur.execute(TABLES_SQL)
        tables = {r.pop("relname"): {k: float(v or 0) for k, v in r.items()} for r in cur.fetchall()}
    conn.commit()

    plans = {}
    for s in statements:
        plan = explain(conn, s["query"])
        if plan is not None:
            plans[s["queryid"]] = plan
    conn.commit()

    for s in statements:
        for k in ("calls", "total_exec_time", "stddev_exec_time", "rows",
                  "shared_blks_hit", "shared_blks_read", "temp_blks_written"):
            s[k] = float(s[k])
    return {"ts": time.time(), "statements": statements, "plans": plans, "tables": tables}


def collect(dsn: str, out_path: Path, interval_s: int, duration_s: int | None = None,
            stop_event=None) -> int:
    """Append a snapshot to `out_path` (JSON lines) every `interval_s` seconds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + duration_s if duration_s else None
    n = 0
    with psycopg.connect(dsn, application_name="sentinel-collector") as conn, out_path.open("a") as fh:
        while True:
            started = time.time()
            fh.write(json.dumps(take_snapshot(conn)) + "\n")
            fh.flush()
            n += 1
            if deadline and time.time() >= deadline:
                break
            if stop_event is not None and stop_event.is_set():
                break
            remaining = interval_s - (time.time() - started)
            if stop_event is not None:
                if stop_event.wait(max(0.0, remaining)):
                    break
            else:
                time.sleep(max(0.0, remaining))
    return n


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
