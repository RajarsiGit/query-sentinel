"""Synthetic OLTP workload with scheduled "chaos" that reproduces real regression causes.

Chaos events, each paired with a recovery so the data has both onsets and quiet periods:

| event              | what it does                                         | typical victim         |
|--------------------|------------------------------------------------------|------------------------|
| bulk_load_events   | inserts a large batch of events, stats left stale    | events-by-kind count   |
| bulk_load_orders   | inserts a burst of recent orders                     | status / top-customer  |
| drop_index         | drops an index the app relies on                     | customer lookups       |
| bloat_orders       | mass UPDATE with autovacuum off -> dead tuples       | anything on orders     |
| starve_work_mem    | work_mem -> 64kB, sorts and hashes spill to disk     | aggregates, sorts      |
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import psycopg

log = logging.getLogger("sentinel.workload")

REGIONS = ["in-south", "in-west", "in-east", "in-north", "apac", "us", "ca"]
CATEGORIES = [f"cat-{i:02d}" for i in range(40)]
STATUSES = ["new", "paid", "shipped", "delivered", "cancelled", "refunded"]
EVENT_KINDS = ["view", "click", "search", "cart_add", "cart_remove", "checkout", "login", "logout"]


def seed(dsn: str, customers: int = 20_000, products: int = 2_000, orders: int = 100_000,
         events: int = 200_000) -> None:
    """Populate the schema with set-based inserts (fast) and ANALYZE."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE order_items, orders, events, products, customers RESTART IDENTITY CASCADE")
        conn.execute(
            "INSERT INTO customers (region, created_at) "
            "SELECT (%s::text[])[1 + (random() * 6)::int], now() - random() * interval '2 years' "
            "FROM generate_series(1, %s)", (REGIONS, customers))
        conn.execute(
            "INSERT INTO products (category, price) "
            "SELECT (%s::text[])[1 + (random() * 39)::int], round((5 + random() * 495)::numeric, 2) "
            "FROM generate_series(1, %s)", (CATEGORIES, products))
        conn.execute(
            "INSERT INTO orders (customer_id, status, total, created_at) "
            "SELECT 1 + (random() * (%s - 1))::bigint, (%s::text[])[1 + (random() * 5)::int], "
            "       round((10 + random() * 990)::numeric, 2), now() - random() * interval '180 days' "
            "FROM generate_series(1, %s)", (customers, STATUSES, orders))
        conn.execute(
            "INSERT INTO order_items (order_id, product_id, qty, price) "
            "SELECT o.id, 1 + (random() * (%s - 1))::bigint, 1 + (random() * 4)::int, "
            "       round((5 + random() * 495)::numeric, 2) "
            "FROM orders o, generate_series(1, 3)", (products,))
        conn.execute(
            "INSERT INTO events (customer_id, kind, created_at) "
            "SELECT 1 + (random() * (%s - 1))::bigint, (%s::text[])[1 + (random() * 7)::int], "
            "       now() - random() * interval '7 days' "
            "FROM generate_series(1, %s)", (customers, EVENT_KINDS, events))
        conn.execute("VACUUM ANALYZE")
        conn.execute("SELECT pg_stat_statements_reset()")
    log.info("seeded: %s customers, %s orders, %s events", customers, orders, events)


# ---------------------------------------------------------------------------
# Query mix. Each entry: (weight, sql, param factory). Params are bound server-side,
# so pg_stat_statements groups every execution under one normalised queryid.
# ---------------------------------------------------------------------------
@dataclass
class Bounds:
    customers: int
    products: int
    orders: int


def _mix(b: Bounds) -> list[tuple[int, str, Callable[[], tuple]]]:
    r = random.randint
    return [
        (20, "SELECT id, status, total, created_at FROM orders WHERE customer_id = %s "
             "ORDER BY created_at DESC LIMIT 20", lambda: (r(1, b.customers),)),
        (15, "SELECT o.id, p.category, i.qty, i.price FROM orders o "
             "JOIN order_items i ON i.order_id = o.id JOIN products p ON p.id = i.product_id "
             "WHERE o.id = %s", lambda: (r(1, b.orders),)),
        (6, "SELECT count(*) FROM orders WHERE status = %s AND created_at > now() - interval '1 day'",
            lambda: (random.choice(STATUSES),)),
        (3, "SELECT p.category, sum(i.qty * i.price) FROM orders o "
            "JOIN order_items i ON i.order_id = o.id JOIN products p ON p.id = i.product_id "
            "WHERE o.created_at > now() - interval '2 days' GROUP BY p.category ORDER BY 2 DESC", lambda: ()),
        (15, "SELECT id, kind, created_at FROM events WHERE customer_id = %s "
             "ORDER BY created_at DESC LIMIT 50", lambda: (r(1, b.customers),)),
        (4, "SELECT count(*) FROM events WHERE kind = %s AND created_at > now() - interval '1 hour'",
            lambda: (random.choice(EVENT_KINDS),)),
        (2, "SELECT customer_id, sum(total) AS spend FROM orders "
            "WHERE created_at > now() - interval '30 days' GROUP BY customer_id ORDER BY spend DESC LIMIT 10",
            lambda: ()),
        (10, "SELECT id, price FROM products WHERE category = %s AND price BETWEEN %s AND %s",
             lambda: (random.choice(CATEGORIES), r(5, 200), r(200, 500))),
        (8, "INSERT INTO events (customer_id, kind) VALUES (%s, %s)",
            lambda: (r(1, b.customers), random.choice(EVENT_KINDS))),
        (4, "INSERT INTO orders (customer_id, status, total) VALUES (%s, 'new', %s)",
            lambda: (r(1, b.customers), r(10, 1000))),
        (5, "UPDATE orders SET status = %s WHERE id = %s",
            lambda: (random.choice(STATUSES), r(1, b.orders))),
        (3, "SELECT c.region, count(o.id) FROM customers c LEFT JOIN orders o ON o.customer_id = c.id "
            "WHERE c.id BETWEEN %s AND %s GROUP BY c.region",
            lambda: (lambda lo: (lo, lo + 200))(r(1, b.customers - 200))),
    ]


def _bounds(dsn: str) -> Bounds:
    with psycopg.connect(dsn) as conn:
        c, p, o = conn.execute(
            "SELECT (SELECT max(id) FROM customers), (SELECT max(id) FROM products), (SELECT max(id) FROM orders)"
        ).fetchone()
    return Bounds(c, p, o)


def _worker(dsn: str, bounds: Bounds, stop: threading.Event, rate_per_s: float) -> None:
    mix = _mix(bounds)
    weights = [m[0] for m in mix]
    pause = 1.0 / rate_per_s if rate_per_s > 0 else 0.0
    with psycopg.connect(dsn, autocommit=True, application_name="shop-app") as conn:
        while not stop.is_set():
            _, sql, params = random.choices(mix, weights=weights, k=1)[0]
            try:
                conn.execute(sql, params(), prepare=False)
            except psycopg.Error as exc:  # chaos can briefly break things; keep going
                log.debug("query failed: %s", exc)
            if pause:
                time.sleep(pause * random.uniform(0.5, 1.5))


# ---------------------------------------------------------------------------
# Chaos
# ---------------------------------------------------------------------------
def _bulk_load_events(conn: psycopg.Connection, b: Bounds) -> Callable[[], None]:
    conn.execute("ALTER TABLE events SET (autovacuum_enabled = false)")
    n = random.choice([150_000, 300_000, 500_000])
    conn.execute(
        "INSERT INTO events (customer_id, kind, created_at) "
        "SELECT 1 + (random() * (%s - 1))::bigint, (%s::text[])[1 + (random() * 7)::int], "
        "       now() - random() * interval '30 minutes' FROM generate_series(1, %s)",
        (b.customers, EVENT_KINDS, n))

    def recover() -> None:
        conn.execute("ALTER TABLE events RESET (autovacuum_enabled)")
        conn.execute("ANALYZE events")
    return recover


def _bulk_load_orders(conn: psycopg.Connection, b: Bounds) -> Callable[[], None]:
    n = random.choice([50_000, 100_000])
    conn.execute(
        "INSERT INTO orders (customer_id, status, total, created_at) "
        "SELECT 1 + (random() * (%s - 1))::bigint, (%s::text[])[1 + (random() * 5)::int], "
        "       round((10 + random() * 990)::numeric, 2), now() - random() * interval '12 hours' "
        "FROM generate_series(1, %s)", (b.customers, STATUSES, n))
    return lambda: conn.execute("ANALYZE orders")


_INDEXES = {
    "orders_customer_idx": "CREATE INDEX orders_customer_idx ON orders (customer_id, created_at DESC)",
    "events_customer_idx": "CREATE INDEX events_customer_idx ON events (customer_id, created_at DESC)",
    "orders_created_idx": "CREATE INDEX orders_created_idx ON orders (created_at)",
}


def _drop_index(conn: psycopg.Connection, b: Bounds) -> Callable[[], None]:
    name = random.choice(list(_INDEXES))
    conn.execute(f"DROP INDEX IF EXISTS {name}")
    return lambda: conn.execute(_INDEXES[name].replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS"))


def _bloat_orders(conn: psycopg.Connection, b: Bounds) -> Callable[[], None]:
    conn.execute("ALTER TABLE orders SET (autovacuum_enabled = false)")
    conn.execute("UPDATE orders SET total = total WHERE id % 3 = 0")

    def recover() -> None:
        conn.execute("ALTER TABLE orders RESET (autovacuum_enabled)")
        conn.execute("VACUUM ANALYZE orders")
    return recover


def _starve_work_mem(conn: psycopg.Connection, b: Bounds) -> Callable[[], None]:
    conn.execute("ALTER SYSTEM SET work_mem = '64kB'")
    conn.execute("SELECT pg_reload_conf()")

    def recover() -> None:
        conn.execute("ALTER SYSTEM RESET work_mem")
        conn.execute("SELECT pg_reload_conf()")
    return recover


CHAOS = {
    "bulk_load_events": _bulk_load_events,
    "bulk_load_orders": _bulk_load_orders,
    "drop_index": _drop_index,
    "bloat_orders": _bloat_orders,
    "starve_work_mem": _starve_work_mem,
}


def _chaos_loop(dsn: str, bounds: Bounds, stop: threading.Event, every_s: int, hold_s: int,
                events_log: list[dict]) -> None:
    """Every ~`every_s` seconds, inject one chaos event, hold it for `hold_s`, then recover."""
    with psycopg.connect(dsn, autocommit=True, application_name="sentinel-chaos") as conn:
        # quiet period first so every query gets a clean baseline
        if stop.wait(every_s):
            return
        while not stop.is_set():
            name = random.choice(list(CHAOS))
            log.info("chaos: %s", name)
            started = time.time()
            recover = CHAOS[name](conn, bounds)
            events_log.append({"event": name, "start": started})
            stop.wait(hold_s)
            recover()
            events_log[-1]["end"] = time.time()
            log.info("chaos: recovered from %s", name)
            if stop.wait(random.uniform(0.5, 1.5) * every_s):
                break
        # leave the database in a clean state
        conn.execute("ALTER SYSTEM RESET work_mem")
        conn.execute("SELECT pg_reload_conf()")
        for ddl in _INDEXES.values():
            conn.execute(ddl.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS"))


def run(dsn: str, stop: threading.Event, workers: int = 8, rate_per_worker: float = 40.0,
        chaos_every_s: int = 300, chaos_hold_s: int = 150) -> tuple[list[threading.Thread], list[dict]]:
    """Start workers and the chaos loop in background threads. Returns (threads, chaos_log)."""
    bounds = _bounds(dsn)
    chaos_log: list[dict] = []
    threads = [threading.Thread(target=_worker, args=(dsn, bounds, stop, rate_per_worker), daemon=True)
               for _ in range(workers)]
    if chaos_every_s > 0:
        threads.append(threading.Thread(target=_chaos_loop,
                                        args=(dsn, bounds, stop, chaos_every_s, chaos_hold_s, chaos_log),
                                        daemon=True))
    for t in threads:
        t.start()
    return threads, chaos_log
