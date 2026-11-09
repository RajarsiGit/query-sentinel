"""End-to-end smoke test against a real Postgres 16 (used by CI).

seed -> short workload with fast chaos -> dataset -> local pipeline -> live scoring under load.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=6)
    a = p.parse_args()
    py = sys.executable
    sh(py, "-m", "sentinel.cli", "generate", "--reset", "--minutes", str(a.minutes), "--interval", "10",
       "--customers", "5000", "--orders", "30000", "--events", "50000",
       "--chaos-every", "60", "--chaos-hold", "40", "--workers", "4", "--rate", "30")
    sh(py, "-m", "sentinel.cli", "build-dataset")
    sh(py, "scripts/local_pipeline.py", "--data", "data/dataset.csv", "--workdir", "build/smoke")
    model = Path("build/smoke/model/xgboost-model")
    assert model.exists(), "model was not produced"

    # Score while the workload is running, as in production.
    from sentinel import workload
    from sentinel.config import settings
    from sentinel.scorer import LocalPredictor, score_once

    state = Path("build/smoke/scorer-state.jsonl")
    state.unlink(missing_ok=True)
    stop = threading.Event()
    threads, _ = workload.run(settings.dsn, stop, workers=4, rate_per_worker=30, chaos_every_s=0)
    predictor = LocalPredictor(model)
    results = []
    try:
        for _ in range(4):
            results = score_once(settings, predictor, state, publish=False)
            time.sleep(10)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=30)
    assert results, "scorer produced no results under load"
    for r in sorted(results, key=lambda r: -r["probability"])[:5]:
        print(f"  {r['status']:<9} p={r['probability']:.2f} {r['query'][:80]}")
    print(f"smoke test OK: scored {len(results)} queries")


if __name__ == "__main__":
    main()
