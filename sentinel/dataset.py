"""Snapshots (JSON lines) -> labelled CSV that the SageMaker pipeline consumes."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .collector import load_snapshots
from .features import FEATURE_COLUMNS, build_feature_rows
from .labeling import add_labels

DATASET_COLUMNS = ["ts", "queryid", "label", *FEATURE_COLUMNS]


def build_dataset(snapshot_files: list[Path], out_csv: Path, *, min_calls: int, baseline_windows: int,
                  horizon: int, ratio: float) -> dict[str, int]:
    """Each snapshot file is one independent run; features never cross run boundaries."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DATASET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for path in snapshot_files:
            snaps = load_snapshots(path)
            rows = add_labels(build_feature_rows(snaps, min_calls, baseline_windows), horizon, ratio)
            for r in rows:
                writer.writerow(r)
                stats["rows"] += 1
                stats["positives"] += r["label"]
            stats["snapshots"] += len(snaps)
            stats["queries"] += len({r["queryid"] for r in rows})
    return dict(stats)


def summarise(stats: dict[str, int]) -> str:
    rows, pos = stats.get("rows", 0), stats.get("positives", 0)
    rate = pos / rows if rows else 0.0
    return f"{rows} rows, {pos} positives ({rate:.1%}), {stats.get('snapshots', 0)} snapshots"
