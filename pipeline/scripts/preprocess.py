"""Pipeline step 1 — time-based split + drift baseline.

Runs in the SageMaker scikit-learn processing container (pandas + numpy available).
Input : /opt/ml/processing/input/data/dataset.csv   (header: ts, queryid, label, <features>)
Output: /opt/ml/processing/train/train.csv          (no header, label first — built-in XGBoost format)
        /opt/ml/processing/validation/validation.csv
        /opt/ml/processing/test/test.csv            (with header, for evaluation)
        /opt/ml/processing/baseline/baseline_stats.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

BASE = "/opt/ml/processing"


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Split on time, never randomly: a random split leaks future table sizes into training."""
    df = df.sort_values("ts").reset_index(drop=True)
    t1 = df["ts"].quantile(train_frac)
    t2 = df["ts"].quantile(train_frac + val_frac)
    return df[df["ts"] <= t1], df[(df["ts"] > t1) & (df["ts"] <= t2)], df[df["ts"] > t2]


def baseline_stats(train: pd.DataFrame, features: list[str], bins: int = 10) -> dict:
    """Quantile bin edges + expected proportions per feature, consumed by sentinel.drift."""
    out = {}
    for f in features:
        values = train[f].astype(float).to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
        out[f] = {"edges": edges.tolist(), "expected": (counts / max(len(values), 1)).tolist()}
    return {"features": out, "n_train": len(train)}


def main(base: str = BASE) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    a, _ = p.parse_known_args()

    files = sorted(glob.glob(f"{base}/input/data/*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    features = [c for c in df.columns if c not in ("ts", "queryid", "label")]
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train, val, test = time_split(df, a.train_frac, a.val_frac)
    for name, part in (("train", train), ("validation", val)):
        os.makedirs(f"{base}/{name}", exist_ok=True)
        part[["label", *features]].to_csv(f"{base}/{name}/{name}.csv", header=False, index=False)
    os.makedirs(f"{base}/test", exist_ok=True)
    test[["label", *features]].to_csv(f"{base}/test/test.csv", index=False)

    os.makedirs(f"{base}/baseline", exist_ok=True)
    stats = baseline_stats(train, features)
    stats["positive_rate_train"] = float(train["label"].mean())
    stats["scale_pos_weight"] = float((1 - train["label"].mean()) / max(train["label"].mean(), 1e-6))
    with open(f"{base}/baseline/baseline_stats.json", "w") as fh:
        json.dump(stats, fh)

    print(json.dumps({"rows": {"train": len(train), "validation": len(val), "test": len(test)},
                      "positive_rate": {"train": float(train["label"].mean()),
                                        "validation": float(val["label"].mean()),
                                        "test": float(test["label"].mean())}}))


if __name__ == "__main__":
    main(os.environ.get("SENTINEL_PROCESSING_BASE", BASE))
