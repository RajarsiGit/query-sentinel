"""Pipeline step 3 — evaluate on the held-out (latest) time slice.

Runs in the SageMaker XGBoost container (xgboost, numpy, pandas, scikit-learn available).
Input : /opt/ml/processing/model/model.tar.gz, /opt/ml/processing/test/test.csv
Output: /opt/ml/processing/evaluation/evaluation.json

Reports the model AND a heuristic baseline, because "XGBoost got 0.6 PR-AUC" means
nothing unless you know a two-line rule gets 0.3 (or 0.65).
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

BASE = "/opt/ml/processing"


def heuristic_score(df: pd.DataFrame) -> np.ndarray:
    """What a DBA might alert on without ML: latency creeping up while tables grow or stats age."""
    creep = np.clip((df["ratio_to_baseline"].to_numpy() - 1.0) / 0.5, 0, 1)
    growth = (df["tuple_growth_rate"].to_numpy() > 0).astype(float)
    stale = np.clip(df["mod_since_analyze_ratio"].to_numpy() / 0.2, 0, 1)
    spill = (df["temp_blks_per_call"].to_numpy() > 0).astype(float)
    return 0.5 * creep + 0.2 * growth + 0.2 * stale + 0.1 * spill


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    has_both = len(np.unique(y)) == 2
    pred = (p >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)) if has_both else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if has_both else 0.0,
        "precision_at_threshold": float(precision_score(y, pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y, pred, zero_division=0)),
    }


def top_features(booster: xgb.Booster, names: list[str], k: int = 10) -> list[list]:
    gain = booster.get_score(importance_type="gain")
    named = {names[int(f[1:])] if f.startswith("f") and f[1:].isdigit() else f: v for f, v in gain.items()}
    return [[n, round(v, 3)] for n, v in sorted(named.items(), key=lambda kv: -kv[1])[:k]]


def main(base: str = BASE) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.7)
    a, _ = p.parse_known_args()

    with tarfile.open(f"{base}/model/model.tar.gz") as tar:
        try:
            tar.extractall(f"{base}/model", filter="data")
        except TypeError:  # older Python in the container: no extraction filters
            tar.extractall(f"{base}/model")
    booster = xgb.Booster()
    booster.load_model(f"{base}/model/xgboost-model")

    test = pd.read_csv(f"{base}/test/test.csv")
    y = test["label"].to_numpy()
    X = test.drop(columns=["label"])
    model_p = booster.predict(xgb.DMatrix(X.to_numpy()))

    model_m = metrics(y, model_p, a.threshold)
    heur_m = metrics(y, heuristic_score(X), a.threshold)
    report = {
        # nested {"value": ...} shape is what SageMaker ModelMetrics / the registry UI expect
        "metrics": {k: {"value": v} for k, v in model_m.items()},
        "heuristic": {k: {"value": v} for k, v in heur_m.items()},
        "test_rows": len(test),
        "test_positive_rate": float(y.mean()) if len(y) else 0.0,
        "top_features_by_gain": top_features(booster, list(X.columns)),
    }
    os.makedirs(f"{base}/evaluation", exist_ok=True)
    with open(f"{base}/evaluation/evaluation.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(os.environ.get("SENTINEL_PROCESSING_BASE", BASE))
