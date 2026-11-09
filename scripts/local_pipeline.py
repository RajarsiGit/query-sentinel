"""Run the pipeline's three steps on this machine, with the same scripts and file layout.

Useful for iterating on features without paying for SageMaker jobs, and as the CI smoke test.
The Train step mimics the built-in XGBoost container: label-first headerless CSV in, a
`xgboost-model` file packed into model.tar.gz out.

    python scripts/local_pipeline.py --data data/dataset.csv --workdir build/local-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "pipeline" / "scripts"


def run_step(script: str, base: Path, *args: str) -> None:
    env = {**os.environ, "SENTINEL_PROCESSING_BASE": str(base)}
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], env=env, check=True)


def train(base: Path, params: dict, num_round: int) -> Path:
    def load(name):
        arr = np.loadtxt(base / name / f"{name}.csv", delimiter=",", ndmin=2)
        return xgb.DMatrix(arr[:, 1:], label=arr[:, 0])

    dtrain, dval = load("train"), load("validation")
    booster = xgb.train(params, dtrain, num_boost_round=num_round, evals=[(dtrain, "train"), (dval, "validation")],
                        early_stopping_rounds=20, verbose_eval=False)
    model_dir = base / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "xgboost-model"))
    with tarfile.open(model_dir / "model.tar.gz", "w:gz") as tar:
        tar.add(model_dir / "xgboost-model", arcname="xgboost-model")
    return model_dir / "xgboost-model"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/dataset.csv")
    p.add_argument("--workdir", default="build/local-run")
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--num-round", type=int, default=300)
    p.add_argument("--threshold", type=float, default=0.7)
    a = p.parse_args()

    base = Path(a.workdir).resolve()
    shutil.rmtree(base, ignore_errors=True)
    (base / "input" / "data").mkdir(parents=True)
    shutil.copy(a.data, base / "input" / "data" / "dataset.csv")

    run_step("preprocess.py", base)
    stats = json.loads((base / "baseline" / "baseline_stats.json").read_text())
    params = {"objective": "binary:logistic", "eval_metric": "aucpr", "max_depth": a.max_depth, "eta": a.eta,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
              "scale_pos_weight": stats["scale_pos_weight"]}
    model_file = train(base, params, a.num_round)
    run_step("evaluate.py", base, "--threshold", str(a.threshold))

    report = json.loads((base / "evaluation" / "evaluation.json").read_text())
    m, h = report["metrics"]["pr_auc"]["value"], report["heuristic"]["pr_auc"]["value"]
    print(f"\nmodel PR-AUC {m:.3f} vs heuristic {h:.3f} -> gate {'PASS' if m >= max(h, 0.4) else 'FAIL'}")
    print(f"model file: {model_file}")


if __name__ == "__main__":
    main()
