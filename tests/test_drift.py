import random

import numpy as np
import pandas as pd

from pipeline.scripts.preprocess import baseline_stats, time_split
from sentinel.drift import drift_report, psi


def test_psi_zero_for_identical_distributions():
    assert psi([0.25] * 4, [0.25] * 4) == 0


def test_baseline_and_drift_use_same_binning():
    rng = random.Random(0)
    train = pd.DataFrame({"x": [rng.gauss(0, 1) for _ in range(5000)]})
    base = baseline_stats(train, ["x"])
    same = [{"x": rng.gauss(0, 1)} for _ in range(5000)]
    shifted = [{"x": rng.gauss(1.5, 1)} for _ in range(5000)]
    assert drift_report(base, same)["x"] < 0.05
    assert drift_report(base, shifted)["x"] > 0.2


def test_time_split_never_puts_future_in_train():
    df = pd.DataFrame({"ts": np.arange(100), "label": 0})
    train, val, test = time_split(df, 0.7, 0.15)
    assert train["ts"].max() < val["ts"].min() <= val["ts"].max() < test["ts"].min()
