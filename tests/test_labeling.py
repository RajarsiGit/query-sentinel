from sentinel.labeling import add_labels


def row(idx, mean, baseline=1.0):
    return {"queryid": "q", "snap_idx": idx, "window_mean_ms": mean, "baseline_ms": baseline,
            "ratio_to_baseline": mean / baseline}


def test_onset_within_horizon_is_positive():
    rows = [row(1, 1), row(2, 1), row(3, 1), row(4, 2.0)]
    labels = {r["snap_idx"]: r["label"] for r in add_labels(rows, horizon=2, ratio=1.5)}
    assert labels[2] == 1 and labels[3] == 1  # spike at 4 is within 2 snapshots
    assert labels[1] == 0                     # spike at 4 is 3 snapshots away


def test_already_regressed_rows_are_excluded():
    rows = [row(1, 1), row(2, 3.0), row(3, 3.0)]
    out = add_labels(rows, horizon=2, ratio=1.5)
    assert [r["snap_idx"] for r in out] == [1]


def test_rows_without_future_are_dropped_not_negative():
    rows = [row(1, 1), row(2, 1)]
    out = add_labels(rows, horizon=3, ratio=1.5)
    assert [r["snap_idx"] for r in out] == [1]


def test_gaps_respect_horizon_in_snapshot_units():
    rows = [row(1, 1), row(10, 5.0)]
    assert add_labels(rows, horizon=3, ratio=1.5) == []
