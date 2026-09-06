"""
Sequential (recursive) inference for the PM2.5-lag model.

At test time the true current PM2.5 is unknown except for the very first test
hour of every station (it equals the last train row's target).  We therefore
walk forward hour by hour: the model's prediction for hour t (= PM2.5 at t+1)
becomes the `pm25_lag1` input of the row at t+1, and the rolling PM2.5
features are recomputed from that running true-then-predicted sequence.

Gap handling: the walk runs on the full hourly grid.  If the row at hour t is
absent from the file, no prediction is produced for it and the PM2.5 estimate
at t+1 stays NaN, so the next row's lag-1 is NaN exactly as in training.

All 12 stations are advanced together, one model call per hour.
"""
import warnings
import numpy as np
import pandas as pd

import config as C
from src.features import PM25_HIST, pm25_features_from_matrix

PM25_FEATURE_ORDER = list(pm25_features_from_matrix(pd.DataFrame(np.zeros((30, 1)))).keys())


def pm25_features_from_buffer(buf: np.ndarray) -> np.ndarray:
    """buf: (PM25_HIST x S) history, last row = PM2.5 at the current hour.
    Returns (S x n_pm25_features) in PM25_FEATURE_ORDER.  Must mirror
    `src.features.pm25_features_from_matrix` exactly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lag1, lag2, lag3 = buf[-1], buf[-2], buf[-3]
        rm3 = np.nanmean(buf[-3:], axis=0)
        rm6 = np.nanmean(buf[-6:], axis=0)
        rm24 = np.nanmean(buf, axis=0)
        rmax6 = np.nanmax(buf[-6:], axis=0)
    cols = {
        "pm25_lag1": lag1, "pm25_lag2": lag2, "pm25_lag3": lag3,
        "pm25_rm3": rm3, "pm25_rm6": rm6, "pm25_rm24": rm24, "pm25_rmax6": rmax6,
        "pm25_diff1": lag1 - lag2, "pm25_dev24": lag1 - rm24,
    }
    return np.column_stack([cols[k] for k in PM25_FEATURE_ORDER])


def walk_forward(data: pd.DataFrame, base_feats, predict_fn, start, end, pm25_feats_check=True):
    """
    Parameters
    ----------
    data       : full feature frame (train+test) from build_features
    base_feats : list of non-PM2.5 feature names, in the model's training order
    predict_fn : callable X(np.ndarray [n, n_base + n_pm25]) -> predictions (raw scale)
    start, end : walk window [start, end) as timestamps.  History before
                 `start` uses TRUE reconstructed PM2.5 (train targets).
    Returns
    -------
    pd.Series of predictions indexed like `data` for rows inside the window.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    stations = C.STATIONS
    S = len(stations)
    t0 = start - pd.Timedelta(hours=PM25_HIST)
    hours = pd.date_range(t0, end - pd.Timedelta("1h"), freq="h")
    T = len(hours)

    win = data[(data.observation_timestamp >= t0) & (data.observation_timestamp < end)]
    # (T x S) matrix of true PM2.5 at hour t (= previous row's target on the
    # hourly grid, NaN where the previous hour is absent) - same definition as
    # in src/features.py
    pm_true = (win.pivot(index="observation_timestamp", columns="station", values=C.TARGET)
                  .reindex(index=hours, columns=stations).shift(1).to_numpy(float))

    # position of each row of `data` in the grid
    row_idx = win.index.to_numpy()
    t_pos = ((win.observation_timestamp - t0) / pd.Timedelta("1h")).astype(int).to_numpy()
    s_pos = pd.Categorical(win.station, categories=stations).codes
    X_base = np.full((T, S, len(base_feats)), np.nan)
    X_base[t_pos, s_pos] = win[base_feats].to_numpy(float)
    present = np.zeros((T, S), bool)
    present[t_pos, s_pos] = True
    out = np.full((T, S), np.nan)

    # running estimate: true values before `start` (and AT `start`), NaN after
    pm_est = pm_true.copy()
    i_start = int((start - t0) / pd.Timedelta("1h"))
    pm_est[i_start + 1:] = np.nan

    for i in range(i_start, T):
        buf = pm_est[i - PM25_HIST + 1:i + 1]
        pm_feats = pm25_features_from_buffer(buf)
        m = present[i]
        if not m.any():
            continue
        X = np.hstack([X_base[i][m], pm_feats[m]])
        pred = predict_fn(X)
        out[i, m] = pred
        if i + 1 < T:
            pm_est[i + 1, m] = pred      # prediction for row t is PM2.5 at t+1

    preds = pd.Series(out[t_pos, s_pos], index=row_idx)
    return preds[win.observation_timestamp >= start]


def check_buffer_matches_matrix(data: pd.DataFrame, n_hours=400):
    """Assert the buffer implementation reproduces the vectorised features on
    a slice of train (true history everywhere)."""
    d = data[data.is_test == 0]
    start = d.observation_timestamp.min() + pd.Timedelta(hours=PM25_HIST + 5)
    end = start + pd.Timedelta(hours=n_hours)
    win = d[(d.observation_timestamp >= start - pd.Timedelta(hours=PM25_HIST)) & (d.observation_timestamp < end)]
    hours = pd.date_range(start - pd.Timedelta(hours=PM25_HIST), end - pd.Timedelta("1h"), freq="h")
    pm = (win.pivot(index="observation_timestamp", columns="station", values=C.TARGET)
             .reindex(index=hours, columns=C.STATIONS).shift(1))
    ref = pm25_features_from_matrix(pm)
    for i in range(PM25_HIST, len(hours)):
        buf = pm.to_numpy(float)[i - PM25_HIST + 1:i + 1]
        mine = pm25_features_from_buffer(buf)
        for j, k in enumerate(PM25_FEATURE_ORDER):
            a, b = mine[:, j], ref[k].to_numpy(float)[i]
            assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1), atol=1e-6), (k, hours[i])
    print("[recursive] buffer features match vectorised features on", n_hours, "hours")
