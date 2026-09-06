"""
Two-stage "nowcast -> forecast" stacking (causal, no exposure bias).

Stage 1 (nowcast): LightGBM predicts PM2.5 AT the observation hour
    (reconstructed from the previous row's target) from the BASE features.
    For training rows the nowcast is produced OUT-OF-FOLD over K contiguous
    time blocks, so stage 2 sees stage-1 errors of realistic size, exactly
    as it will on the test rows (where stage 1 is trained on all of train).
Stage 2 (forecast): the usual models predict PM2_5_next_hour from BASE
    features plus features derived from the stage-1 nowcast series on the
    hourly grid (current estimate, lags 1-3, rolling means, 1 h change).

Unlike the recursive model (src/recursive.py) nothing is fed back, so the
train and test distributions of every feature match.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C

K_BLOCKS = 4
STAGE1_ROUNDS = 350
STAGE1_LR = 0.05
NOWCAST_FEATS = ["nc_lag0", "nc_lag1", "nc_lag2", "nc_lag3", "nc_rm3", "nc_rm6", "nc_rm24",
                 "nc_diff1", "nc_dev24", "nc_minus_pm10"]


def _fit(X, y, feats, seed=C.SEED):
    p = dict(C.LGB_PARAMS, learning_rate=STAGE1_LR, seed=seed)
    return lgb.train(p, lgb.Dataset(X, y, feature_name=feats, categorical_feature=["station_code"]), STAGE1_ROUNDS)


def stage1_nowcast(data, feats, train_mask, apply_mask, verbose=True):
    """Nowcast (PM2.5 at hour t) for train rows (out-of-fold) and apply rows (model on all train rows)."""
    y = data["pm25_lag1"].to_numpy(float)
    lab = train_mask & ~np.isnan(y)
    nc = np.full(len(data), np.nan)
    ts = data.observation_timestamp.to_numpy().astype("int64")
    edges = np.quantile(ts[lab], np.linspace(0, 1, K_BLOCKS + 1))
    blk = np.clip(np.searchsorted(edges, ts, side="right") - 1, 0, K_BLOCKS - 1)
    X = data[feats].to_numpy(float)
    for b in range(K_BLOCKS):
        tr = lab & (blk != b); te = train_mask & (blk == b)
        nc[te] = _fit(X[tr], y[tr], feats).predict(X[te])
        if verbose:
            print(f"    [stage1] block {b}: fit on {tr.sum():,} rows -> out-of-fold nowcast for {te.sum():,}", flush=True)
    nc[apply_mask] = _fit(X[lab], y[lab], feats).predict(X[apply_mask])
    return np.clip(nc, 0, None)


def nowcast_features(data, nc):
    """Grid-based (gap-aware) features from the nowcast series, aligned to `data`."""
    d = data[["station", "observation_timestamp", "PM10"]].copy(); d["nc"] = nc
    full_ts = pd.date_range(d.observation_timestamp.min(), d.observation_timestamp.max(), freq="h")
    pv = d.pivot(index="observation_timestamp", columns="station", values="nc").reindex(index=full_ts, columns=C.STATIONS)
    F = {"nc_lag0": pv, "nc_lag1": pv.shift(1), "nc_lag2": pv.shift(2), "nc_lag3": pv.shift(3),
         "nc_rm3": pv.rolling(3, min_periods=1).mean(), "nc_rm6": pv.rolling(6, min_periods=1).mean(),
         "nc_rm24": pv.rolling(24, min_periods=1).mean()}
    F["nc_diff1"] = F["nc_lag0"] - F["nc_lag1"]; F["nc_dev24"] = F["nc_lag0"] - F["nc_rm24"]
    idx = pd.MultiIndex.from_product([C.STATIONS, full_ts], names=["station", "observation_timestamp"])
    fd = pd.DataFrame({k: v.to_numpy().T.ravel() for k, v in F.items()}, index=idx)
    out = d.join(fd, on=["station", "observation_timestamp"])
    out["nc_minus_pm10"] = out["nc_lag0"] - out["PM10"]
    return out[NOWCAST_FEATS]


def add_stacked_features(data, base_feats, train_mask, apply_mask):
    """Return (data with nowcast features appended, extended feature list)."""
    nc = stage1_nowcast(data, base_feats, train_mask, apply_mask)
    NF = nowcast_features(data, nc)
    data = data.copy()
    for c in NOWCAST_FEATS:
        data[c] = NF[c].to_numpy(float)
    data["nc_raw"] = nc
    return data, base_feats + NOWCAST_FEATS
