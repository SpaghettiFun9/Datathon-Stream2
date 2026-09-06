"""
Two-stage "nowcast -> forecast" stacking (causal, no exposure bias).

Stage 1 (nowcast): a small ensemble of gradient-boosted regressors (LightGBM,
    XGBoost, optionally CatBoost) predicts PM2.5 AT the observation hour
    (reconstructed from the previous row's target) from the BASE features.
    For training rows the nowcast is produced OUT-OF-FOLD over K contiguous
    time blocks, so stage 2 sees stage-1 errors of realistic size, exactly
    as it will on the test rows (where stage 1 is trained on all of train).
    The ensemble mean is a strong current-PM2.5 estimate and the cross-library
    disagreement is used as a cheap nowcast-uncertainty signal (nc_std*).
Stage 2 (forecast): the usual models predict PM2_5_next_hour from BASE
    features plus features derived from the stage-1 nowcast series on the
    hourly grid (current estimate, lags, rolling moments, city-wide stats).

Unlike the recursive model (src/recursive.py) nothing is fed back, so the
train and test distributions of every feature match.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
from src.models import train_xgb, predict_xgb, train_cat, predict_cat, XGB_PARAMS, CAT_PARAMS

K_BLOCKS = 4                    # contiguous out-of-fold blocks for stage 1
STAGE1_LR = 0.05
# Stage-1 nowcasters.  A diverse ensemble of libraries produces a better
# current-PM2.5 estimate, and the disagreement between them is itself a useful
# stage-2 signal (nc_std*).  Override with the STAGE1_MODELS env var
# (comma-separated subset of lgb,xgb,cat).
STAGE1_MODELS = tuple(os.environ.get("STAGE1_MODELS", "lgb,xgb").split(","))
STAGE1_ROUNDS = {"lgb": 350, "xgb": 500, "cat": 800}

# Features derived from the stage-1 nowcast series (grid-aligned, gap-aware)
# and from the cross-library disagreement (a practical nowcast-uncertainty proxy).
NOWCAST_FEATS = [
    "nc_lag0", "nc_lag1", "nc_lag2", "nc_lag3", "nc_lag6",
    "nc_rm3", "nc_rm6", "nc_rm12", "nc_rm24", "nc_rstd6", "nc_rstd24",
    "nc_rmax6", "nc_rmin24", "nc_diff1", "nc_diff3", "nc_dev24",
    "nc_minus_pm10", "nc_city_mean", "nc_city_dev", "nc_city_mean_lag1",
    "nc_city_mean_rm6", "nc_std0", "nc_std_rm6", "nc_std_city_mean",
]


def _fit(X, y, feats, seed=C.SEED, model="lgb"):
    """Train one nowcaster on the supplied rows (no validation set: fixed rounds)."""
    if model == "lgb":
        p = dict(C.LGB_PARAMS, learning_rate=STAGE1_LR, seed=seed, feature_fraction=0.6)
        return lgb.train(p, lgb.Dataset(X, y, feature_name=feats, categorical_feature=["station_code"]),
                         STAGE1_ROUNDS["lgb"])
    if model == "xgb":
        p = dict(XGB_PARAMS, learning_rate=STAGE1_LR, seed=seed, nthread=C.N_THREADS)
        b, _ = train_xgb(X, y, num_rounds=STAGE1_ROUNDS["xgb"], params=p)
        return b
    if model == "cat":
        p = dict(CAT_PARAMS, learning_rate=STAGE1_LR, random_seed=seed, thread_count=C.N_THREADS)
        b, _ = train_cat(X, y, num_rounds=STAGE1_ROUNDS["cat"], params=p)
        return b
    raise ValueError(model)


def _predict(booster, X, model):
    if model == "lgb":
        return booster.predict(X)
    if model == "xgb":
        return predict_xgb(booster, X)
    if model == "cat":
        return predict_cat(booster, X)
    raise ValueError(model)


def stage1_nowcast(data, feats, train_mask, apply_mask, models=STAGE1_MODELS, verbose=True):
    """Nowcast (PM2.5 at hour t) for train rows (out-of-fold) and apply rows (full fit).

    Returns a dict: one out-of-fold/clipped array per model name plus an
    "ensemble" key holding the mean of all members.
    """
    y = data["pm25_lag1"].to_numpy(float)
    lab = train_mask & ~np.isnan(y)
    ts = data.observation_timestamp.to_numpy().astype("int64")
    edges = np.quantile(ts[lab], np.linspace(0, 1, K_BLOCKS + 1))
    blk = np.clip(np.searchsorted(edges, ts, side="right") - 1, 0, K_BLOCKS - 1)
    X = data[feats].to_numpy(float)
    models = tuple(models)
    out = {m: np.full(len(data), np.nan) for m in models}
    for m in models:
        for b in range(K_BLOCKS):
            tr = lab & (blk != b); te = train_mask & (blk == b)
            if not tr.any() or not te.any():
                continue
            bst = _fit(X[tr], y[tr], feats, model=m)
            out[m][te] = _predict(bst, X[te], m)
            if verbose:
                print(f"    [stage1:{m}] block {b}: fit on {tr.sum():,} -> oof nowcast for {te.sum():,}", flush=True)
        if apply_mask.any():
            bst = _fit(X[lab], y[lab], feats, model=m)
            out[m][apply_mask] = _predict(bst, X[apply_mask], m)
            if verbose:
                print(f"    [stage1:{m}] full fit on {lab.sum():,} -> {apply_mask.sum():,} apply rows", flush=True)
        out[m] = np.clip(out[m], 0, None)
    stacked = np.ma.masked_invalid(np.stack([out[m] for m in models], axis=0))
    out["ensemble"] = stacked.mean(axis=0).filled(np.nan)
    return out


def nowcast_features(data, nc):
    """Grid-based (gap-aware) features from the nowcast series, aligned to `data`.

    nc may be a 1-D array (single-modeller path) or the dict returned by the
    multi-modeller stage1_nowcast.  With the dict, the ensemble mean drives
    the location features and the cross-modeller std becomes uncertainty features.
    """
    if isinstance(nc, dict):
        members = [m for m in nc if m != "ensemble"]
        nc_arr = nc["ensemble"]
        if len(members) <= 1:
            std_arr = np.zeros(len(data), dtype=float)
        else:
            std_arr = np.nanstd(np.stack([nc[m] for m in members], axis=0), axis=0, ddof=0)
    else:
        nc_arr = nc
        std_arr = np.zeros(len(data), dtype=float)
    d = data[["station", "observation_timestamp", "PM10"]].copy()
    d["nc"] = nc_arr
    d["nc_std"] = std_arr
    full_ts = pd.date_range(d.observation_timestamp.min(), d.observation_timestamp.max(), freq="h")
    pv = d.pivot(index="observation_timestamp", columns="station", values="nc").reindex(index=full_ts, columns=C.STATIONS)
    pstd = d.pivot(index="observation_timestamp", columns="station", values="nc_std").reindex(index=full_ts, columns=C.STATIONS)
    city_mean = pd.DataFrame(np.repeat(pv.mean(axis=1).to_numpy()[:, None], len(C.STATIONS), axis=1),
                             index=pv.index, columns=pv.columns)
    city_std = pd.DataFrame(np.repeat(pstd.mean(axis=1).to_numpy()[:, None], len(C.STATIONS), axis=1),
                            index=pstd.index, columns=pstd.columns)
    F = {"nc_lag0": pv, "nc_lag1": pv.shift(1), "nc_lag2": pv.shift(2), "nc_lag3": pv.shift(3),
         "nc_lag6": pv.shift(6),
         "nc_rm3": pv.rolling(3, min_periods=1).mean(), "nc_rm6": pv.rolling(6, min_periods=1).mean(),
         "nc_rm12": pv.rolling(12, min_periods=1).mean(), "nc_rm24": pv.rolling(24, min_periods=1).mean(),
         "nc_rstd6": pv.rolling(6, min_periods=2).std(), "nc_rstd24": pv.rolling(24, min_periods=2).std(),
         "nc_rmax6": pv.rolling(6, min_periods=1).max(), "nc_rmin24": pv.rolling(24, min_periods=1).min(),
         "nc_city_mean": city_mean, "nc_city_dev": pv - city_mean,
         "nc_city_mean_lag1": city_mean.shift(1), "nc_city_mean_rm6": city_mean.rolling(6, min_periods=1).mean(),
         "nc_std0": pstd, "nc_std_rm6": pstd.rolling(6, min_periods=1).mean(),
         "nc_std_city_mean": city_std}
    F["nc_diff1"] = F["nc_lag0"] - F["nc_lag1"]; F["nc_diff3"] = F["nc_lag0"] - F["nc_lag3"]
    F["nc_dev24"] = F["nc_lag0"] - F["nc_rm24"]
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
    data["nc_raw"] = nc["ensemble"] if isinstance(nc, dict) else nc
    return data, base_feats + NOWCAST_FEATS
