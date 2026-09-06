"""Season-matched, strictly chronological validation folds (see config.FOLDS)."""
import numpy as np
import pandas as pd

import config as C


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def fold_masks(data: pd.DataFrame, fold: dict):
    """Boolean masks (train, val) over `data` for one fold. Test rows are never included."""
    ts = data.observation_timestamp
    is_train_file = data.is_test == 0
    tr = is_train_file & (ts < pd.Timestamp(fold["train_end"]))
    va = is_train_file & (ts >= pd.Timestamp(fold["val_start"])) & (ts < pd.Timestamp(fold["val_end"]))
    return tr.to_numpy(), va.to_numpy()


def tail_report(y, p, qs=(0.95, 0.99)):
    """RMSE restricted to the largest targets - where RMSE is decided."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"all": rmse(y, p)}
    for q in qs:
        thr = np.quantile(y, q)
        m = y >= thr
        out[f"top{int((1-q)*100)}%"] = rmse(y[m], p[m])
        out[f"top{int((1-q)*100)}%_bias"] = float(np.mean(p[m] - y[m]))
    return out
