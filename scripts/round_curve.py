"""Validation RMSE as a function of boosting rounds (stacked LightGBM, winter weights) - used to pick a
round count that is safe on both folds. Writes results/round_curve.json."""
import json, sys
from pathlib import Path
import numpy as np, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_all
from src.features import build_features
from src.validation import fold_masks, rmse
from src.stacking import add_stacked_features

df = load_all(); data0, F = build_features(df)
out = {}
for fold in C.FOLDS:
    tr, va = fold_masks(data0, fold)
    data, feats = add_stacked_features(data0, F["base"], tr, va)
    w = np.where(data.loc[tr, "month"].isin(C.WINTER_MONTHS), C.WINTER_WEIGHT, 1.0)
    dtr = lgb.Dataset(data.loc[tr, feats].to_numpy(float), data.loc[tr, C.TARGET].to_numpy(float), weight=w,
                      feature_name=feats, categorical_feature=["station_code"])
    b = lgb.train(C.LGB_PARAMS, dtr, 1500)
    Xv, yv = data.loc[va, feats].to_numpy(float), data.loc[va, C.TARGET].to_numpy(float)
    curve = {k: rmse(yv, np.clip(b.predict(Xv, num_iteration=k), 0, None)) for k in range(100, 1501, 100)}
    out[fold["name"]] = curve
    print(fold["name"], {k: round(v, 3) for k, v in curve.items()}, flush=True)
json.dump(out, open(C.RESULTS_DIR / "round_curve.json", "w"), indent=2)
