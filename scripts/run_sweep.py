"""
Quick sweep over LightGBM settings / feature subsets / sample weights for the
causal BASE model, on the season-matched folds.  Writes results/sweep_results.json.
Usage: python scripts/run_sweep.py [config names...]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_all
from src.features import build_features
from src.validation import fold_masks, rmse

SEASON_FEATS = ["month", "doy_sin", "doy_cos", "heating_season"]

CONFIGS = {
    "ref_lr05":      dict(params={}, drop=[], winter_w=1.0),
    "reg_small":     dict(params=dict(num_leaves=63, min_data_in_leaf=300, feature_fraction=0.3, lambda_l2=10.0), drop=[], winter_w=1.0),
    "reg_big":       dict(params=dict(num_leaves=255, min_data_in_leaf=200, feature_fraction=0.4, lambda_l2=10.0, learning_rate=0.03), drop=[], winter_w=1.0),
    "no_season":     dict(params={}, drop=SEASON_FEATS, winter_w=1.0),
    "winter_w2":     dict(params={}, drop=[], winter_w=2.0),
    "reg_small_ws":  dict(params=dict(num_leaves=63, min_data_in_leaf=300, feature_fraction=0.3, lambda_l2=10.0), drop=SEASON_FEATS, winter_w=2.0),
    "reg_big_ns":    dict(params=dict(num_leaves=255, min_data_in_leaf=200, feature_fraction=0.4, lambda_l2=10.0, learning_rate=0.03), drop=SEASON_FEATS, winter_w=1.0),
    "reg_big_ns_w2": dict(params=dict(num_leaves=255, min_data_in_leaf=200, feature_fraction=0.4, lambda_l2=10.0, learning_rate=0.03), drop=SEASON_FEATS, winter_w=2.0),
    "ns_no_dow":     dict(params={}, drop=SEASON_FEATS + ["dow", "is_weekend"], winter_w=1.0),
}


def main():
    names = sys.argv[1:] or list(CONFIGS)
    df = load_all()
    data, F = build_features(df, drop=[])   # sweep controls its own drops
    out_path = C.RESULTS_DIR / "sweep_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        cfg = CONFIGS[name]
        feats = [f for f in F["base"] if f not in cfg["drop"]]
        params = dict(C.LGB_PARAMS, learning_rate=0.05); params.update(cfg["params"])
        fold_res = []
        for fold in C.FOLDS:
            t = time.time()
            tr, va = fold_masks(data, fold)
            w = np.where(data.loc[tr, "month"].isin([9, 10, 11, 12, 1, 2]), cfg["winter_w"], 1.0)
            dtr = lgb.Dataset(data.loc[tr, feats].to_numpy(float), data.loc[tr, C.TARGET].to_numpy(float),
                              weight=w, feature_name=feats, categorical_feature=["station_code"], free_raw_data=False)
            dva = lgb.Dataset(data.loc[va, feats].to_numpy(float), data.loc[va, C.TARGET].to_numpy(float), reference=dtr)
            b = lgb.train(params, dtr, C.LGB_MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(C.LGB_EARLY_STOP, verbose=False)])
            p = np.clip(b.predict(data.loc[va, feats].to_numpy(float), num_iteration=b.best_iteration), 0, None)
            r = rmse(data.loc[va, C.TARGET].to_numpy(float), p)
            fold_res.append({"fold": fold["name"], "rmse": r, "best_iter": b.best_iteration, "sec": round(time.time() - t)})
            print(f"[sweep:{name}] {fold['name']} RMSE={r:.4f} iters={b.best_iteration} ({fold_res[-1]['sec']}s)", flush=True)
        m = float(np.mean([f["rmse"] for f in fold_res]))
        print(f"[sweep:{name}] MEAN RMSE={m:.4f}", flush=True)
        results[name] = {"config": cfg, "folds": fold_res, "mean_rmse": m}
        out_path.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
