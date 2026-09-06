"""
Stand-alone validation of the two-stage "nowcast -> forecast" stacking idea
(LightGBM only, no sample weights) on the season-matched folds.  The
implementation lives in src/stacking.py; the final pipeline uses it through
`scripts/run_final.py --stack` and `scripts/run_ensemble_val.py --stack`.

Usage: python scripts/run_stacking.py validate
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_all, verify
from src.features import build_features
from src.validation import fold_masks, rmse, tail_report
from src.stacking import add_stacked_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate"])
    ap.add_argument("--lr2", type=float, default=0.03)
    args = ap.parse_args()
    df = load_all(); verify(df)
    data0, F = build_features(df)
    res = []
    for fold in C.FOLDS:
        t = time.time()
        tr, va = fold_masks(data0, fold)
        data, feats = add_stacked_features(data0, F["base"], tr, va)
        y_nc = data["pm25_lag1"].to_numpy(float); m = va & ~np.isnan(y_nc)
        print(f"  [stack] {fold['name']} stage-1 nowcast RMSE on val: {rmse(y_nc[m], data['nc_raw'].to_numpy(float)[m]):.3f}", flush=True)
        X, y = data[feats].to_numpy(float), data[C.TARGET].to_numpy(float)
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, categorical_feature=["station_code"])
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        b = lgb.train(dict(C.LGB_PARAMS, learning_rate=args.lr2), dtr, C.LGB_MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(C.LGB_EARLY_STOP, verbose=False)])
        p = np.clip(b.predict(X[va], num_iteration=b.best_iteration), 0, None)
        rep = tail_report(y[va], p); rep.update(fold=fold["name"], best_iter=b.best_iteration, sec=round(time.time() - t))
        print(f"[stack] {fold['name']}: RMSE={rep['all']:.4f} top5%={rep['top5%']:.2f} iters={b.best_iteration} ({rep['sec']}s)", flush=True)
        pd.DataFrame({"id": data.loc[va, "id"].to_numpy(), "y": y[va], "pred": p}).to_csv(
            C.RESULTS_DIR / f"valpred_stack_{fold['name']}.csv", index=False)
        res.append(rep)
    print(f"[stack] MEAN RMSE={np.mean([r['all'] for r in res]):.4f}", flush=True)
    json.dump({"lr2": args.lr2, "folds": res}, open(C.RESULTS_DIR / "stacking_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
