"""
Validate a LightGBM + XGBoost + CatBoost ensemble on the season-matched folds.
Usage: python scripts/run_ensemble_val.py --variant A            (causal)
       python scripts/run_ensemble_val.py --variant L            (lead features)
Saves per-model fold predictions to results/ and the summary to
results/ensemble_results_<variant>.json (individual RMSEs, best iterations,
and the RMSE of simple / weight-searched averages).
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_all
from src.features import build_features
from src.models import (train_lgb, predict_lgb, train_xgb, predict_xgb, train_cat, predict_cat, TargetTransform)
from src.validation import fold_masks, rmse, tail_report
from src.stacking import add_stacked_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=["A", "L"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--models", nargs="+", default=["lgb", "xgb", "cat"])
    ap.add_argument("--stack", action="store_true", help="add out-of-fold nowcast (stage-1) features")
    args = ap.parse_args()
    df = load_all()
    data, F = build_features(df, use_lead=(args.variant == "L"))
    feats0 = F["base"] + (F["lead"] if args.variant == "L" else [])
    tag = args.variant + ("S" if args.stack else "")
    out = {"variant": tag, "n_features": len(feats0), "folds": []}
    data0 = data
    pooled = {m: [] for m in args.models}; ys = []
    for fold in C.FOLDS:
        tr, va = fold_masks(data0, fold)
        if args.stack:
            data, feats = add_stacked_features(data0, feats0, tr, va)
        else:
            data, feats = data0, feats0
        X_tr, y_tr = data.loc[tr, feats].to_numpy(float), data.loc[tr, C.TARGET].to_numpy(float)
        X_va, y_va = data.loc[va, feats].to_numpy(float), data.loc[va, C.TARGET].to_numpy(float)
        w_tr = np.where(data.loc[tr, "month"].isin(C.WINTER_MONTHS), C.WINTER_WEIGHT, 1.0)
        fr = {"fold": fold["name"]}
        preds = {}
        for m in args.models:
            t = time.time()
            if m == "lgb":
                b, it = train_lgb(X_tr, y_tr, X_va, y_va, params=dict(C.LGB_PARAMS, learning_rate=args.lr),
                                  feature_names=feats, verbose_every=0, weight=w_tr)
                p = predict_lgb(b, X_va, num_iteration=it)
            elif m == "xgb":
                b, it = train_xgb(X_tr, y_tr, X_va, y_va, weight=w_tr); p = predict_xgb(b, X_va, it)
            elif m == "cat":
                b, it = train_cat(X_tr, y_tr, X_va, y_va, weight=w_tr); p = predict_cat(b, X_va)
            preds[m] = p
            fr[m] = {"rmse": rmse(y_va, p), "best_iter": int(it), "sec": round(time.time() - t)}
            print(f"[ens:{tag}] {fold['name']} {m}: RMSE={fr[m]['rmse']:.4f} iters={it} ({fr[m]['sec']}s)", flush=True)
            pooled[m].append(p)
            pd.DataFrame({"id": data.loc[va, "id"].to_numpy(), "y": y_va, "pred": p}).to_csv(
                C.RESULTS_DIR / f"valpred_ens{tag}_{m}_{fold['name']}.csv", index=False)
        ys.append(y_va)
        avg = np.mean([preds[m] for m in args.models], axis=0)
        fr["avg"] = tail_report(y_va, avg)
        print(f"[ens:{tag}] {fold['name']} AVG: RMSE={fr['avg']['all']:.4f} top5%={fr['avg']['top5%']:.2f}", flush=True)
        out["folds"].append(fr)
    # weight search on pooled fold predictions (coarse grid, sums to 1)
    y = np.concatenate(ys); P = {m: np.concatenate(pooled[m]) for m in args.models}
    best = None
    grid = np.arange(0, 1.01, 0.1)
    for w in itertools.product(grid, repeat=len(args.models)):
        if abs(sum(w) - 1) > 1e-9: continue
        r = rmse(y, sum(wi * P[m] for wi, m in zip(w, args.models)))
        if best is None or r < best[0]: best = (r, dict(zip(args.models, [round(x, 2) for x in w])))
    out["pooled_individual"] = {m: rmse(y, P[m]) for m in args.models}
    out["pooled_equal_avg"] = rmse(y, np.mean([P[m] for m in args.models], axis=0))
    out["pooled_best_weights"] = {"rmse": best[0], "weights": best[1]}
    print(f"[ens:{tag}] pooled individual {out['pooled_individual']}\n"
          f"[ens:{tag}] pooled equal-avg RMSE={out['pooled_equal_avg']:.4f}; best weights {best[1]} -> {best[0]:.4f}", flush=True)
    json.dump(out, open(C.RESULTS_DIR / f"ensemble_results_{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
