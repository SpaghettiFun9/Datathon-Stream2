"""
Season-matched validation experiments.

Usage:  python scripts/run_validation.py --exp A B Alog L
  A    : BASE features only (no PM2.5 history)                 - causal
  Alog : A with log1p target transform                          - causal
  B    : BASE + reconstructed PM2.5 history, recursive walk     - causal
  L    : BASE + LEAD (t+1/t+2 rows) features   *** not a real-time forecast ***
Results are appended to results/validation_results.json and printed.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_all, verify
from src.features import build_features
from src.models import TargetTransform, train_lgb, predict_lgb
from src.validation import fold_masks, rmse, tail_report
from src.recursive import walk_forward, check_buffer_matches_matrix


def run_experiment(name, data, feats, transform, recursive=False, lr=None):
    params = dict(C.LGB_PARAMS)
    if lr:
        params["learning_rate"] = lr
    res = {"experiment": name, "n_features": len(feats), "folds": []}
    all_y, all_p = [], []
    for fold in C.FOLDS:
        t = time.time()
        tr, va = fold_masks(data, fold)
        X_tr, y_tr = data.loc[tr, feats].to_numpy(float), data.loc[tr, C.TARGET].to_numpy(float)
        X_va, y_va = data.loc[va, feats].to_numpy(float), data.loc[va, C.TARGET].to_numpy(float)
        # NOTE for the recursive model: early stopping still uses the TRUE
        # lag features on the validation rows (cheap, non-sequential) - the
        # honest score is then computed with the sequential walk below.
        booster, best_it = train_lgb(X_tr, y_tr, X_va, y_va, params=params, transform=transform,
                                     feature_names=feats, verbose_every=0)
        if recursive:
            n_pm = len([f for f in feats if f.startswith("pm25_")])
            base = [f for f in feats if not f.startswith("pm25_")]
            fn = lambda X: predict_lgb(booster, X, transform, num_iteration=best_it)
            pred_s = walk_forward(data, base, fn, fold["val_start"], fold["val_end"])
            pred = pred_s.reindex(data.index[va]).to_numpy(float)
            assert not np.isnan(pred).any(), "recursive walk left rows unpredicted"
            p_truelag = predict_lgb(booster, X_va, transform, num_iteration=best_it)
            oracle = rmse(y_va, p_truelag)
        else:
            pred = predict_lgb(booster, X_va, transform, num_iteration=best_it)
            oracle = None
        rep = tail_report(y_va, pred)
        rep.update({"fold": fold["name"], "best_iter": int(best_it), "n_train": int(tr.sum()),
                    "n_val": int(va.sum()), "seconds": round(time.time() - t, 1)})
        if oracle is not None:
            rep["rmse_if_true_lag_known"] = oracle
        res["folds"].append(rep)
        all_y.append(y_va); all_p.append(pred)
        print(f"[{name}] {fold['name']}: RMSE={rep['all']:.4f}  top5%={rep['top5%']:.2f} "
              f"top1%={rep['top1%']:.2f}  iters={best_it}  ({rep['seconds']}s)"
              + (f"  [oracle true-lag RMSE={oracle:.4f}]" if oracle else ""))
        # keep fold predictions for later ensembling/analysis
        pd.DataFrame({"id": data.loc[va, "id"].to_numpy(), "y": y_va, "pred": pred}).to_csv(
            C.RESULTS_DIR / f"valpred_{name}_{fold['name']}.csv", index=False)
        # feature importance (gain) for the report
        imp = pd.Series(booster.feature_importance("gain"), index=feats).sort_values(ascending=False)
        imp.to_csv(C.RESULTS_DIR / f"importance_{name}_{fold['name']}.csv", header=["gain"])
    res["mean_rmse"] = float(np.mean([f["all"] for f in res["folds"]]))
    res["pooled_rmse"] = rmse(np.concatenate(all_y), np.concatenate(all_p))
    print(f"[{name}] mean fold RMSE = {res['mean_rmse']:.4f}   pooled = {res['pooled_rmse']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["A", "B"])
    ap.add_argument("--lr", type=float, default=None, help="override learning rate (faster experiments)")
    args = ap.parse_args()
    C.RESULTS_DIR.mkdir(exist_ok=True)

    df = load_all(); verify(df)
    t = time.time()
    data, F = build_features(df, use_lead="L" in args.exp)
    print(f"[features] built {len(F['base'])} base + {len(F['pm25'])} pm25 + {len(F['lead'])} lead features in {time.time()-t:.1f}s")
    check_buffer_matches_matrix(data)

    out_path = C.RESULTS_DIR / "validation_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    for e in args.exp:
        if e == "A":
            r = run_experiment("A_base", data, F["base"], TargetTransform("identity"), lr=args.lr)
        elif e == "Alog":
            r = run_experiment("Alog_base_log1p", data, F["base"], TargetTransform("log1p"), lr=args.lr)
        elif e == "B":
            r = run_experiment("B_base+pm25_recursive", data, F["base"] + F["pm25"], TargetTransform("identity"), recursive=True, lr=args.lr)
        elif e == "Blog":
            r = run_experiment("Blog_base+pm25_recursive_log1p", data, F["base"] + F["pm25"], TargetTransform("log1p"), recursive=True, lr=args.lr)
        elif e == "L":
            r = run_experiment("L_base+lead", data, F["base"] + F["lead"], TargetTransform("identity"), lr=args.lr)
        else:
            raise ValueError(e)
        results = [x for x in results if x["experiment"] != r["experiment"]] + [r]
        out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
