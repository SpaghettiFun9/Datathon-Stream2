"""
Final training + test inference + submission file.

Usage (the exact command used for the submitted file is in README.md):
  python scripts/run_final.py --variant A --models lgb xgb cat --rounds lgb=600 xgb=900 cat=1500
  python scripts/run_final.py --variant B --models lgb --rounds lgb=600      # recursive PM2.5 model
  python scripts/run_final.py --variant L --models lgb xgb cat --rounds ... --out submission_lead.csv

`--rounds` gives the number of boosting rounds per library.  They come from the
season-matched validation runs (best iteration on the folds), inflated ~1.3x
because the final fit sees ~25-40% more data than the largest validation fold.
`--seeds` trains each library once per seed and averages (seed bagging).
`--weights` are the ensemble weights (default: equal).

Steps
  1. load train+test and re-verify the data contract,
  2. build features on the combined hourly grid,
  3. train on ALL labelled rows (through 2016-08-31 22:00),
  4. predict the test rows (variant B uses the sequential walk-forward),
  5. merge predictions onto `sample_submission.csv` by id and write the file.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_all, verify
from src.features import build_features
from src.models import (TargetTransform, train_lgb, predict_lgb, train_xgb, predict_xgb,
                        train_cat, predict_cat, XGB_PARAMS, CAT_PARAMS)
from src.recursive import walk_forward, check_buffer_matches_matrix
from src.stacking import add_stacked_features


def parse_rounds(items):
    return {k: int(v) for k, v in (it.split("=") for it in items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=["A", "B", "L"])
    ap.add_argument("--models", nargs="+", default=["lgb"], choices=["lgb", "xgb", "cat"])
    ap.add_argument("--rounds", nargs="+", required=True, help="e.g. lgb=600 xgb=900 cat=1500")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="ensemble weights, same order as --models")
    ap.add_argument("--seeds", type=int, nargs="+", default=[C.SEED])
    ap.add_argument("--transform", default="identity", choices=["identity", "log1p"])
    ap.add_argument("--out", default=str(C.SUBMISSION_CSV))
    ap.add_argument("--stack", action="store_true", help="add out-of-fold nowcast (stage-1) features, see src/stacking.py")
    ap.add_argument("--lead-long", action="store_true", help="variant L: longer forward context features")
    ap.add_argument("--stage1-rounds", type=int, default=None, help="override stage-1 nowcast rounds (default 350)")
    ap.add_argument("--tag", default="", help="suffix for per-library prediction files")
    args = ap.parse_args()
    rounds = parse_rounds(args.rounds)
    weights = args.weights or [1.0 / len(args.models)] * len(args.models)
    assert len(weights) == len(args.models) and abs(sum(weights) - 1) < 1e-6
    C.MODELS_DIR.mkdir(exist_ok=True); C.RESULTS_DIR.mkdir(exist_ok=True)

    df = load_all(); verify(df)
    data, F = build_features(df, use_lead=(args.variant == "L"), lead_long=args.lead_long)
    check_buffer_matches_matrix(data)
    feats = {"A": F["base"], "B": F["base"] + F["pm25"], "L": F["base"] + F["lead"]}[args.variant]
    transform = TargetTransform(args.transform)
    print(f"[final] variant={args.variant} models={args.models} weights={weights} features={len(feats)} "
          f"rounds={rounds} seeds={args.seeds}", flush=True)

    is_tr = (data.is_test == 0).to_numpy()
    is_te = (data.is_test == 1).to_numpy()
    if args.stack:
        assert args.variant != "B", "stacking is an alternative to the recursive model, not a complement"
        t = time.time()
        data, feats = add_stacked_features(data, feats, is_tr, is_te, use_lead=(args.variant == "L"), stage1_rounds=args.stage1_rounds)
        print(f"[final] stage-1 nowcast features added in {time.time()-t:.0f}s -> {len(feats)} features", flush=True)
    X_tr = data.loc[is_tr, feats].to_numpy(float)
    y_tr = data.loc[is_tr, C.TARGET].to_numpy(float)
    w_tr = np.where(data.loc[is_tr, "month"].isin(C.WINTER_MONTHS), C.WINTER_WEIGHT, 1.0)
    X_te = data.loc[is_te, feats].to_numpy(float)
    te_start = data.loc[is_te, "observation_timestamp"].min()
    te_end = data.loc[is_te, "observation_timestamp"].max() + pd.Timedelta("1h")

    model_preds = {}
    for m in args.models:
        seed_preds = []
        for seed in args.seeds:
            t = time.time()
            if m == "lgb":
                params = dict(C.LGB_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
                booster, _ = train_lgb(X_tr, y_tr, params=params, num_rounds=rounds["lgb"], transform=transform,
                                       feature_names=feats, verbose_every=0, weight=w_tr)
                booster.save_model(str(C.MODELS_DIR / f"lgb_{args.variant}_seed{seed}.txt"))
                if seed == args.seeds[0]:   # feature importance of the final LightGBM model (for the report)
                    pd.Series(booster.feature_importance("gain"), index=feats, name="gain").sort_values(ascending=False).to_csv(
                        C.RESULTS_DIR / f"importance_final_{args.variant}{'S' if args.stack else ''}_lgb.csv")
                fn = lambda X: predict_lgb(booster, X, transform)
            elif m == "xgb":
                booster, _ = train_xgb(X_tr, transform.fwd(y_tr), num_rounds=rounds["xgb"], params=dict(XGB_PARAMS, seed=seed), weight=w_tr)
                booster.save_model(str(C.MODELS_DIR / f"xgb_{args.variant}_seed{seed}.json"))
                fn = lambda X: np.clip(transform.inv(predict_xgb(booster, X)), 0, None)
            elif m == "cat":
                booster, _ = train_cat(X_tr, transform.fwd(y_tr), num_rounds=rounds["cat"], params=dict(CAT_PARAMS, random_seed=seed), weight=w_tr)
                booster.save_model(str(C.MODELS_DIR / f"cat_{args.variant}_seed{seed}.cbm"))
                fn = lambda X: np.clip(transform.inv(predict_cat(booster, X)), 0, None)
            print(f"[final] {m} seed {seed} trained in {time.time()-t:.0f}s", flush=True)
            if args.variant == "B":
                t = time.time()
                base = [f for f in feats if not f.startswith("pm25_")]
                p = walk_forward(data, base, fn, te_start, te_end).reindex(data.index[is_te]).to_numpy(float)
                print(f"[final] recursive walk over test in {time.time()-t:.0f}s", flush=True)
            else:
                p = fn(X_te)
            assert not np.isnan(p).any()
            seed_preds.append(p)
        model_preds[m] = np.mean(seed_preds, axis=0)
        # per-library test predictions (seed-averaged) so that blend weights can be changed
        # afterwards with scripts/blend.py without retraining
        pd.DataFrame({"id": data.loc[is_te, "id"].to_numpy(), C.TARGET: model_preds[m]}).to_csv(
            C.RESULTS_DIR / f"testpred_{args.variant}{'S' if args.stack else ''}{args.tag}_{m}.csv", index=False)
    pred = sum(w * model_preds[m] for w, m in zip(weights, args.models))

    sub = pd.DataFrame({"id": data.loc[is_te, "id"].to_numpy(), C.TARGET: pred})
    ss = pd.read_csv(C.SAMPLE_SUBMISSION_CSV)
    out = ss[["id"]].merge(sub, on="id", how="left")
    assert len(out) == len(ss) and out[C.TARGET].notna().all() and set(out.id) == set(ss.id)
    out.to_csv(args.out, index=False)
    print(f"[final] wrote {args.out}: {len(out):,} rows; mean={out[C.TARGET].mean():.2f} "
          f"std={out[C.TARGET].std():.2f} min={out[C.TARGET].min():.2f} max={out[C.TARGET].max():.2f}", flush=True)
    info = {"variant": args.variant, "stack": args.stack, "lead_long": args.lead_long, "stage1_rounds": args.stage1_rounds,
            "models": args.models, "weights": weights, "rounds": rounds,
            "seeds": args.seeds, "transform": args.transform, "n_features": len(feats),
            "lgb_params": C.LGB_PARAMS, "xgb_params": XGB_PARAMS, "cat_params": CAT_PARAMS,
            "dropped_features": C.DROP_FEATURES, "winter_weight": C.WINTER_WEIGHT, "winter_months": C.WINTER_MONTHS,
            "features": feats, "output": args.out,
            "per_model_test_mean": {m: float(v.mean()) for m, v in model_preds.items()}}
    json.dump(info, open(C.RESULTS_DIR / f"final_model_info_{args.variant}{'S' if args.stack else ''}{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
