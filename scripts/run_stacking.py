"""
Two-stage "nowcast -> forecast" stacking (causal, no exposure bias).

Stage 1 (nowcast): LightGBM predicts PM2.5 AT the observation hour
    (reconstructed from the previous row's target) from the BASE features.
    For training rows the nowcast is produced OUT-OF-FOLD over K contiguous
    time blocks, so stage 2 sees stage-1 errors of realistic size, exactly
    as it will on the test rows (where stage 1 is trained on all of train).
Stage 2 (forecast): LightGBM predicts PM2_5_next_hour from BASE features plus
    features derived from the stage-1 nowcast series on the hourly grid
    (current estimate, lags 1-3, rolling means, 1 h change).

Usage:
  python scripts/run_stacking.py validate            # season-matched folds
  python scripts/run_stacking.py final --rounds1 400 --rounds2 750 --out submission_stack.csv
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

K_BLOCKS = 4
NOWCAST_FEATS = ["nc_lag0", "nc_lag1", "nc_lag2", "nc_lag3", "nc_rm3", "nc_rm6", "nc_rm24", "nc_diff1", "nc_dev24",
                 "nc_minus_pm10"]


def fit_lgb(X, y, rounds, lr, feats, seed=C.SEED):
    p = dict(C.LGB_PARAMS, learning_rate=lr, seed=seed)
    d = lgb.Dataset(X, y, feature_name=feats, categorical_feature=["station_code"])
    return lgb.train(p, d, rounds)


def stage1_nowcast(data, feats, train_mask, apply_mask, rounds, lr):
    """Return nowcast (PM2.5 at hour t) for train rows (OOF) and apply rows (full model)."""
    y = data["pm25_lag1"].to_numpy(float)
    lab = train_mask & ~np.isnan(y)
    nc = np.full(len(data), np.nan)
    # contiguous time blocks over the labelled training rows
    ts = data.observation_timestamp.to_numpy()
    edges = np.quantile(ts[lab].astype("int64"), np.linspace(0, 1, K_BLOCKS + 1))
    blk = np.clip(np.searchsorted(edges, ts.astype("int64"), side="right") - 1, 0, K_BLOCKS - 1)
    X = data[feats].to_numpy(float)
    for b in range(K_BLOCKS):
        tr = lab & (blk != b); te = train_mask & (blk == b)
        m = fit_lgb(X[tr], y[tr], rounds, lr, feats)
        nc[te] = m.predict(X[te])
        print(f"    stage1 block {b}: trained on {tr.sum():,} -> nowcast {te.sum():,} rows", flush=True)
    m = fit_lgb(X[lab], y[lab], rounds, lr, feats)
    nc[apply_mask] = m.predict(X[apply_mask])
    return np.clip(nc, 0, None)


def nowcast_features(data, nc):
    """Grid-based (gap-aware) features from the nowcast series."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "final"])
    ap.add_argument("--rounds1", type=int, default=350)
    ap.add_argument("--lr1", type=float, default=0.05)
    ap.add_argument("--rounds2", type=int, default=None, help="final only; from validation best_iter")
    ap.add_argument("--lr2", type=float, default=0.03)
    ap.add_argument("--out", default="submission_stack.csv")
    args = ap.parse_args()
    df = load_all(); verify(df)
    data, F = build_features(df)
    base = F["base"]
    if args.mode == "validate":
        res = []
        for fold in C.FOLDS:
            t = time.time()
            tr, va = fold_masks(data, fold)
            nc = stage1_nowcast(data, base, tr, va, args.rounds1, args.lr1)
            y_nc = data["pm25_lag1"].to_numpy(float)
            m = va & ~np.isnan(y_nc)
            print(f"  [stack] {fold['name']} stage-1 nowcast RMSE on val: {rmse(y_nc[m], nc[m]):.3f}", flush=True)
            NF = nowcast_features(data, nc)
            X = np.hstack([data[base].to_numpy(float), NF.to_numpy(float)]); feats = base + NOWCAST_FEATS
            y = data[C.TARGET].to_numpy(float)
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
        json.dump({"rounds1": args.rounds1, "lr1": args.lr1, "lr2": args.lr2, "folds": res},
                  open(C.RESULTS_DIR / "stacking_results.json", "w"), indent=2)
    else:
        assert args.rounds2, "--rounds2 required for final"
        is_tr = (data.is_test == 0).to_numpy(); is_te = (data.is_test == 1).to_numpy()
        nc = stage1_nowcast(data, base, is_tr, is_te, args.rounds1, args.lr1)
        NF = nowcast_features(data, nc)
        X = np.hstack([data[base].to_numpy(float), NF.to_numpy(float)]); feats = base + NOWCAST_FEATS
        y = data[C.TARGET].to_numpy(float)
        b = fit_lgb(X[is_tr], y[is_tr], args.rounds2, args.lr2, feats)
        p = np.clip(b.predict(X[is_te]), 0, None)
        sub = pd.DataFrame({"id": data.loc[is_te, "id"].to_numpy(), C.TARGET: p})
        ss = pd.read_csv(C.SAMPLE_SUBMISSION_CSV)
        out = ss[["id"]].merge(sub, on="id", how="left"); assert out[C.TARGET].notna().all()
        out.to_csv(args.out, index=False); print(f"[stack] wrote {args.out} mean={p.mean():.2f}")


if __name__ == "__main__":
    main()
