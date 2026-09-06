"""
Post-hoc analysis of validation predictions for the methodology report.
Reads results/valpred_ensA_<model>_<fold>.csv (written by run_ensemble_val.py)
and the feature-importance files, writes results/analysis.md.
Usage: python scripts/analyze_results.py [--weights lgb=0.5 xgb=0.2 cat=0.3]
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_all
from src.validation import rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["lgb=0.34", "xgb=0.33", "cat=0.33"])
    args = ap.parse_args()
    W = {k: float(v) for k, v in (x.split("=") for x in args.weights)}
    df = load_all()
    meta = df[["id", "station", "observation_timestamp", "hour", "month"]]
    lines = ["# Validation analysis (season-matched folds, ensemble of " + ", ".join(f"{k}:{v}" for k, v in W.items()) + ")\n"]
    allp = []
    for fold in C.FOLDS:
        parts = {}
        for m in W:
            f = C.RESULTS_DIR / f"valpred_ensAS_{m}_{fold["name"]}.csv"
            if f.exists():
                parts[m] = pd.read_csv(f)
        if not parts:
            continue
        base = next(iter(parts.values()))[["id", "y"]].copy()
        base["pred"] = sum(W[m] * parts[m]["pred"].to_numpy() for m in parts) / sum(W[m] for m in parts)
        base["fold"] = fold["name"]
        allp.append(base)
    p = pd.concat(allp).merge(meta, on="id")
    p["err"] = p.pred - p.y
    lines.append(f"Pooled RMSE: **{rmse(p.y, p.pred):.3f}**  (n={len(p):,})\n")
    for name, key in [("fold", "fold"), ("station", "station"), ("month", "month"), ("hour", "hour")]:
        g = p.groupby(key).apply(lambda d: pd.Series({"n": len(d), "mean_y": d.y.mean(), "RMSE": rmse(d.y, d.pred), "bias": d.err.mean()}), include_groups=False)
        lines.append(f"\n## RMSE by {name}\n\n" + g.round(2).to_markdown() + "\n")
    # tail behaviour
    lines.append("\n## Tail behaviour (validation rows with the largest true values)\n")
    rows = []
    for q in (0.90, 0.95, 0.99):
        thr = p.y.quantile(q); d = p[p.y >= thr]
        rows.append({"quantile": q, "threshold": round(thr, 1), "n": len(d), "RMSE": rmse(d.y, d.pred), "mean_bias": d.err.mean()})
    lines.append(pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n")
    lines.append("\nNegative bias on the top quantiles = the model under-calls the most severe hours; "
                 "this is the expected behaviour of a conditional-mean forecaster without the current PM2.5 reading.\n")
    # feature importance (LightGBM gain, averaged over folds)
    fin = C.RESULTS_DIR / "importance_final_AS_lgb.csv"
    if fin.exists():
        imp = pd.read_csv(fin, index_col=0)["gain"].sort_values(ascending=False)
        imp = imp / imp.sum() * 100
        lines.append("\n## Top-25 features by LightGBM gain (% of total) - final stacked model\n\n" + imp.head(25).round(2).to_frame("gain_%").to_markdown() + "\n")
    (C.RESULTS_DIR / "analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
