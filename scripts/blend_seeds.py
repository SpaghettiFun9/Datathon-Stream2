"""
Combine per-library test predictions from several run_final.py invocations
(different --tag / seed sets) into seed-weighted member averages, then blend.
Usage:
  python scripts/blend_seeds.py --member xgb=LS:3,LS_s2:2 --member cat=LS:3,LS_s2:2 --member lgb=LS:3 \
                                --weights lgb=0.25 xgb=0.30 cat=0.45 --out results/experiments/submission_LS_w_5seed.csv
`tag:n` = the prediction file results/testpred_<tag>_<model>.csv is an average of n seeds.
"""
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--member", action="append", required=True); ap.add_argument("--weights", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
W = {k: float(v) for k, v in (x.split("=") for x in a.weights)}; assert abs(sum(W.values()) - 1) < 1e-6
ss = pd.read_csv(C.SAMPLE_SUBMISSION_CSV)[["id"]]
out = ss.copy(); out[C.TARGET] = 0.0
for spec in a.member:
    m, parts = spec.split("="); acc = 0.0; n_tot = 0
    for part in parts.split(","):
        tag, n = part.split(":"); n = int(n)
        p = ss.merge(pd.read_csv(C.RESULTS_DIR / f"testpred_{tag}_{m}.csv"), on="id", how="left")[C.TARGET].to_numpy()
        acc = acc + n * p; n_tot += n
    member = acc / n_tot
    print(f"[blend_seeds] {m}: {n_tot} seeds from {parts}, weight {W[m]}")
    out[C.TARGET] += W[m] * member
assert out[C.TARGET].notna().all() and len(out) == len(ss)
out.to_csv(a.out, index=False)
print(f"[blend_seeds] -> {a.out}: mean={out[C.TARGET].mean():.2f} std={out[C.TARGET].std():.2f}")
