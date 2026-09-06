"""
Blend saved per-library test predictions (results/testpred_<tag>_<model>.csv,
written by scripts/run_final.py) with given weights into a submission file.
Usage: python scripts/blend.py --tag LS --weights lgb=0.35 xgb=0.35 cat=0.30 --out submission.csv
"""
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--weights", nargs="+", required=True); ap.add_argument("--out", default=str(C.SUBMISSION_CSV))
a = ap.parse_args()
W = {k: float(v) for k, v in (x.split("=") for x in a.weights)}
assert abs(sum(W.values()) - 1) < 1e-6, "weights must sum to 1"
ss = pd.read_csv(C.SAMPLE_SUBMISSION_CSV)[["id"]]
out = ss.copy(); out[C.TARGET] = 0.0
for m, w in W.items():
    p = pd.read_csv(C.RESULTS_DIR / f"testpred_{a.tag}_{m}.csv")
    out[C.TARGET] += w * ss.merge(p, on="id", how="left")[C.TARGET].to_numpy()
assert out[C.TARGET].notna().all() and len(out) == len(ss)
# guard against a member silently dropping out (weights must all have been applied)
assert 60 < out[C.TARGET].mean() < 130, f"implausible blend mean {out[C.TARGET].mean():.1f}: a member is missing"
out.to_csv(a.out, index=False)
print(f"[blend] {a.tag} {W} -> {a.out}: mean={out[C.TARGET].mean():.2f} std={out[C.TARGET].std():.2f}")
