"""
Causality check for the CAUSAL feature set (variant A, the model behind
results/experiments/submission_causal_LB23.43549.csv): deleting every row after
a cut-off must leave all features of the rows before the cut-off unchanged.
If any feature peeked at later rows, this would fail.  Also confirms that no
LEAD feature name leaks into the "base" list.

Variant L (the final submission) deliberately fails this property - it uses
the following hours' readings - which is exactly why the two variants are kept
separate and documented separately.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_all
from src.features import build_features

CUT = pd.Timestamp("2016-12-01")

df = load_all()
full, F = build_features(df)
trunc, _ = build_features(df[df.observation_timestamp < CUT].copy())
feats = F["base"]
assert not any(("lead" in f) or f.endswith(("_c5", "_c13")) for f in feats), "lead feature in base list"
a = full.loc[full.observation_timestamp < CUT, ["id"] + feats].set_index("id").sort_index()
b = trunc.set_index("id")[feats].sort_index()
assert a.index.equals(b.index)
diff = (a.to_numpy(float) != b.to_numpy(float)) & ~(np.isnan(a.to_numpy(float)) & np.isnan(b.to_numpy(float)))
bad = [feats[j] for j in np.where(diff.any(axis=0))[0]]
assert not bad, f"features depend on later rows: {bad}"
print(f"[causality] OK - {len(feats)} base features on {len(a):,} rows before {CUT.date()} are unchanged when all later rows are removed")
