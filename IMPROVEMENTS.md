# `improve` branch — candidates beyond the submitted model

`main` is frozen at the file that scored **17.95550** (LightGBM-only
adjacent-row stacked model, 2,000 rounds, 3 seeds).  This branch holds the
follow-up experiments; each candidate is a separate file under
`results/experiments/` so any of them can be uploaded and compared on the
leaderboard without touching `submission.csv`.

| candidate | what changes vs. `main` | status | file |
|---|---|---|---|
| 3-way blend | adds XGBoost (1,500 rounds) and CatBoost (1,800 rounds) members, equal weights; fold-1 validation 22.24 vs 22.60 for LightGBM alone | full-data XGBoost/CatBoost fits in progress | `results/experiments/submission_LS_3way.csv` |
| long context | `--lead-long` (12/24 h leads, forward means, 25 h centred means, +58 features) and stage-1 nowcast trained for 1,000 rounds instead of 350 | full-data LightGBM fit in progress; season-matched validation not yet run | `results/experiments/submission_LS_long.csv` |

Commands (branch `improve`):

```bash
# 3-way blend of the main configuration (per-library predictions saved by run_final.py)
python scripts/run_final.py --variant L --stack --models lgb xgb cat --rounds lgb=2000 xgb=1500 cat=1800 --weights 0.34 0.33 0.33 --seeds 42 7 2024 --out results/experiments/submission_LS_3way.csv
# or, if results/testpred_LS_{lgb,xgb,cat}.csv already exist:
python scripts/blend.py --tag LS --weights lgb=0.34 xgb=0.33 cat=0.33 --out results/experiments/submission_LS_3way.csv

# long-context variant
python scripts/run_final.py --variant L --stack --lead-long --stage1-rounds 1000 --models lgb --rounds lgb=2000 --seeds 42 7 2024 --tag _long --out results/experiments/submission_LS_long.csv
# its validation (fixed rounds, both folds)
python scripts/run_ensemble_val.py --variant L --stack --lead-long --stage1-rounds 1000 --models lgb --fixed-rounds lgb=2000 --tag _long
```

Further ideas, in order of expected value:
1. Third-stage smoothing: feed the stage-2 predictions of neighbouring rows
   (t−2…t+2) back as features (a second pass of the same stacking).
2. Stage-1 ensemble (LightGBM + CatBoost nowcasts averaged before stage 2).
3. Per-station calibration of the PM2.5/PM10 ratio drift (yearly re-weighting).
4. Bayesian hyper-parameter search of stage 2 on the season-matched folds.
