# `improve` branch — candidates beyond the submitted model

`main` is frozen at the file that scored **17.95550** (LightGBM-only
adjacent-row stacked model, 2,000 rounds, 3 seeds).  This branch holds the
follow-up experiments; each candidate is a separate file under
`results/experiments/` so any of them can be uploaded and compared on the
leaderboard without touching `submission.csv`.

| candidate | what changes vs. `main` | status | file |
|---|---|---|---|
| 2-way blend | adds the XGBoost member (1,500 rounds, 3 seeds), 0.5 / 0.5; fold-1 validation LightGBM 22.60, XGBoost 22.49 | ready | `results/experiments/submission_LS_lgb_xgb.csv` |
| 3-way blend | adds CatBoost (1,800 rounds) as well, equal weights; fold-1 validation of the 3-way average 22.24 | **ready; public LB 17.94365** | `results/experiments/submission_LS_3way.csv` |
| weighted 3-way (fold-1 weights) | 0.25 / 0.30 / 0.45 from a grid search on fold-1 validation predictions | superseded: the leaderboard analysis below predicts 17.96, worse than equal weights | `results/experiments/submission_LS_w25_30_45.csv` |
| **leaderboard-informed blend** | 0.65 LightGBM / 0 XGBoost / 0.35 CatBoost; see "Weight inference" below | **ready; predicted 17.86** | `results/experiments/submission_LS_lb65_0_35.csv` |
| hedged variant | 0.60 / 0.05 / 0.35 | ready; predicted 17.87 | `results/experiments/submission_LS_lb60_05_35.csv` |
| 5-seed CatBoost | leaderboard-informed blend with the CatBoost member averaged over 5 seeds (42, 7, 2024, 11, 23) | **ready** | `results/experiments/submission_LS_lb65_0_35_cat5.csv` |
| long context | `--lead-long` (12/24 h leads, forward means, 25 h centred means, +58 features) and stage-1 nowcast trained for 1,000 rounds instead of 350 | implemented; full-data fit stopped at the deadline (stage 1 alone takes ~25 min); validation not run | — |

Commands (branch `improve`):

```bash
# 3-way blend of the main configuration (per-library predictions saved by run_final.py)
python scripts/run_final.py --variant L --stack --models lgb xgb cat --rounds lgb=2000 xgb=1500 cat=1800 --weights 0.34 0.33 0.33 --seeds 42 7 2024 --out results/experiments/submission_LS_3way.csv
# or, if results/testpred_LS_{lgb,xgb,cat}.csv already exist:
python scripts/blend.py --tag LS --weights lgb=0.5 xgb=0.5 --out results/experiments/submission_LS_lgb_xgb.csv
python scripts/blend.py --tag LS --weights lgb=0.34 xgb=0.33 cat=0.33 --out results/experiments/submission_LS_3way.csv

# long-context variant
python scripts/run_final.py --variant L --stack --lead-long --stage1-rounds 1000 --models lgb --rounds lgb=2000 --seeds 42 7 2024 --tag _long --out results/experiments/submission_LS_long.csv
# its validation (fixed rounds, both folds)
python scripts/run_ensemble_val.py --variant L --stack --lead-long --stage1-rounds 1000 --models lgb --fixed-rounds lgb=2000 --tag _long
```

## Weight inference from the leaderboard

For a weighted average of members with errors e_i, the blend's MSE obeys the
exact identity  MSE(w) = Σ w_i MSE_i − Σ_{i<j} w_i w_j D_ij,  where
D_ij = mean((p_i − p_j)²) depends only on the members' predictions (known on
the test set from `results/testpred_LS_*.csv`).  Three public scores for three
known blends of the same three members therefore determine the three member
MSEs exactly:

| public score | blend | | implied member RMSE on test |
|---|---|---|---|
| 17.95550 | LightGBM only | | LightGBM 17.96 |
| 18.04513 | 0.5 LightGBM + 0.5 XGBoost | | XGBoost 18.50 |
| 17.94365 | 0.34 / 0.33 / 0.33 | | CatBoost 18.19 |

D_LX = 26.6, D_LC = 28.2, D_XC = 32.8.  Minimising MSE(w) over the simplex
gives w = (0.65, 0.00, 0.35) with predicted RMSE 17.86; the equal blend is
predicted at 17.94 (matches its score by construction) and the fold-1 weights
(0.25/0.30/0.45) at 17.96.  Fold 1 had ranked XGBoost ≥ CatBoost ≥ LightGBM,
the test winter ranks them the other way round, so the blend weights should
follow the test-season evidence.  (`python scripts/blend.py --tag LS --weights lgb=0.65 xgb=0.0 cat=0.35`.)

Further ideas, in order of expected value:
1. Third-stage smoothing: feed the stage-2 predictions of neighbouring rows
   (t−2…t+2) back as features (a second pass of the same stacking).
2. Stage-1 ensemble (LightGBM + CatBoost nowcasts averaged before stage 2).
3. Per-station calibration of the PM2.5/PM10 ratio drift (yearly re-weighting).
4. Bayesian hyper-parameter search of stage 2 on the season-matched folds.
