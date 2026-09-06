# Beijing Multi-Site Air Quality — PM2.5 next-hour forecast (Stream 2)

Forecast `PM2_5_next_hour` (µg/m³) one hour ahead at 12 Beijing monitoring
stations, scored by RMSE on a hidden, chronologically later test period
(Sep 2016 – Feb 2017).

**Final leaderboard submission:** `submission.csv`, produced by
`scripts/run_final.py` with the exact command in "Reproduce" below.
See `METHODOLOGY.md` for the full write-up (data, features, validation,
models, results, limitations) and `DISCLOSURE.md` for the required disclosure.

## Repository layout

```
train.csv, test(1).csv, sample_submission.csv   competition files (unchanged)
config.py            paths, seeds, fold definitions, model hyper-parameters, feature switches
src/data.py          loading + integrity checks (train->test continuity, gaps, duplicates)
src/features.py      feature engineering on a per-station hourly grid (gap-aware lags/rollings,
                     cross-station stats, reconstructed PM2.5 history, optional lead features)
src/validation.py    season-matched chronological folds, RMSE + tail diagnostics
src/models.py        LightGBM / XGBoost / CatBoost wrappers
src/recursive.py     sequential walk-forward inference for the recursive PM2.5-lag model (rejected variant)
src/stacking.py      two-stage nowcast -> forecast stacking (out-of-fold stage-1 nowcast features)
scripts/run_validation.py   experiments A / B / L / log-target on the two winter folds
scripts/run_sweep.py        hyper-parameter & feature-subset sweep (causal model)
scripts/run_ensemble_val.py LightGBM+XGBoost+CatBoost ensemble validation & weight search (--stack)
scripts/run_stacking.py     stand-alone validation of the two-stage stacking idea
scripts/round_curve.py      validation RMSE vs. boosting rounds (used to fix the final round counts)
scripts/analyze_results.py  per-station / per-hour / tail error analysis -> results/analysis.md
scripts/check_causality.py  asserts that no submitted feature depends on later rows
scripts/run_final.py        train on ALL train rows, predict test, write the submission
results/             validation logs, JSON summaries, feature importances
submission.csv       FINAL submission (causal model)
```

No intermediate/cleaned datasets are stored: every feature is regenerated
deterministically from the raw CSVs by `src/features.py` in ~2 seconds.

## Reproduce

```bash
pip install -r requirements.txt          # Python >= 3.10; tested on 3.14.4 / macOS arm64
python scripts/run_final.py --variant A --stack --models lgb xgb cat --rounds lgb=1000 xgb=700 cat=1300 --weights 0.3 0.5 0.2 --seeds 42 7 2024
```

That single command (about 10 minutes on a 10-core laptop) regenerates
`submission.csv` bit-for-bit up to floating-point noise from library
threading (verified: two consecutive runs agree to within 1e-6 on every row).

Optional — re-run the experiments behind the design decisions:

```bash
python scripts/run_validation.py --exp A B Alog --lr 0.05     # causal / recursive / log-target
python scripts/run_sweep.py                                    # hyper-parameters & feature drops
python scripts/run_stacking.py validate                        # two-stage stacking
python scripts/run_ensemble_val.py --variant A --stack --lr 0.03   # final config: stacked 3-library ensemble
python scripts/round_curve.py                                  # RMSE vs rounds -> final round counts
python scripts/check_causality.py                              # proves no feature uses later rows
python scripts/analyze_results.py --weights lgb=0.3 xgb=0.5 cat=0.2       # error analysis of the final config
```

Random seeds: `config.SEED = 42` for every library (LightGBM `seed`,
`bagging_seed`, `feature_fraction_seed`; XGBoost `seed`; CatBoost `random_seed`).
Thread count: `N_THREADS` env var (default 10) — only affects speed.

## Approach in one paragraph

The files contain no "current PM2.5" column, so the model must infer the
present state of the aerosol from the co-measured pollutants (PM10, CO, NO2,
SO2, O3), weather, wind and their recent history, then project it one hour
ahead.  Every temporal feature is computed on a complete hourly grid per
station so that lags never bridge a real gap.  Cross-station statistics at the
same hour capture city-wide pollution episodes.  A first-stage model
*nowcasts* the current PM2.5 (its label is reconstructed from the previous
row's target) and its out-of-fold estimates feed the second-stage forecaster,
which avoids the exposure bias that sinks a naive recursive model.
Validation uses two season-matched folds (train strictly before, validate on
the following Sep–Feb) because the test period is autumn/winter while the last
months of train are summer.  The final model is a weighted average of
LightGBM, XGBoost and CatBoost second-stage regressors trained on all labelled
rows, with Sep–Feb rows weighted 2x.

## Key validation results (RMSE, µg/m³)

| step | F1 (2014-15) | F2 (2015-16) | mean |
|---|---|---|---|
| oracle: model given the *true* current PM2.5 (not achievable) | 17.54 | 21.07 | 19.31 |
| causal LightGBM, 216 features | 26.60 | 30.68 | 28.64 |
| + recursive reconstructed-PM2.5 lags (rejected: exposure bias) | 32.06 | 43.83 | 37.94 |
| − calendar-season features, bigger regularised trees | 26.55 | 28.63 | 27.59 |
| + Sep–Feb sample weight 2 | 26.49 | 28.28 | 27.38 |
| + two-stage nowcast stacking (LightGBM) | 26.22 | 27.03 | 26.63 |
| + XGBoost / CatBoost, weighted 0.3 / 0.5 / 0.2 → **final** | **25.93** | **26.90** | **26.42** |

Full table and discussion in `METHODOLOGY.md` §5; per-station / per-hour / tail
diagnostics in `results/analysis.md`.

## Lead features (NOT used in `submission.csv`)

`build_features(..., use_lead=True)` can also build features from the
*following* rows of the file (pollutant/weather readings one to six hours
after the observation hour).  Those values are not available in a real-time
forecast, so `submission.csv` does not use them.  The code path is kept,
switched off by default, to document the experiment: season-matched RMSE
23.17 / 20.04 (pooled 21.65) versus 26.42 for the causal model.
The file `results/experiments/submission_lead_features.csv` was generated with

```bash
python scripts/run_final.py --variant L --models lgb --rounds lgb=1500 --out results/experiments/submission_lead_features.csv
```

and is kept only as the record of that experiment; see METHODOLOGY.md
§7 "What we did not submit".
