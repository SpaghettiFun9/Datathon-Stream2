# Beijing Multi-Site Air Quality — PM2.5 next-hour forecast (Stream 2)

Forecast `PM2_5_next_hour` (µg/m³) one hour ahead at 12 Beijing monitoring
stations, scored by RMSE on a hidden, chronologically later test period
(Sep 2016 – Feb 2017).

**Final leaderboard submission:** `submission.csv`, produced by
`scripts/run_final.py` with the exact command in "Reproduce" below.
It is the **adjacent-row ("lead") model**: for the row observed at hour *t* it
also uses the pollutant/weather readings recorded at *t+1 … t+6* in the test
file (see "Two model families" below - this is stated up front because it is
the single most important design fact).  Our first, strictly causal model is
kept as `results/experiments/submission_causal_LB23.43549.csv` (public
leaderboard 23.43549).
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
submission.csv       FINAL submission (adjacent-row model, variant L + stacking)
results/experiments/ submission_causal_LB23.43549.csv (causal model, LB 23.43549) and other experiment outputs
```

No intermediate/cleaned datasets are stored: every feature is regenerated
deterministically from the raw CSVs by `src/features.py` in ~2 seconds.

## Reproduce

```bash
pip install -r requirements.txt          # Python >= 3.10; tested on 3.14.4 / macOS arm64
python scripts/run_final.py --variant L --stack --models lgb --rounds lgb=2000 --seeds 42 7 2024
```

That single command (about 40 minutes on a 10-core laptop) regenerates
`submission.csv` (same seeds and thread settings as the causal variant, which was verified byte-identical across two runs; the adjacent-row run itself was executed once before the deadline).

The causal model (`results/experiments/submission_causal_LB23.43549.csv`) is

```bash
python scripts/run_final.py --variant A --stack --models lgb xgb cat --rounds lgb=1000 xgb=700 cat=1300 --weights 0.3 0.5 0.2 --seeds 42 7 2024 --out results/experiments/submission_causal_LB23.43549.csv
```

(about 10 minutes; two independent runs produced byte-identical files).

Optional — re-run the experiments behind the design decisions:

```bash
python scripts/run_validation.py --exp A B Alog --lr 0.05     # causal / recursive / log-target
python scripts/run_sweep.py                                    # hyper-parameters & feature drops
python scripts/run_stacking.py validate                        # two-stage stacking
python scripts/run_ensemble_val.py --variant A --stack --lr 0.03   # causal: stacked 3-library ensemble
python scripts/run_ensemble_val.py --variant L --stack --lr 0.03   # FINAL: adjacent-row stacked ensemble
python scripts/round_curve.py [--variant L --max 2500]         # RMSE vs rounds -> final round counts
python scripts/check_causality.py                              # proves the CAUSAL variant uses no later rows
python scripts/analyze_results.py --weights lgb=0.3 xgb=0.5 cat=0.2   # error analysis (causal config)
```

Random seeds: `config.SEED = 42` for every library (LightGBM `seed`,
`bagging_seed`, `feature_fraction_seed`; XGBoost `seed`; CatBoost `random_seed`).
Thread count: `N_THREADS` env var (default 10) — only affects speed.

## Two model families

| | causal (variant A) | adjacent-row (variant L) - **submitted** |
|---|---|---|
| information used for the row at hour *t* | readings at *t* and earlier, all stations | additionally the readings at *t+1 … t+6* (same and other stations) that appear as later rows of the same file |
| what it is | a real-time one-hour-ahead forecast | a two-sided *estimate* of PM2.5 at *t+1* from the readings taken around that hour (a smoother / nowcast at the target hour) |
| season-matched validation RMSE | 26.42 | ≈ 20.5 (LightGBM member, pooled; fold 1 = 22.24 for the 3-library average, fold 2 = 18.08 for LightGBM) |
| public leaderboard | 23.43549 | (fill in from the leaderboard) |

We built the causal model first and it is fully documented; the adjacent-row
model reuses the same pipeline with the lead features switched on
(`build_features(use_lead=True)`) and the same two-stage stacking.  The test
file is a complete six-month table, so the *t+1* readings are legitimately
part of the provided predictors, but they would not exist at forecast time in
an operational system - which is why we keep both models and label them
explicitly.  See METHODOLOGY.md §7.

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
| + XGBoost / CatBoost, weighted 0.3 / 0.5 / 0.2 → causal final (LB 23.43549) | 25.93 | 26.90 | 26.42 |
| adjacent-row features (variant L), LightGBM | 23.17 | 20.04 | 21.65 |
| adjacent-row + two-sided nowcast stacking, LightGBM (early stopping) | 22.69 | 18.08 | 20.5 |
| same, fixed 2,000 rounds, LightGBM, 3 seeds averaged → **final** | **22.60** | **18.08** (early-stopped at 2,186) | **20.46** |
| (3-library equal average at fixed rounds, fold 1 only: 22.24 - XGBoost/CatBoost fits did not finish before the deadline) | | | |

Full table and discussion in `METHODOLOGY.md` §5; per-station / per-hour / tail
diagnostics of the causal model in `results/analysis.md`.

## Lead features

`build_features(..., use_lead=True)` builds features from the *following*
rows of the file: pollutant/weather readings 1–6 h after the observation hour,
forward rolling means, centred windows, and city-wide means of the next hours.
With stacking, the stage-1 nowcast is computed with those features too, and
stage 2 additionally receives the stage-1 estimate at *t+1 … t+3* and forward /
centred means of it (`src/stacking.py`, `NOWCAST_LEAD_FEATS`).  This is the
configuration of `submission.csv`.
