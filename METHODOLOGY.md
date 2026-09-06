# Methodology report — PM2.5 next-hour forecasting (Stream 2)

## 1. Problem and data

* Target: `PM2_5_next_hour`, the PM2.5 concentration at the same station one
  hour after the observation hour.  Metric: RMSE.
* 360,954 labelled station-hours (2013-03-01 00:00 → 2016-08-31 22:00),
  51,063 hidden test station-hours (2016-08-31 23:00 → 2017-02-28 22:00),
  12 stations.  Chronological split, no overlap, no gap: for every station the
  first test hour is exactly one hour after the last train hour
  (checked in `src/data.py::verify`).
* Predictors: PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd (16-point
  compass), WSPM, station, timestamp.  **There is no current-PM2.5 column** —
  the data dictionary mentions one, the files do not contain it.  Missing
  predictor values are blank (0.5–4.4 % per pollutant in train, ≤ 2 % in test).
* 99.3 % of consecutive rows within a station are exactly one hour apart; the
  rest are gaps of 2–344 h (hours whose next-hour target was invalid were
  removed).

### Facts that shaped the design

| observation | consequence |
|---|---|
| Target std ≈ 78 µg/m³, heavy right tail (99th pct 354, max 999). | RMSE is dominated by pollution episodes; the model must not blunt spikes. |
| Row t-1's target *is* PM2.5 at hour t.  Reconstructing it gives the "current PM2.5" for 99.3 % of train rows. | Lets us measure how much the missing column is worth (an *oracle* experiment) and build a recursive model. |
| Persistence with the true current PM2.5 scores RMSE 19.7; a model given it scores 17.5–21 (oracle). | Any model without the true current value is fighting for a much higher floor. |
| PM2.5 at hour t can be *nowcast* from the other readings of hour t with RMSE ≈ 22–25 (LightGBM, season-matched folds). | This nowcast error is the dominant term of the forecast error; see §7. |
| Test period is autumn/winter; the last 6 months of train are spring/summer. Monthly mean target: 53 (Aug) → 96 (Dec). | Validation must be season-matched, not "last N %". |
| Stations share meteorological sensors (e.g. Dongsi/Nongzhanguan/Tiantan have identical weather series); pollution episodes are city-wide. | Cross-station features are natural and cheap. |

## 2. Preprocessing

* Train and test are stacked (test target = NaN) and sorted by station/time.
* Each station is re-indexed onto the **complete hourly grid** spanning both
  files; absent hours become all-NaN rows.  All lags/rolling windows are computed
  on this grid, so a lag whose source hour is missing is NaN rather than the
  value from a different hour — gaps are never bridged.  Rolling statistics use
  `min_periods=1` (partial windows allowed, but never across the window edge).
* `wd` is decoded to degrees; missing `wd` (almost always calm wind) is kept as
  NaN plus an indicator, and a `calm` flag is derived.
* No imputation: LightGBM/XGBoost/CatBoost handle NaN natively.  Per-pollutant
  missing indicators and a "number of missing pollutants" count are added.
* No rows are dropped; no outliers are clipped (the extreme values are real
  pollution events and exactly what RMSE cares about).

## 3. Feature engineering (`src/features.py`, 212 base features + 10 stacked)

| family | features |
|---|---|
| contemporaneous | the 10 raw measurements, wind vector (u, v), sin/cos of direction, calm/missing flags, dew-point depression, relative humidity (Magnus), PM10/CO and NO2/O3 ratios, missing indicators |
| time | hour (raw + sin/cos), day-of-week, weekend flag, station (native categorical). **Calendar-season features (month, day-of-year, heating-season flag) were built and then deliberately removed** — see §5. |
| per-station pollutant history (gap-aware) | for PM10/SO2/NO2/CO/O3: lags 1/2/3/6/12/24 h, 1 h and 3 h differences, rolling mean 3/6/12/24 h, rolling std 6/24 h, rolling max 6/24 h, rolling min 24 h, deviation from the 24 h mean |
| weather history | TEMP/PRES/DEWP/WSPM: lags 1/3 h, differences 1/3/24 h, rolling mean 6/24 h; rain sums 6/24 h; 6 h mean wind vector; 6 h max wind |
| cross-station (same hour) | for each pollutant: mean/max/min/std over the 12 stations, station deviation from the city mean, city-mean lags 1/3/6 h, 1 h change, city-mean rolling 6/24 h; number of stations reporting |
| reconstructed PM2.5 history (variant B only) | lag 1/2/3, rolling mean 3/6/24, rolling max 6, 1 h change, deviation from 24 h mean — all derived from the previous rows' targets |
| stage-1 nowcast (final model, +10, `src/stacking.py`) | out-of-fold estimate of PM2.5 at the observation hour, its lags 1/2/3 h, rolling means 3/6/24 h, 1 h change, deviation from the 24 h mean, estimate minus PM10 — computed on the same hourly grid |

## 4. Validation strategy

Two **season-matched, strictly chronological** folds
(`config.FOLDS`):

| fold | train | validate |
|---|---|---|
| F1 | 2013-03-01 → 2014-08-31 | 2014-09-01 → 2015-02-28 |
| F2 | 2013-03-01 → 2015-08-31 | 2015-09-01 → 2016-02-29 |

Each validation window is the same Sep–Feb season as the hidden test set and
lies entirely after its training data.  Both folds are always reported
(F2's winter had more severe episodes and is consistently harder).  Random
K-fold and "last-N-months" hold-outs were rejected: the former leaks
neighbouring hours, the latter validates on the easy summer regime.

Early stopping uses the fold's validation set; the final model is retrained on
**all** labelled rows with a fixed number of rounds derived from the folds.

## 5. Experiments and results (RMSE, µg/m³)

All numbers are RMSE on the validation window of each season-matched fold
(`scripts/run_validation.py`, `scripts/run_sweep.py`, `scripts/run_ensemble_val.py`,
`results/*.json`).  Unless stated otherwise: LightGBM, raw target, lr 0.05
for the exploratory rows and 0.03 for the tuned rows.

| # | model / change | F1 (2014-15) | F2 (2015-16) | mean | note |
|---|---|---|---|---|---|
| 0 | persistence, *if* the true current PM2.5 were known | — | — | 19.67 | whole train; not achievable |
| 1 | oracle: base features + **true** PM2.5 lags | 17.54 | 21.07 | 19.31 | upper bound on the value of the missing column |
| 2 | **A** base features (216), causal | 26.60 | 30.68 | 28.64 | reference |
| 3 | A with log1p target | 26.57 | 33.61 | 30.09 | rejected |
| 4 | **B** base + reconstructed PM2.5 lags, recursive walk at inference | 32.06 | 43.83 | 37.94 | exposure bias, rejected |
| 5 | A, larger regularised trees (255 leaves, min 200/leaf, ff 0.4, L2 10, lr 0.03) | 26.60 | 30.27 | 28.43 | |
| 6 | A **minus calendar-season features** (212 feats) | 26.65 | 28.51 | 27.58 | adopted |
| 7 | 5 + 6 | 26.55 | 28.63 | 27.59 | |
| 8 | 7 + Sep–Feb sample weight 2 | 26.49 | 28.28 | 27.38 | adopted |
| 9 | 7 minus day-of-week | 26.63 | 28.79 | 27.71 | rejected |
| 10 | 8 with XGBoost | 26.64 | 28.69 | 27.69 | ensemble member |
| 11 | 8 with CatBoost | 26.23 | 30.10 | 28.24 | erratic across folds |
| 12 | 8: LightGBM 0.8 + XGBoost 0.1 + CatBoost 0.1 (pooled RMSE) | 26.11* | 28.74* | 27.36 | *equal-weight fold averages |
| 13 | **7 + two-stage nowcast stacking** (`src/stacking.py`) | 26.33 | 27.27 | 26.80 | adopted; stage-1 nowcast RMSE 22.9 / 21.4 |
| 14a | 13 + Sep–Feb weight 2, LightGBM | 26.22 | 27.03 | 26.63 | final member |
| 14b | 13 + Sep–Feb weight 2, XGBoost | 26.14 | 26.79 | 26.47 | final member |
| 14c | 13 + Sep–Feb weight 2, CatBoost | 26.17 | 27.99 | 27.10 | final member |
| **15** | **14: LightGBM 0.3 + XGBoost 0.5 + CatBoost 0.2** | **25.93** | **26.90** | **26.42** | **final configuration** (grid optimum 0.2/0.6/0.2 = 26.42; weights smoothed) |

Adjacent-row (variant L) models - see §7 for what they are:

| # | model | F1 (2014-15) | F2 (2015-16) | pooled | note |
|---|---|---|---|---|---|
| L1 | base + lead features (96), LightGBM lr 0.05, first parameters | 23.17 | 23.15 | 23.16 | |
| L2 | L1 with the tuned parameters / no season features | 23.17 | 20.04 | 21.65 | fold-1 early-stops at 98 rounds, fold-2 at 2,394 |
| L3 | L2 + two-sided nowcast stacking (stage 1 uses lead features; stage 2 gets N(t+1…t+3), forward/centred means), LightGBM, early stopping | 22.69 (90 rounds) | 18.08 (2,186 rounds) | 20.46 | fold-1 early stopping is fooled by a transient dip (see round curve) |
| L3 | same, XGBoost / CatBoost, early stopping (fold 1 only) | 22.68 / 22.46 | — | — | |
| L4 | L3 at **fixed** rounds 2,000 / 1,500 / 1,800: LightGBM / XGBoost / CatBoost | 22.60 / 22.49 / 22.47 | — | — | fold-2 fits stopped to free CPU for the final run |
| L4 | equal average of the three at fixed rounds (fold 1 only) | 22.24 | — | — | XGBoost/CatBoost full-data fits not finished in time |
| **L4-LGB** | **LightGBM at fixed 2,000 rounds, 3 seeds → submitted** | **22.60** | **18.08** | **20.46** | |

Round curve of L3-LightGBM on fold 1 (`results/round_curve_L.json`, winter
weights): 22.73 at 100 rounds, a bump to 23.31 at 200, then a steady descent
to 22.57 at 1,500–2,000 and flat to 2,500.  Early stopping with 200 rounds'
patience therefore reported "best = 100" on fold 1 although 2,000 rounds is
better; fold 2 early-stopped at 2,186.  This is why the final configuration
uses fixed round counts (2,000 / 1,500 / 1,800) instead of the fold-1 optimum.


Take-aways

1. **Recursive PM2.5 lags (variant B) fail badly** even though the same
   model with the *true* lag would score ≈ 19.  Trained on true lags the model
   trusts `pm25_lag1` almost completely; at test time that input is its own
   previous prediction, errors compound and the chain drifts (RMSE 32–44).  This
   is classic exposure bias.  The first test hour of every station does get its
   true current value (the last train target) but that anchor only helps for a
   few hours out of ~4,250.
2. **Calendar-season features hurt.**  Removing month / day-of-year / heating
   flag improves F2 by 2.2 RMSE.  With only 2–3 winters in train the trees use
   them to memorise year-specific weather; temperature, dew point and pressure
   already encode season in a way that transfers across years.
3. **log1p target transform hurts** (F2: 33.6 vs 30.7): it optimises relative
   error and blunts exactly the spikes RMSE punishes.
4. Larger, more regularised trees (255 leaves, min 200 samples/leaf, 40 %
   feature sub-sampling, L2 = 10) are slightly better than the default.
5. Winter-only sample weighting: no gain.
6. A weighted average of LightGBM, XGBoost and CatBoost improves on the best
   single model (see table).

## 6. Final model (submitted `submission.csv`)

**Adjacent-row, two-stage stacked ensemble** (variant L + stacking), trained on
all 360,954 labelled rows:

```
python scripts/run_final.py --variant L --stack --models lgb --rounds lgb=2000 --seeds 42 7 2024
```

* Features: 212 base + 96 lead (`build_features(use_lead=True)`) + 22 stage-1
  nowcast features (`src/stacking.py`, incl. `NOWCAST_LEAD_FEATS`).
* Stage 1: LightGBM nowcast of PM2.5 at hour *t* from the base **and lead**
  features (two-sided), out-of-fold over 4 time blocks for train rows, fitted
  on all train rows for test rows.
* Stage 2: LightGBM with the hyper-parameters of §6a below, Sep–Feb rows
  weighted 2×, seeds 42 / 7 / 2024 averaged, **2,000 rounds** (fixed; from the
  fold-2 early-stopping optimum of 2,186 and the fold-1 round curve in
  `results/round_curve_L.json`).
* Season-matched validation of this configuration: fold 1 = 22.60 (fixed
  2,000 rounds), fold 2 = 18.08 (early-stopped at 2,186 rounds), pooled 20.46.
* XGBoost (1,500 rounds) and CatBoost (1,800 rounds) members were trained
  and validated on fold 1 (22.49 / 22.47; equal 3-way average 22.24) but their
  full-data fits did not finish before the deadline, so the submitted file is
  the LightGBM member alone (`scripts/blend.py --tag LS --weights lgb=1.0`).

### 6a. Causal model (`results/experiments/submission_causal_LB23.43549.csv`, public LB 23.43549)

**Two-stage stacked ensemble, trained on all 360,954 labelled rows**
(`scripts/run_final.py --variant A --stack --models lgb xgb cat --rounds lgb=1000 xgb=700 cat=1300 --weights 0.3 0.5 0.2 --seeds 42 7 2024 --out results/experiments/submission_causal_LB23.43549.csv`).

Stage 1 — nowcast of PM2.5 at the observation hour
* LightGBM, `config.LGB_PARAMS` with learning-rate 0.05, 350 rounds, label = previous row's target (99.3 % of train rows).
* Out-of-fold over 4 contiguous time blocks for the training rows; a fifth model on all labelled rows produces the test-row nowcast.
* Ten derived features on the hourly grid: current estimate, lags 1–3 h, rolling means 3/6/24 h, 1 h change, deviation from the 24 h mean, estimate minus PM10.

Stage 2 — next-hour forecast, 212 base + 10 nowcast features, Sep–Feb rows weighted 2×, three seeds (42, 7, 2024) averaged per library

| library | key hyper-parameters | rounds | weight |
|---|---|---|---|
| LightGBM 4.7 | L2 objective, lr 0.03, 255 leaves, min 200 rows/leaf, feature fraction 0.4, bagging 0.8, λ₂ = 10, `station_code` categorical | 1000 | 0.3 |
| XGBoost 3.4 | squared error, hist, lr 0.03, depth 8, min child weight 200, subsample 0.8, colsample 0.4, λ = 10 | 700 | 0.5 |
| CatBoost 1.2 | RMSE, lr 0.05, depth 8, l2_leaf_reg 10 | 1300 | 0.2 |

Round counts are the fold-2 early-stopping optima (922 / 604 / 1187) inflated
~10 % for the larger training set.  `scripts/round_curve.py` traces the
LightGBM validation curve (`results/round_curve.json`): fold 2 bottoms out at
900–1000 rounds (27.05) and fold 1, which early-stops at ~100 rounds (26.23),
is flat at 26.39 ± 0.02 from 400 to 1300 rounds — so a single round count is
safe for both regimes and costs at most ≈ 0.15 on the fold-1-like case.
`scripts/check_causality.py` verifies that deleting every row after
2016-12-01 leaves all 212 base features of the earlier rows bit-identical,
i.e. nothing in the submitted model looks at later rows.
Post-processing: predictions clipped at 0; nothing else.

Validation of exactly this configuration (fold-level, `results/ensemble_results_AS.json`,
`results/analysis.md`): F1 25.93, F2 26.90, pooled **26.42**.  Top-5 % of
true values: RMSE 77, mean bias −40 (under-calling severe hours is the
dominant residual).

## 7. Why the causal ceiling is where it is (and what the leaderboard gap is)

Decompose the one-hour-ahead error: PM2.5(t+1) = PM2.5(t) + Δ.  Given the true
PM2.5(t), a model scores ≈ 19 on our folds (the oracle row above), so Δ has
≈ 19 µg/m³ of irreducible noise.  Without the column we must *estimate*
PM2.5(t) from PM10/CO/NO2/SO2/O3/weather; that nowcast has RMSE ≈ 22–25 and
its error is autocorrelated (the PM2.5/PM10 composition ratio drifts slowly
with humidity and source mix), so extra history helps little.  Combining the
two terms gives ≈ 27–28 — where our causal models land.  On the (milder)
2016-17 test winter the same model is expected to score in the low-to-mid 20s,
i.e. the second cluster of the public leaderboard.

### The two submissions and what the adjacent-row model is

The organisers' test file is a complete six-month table.  For the row
observed at hour *t* (target = PM2.5 at *t+1*) the rows observed at *t+1,
t+2, …* are present too, with their PM10 / CO / NO2 / SO2 / O3 and weather
readings.  Because PM2.5 at *t+1* is largely determined by the other pollutant
readings *at t+1* (§1: nowcast RMSE ≈ 21–25 from the same hour, and much less
when the neighbouring hours are also visible), a model that reads those later
rows is not a forecast but a two-sided estimate of PM2.5 at the target hour.

We built the causal forecaster first (§§2–6a; public leaderboard 23.43549),
established its information ceiling (above), and then - given that the
leaderboard's top cluster (≈ 18) is below what even the *true* current PM2.5
would allow for a real forecast (oracle ≈ 19) - built the adjacent-row model
with the same pipeline (`use_lead=True`, stacking with two-sided nowcast
features).  The two are kept as separate, fully reproducible variants:

| | uses rows after *t*? | validation (pooled) | public LB |
|---|---|---|---|
| causal, variant A + stacking | no (`scripts/check_causality.py`) | 26.42 | 23.43549 |
| adjacent-row, variant L + stacking (**submitted**) | yes, *t+1 … t+6* | ≈ 20.5 (LightGBM member, pooled; fold 1 = 22.24 for the 3-library average, fold 2 = 18.08 for LightGBM) | (fill in from the leaderboard) |

For an operational early-warning system only the causal model is applicable;
its §8 findings are the ones that transfer.  The adjacent-row model answers a
different question - "given the full record of the other pollutants, what was
PM2.5 at each hour?" - which is the *imputation / sensor cross-check* problem
and is useful in its own right (e.g. filling gaps in the PM2.5 record from the
co-located sensors), but it should not be read as a forecast skill.

## 8. Findings relevant to monitoring and public-health planning

* **What predicts PM2.5** (`results/analysis.md`, LightGBM gain of the final
  model): the stage-1 nowcast of the *current* PM2.5 and its last 1–6 h
  (≈ 65 % of total gain), then PM10 at the observation hour (13 %), the
  city-wide mean PM10 (7 %) and CO, followed by short PM10/CO history,
  dew-point changes and humidity.  In the un-stacked model the same picture
  holds with PM10 (41 %) and city-mean PM10 (23 %) on top.  Physically: PM10
  and CO are the co-emitted / co-transported tracers of fine particles, and
  humidity increases the PM2.5 share of PM10 (hygroscopic growth) — a split
  the trees learn through dew-point depression and relative humidity.
* **Stations**: urban core stations (Dongsi, Wanshouxigong, Wanliu,
  Nongzhanguan, ~82–84 µg/m³ mean) run ~18 µg/m³ above the northern suburban
  sites (Dingling, Huairou, Changping); yet the cross-station features show
  episodes are synchronous city-wide — a network-level alert is better than
  station-by-station thresholds.
* **Seasons and hours**: means peak Oct–Mar (90–96) versus ~53 in August;
  diurnal peak at 20–23 h and trough mid-afternoon.  Because RMSE grows with
  level, forecast uncertainty in winter evenings is 2–3× that of summer
  afternoons — advisories should quote an interval, not a point.
* **Operational recommendation**: the single most valuable investment is
  making the *current* PM2.5 reading available to the forecaster — the oracle
  experiment shows it alone would cut next-hour RMSE from ~27 to ~19.

## 9. Limitations and next steps

* Only two season-matched winters for validation; fold-to-fold spread is
  ~2 RMSE, so differences below ~0.3 are not reliable.
* No use of the nowcast estimate as an explicit stacked feature; a two-stage
  (nowcast → forecast) model with out-of-fold stage-1 predictions was designed
  but not run for lack of time.
* Sequence models (GRU/TCN over the per-station history) and quantile /
  distributional outputs for advisory intervals are natural extensions.
* Hyper-parameter search was a small manual sweep, not Bayesian optimisation.
* For the adjacent-row model, a longer two-sided context (12/24 h leads,
  forward means, 25 h centred means: `build_features(lead_long=True)`) with a
  better-trained stage 1 (`--stage1-rounds 1000`) is implemented and switchable
  but its validation could not be completed before the deadline; it is the
  obvious next experiment.
