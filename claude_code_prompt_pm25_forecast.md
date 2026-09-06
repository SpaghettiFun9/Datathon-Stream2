# PM2.5 Next-Hour Forecasting — Build Instructions for Claude Code

> **How to use this**: paste this whole document into Claude Code chat as your first message, or save it as `PROMPT.md` / fold it into `CLAUDE.md` in your project root. It's self-contained — it doesn't assume you have any other context from elsewhere.

## Objective

Forecast `PM2_5_next_hour` (µg/m³, one hour ahead, same station) for a hidden test set, scored by RMSE. Lower is better. This is a live datathon with a public leaderboard — current standings cluster like this:

| Rank | RMSE |
|---|---|
| 1 | 18.13847 |
| 2 | 18.63924 |
| 3 | 18.69571 |
| 4 | 19.14834 |
| 5 | 24.19524 |
| 6 | 24.63263 |
| 7 | 24.79790 |

Treat **RMSE 18–19** as the target to beat, and note the visible gap down to ~24-25 — there's a real difference in approach between those two clusters, not just tuning (see "Why the lag-reconstruction approach is likely necessary" below).

## Files

Locate these in the project directory (names may vary slightly, e.g. `test.csv` vs `test(1).csv` — check what's actually there rather than assuming):
- A labeled training file (~360,954 rows)
- An unlabeled test file (~51,063 rows)
- `sample_submission.csv` (columns: `id`, `PM2_5_next_hour`)

## Hard constraints — follow these throughout

- **No web search, no external datasets, no attempt to identify or locate the original source of this data.** Work only from the provided files.
- **No attempt to recover hidden test targets** by any means other than legitimate modeling from the provided files.
- These aren't optional style preferences — they're competition integrity rules. Don't route around them even if it would technically improve the score.

## Verified facts about the data

I've already inspected this data directly (not guessing) — re-verify these against your local copies as a sanity check, but expect them to hold:

- **Columns actually present**: `id, observation_timestamp, station, year, month, day, hour, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM` plus `PM2_5_next_hour` in train only. **There is no `current_PM2_5` column in either file**, even if a data dictionary you've seen describes one. Design around this — it's the central challenge of the problem, not an oversight to work around trivially.
- **12 stations**: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong.
- **Chronological split, no gap**: train spans 2013-03-01 00:00 to 2016-08-31 22:00; test spans 2016-08-31 23:00 to 2017-02-28 22:00. For every station, the last train timestamp is exactly one hour before that station's first test timestamp — a clean continuation, no overlap.
- **Important consequence**: for each station's very first test row, the true current-hour PM2.5 is exactly that station's last train row's `PM2_5_next_hour` value. That's known, not approximated — use it directly.
- **Missingness (train / test)**: PM10 0.5%/0.6%, SO2 1.3%/0.9%, NO2 2.1%/1.3%, CO 4.4%/1.5%, O3 2.3%/2.0%, TEMP/PRES/DEWP/RAIN/WSPM ~0.05%/~0.4%, `wd` 0.2%/2.0%. Test has proportionally more missing weather and wind data than train. ~7.7% of train rows are missing at least one pollutant; ~0.3% are missing all five at once (short sensor outages). When `wd` is missing, `WSPM` is almost always near zero (median 0.1 m/s) — missing direction ≈ calm wind, not a random hole.
- **Time gaps**: ~99.3% (train) / ~99.25% (test) of consecutive rows within a station are exactly 1 hour apart. The rest have gaps from 2 hours up to 344 hours (train) / 73 hours (test). Any lag/rolling feature must be gap-aware — null it rather than silently bridging a real gap.
- **Correlation with `PM2_5_next_hour`** (train): PM10 0.85, CO 0.76, NO2 0.64, SO2 0.50, WSPM −0.28, O3 −0.13, TEMP −0.12, DEWP 0.12, RAIN −0.03, PRES 0.02.
- **Seasonality**: strong heating-season effect — monthly mean target as low as ~53 µg/m³ in August, up in the low-to-mid 90s in Oct/Nov/Dec/Mar.
- **Diurnal cycle**: lowest (~72-73 µg/m³) mid-afternoon (14-16h), highest (~84-86 µg/m³) at night (20-23h).
- **Station baselines** differ by ~18 µg/m³ between the most-polluted (Dongsi, ~84) and least-polluted (Dingling, ~65) station on average.
- **The validation trap**: test (Sep 2016–Feb 2017) is entirely the high-pollution autumn/winter season. The chunk of train right before the cutoff (Mar–Aug 2016) is entirely the low-pollution spring/summer season. Holding out "the last N% of train by time" as validation will validate on an easy, low-variance regime and look much better than real test performance. Build season-matched validation folds instead (below) — don't skip this, it will silently mislead every decision downstream.

## Why the lag-reconstruction / recursive approach is very likely necessary

Train's target has std ≈ 78 µg/m³. A simple linear fit off PM10 alone (r=0.85, R²≈0.72) would still leave residual error somewhere in the mid-30s to mid-40s µg/m³ — nowhere near RMSE 18-19. Getting into that range needs something closer to an actual short-term PM2.5 history, even though it isn't given as a column. The leaderboard gap between the ~18-19 cluster and the ~24-25 cluster is consistent with the top teams having found a way to approximate "how high was PM2.5 very recently at this station" — most plausibly by exploiting the fact above (each row's `PM2_5_next_hour` is next hour's true current PM2.5, so it can be reconstructed as a lag feature for training and propagated forward at test time). **Treat implementing this correctly as a priority, not a stretch goal** — it's likely the main gap between the two clusters on the board.

## Build this, in order

### 1. Load & re-verify
Load the files, re-run the checks above (column list, missingness, per-station time gaps, train→test continuity) against the real local data, and report any differences before proceeding — the numbers above are a strong prior, not a guarantee.

### 2. Time-aware validation harness — build and lock this before touching models
No random K-fold. No "last N% of train by time" either — use season-matched holdouts:
- **Fold 1**: train on rows before 2014-09-01, validate on 2014-09-01 → 2015-03-01.
- **Fold 2**: train on rows before 2015-09-01, validate on 2015-09-01 → 2016-03-01.
- Report both folds' RMSE, not just the average — check they're reasonably consistent with each other.
- Only after you're happy with an approach, retrain on ALL of train (through 2016-08-31) for the actual submission.

### 3. Feature engineering — safe baseline features
These use identical logic on train and test, no leakage risk:
- Cyclical hour-of-day (sin/cos, period 24) and month or day-of-year (sin/cos, period 12 or 365). A blunt heating-season flag (~Nov–Mar) is worth trying given the seasonality above.
- `station` as a categorical feature — try native categorical handling first if your library supports it.
- Decode 16-point `wd` to degrees and encode cyclically; add a "calm" flag for near-zero `WSPM` / missing `wd`.
- Lag (1h/2h/3h) and rolling (mean+std over 3h/6h/12h/24h) features of PM10/CO/NO2/SO2/O3, computed per-station in chronological order. Mask/null any value whose window crosses a real time gap.
- Missing-value indicator flags per pollutant, and prefer a model with native NaN support over manual imputation — imputing the drivers of a right-skewed target tends to blur exactly the signal behind the extreme events RMSE punishes hardest.

### 4. Feature engineering — the lag-reconstruction features (build carefully, this is the important part)

**For TRAIN**: create `reconstructed_current_pm25` as, per station in chronological order, the PREVIOUS row's `PM2_5_next_hour` — but only when the previous row's timestamp is exactly 1 hour earlier (null it otherwise). Build lag-2/lag-3 and short rolling stats of this reconstructed series the same way.

**For TEST inference**, walk forward sequentially per station in chronological order — this cannot be a single vectorized batch prediction, it's inherently sequential:
- The first test row of each station gets a **real** current-PM2.5 value: that station's last train row's true `PM2_5_next_hour` (see the continuity fact above). Not an approximation — use it directly.
- For each subsequent row: if the gap from the previous row is exactly 1 hour, use the model's own prediction for the previous row as this row's lag-1 feature (and roll lag-2/lag-3/rolling stats forward using the running true-then-predicted sequence). If the gap isn't 1 hour, the chain is broken — null the lag features for that row (falling back to the no-lag feature set) and restart from the next row.

**Compare two model variants on the season-matched validation folds**, simulating the exact sequential walk on the validation window itself (not a row-wise shortcut, or the comparison isn't honest):
- **Model A**: no PM2.5 lag at all — contemporaneous + time + station + wind features only.
- **Model B**: Model A's features plus the reconstructed/recursive lag features.

Expect B to substantially beat A if the leaderboard reasoning above is right. If it doesn't, suspect a bug in the sequential-walk implementation before concluding the idea doesn't work — this is the most bug-prone part of the whole pipeline (off-by-one timing errors and gap-handling mistakes are the likely failure modes).

### 5. Modeling
Gradient-boosted trees are the right tool here — fast, native handling of missing values and mixed categorical/numeric features, hard to beat on tabular data at this size. Try LightGBM and/or CatBoost (convenient native categorical handling for `station`/`wd`) and/or XGBoost — install whichever isn't already available (`pip install lightgbm catboost xgboost`; if environment/network restrictions block installation, `sklearn`'s `HistGradientBoostingRegressor` is a solid native-NaN-handling fallback). Use early stopping against the season-matched validation folds, not training loss. A simple average ensemble of 2-3 models (different seeds/algorithms) is often worth trying once you have a couple of solid individual models.

### 6. Pay attention to the tail
RMSE punishes large misses hardest, and the brief explicitly cares about not under/over-calling severe pollution events. After validating, check residuals specifically on the top 1-5% of target values in the validation folds, not just overall RMSE — a model that's fine on typical days but blunts the spikes loses the most exactly where it matters. If that's happening, try log1p-transforming the target for training and back-transforming predictions — but verify empirically on the validation folds that it actually improves raw-scale RMSE, since it doesn't always.

### 7. Final submission
Retrain your chosen approach on the FULL train set (through 2016-08-31), run the sequential walk-forward inference across the entire test set in chronological order per station, and write `id, PM2_5_next_hour`. Merge predictions onto `id` rather than trusting row order, and confirm row count and id set match `sample_submission.csv` exactly before finishing.

## Deliverables

- A reproducible script or notebook covering the pipeline above.
- Validation RMSE reported at each meaningful step (baseline features → +lag/recursive features → model/ensemble choices), so progress is auditable.
- Final `submission.csv` matching `sample_submission.csv`'s format exactly.
- A short summary of what was tried, what the season-matched validation RMSE was, and what you'd try next with more time.
