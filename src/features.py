"""
Feature engineering.

Design
------
All temporal features are computed on a *regular hourly grid* per station:
every station is re-indexed onto the complete hourly range spanning train and
test.  Hours that are absent from the files become all-NaN rows.  A `shift(k)`
on that grid is therefore automatically gap-aware - a lag whose source hour is
missing is NaN instead of silently borrowing the value from 2 (or 300) hours
earlier.  Rolling windows use `min_periods=1` so they average whatever is
available inside the window without bridging beyond it.

Internally each raw column is a (T x S) DataFrame (rows = hours, columns =
stations) so that pandas' column-wise shift / rolling gives per-station
operations in one vectorised call.  Cross-station ("city-wide") features are
just row-wise statistics of the same matrices.

Three feature families are produced:
  * BASE      - contemporaneous + time + station + wind + gap-aware lags/rollings
                of the observable pollutants & weather + cross-station stats.
                Uses only information available at or before the observation
                hour.  Identical logic for train and test.
  * PM25 lags - `pm25_lag1` is the PREVIOUS row's target (= true PM2.5 at the
                observation hour), reconstructed from the train file.  On the
                combined grid the shift automatically gives the first test hour
                of every station its true value from the last train row.  Later
                test hours are NaN here and are filled by the recursive walk in
                `src/recursive.py`.
  * LEAD      - optional, OFF by default.  Values from the *next* rows of the
                file (t+1, t+2).  These are not available in a real-time
                forecast, so they are kept behind a switch and documented
                separately; see README "Lead features".
"""
import numpy as np
import pandas as pd

import config as C

PM25_HIST = 24  # hours of PM2.5 history the recursive features look back over


# --------------------------------------------------------------------------- helpers
def _grid(df: pd.DataFrame):
    """Return (full hourly index, station list, dict of (T x S) matrices)."""
    stations = C.STATIONS
    full_ts = pd.date_range(df.observation_timestamp.min(), df.observation_timestamp.max(), freq="h")
    mats = {}
    cols = C.POLLUTANTS + C.MET_VARS + ["wd_deg", C.TARGET, "present"]
    for col in cols:
        mats[col] = (
            df.pivot(index="observation_timestamp", columns="station", values=col)
            .reindex(index=full_ts, columns=stations)
            .astype(float)
        )
    return full_ts, stations, mats


def _long(mat: pd.DataFrame) -> np.ndarray:
    """(T x S) matrix -> station-major flat vector matching the long index."""
    return mat.to_numpy().T.ravel()


def pm25_features_from_matrix(pm: pd.DataFrame) -> dict:
    """
    PM2.5-history features from a (T x S) matrix whose row t holds the PM2.5
    concentration AT hour t (i.e. the value known when forecasting t+1).
    The exact same definitions are re-implemented on a small numpy buffer in
    `src/recursive.py`; `scripts/run_validation.py` asserts they agree.
    """
    out = {
        "pm25_lag1": pm,
        "pm25_lag2": pm.shift(1),
        "pm25_lag3": pm.shift(2),
        "pm25_rm3": pm.rolling(3, min_periods=1).mean(),
        "pm25_rm6": pm.rolling(6, min_periods=1).mean(),
        "pm25_rm24": pm.rolling(PM25_HIST, min_periods=1).mean(),
        "pm25_rmax6": pm.rolling(6, min_periods=1).max(),
    }
    out["pm25_diff1"] = out["pm25_lag1"] - out["pm25_lag2"]
    out["pm25_dev24"] = out["pm25_lag1"] - out["pm25_rm24"]
    return out


# --------------------------------------------------------------------------- main
def build_features(df: pd.DataFrame, use_lead: bool = False, drop=None):
    """
    Parameters
    ----------
    df : stacked train+test frame from `src.data.load_all()`.
    use_lead : also build LEAD features (t+1 / t+2 rows).  Default False.
    drop : feature names to leave out of the "base" list (default config.DROP_FEATURES).

    Returns
    -------
    data : df with feature columns appended (same row order as input)
    feats : dict of feature-name lists: {"base": [...], "pm25": [...], "lead": [...]}
    """
    df = df.copy()
    df["wd_deg"] = df["wd"].map(C.WD_DEGREES).astype(float)
    df["present"] = 1.0

    full_ts, stations, M = _grid(df)
    long_index = pd.MultiIndex.from_product([stations, full_ts], names=["station", "observation_timestamp"])
    F = {}  # name -> flat vector on long_index
    base, pm25_feats, lead = [], [], []

    def add(name, mat, family):
        F[name] = _long(mat)
        family.append(name)

    # ---------------------------------------------------------------- pollutants
    for p in C.POLLUTANTS:
        x = M[p]
        for k in (1, 2, 3, 6, 12, 24):
            add(f"{p}_lag{k}", x.shift(k), base)
        add(f"{p}_diff1", x - x.shift(1), base)
        add(f"{p}_diff3", x - x.shift(3), base)
        for w in (3, 6, 12, 24):
            add(f"{p}_rm{w}", x.rolling(w, min_periods=1).mean(), base)
        for w in (6, 24):
            add(f"{p}_rs{w}", x.rolling(w, min_periods=2).std(), base)
            add(f"{p}_rmax{w}", x.rolling(w, min_periods=1).max(), base)
        add(f"{p}_rmin24", x.rolling(24, min_periods=1).min(), base)
        add(f"{p}_dev24", x - x.rolling(24, min_periods=1).mean(), base)

        # cross-station (city-wide) statistics at the same hour
        city_mean = x.mean(axis=1)
        cm = pd.DataFrame(np.repeat(city_mean.to_numpy()[:, None], len(stations), axis=1), index=x.index, columns=x.columns)
        add(f"{p}_city_mean", cm, base)
        add(f"{p}_city_max", pd.DataFrame(np.repeat(x.max(axis=1).to_numpy()[:, None], len(stations), axis=1), index=x.index, columns=x.columns), base)
        add(f"{p}_city_min", pd.DataFrame(np.repeat(x.min(axis=1).to_numpy()[:, None], len(stations), axis=1), index=x.index, columns=x.columns), base)
        add(f"{p}_city_std", pd.DataFrame(np.repeat(x.std(axis=1).to_numpy()[:, None], len(stations), axis=1), index=x.index, columns=x.columns), base)
        add(f"{p}_city_dev", x - cm, base)
        for k in (1, 3, 6):
            add(f"{p}_city_mean_lag{k}", cm.shift(k), base)
        add(f"{p}_city_mean_diff1", cm - cm.shift(1), base)
        add(f"{p}_city_mean_rm6", cm.rolling(6, min_periods=1).mean(), base)
        add(f"{p}_city_mean_rm24", cm.rolling(24, min_periods=1).mean(), base)

        if use_lead:
            for k in (1, 2, 3, 6):
                add(f"{p}_lead{k}", x.shift(-k), lead)
            add(f"{p}_lead1_diff", x.shift(-1) - x, lead)
            add(f"{p}_lead2_diff", x.shift(-2) - x.shift(-1), lead)
            fwd = x[::-1]  # reversed time -> trailing windows become forward windows
            add(f"{p}_leadrm3", fwd.rolling(3, min_periods=1).mean()[::-1].shift(-1), lead)
            add(f"{p}_leadrm6", fwd.rolling(6, min_periods=1).mean()[::-1].shift(-1), lead)
            add(f"{p}_leadmax3", fwd.rolling(3, min_periods=1).max()[::-1].shift(-1), lead)
            add(f"{p}_c5", x.rolling(5, center=True, min_periods=1).mean(), lead)
            add(f"{p}_c13", x.rolling(13, center=True, min_periods=1).mean(), lead)
            for k in (1, 2, 3):
                add(f"{p}_city_mean_lead{k}", cm.shift(-k), lead)
            add(f"{p}_city_max_lead1", pd.DataFrame(np.repeat(x.max(axis=1).to_numpy()[:, None], len(stations), axis=1), index=x.index, columns=x.columns).shift(-1), lead)
            add(f"{p}_city_dev_lead1", (x - cm).shift(-1), lead)

    # ---------------------------------------------------------------- weather
    for v in ("TEMP", "PRES", "DEWP", "WSPM"):
        x = M[v]
        add(f"{v}_lag1", x.shift(1), base)
        add(f"{v}_lag3", x.shift(3), base)
        add(f"{v}_diff1", x - x.shift(1), base)
        add(f"{v}_diff3", x - x.shift(3), base)
        add(f"{v}_diff24", x - x.shift(24), base)
        add(f"{v}_rm6", x.rolling(6, min_periods=1).mean(), base)
        add(f"{v}_rm24", x.rolling(24, min_periods=1).mean(), base)
        if use_lead:
            add(f"{v}_lead1", x.shift(-1), lead)
            add(f"{v}_lead2", x.shift(-2), lead)
            add(f"{v}_lead1_diff", x.shift(-1) - x, lead)
    rain = M["RAIN"]
    add("RAIN_rsum6", rain.rolling(6, min_periods=1).sum(), base)
    add("RAIN_rsum24", rain.rolling(24, min_periods=1).sum(), base)
    if use_lead:
        add("RAIN_lead1", rain.shift(-1), lead)
        add("wd_deg_lead1", M["wd_deg"].shift(-1), lead)
        _rad1 = np.deg2rad(M["wd_deg"].shift(-1))
        add("wind_u_lead1", -M["WSPM"].shift(-1) * np.sin(_rad1), lead)
        add("wind_v_lead1", -M["WSPM"].shift(-1) * np.cos(_rad1), lead)

    # wind vector components (direction * speed) and their short history
    rad = np.deg2rad(M["wd_deg"])
    u = -M["WSPM"] * np.sin(rad)
    v = -M["WSPM"] * np.cos(rad)
    add("wind_u", u, base)
    add("wind_v", v, base)
    add("wind_u_rm6", u.rolling(6, min_periods=1).mean(), base)
    add("wind_v_rm6", v.rolling(6, min_periods=1).mean(), base)
    add("WSPM_rmax6", M["WSPM"].rolling(6, min_periods=1).max(), base)

    # number of stations reporting PM10 this hour (sensor-network health)
    n_rep = M["PM10"].notna().sum(axis=1).astype(float)
    add("n_stations_reporting", pd.DataFrame(np.repeat(n_rep.to_numpy()[:, None], len(stations), axis=1), index=full_ts, columns=stations), base)

    # ---------------------------------------------------------------- PM2.5 history (reconstructed)
    # Row t's target is PM2.5 at t+1  ==>  shifting the target down by one hour
    # gives PM2.5 AT hour t, i.e. the "current PM2.5" that the files omit.
    pm_true = M[C.TARGET].shift(1)
    for name, mat in pm25_features_from_matrix(pm_true).items():
        add(name, mat, pm25_feats)

    # ---------------------------------------------------------------- join back onto rows
    feat_df = pd.DataFrame(F, index=long_index)
    data = df.join(feat_df, on=["station", "observation_timestamp"])

    # ---------------------------------------------------------------- row-wise (no history) features
    ts = data.observation_timestamp
    data["hour_sin"] = np.sin(2 * np.pi * data.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data.hour / 24)
    doy = ts.dt.dayofyear
    data["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    data["dow"] = ts.dt.dayofweek
    data["is_weekend"] = (data.dow >= 5).astype(int)
    data["heating_season"] = data.month.isin([11, 12, 1, 2, 3]).astype(int)
    data["station_code"] = pd.Categorical(data.station, categories=C.STATIONS).codes
    data["wd_sin"] = np.sin(np.deg2rad(data.wd_deg))
    data["wd_cos"] = np.cos(np.deg2rad(data.wd_deg))
    data["wd_missing"] = data.wd_deg.isna().astype(int)
    data["calm"] = ((data.WSPM.fillna(0) < 0.3) | data.wd_deg.isna()).astype(int)
    data["dewp_dep"] = data.TEMP - data.DEWP
    # relative humidity from Magnus formula (hygroscopic growth changes PM2.5/PM10 ratio)
    data["rh"] = 100 * np.exp(17.625 * data.DEWP / (243.04 + data.DEWP)) / np.exp(17.625 * data.TEMP / (243.04 + data.TEMP))
    data["n_missing_poll"] = data[C.POLLUTANTS].isna().sum(axis=1)
    for p in C.POLLUTANTS:
        data[f"{p}_isna"] = data[p].isna().astype(int)
    data["PM10_over_CO"] = data.PM10 / (data.CO + 1)
    data["NO2_over_O3"] = data.NO2 / (data.O3 + 1)

    row_feats = (
        C.POLLUTANTS + C.MET_VARS
        + ["hour", "month", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow", "is_weekend",
           "heating_season", "station_code", "wd_deg", "wd_sin", "wd_cos", "wd_missing", "calm",
           "dewp_dep", "rh", "n_missing_poll", "PM10_over_CO", "NO2_over_O3"]
        + [f"{p}_isna" for p in C.POLLUTANTS]
    )
    drop = C.DROP_FEATURES if drop is None else drop
    feats = {"base": [f for f in row_feats + base if f not in drop], "pm25": pm25_feats, "lead": lead}
    return data, feats
