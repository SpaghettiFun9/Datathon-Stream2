"""
Data loading and integrity checks.

`load_all()` returns one DataFrame containing train and test stacked together
(test rows have NaN target and `is_test == 1`).  Stacking them is what lets the
per-station lag features flow seamlessly from the last train hour into the
first test hour - the two files are a clean chronological continuation.
"""
import numpy as np
import pandas as pd

import config as C


def load_all() -> pd.DataFrame:
    train = pd.read_csv(C.TRAIN_CSV, parse_dates=["observation_timestamp"])
    test = pd.read_csv(C.TEST_CSV, parse_dates=["observation_timestamp"])
    train["is_test"] = 0
    test["is_test"] = 1
    test[C.TARGET] = np.nan
    df = pd.concat([train, test], ignore_index=True)
    df = df.sort_values(["station", "observation_timestamp"]).reset_index(drop=True)
    # Pandas 3 defaults string columns to the new "str" dtype; keep plain object
    # so that the categorical / mapping code below behaves identically everywhere.
    df["station"] = df["station"].astype(object)
    df["wd"] = df["wd"].astype(object)
    return df


def verify(df: pd.DataFrame) -> None:
    """Re-check the structural facts the whole pipeline relies on."""
    tr, te = df[df.is_test == 0], df[df.is_test == 1]
    assert sorted(tr.station.unique()) == C.STATIONS, "unexpected station set"
    assert sorted(te.station.unique()) == C.STATIONS
    assert "current_PM2_5" not in df.columns, "spec says no current PM2.5 column"
    assert not df.duplicated(["station", "observation_timestamp"]).any(), "duplicate station-hours"
    # train -> test continuity: for every station the first test hour is exactly
    # one hour after the last train hour, so the last train target IS the true
    # current PM2.5 of the first test row.
    for st in C.STATIONS:
        last_tr = tr.loc[tr.station == st, "observation_timestamp"].max()
        first_te = te.loc[te.station == st, "observation_timestamp"].min()
        assert first_te - last_tr == pd.Timedelta("1h"), f"continuity broken at {st}"
    print(f"[verify] train rows={len(tr):,}  test rows={len(te):,}  stations={len(C.STATIONS)}")
    print(f"[verify] train span {tr.observation_timestamp.min()} -> {tr.observation_timestamp.max()}")
    print(f"[verify] test  span {te.observation_timestamp.min()} -> {te.observation_timestamp.max()}")
    gaps = df.groupby("station").observation_timestamp.diff() / pd.Timedelta("1h")
    print(f"[verify] consecutive rows exactly 1h apart: {(gaps.dropna() == 1).mean():.4f}")
    print("[verify] all continuity checks passed")
