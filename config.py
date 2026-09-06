"""
Central configuration for the PM2.5 next-hour forecasting pipeline.

Everything that affects reproducibility (paths, seeds, fold boundaries,
model hyper-parameters, feature-set switches) lives here so that
`scripts/run_final.py` can regenerate the exact submission.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- data files
# Paths resolve in this order:
#   1. environment variables DATA_DIR (where the three CSVs live) / WORK_DIR (outputs)
#   2. Kaggle: the CSVs are searched under /kaggle/input/** and outputs go to /kaggle/working
#   3. local checkout: CSVs and outputs live next to this file
KAGGLE_INPUT = Path("/kaggle/input")
ON_KAGGLE = KAGGLE_INPUT.exists()


def _find_input(*names):
    """First file under /kaggle/input (any depth) whose name is one of `names`."""
    for name in names:
        hits = sorted(p for p in KAGGLE_INPUT.rglob(name) if p.is_file())
        if hits:
            return hits[0]
    raise FileNotFoundError(f"none of {names} found under {KAGGLE_INPUT}; set DATA_DIR")


if os.environ.get("DATA_DIR"):
    _D = Path(os.environ["DATA_DIR"])
    TRAIN_CSV, SAMPLE_SUBMISSION_CSV = _D / "train.csv", _D / "sample_submission.csv"
    TEST_CSV = _D / "test.csv" if (_D / "test.csv").exists() else _D / "test(1).csv"
elif ON_KAGGLE:
    TRAIN_CSV = _find_input("train.csv")
    TEST_CSV = _find_input("test.csv", "test(1).csv")
    SAMPLE_SUBMISSION_CSV = _find_input("sample_submission.csv")
else:
    TRAIN_CSV = ROOT / "train.csv"
    TEST_CSV = ROOT / "test(1).csv"          # file name as delivered by the organisers
    SAMPLE_SUBMISSION_CSV = ROOT / "sample_submission.csv"

WORK = Path(os.environ["WORK_DIR"]) if os.environ.get("WORK_DIR") else (Path("/kaggle/working") if ON_KAGGLE else ROOT)
RESULTS_DIR = WORK / "results"
MODELS_DIR = WORK / "models"
SUBMISSION_CSV = WORK / "submission.csv"

# ---------------------------------------------------------------- reproducibility
SEED = 42
N_THREADS = int(os.environ.get("N_THREADS", 10))   # override with env var when running experiments in parallel

# ---------------------------------------------------------------- domain constants
STATIONS = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng",
    "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong",
]
POLLUTANTS = ["PM10", "SO2", "NO2", "CO", "O3"]
MET_VARS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
TARGET = "PM2_5_next_hour"

# 16-point compass -> degrees (clockwise from north)
WD_DEGREES = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5, "E": 90.0, "ESE": 112.5,
    "SE": 135.0, "SSE": 157.5, "S": 180.0, "SSW": 202.5, "SW": 225.0,
    "WSW": 247.5, "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

# Calendar-season features are deliberately EXCLUDED from the model: with only
# 2-3 winters of training data they let the trees memorise year-specific
# weather patterns and hurt season-matched validation by ~1 RMSE
# (see results/sweep_results.json, config "no_season").
DROP_FEATURES = ["month", "doy_sin", "doy_cos", "heating_season"]

# Rows observed in the same season as the test set (Sep-Feb) get this sample
# weight during training (1.0 = no weighting).  Sweep: 27.38 vs 27.59 RMSE.
WINTER_MONTHS = [9, 10, 11, 12, 1, 2]
WINTER_WEIGHT = 2.0

# ---------------------------------------------------------------- validation
# The hidden test set is Sep-2016 .. Feb-2017 (autumn/winter, high pollution).
# The months of train immediately before it are spring/summer (low pollution),
# so a naive "last N months" hold-out is far too optimistic.  We instead use
# two SEASON-MATCHED folds: each validates on a Sep..Feb window and trains on
# everything strictly before it (no future information leaks into training).
FOLDS = [
    {"name": "F1_2014-15", "train_end": "2014-09-01", "val_start": "2014-09-01", "val_end": "2015-03-01"},
    {"name": "F2_2015-16", "train_end": "2015-09-01", "val_start": "2015-09-01", "val_end": "2016-03-01"},
]

# ---------------------------------------------------------------- model hyper-parameters
LGB_PARAMS = dict(
    objective="regression",          # plain L2 -> directly optimises RMSE
    learning_rate=0.03,
    num_leaves=255,                  # chosen by the sweep in scripts/run_sweep.py ("reg_big")
    min_data_in_leaf=200,
    feature_fraction=0.4,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    max_bin=255,
    verbose=-1,
    num_threads=N_THREADS,
    seed=SEED,
)
LGB_MAX_ROUNDS = 6000
LGB_EARLY_STOP = 200
