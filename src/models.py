"""Thin wrappers around the gradient-boosting libraries used in the pipeline."""
import numpy as np
import lightgbm as lgb

import config as C


class TargetTransform:
    """Optional monotone transform of the target (identity or log1p).

    RMSE is evaluated on the raw scale, so any transform must be verified
    empirically on the season-matched folds rather than assumed to help.
    """

    def __init__(self, kind="identity"):
        assert kind in ("identity", "log1p")
        self.kind = kind

    def fwd(self, y):
        return np.log1p(y) if self.kind == "log1p" else y

    def inv(self, z):
        return np.expm1(z) if self.kind == "log1p" else z


def train_lgb(X_tr, y_tr, X_va=None, y_va=None, params=None, num_rounds=None,
              transform=TargetTransform(), feature_names=None, verbose_every=200, weight=None):
    """
    Train a LightGBM regressor.
    * If a validation set is given: early-stop on it and return the booster
      plus the best iteration.
    * Otherwise train for exactly `num_rounds` (used for the final fit on all
      of train, where `num_rounds` comes from the validation runs).
    """
    params = dict(params or C.LGB_PARAMS)
    cat = ["station_code"] if feature_names and "station_code" in feature_names else "auto"
    dtr = lgb.Dataset(X_tr, transform.fwd(y_tr), weight=weight, feature_name=feature_names, categorical_feature=cat, free_raw_data=False)
    callbacks = [lgb.log_evaluation(verbose_every)] if verbose_every else []
    if X_va is not None:
        dva = lgb.Dataset(X_va, transform.fwd(y_va), reference=dtr)
        callbacks.append(lgb.early_stopping(C.LGB_EARLY_STOP, verbose=False))
        booster = lgb.train(params, dtr, num_boost_round=num_rounds or C.LGB_MAX_ROUNDS,
                            valid_sets=[dva], valid_names=["val"], callbacks=callbacks)
        return booster, booster.best_iteration
    booster = lgb.train(params, dtr, num_boost_round=num_rounds, callbacks=callbacks)
    return booster, num_rounds


def predict_lgb(booster, X, transform=TargetTransform(), num_iteration=None):
    p = transform.inv(booster.predict(X, num_iteration=num_iteration))
    return np.clip(p, 0, None)          # concentrations cannot be negative


# --------------------------------------------------------------------------- other libraries (ensemble members)
XGB_PARAMS = dict(
    objective="reg:squarederror", tree_method="hist", learning_rate=0.03, max_depth=8,
    min_child_weight=200, subsample=0.8, colsample_bytree=0.4, reg_lambda=10.0,
    nthread=C.N_THREADS, seed=C.SEED,
)
CAT_PARAMS = dict(
    loss_function="RMSE", learning_rate=0.05, depth=8, l2_leaf_reg=10.0,
    iterations=5000, random_seed=C.SEED, thread_count=C.N_THREADS, verbose=0,
    allow_writing_files=False,
)


def train_xgb(X_tr, y_tr, X_va=None, y_va=None, num_rounds=None, params=None, weight=None):
    import xgboost as xgb
    params = dict(params or XGB_PARAMS)
    dtr = xgb.DMatrix(X_tr, y_tr, weight=weight)
    if X_va is not None:
        dva = xgb.DMatrix(X_va, y_va)
        b = xgb.train(params, dtr, num_boost_round=num_rounds or C.LGB_MAX_ROUNDS, evals=[(dva, "val")],
                      early_stopping_rounds=C.LGB_EARLY_STOP, verbose_eval=False)
        return b, b.best_iteration + 1
    return xgb.train(params, dtr, num_boost_round=num_rounds), num_rounds


def predict_xgb(b, X, num_iteration=None):
    import xgboost as xgb
    kw = {"iteration_range": (0, num_iteration)} if num_iteration else {}
    return np.clip(b.predict(xgb.DMatrix(X), **kw), 0, None)


def train_cat(X_tr, y_tr, X_va=None, y_va=None, num_rounds=None, params=None, weight=None):
    from catboost import CatBoostRegressor
    params = dict(params or CAT_PARAMS)
    if num_rounds:
        params["iterations"] = num_rounds
    m = CatBoostRegressor(**params)
    if X_va is not None:
        m.fit(X_tr, y_tr, sample_weight=weight, eval_set=(X_va, y_va), early_stopping_rounds=C.LGB_EARLY_STOP, use_best_model=True)
        return m, m.get_best_iteration() + 1
    m.fit(X_tr, y_tr, sample_weight=weight)
    return m, num_rounds


def predict_cat(m, X, num_iteration=None):
    return np.clip(m.predict(X), 0, None)
