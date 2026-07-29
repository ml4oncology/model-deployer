"""
Module to train and tune the models using K-fold cross validation
"""

import warnings
from functools import partial
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger, ScreenLogger
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

algs = {
    "Ridge": LogisticRegression,
    "LASSO": LogisticRegression,
    "XGB": XGBClassifier,
    "LGBM": LGBMClassifier,
    "RF": RandomForestClassifier,
}

# model parameters
model_tuning_param = {
    "Ridge": {"C": (0.0001, 1)},
    "LASSO": {"C": (0.0001, 1)},
    "XGB": {
        "n_estimators": (50, 100),
        "max_depth": (3, 6),
        "learning_rate": (0.01, 0.3),
        "min_split_loss": (0, 0.5),
        "min_child_weight": (6, 100),
        "reg_lambda": (0, 1),
        "reg_alpha": (0, 1000),
    },
    "LGBM": {
        "n_estimators": (50, 200),
        "max_depth": (3, 6),
        "learning_rate": (0.01, 0.3),
        "num_leaves": (10, 40),
        "min_data_in_leaf": (10, 50),
        "feature_fraction": (0.5, 1),
        "bagging_fraction": (0.5, 1),
        "bagging_freq": (0, 10),
        "reg_lambda": (0, 1),
        "reg_alpha": (0, 1000),
    },
    "RF": {
        "n_estimators": (50, 100),
        "max_depth": (3, 6),
        "min_samples_leaf": (10, 50),
    },
    "SVC": {
        "C": (0.0001, 1),
        "kernel": (0, 3.99),
    },
}
bayesopt_param = {
    "Ridge": {"init_points": 2, "n_iter": 10},
    "LASSO": {"init_points": 2, "n_iter": 10},
    "XGB": {"init_points": 15, "n_iter": 200},
    "LGBM": {"init_points": 15, "n_iter": 200},
    "RF": {"init_points": 10, "n_iter": 50},
    "SVC": {"init_points": 10, "n_iter": 50},
}
model_static_param = {
    "Ridge": {
        "penalty": "l2",
        "class_weight": "balanced",
        "max_iter": 500,
        "random_state": 42,
    },
    "LASSO": {
        "penalty": "l1",
        "solver": "saga",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42,
    },
    "XGB": {
        "random_state": 42,
    },
    "LGBM": {"random_state": 42, "verbosity": -1},
    "RF": {"random_state": 42, "n_jobs": -1, "warm_start": True},
    "SVC": {"random_state": 42, "probability": True},
}


###############################################################################
# Training
###############################################################################
def train_model(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    metainfo: pd.DataFrame,
    alg: str,
    best_params: dict,
    calibrate: bool = True,
):
    models = {}
    for target, label in Y.items():
        kwargs = {**model_static_param[alg], **best_params[alg]}
        if alg in ["XGB", "LGBM"]:
            kwargs["scale_pos_weight"] = sum(label == 0) / sum(label == 1)

        models[target] = cross_validate(
            X, label, metainfo, alg, calibrate=calibrate, **kwargs
        )

    return models


def train_models(
    X: pd.DataFrame, Y: pd.DataFrame, metainfo: pd.DataFrame, best_params: dict
):
    return {alg: train_model(X, Y, metainfo, alg, best_params) for alg in algs}


def cross_validate(
    X: pd.DataFrame,
    Y: pd.Series,
    metainfo: pd.DataFrame,
    alg: str,
    calibrate: bool = False,
    **kwargs,
):
    models = []
    for fold in metainfo["cv_folds"].unique():
        mask = metainfo["cv_folds"] == fold

        # get the data splits
        X_train, X_valid = X[~mask].copy(), X[mask].copy()
        Y_train, Y_valid = Y[~mask].copy(), Y[mask].copy()

        # train the model
        if alg == "XGB":
            kwargs["early_stopping_rounds"] = 10
            fit_kwargs = {
                "eval_set": [(X_valid, Y_valid)],
                "verbose": 0,
            }
        elif alg == "LGBM":
            fit_kwargs = {
                "eval_set": [(X_valid, Y_valid)],
                "callbacks": [lgb.early_stopping(stopping_rounds=10, verbose=False)],
            }
        else:
            fit_kwargs = {}

        model = algs[alg](**kwargs)
        model.fit(X_train, Y_train, **fit_kwargs)

        if calibrate:
            model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
            model.fit(X_valid, Y_valid)

        models.append(model)

    return models


###############################################################################
# Hyperparameter tuning
###############################################################################
def tune_params(
    alg: str,
    X: pd.DataFrame,
    Y: pd.Series,
    metainfo: pd.DataFrame,
    log_dir: str = "./logs/bayes_opt",
    verbose: int = 2,
):
    """Tunes hyperparameters for a given algorithm using Bayesian Optimization."""
    hyperparam_config = model_tuning_param[alg]
    data = (X, Y, metainfo)
    bo = BayesianOptimization(
        f=partial(eval_func, alg=alg, data=data),
        pbounds=hyperparam_config,
        verbose=verbose,
        random_state=42,
        allow_duplicate_points=True,
    )

    # log the progress
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger1 = JSONLogger(path=f"{log_dir}/{alg}-bayesopt.log")
    logger2 = ScreenLogger(verbose=verbose, is_constrained=False)
    for bo_logger in [logger1, logger2]:
        bo.subscribe(Events.OPTIMIZATION_START, bo_logger)
        bo.subscribe(Events.OPTIMIZATION_STEP, bo_logger)
        bo.subscribe(Events.OPTIMIZATION_END, bo_logger)

    bo.maximize(**bayesopt_param[alg])
    best_param = bo.max["params"]

    return convert_params(best_param)


def convert_params(params):
    int_params = [
        "n_estimators",
        "max_depth",
        "num_leaves",
        "min_child_weight",
        "min_data_in_leaf",
        "min_samples_leaf",
        "bagging_freq",
    ]
    svc_kernels = ["linear", "poly", "rbf", "sigmoid"]
    for param, value in params.items():
        if param in int_params:
            params[param] = int(value)
        if param == "kernel":
            params[param] = svc_kernels[int(value)]
    return params


def eval_func(alg: str, data: tuple[pd.DataFrame, pd.Series, pd.Series], **kwargs):
    X, Y, metainfo = data

    kwargs = {**model_static_param[alg], **convert_params(kwargs)}
    models = cross_validate(X, Y, metainfo, alg, **kwargs)

    result = []
    for fold, model in enumerate(models):
        mask = metainfo["cv_folds"] == fold
        X_valid, Y_valid = X[mask], Y[mask]
        assert model.classes_[1] == 1  # positive class is at index 1
        pred_prob = model.predict_proba(X_valid)[:, 1]
        result.append(roc_auc_score(Y_valid, pred_prob))

    return np.mean(result)
