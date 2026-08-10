"""
Module to evaluate models and compute prediction thresholds
"""

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def predict(models, data):
    # average across the folds
    return np.mean([m.predict_proba(data)[:, 1] for m in models], axis=0)


def _bootstrap_auc_ci(y_true, y_pred, n_samples=1000, seed=42):
    """Compute 95% CI for AUROC via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    idxs = np.arange(n)
    aucs = []
    for _ in range(n_samples):
        idx = rng.choice(idxs, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi

def evaluate_test(model, X, Y, n_bootstrap=1000, seed=42):
    result = {}
    for target, label in Y.items():
        pred = predict(model[target], X)
        auroc = roc_auc_score(label, pred)
        lo, hi = _bootstrap_auc_ci(label, pred, n_samples=n_bootstrap, seed=seed)
        result[target] = {
            "AUPRC": average_precision_score(label, pred),
            "AUROC": f"{auroc:.4f} ({lo:.4f}, {hi:.4f})",
            "Brier": brier_score_loss(label, pred),
        }
    return pd.DataFrame(result)


def evaluate_valid(
    models: dict, X: pd.DataFrame, Y: pd.DataFrame, metainfo: pd.DataFrame
):
    result = {}
    for alg, estimators in models.items():
        output = _evaluate_all_targets(estimators, X, Y, metainfo)
        result[alg] = pd.DataFrame(output)
    return pd.concat(result).T


def _evaluate_all_targets(
    models: dict, X: pd.DataFrame, Y: pd.DataFrame, metainfo: pd.DataFrame
):
    """Evaluate models for each target by averaging the performance across cross-validation folds"""
    result = {}
    for target, label in Y.items():
        output = _evaluate_across_folds(models[target], X, label, metainfo)
        result[target] = pd.DataFrame(output).mean(axis=1)
    return result


def _evaluate_across_folds(
    models: list, X: pd.DataFrame, Y: pd.Series, metainfo: pd.DataFrame
):
    """Evaluate models for each fold using it's associated validation set"""
    result = {}
    for fold, model in enumerate(models):
        mask = metainfo["cv_folds"] == fold
        X_valid, Y_valid = X[mask], Y[mask]
        result[fold] = _evaluate(model, X_valid, Y_valid)
    return result


def _evaluate(model, data: pd.DataFrame, label: pd.Series):
    pred = model.predict_proba(data)[:, 1]
    return {
        "AUPRC": average_precision_score(label, pred),
        "AUROC": roc_auc_score(label, pred),
    }


###############################################################################
# Thresholding
###############################################################################
def compute_threshold(pred: Sequence[float], desired_alarm_rate: float):
    """Compute the prediction threshold based on desired alarm rate"""
    for pred_threshold in np.arange(0, 1.0, 0.001):
        alarm_rate = np.mean(pred > pred_threshold)
        if np.isclose(alarm_rate, desired_alarm_rate, atol=0.005):
            return pred_threshold, alarm_rate
