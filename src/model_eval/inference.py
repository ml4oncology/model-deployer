"""
Module to generate predictions
"""

import pickle
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
from deployer.loader import Model
from sklearn.base import BaseEstimator

ScikitModel = TypeVar("ScikitModel", bound=BaseEstimator)

ANCHOR_META_COLS = {"clinic": ["mrn", "next_sched_trt_date", "clinic_date"], "treatment": ["mrn", "treatment_date"]}


def predict(data: pd.DataFrame, models: list[ScikitModel]):
    # average across the folds
    return np.mean([m.predict_proba(data)[:, 1] for m in models], axis=0)


def get_model_output(
    model: Model, df: pd.DataFrame, thresholds: pd.DataFrame, pred_fn: Callable | None = None
) -> dict[str, pd.DataFrame]:
    """
    TODO: set data_pull_day as the assessment date for treatment date anchor
    """
    if pred_fn is None:
        pred_fn = predict

    model_input = df[model.model_features].copy()  # reorder model features according to the order used in training
    meta_cols = ANCHOR_META_COLS[model.anchor]
    model_output = df[meta_cols].copy()

    # Drop any row that contains NaN => to work with RF
    # NOTE: mostly when ['height', 'weight', 'body_surface_area'] is missing
    # TODO: impute them instead of dropping
    model_input = model_input.dropna()
    model_output = model_output.loc[model_input.index]

    # Generate prediction probabilities
    model_output["ed_pred_prob"] = pred_fn(model_input, model.model)

    # Generate binary predictions based on these pre-defined thresholds
    for _, row in thresholds.iterrows():
        assert row["labels"] == "ED_visit"
        alarm_rate = row["alarm_rate"]
        pred_thresh = row["prediction_threshold"]
        model_output[f"ed_pred_alarm_{alarm_rate}"] = (model_output["ed_pred_prob"] > pred_thresh).astype(int)

    return {
        "model_input": model_input,
        "model_output": model_output,
    }


def get_symp_model_output(df: pd.DataFrame, thresholds: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    # TODO: support trying out multiple different models per target
    # TODO: create a config file that maps targets with the model names
    #       (i.e. Pain: [LGBM_pain.pkl, Mistral_pain.pkl, etc])

    # load the model from disk
    filename = "LGBM_symp.pkl"
    with open(f"{model_dir}/{filename}", "rb") as file:
        model = pickle.load(file)

    # TODO: Ask Noke to provide just the models (no need for rest of the stuff in here)
    # reformat the model object
    symps = ["Pain", "Tired", "Nausea", "Depress", "Anxious", "Drowsy", "Appetite", "WellBeing", "SOB"]
    lgbm_models = {symp: model[("lgbm", f"Label_{symp}_3pt_change")]["best_model"] for symp in symps}
    calib_models = {symp: model[("lgbm", f"Label_{symp}_3pt_change")]["best_ir_model"] for symp in symps}

    # Separate patient id and visit date information
    result = df[["mrn", "treatment_date"]].copy()

    for symp in symps:
        model = lgbm_models[symp]
        calibrator = calib_models[symp]

        # Reorder model features according to the order used in training
        x = df[model.feature_name_]

        # Generate predictions and combine with patient info
        pred_prob = model.predict_proba(x)[:, 1]  # probability of the positive class
        calib_pred_prob = calibrator.transform(pred_prob)
        result[f"{symp}_pred_prob"] = calib_pred_prob

        # Generate binary predictions based on pre-defined thresholds
        thresh = thresholds[f"Label_{symp}_3pt_change"]
        result[f"{symp}_pred"] = (result[f"{symp}_pred_prob"] > thresh).astype(int)

    return result
