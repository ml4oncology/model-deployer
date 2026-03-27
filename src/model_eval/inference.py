"""
Module to generate predictions
"""

import matplotlib  # NEW
matplotlib.use('Agg')  # NEW — non-interactive backend, matches ml4u backend

import pickle
from functools import partial  # NEW
from pathlib import Path  # NEW
from typing import Callable, TypeVar

import matplotlib.pyplot as plt  # NEW
import numpy as np
import pandas as pd
import shap  # NEW
from deployer.loader import Model
from make_clinical_dataset.shared.constants import UNIT_MAP  # NEW
from sklearn.base import BaseEstimator

from deployer.model_eval.util import clean_feature_name  # NEW — copy util.py into this repo

ScikitModel = TypeVar("ScikitModel", bound=BaseEstimator)

ANCHOR_META_COLS = {"clinic": ["mrn", "next_sched_trt_date", "clinic_date"], "treatment": ["mrn", "treatment_date"]}
SHAP_SUBDIR = "shap_waterfall"  # NEW — subdirectory name inside output_dir


def predict(data: pd.DataFrame, models: list[ScikitModel]):
    # average across the folds
    return np.mean([m.predict_proba(data)[:, 1] for m in models], axis=0)


# NEW — helper function, mirrors shap_plot() in ml4u backend main.py
def _save_shap_waterfall(
    model: Model,
    model_input: pd.DataFrame,
    explainer: shap.Explainer,
    shap_dir: Path,
    row_idx: int,
    mrn: int,
    clinic_date: pd.Timestamp,
    max_display: int = 10,
) -> None:
    """
    Compute and save a SHAP waterfall plot for a single model input row.
    Mirrors the logic in ml4u backend shap_plot() + ml4u frontend component.py.
    """
    # Step 1: Extract the single row and compute SHAP values
    input_row = model_input.iloc[[row_idx]].astype(float)
    shap_values = explainer(input_row)

    # Step 2: Unnormalize input features for display (from ml4u backend shap_plot)
    norm_cols = model.prep.norm_cols
    input_row[norm_cols] = model.prep.scaler.inverse_transform(input_row[norm_cols])
    input_data = input_row.iloc[0].tolist()

    # Step 3: Clean feature names + add units (from ml4u backend shap_plot)
    unit_map = {feat: unit for unit, feats in UNIT_MAP.items() for feat in feats}
    feature_names = []
    for feat in shap_values.feature_names:
        res = clean_feature_name(feat)
        if feat in unit_map:
            res = f"{res} ({unit_map[feat]})"
        feature_names.append(res)

    # Step 4: Rebuild shap.Explanation with cleaned names + unnormalized data
    # (mirrors component.py frontend reconstruction)
    explanation = shap.Explanation(
        values=shap_values.values[0],
        base_values=float(shap_values.base_values[0]),
        data=np.array(input_data),
        feature_names=feature_names,
    )

    # Step 5: Plot and save (mirrors component.py shap.plots.waterfall call)
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout(pad=0.5)

    clinic_date_str = pd.Timestamp(clinic_date).strftime('%Y%m%d')
    filename = shap_dir / f"shap_mrn_{mrn}_clinic_{clinic_date_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    plt.close('all')


def get_model_output(
    model: Model,
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    pred_fn: Callable | None = None,
    output_dir: str | Path | None = None,  # NEW
) -> dict[str, pd.DataFrame]:
    """
    TODO: set data_pull_day as the assessment date for treatment date anchor
    """
    if pred_fn is None:
        pred_fn = predict

    model_input = df[model.model_features].copy() # reorder model features according to the order used in training
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

    # NEW — compute and save SHAP waterfall plots if output_dir is provided
    if output_dir is not None:
        shap_dir = Path(output_dir) / SHAP_SUBDIR
        shap_dir.mkdir(parents=True, exist_ok=True)

        pred_fn_for_shap = partial(predict, models=model.model)
        bg_data = model.orig_x.sample(min(10000, len(model.orig_x))).astype(float)
        explainer = shap.Explainer(pred_fn_for_shap, bg_data)

        for row_idx, (idx, row) in enumerate(model_output.iterrows()):
            _save_shap_waterfall(
                model=model,
                model_input=model_input,
                explainer=explainer,
                shap_dir=shap_dir,
                row_idx=row_idx,
                mrn=int(row["mrn"]),
                clinic_date=row["clinic_date"],
            )

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
