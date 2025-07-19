"""
Script to load the models and generate predictions
"""

import pickle

import numpy as np
from deployer.model_eval.calc_shap import calc_plot_mean_shap_values


def predict(models, data):
    # average across the folds
    return np.mean([m.predict_proba(data)[:, 1] for m in models], axis=0)


def get_model_output(model, df, thresholds, fig_dir):
    if model.anchor == "clinic":
        meta_cols = ["mrn", "tx_sched_date", "clinic_date"]
    elif model.anchor == "treatment":
        meta_cols = ["mrn", "treatment_date"]

    # Separate patient id and visit date information
    result = df[meta_cols].copy()

    # Reorder model features according to the order used in training
    x = df[model.model_features]

    # Drop any row that contains NaN => to work with RF
    # NOTE: mostly when ['height', 'weight', 'body_surface_area'] is missing
    # TODO: impute them instead of dropping
    x = x.dropna()
    result = result[result.index.isin(x.index)]

    # Generate predictions and combine with the result
    result["ed_pred_prob"] = predict(model.model, x)  # probability of the positive class

    # Generate binary predictions based on these pre-defined thresholds
    for _, row in thresholds.iterrows():
        assert row["labels"] == "ED_visit"
        alarm_rate = row["alarm_rate"]
        result[f"ed_pred_{alarm_rate}"] = (result["ed_pred_prob"] > row["prediction_threshold"]).astype(int)

    # Plot SHAP values
    # shap_values = calc_plot_mean_shap_values(X, model, result, fig_dir)

    return result


def get_symp_model_output(df, thresholds, model_dir):
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
