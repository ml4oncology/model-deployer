"""
Script to load the models and generate predictions
"""
import pickle

import pandas as pd


def get_ED_visit_model_output(df, thresholds, model_dir):
    # TODO: support trying out multiple different models per target
    # TODO: create a config file that maps targets with the model names 
    #       (i.e. ED_visit: [XGB_ED_visit.pkl, Mistral_ED_visit.pkl, etc])

    # Load the model from disk
    # NOTE: Ensure XGBoost version 2.0.3 is installed (pip install xgboost==2.0.3 --user)
    filename = 'XGB_ED_visit.pkl'
    with open(f'{model_dir}/{filename}', "rb") as file:
        model = pickle.load(file)
    
    # Separate patient id and visit date information
    result = df[['mrn', 'treatment_date']].copy()

    # Reorder model features according to the order used in training
    X = df[model.get_booster().feature_names]
    
    # Generate predictions and combine with the result
    result['ed_pred_prob'] = model.predict_proba(X)[:, 1] # probability of the positive class

    # Generate binary predictions based on these pre-defined thresholds
    for label, thresh in thresholds.items():
        assert label == 'ED_visit'
        result['ed_pred'] = (result['ed_pred_prob'] > thresh).astype(int)

    return result


def get_symp_model_output(df, thresholds, model_dir):
    # TODO: support trying out multiple different models per target
    # TODO: create a config file that maps targets with the model names 
    #       (i.e. Pain: [LGBM_pain.pkl, Mistral_pain.pkl, etc])

    # load the model from disk
    filename = 'LGBM_symp.pkl'
    with open(f'{model_dir}/{filename}', "rb") as file:
        model = pickle.load(file)

    # TODO: Ask Noke to provide just the models (no need for rest of the stuff in here)
    # reformat the model object
    symps = ['Pain', 'Tired', 'Nausea', 'Depress', 'Anxious', 'Drowsy', 'Appetite', 'WellBeing', 'SOB']
    lgbm_models = {symp: model[('lgbm', f'Label_{symp}_3pt_change')]['best_model'] for symp in symps}
    calib_models = {symp: model[('lgbm', f'Label_{symp}_3pt_change')]['best_ir_model'] for symp in symps}

    # Separate patient id and visit date information
    result = df[['mrn', 'treatment_date']].copy()

    for symp in symps:
        model = lgbm_models[symp]
        calibrator = calib_models[symp]

        # Reorder model features according to the order used in training
        X = df[model.feature_name_]
        
        # Generate predictions and combine with patient info
        pred_prob = model.predict_proba(X)[:, 1] # probability of the positive class
        calib_pred_prob = calibrator.transform(pred_prob)
        result[f'{symp}_pred_prob'] = calib_pred_prob

        # Generate binary predictions based on pre-defined thresholds
        thresh = thresholds[f'Label_{symp}_3pt_change']
        result[f'{symp}_pred'] = (result[f'{symp}_pred_prob'] > thresh).astype(int)

    return result