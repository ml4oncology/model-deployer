"""
Script to load the models and generate predictions
"""
import pickle

import numpy as np

from model_eval.calc_shap import calc_plot_mean_shap_values


def predict(models, data):
    # average across the folds
    return np.mean([m.predict_proba(data)[:, 1] for m in models], axis=0)

def get_ED_visit_model_output(model, df, thresholds, fig_dir, anchor):
    if anchor == "clinic":
        meta_cols = ['mrn', 'tx_sched_date','clinic_date']
    elif anchor == "treatment":
        meta_cols = ['mrn', 'treatment_date']
        
    # Separate patient id and visit date information
    result = df[meta_cols].copy()
    
    # Reorder model features according to the order used in training
    X = df[model.model_features]
    
    # Drop any row that contains NaN => to work with RF 
    X = X.dropna() # drop rows with nan values
    result = result[result.index.isin(X.index)]
    
    # Generate predictions and combine with the result
    result['ed_pred_prob'] = predict(model.model, X) # probability of the positive class
    
    # Generate binary predictions based on these pre-defined thresholds
    ct=1
    for label, thresh in thresholds.items():
        assert label == 'ED_visit'
        result[f'ed_pred_{10*ct}'] = (result['ed_pred_prob'] > thresh).astype(int)
        ct+=1
    
    if anchor == "clinic":
        ############### SHAP ##################
        # shap_values = calc_plot_mean_shap_values(X, model, result, fig_dir)
        pass
        
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