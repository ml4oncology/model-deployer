"""
Final processing script 
"""
import pickle

import pandas as pd
from datetime import timedelta 

from ml_common.engineer import get_change_since_prev_session
from ml_common.prep import fill_missing_data_heuristically

from data_prep.build import build_features
from data_prep.combine import combine_features
from data_prep.prep import encode_regimens, encode_intent, PrepData, prep_symp_data


def final_process(data_dir, info_dir, train_param_dir, code_dir, proj_name, model_name, data_pull_day, anchor):
    # Build Features
    dart, canc_reg, opis, lab, er_visit = build_features(data_dir, info_dir, proj_name, data_pull_day, anchor)
    
    # Combine Features
    df = combine_features(lab, opis, canc_reg, dart, er_visit, code_dir, data_pull_day, anchor)
    
    #Get changes between treatment sessions
    df = get_change_since_prev_session(df)
    
    # Get missingness features
    # NOTE: we filter out unused features later on in inference.py
    # so easier to just calculate missingness for all columns
    df[df.columns + "_is_missing"] = df.isnull()
    
    # Encode Regimens and Intent
    regimens = pd.read_excel(f'{info_dir}/GI_regimen_feature_list.xlsx')
    df = encode_regimens(df, regimens)
    df = encode_intent(df)
    
    # Remove / reorganize features for symptoms' models
    if model_name == 'symp':
        df = prep_symp_data(df)
    
    if anchor == 'treatment':
        prep = pickle.load(open(f'{train_param_dir}/prep_{model_name}_trt_anchored.pkl', 'rb'))
    elif anchor == 'clinic':
        prep = pickle.load(open(f'{train_param_dir}/prep_{model_name}_clinic_anchored.pkl', 'rb'))
        
    # Transofrm Data: Impute, Normalize and Clip
    clip_threshold = prep.clip_thresh
    normalize_scaler = prep.scaler
    data_imputer = prep.imp.imputer
    
    # extraxt all the columns with order in which they were used; then reorder the pandas dataframe
    # also extract all thresholds from training data for data transformation
    clip_threshold_cols = list(clip_threshold.columns)
    normalize_scaler_cols = list(normalize_scaler.feature_names_in_)
    data_imputer_mean_cols = list(data_imputer['mean'].feature_names_in_)
    data_imputer_mfreq_cols = list(data_imputer['most_frequent'].feature_names_in_)
    
    transformer = PrepData()
    transformer.scaler = normalize_scaler
    transformer.clip_thresh = clip_threshold
    transformer.imp.imputer = data_imputer
    
    transformer.clip_cols = clip_threshold_cols
    transformer.norm_cols = normalize_scaler_cols    
    transformer.imp.impute_cols = {
        'mean': data_imputer_mean_cols, 
        'most_frequent': data_imputer_mfreq_cols,
        'median': []
    }
    
    df = transformer.transform_data(df)
    
    # Fill remaining nan's
    df = fill_missing_data_heuristically(df)
    
    if anchor == 'treatment':
        # keep treatments scheduled for the next day
        mask = df['treatment_date'] == pd.to_datetime(data_pull_day) + timedelta(days=1) 
        df = df[mask]
    
    return df