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
from loader import Config, Model


def final_process(
    config: Config, 
    model: Model,
    data_dir: str, 
    proj_name: str, 
    model_name: str, 
    data_pull_day: str, 
    anchor: str
):
    # Build Features
    feats = build_features(config, data_dir, proj_name, data_pull_day, anchor)
    
    # Combine Features
    df = combine_features(model.prep_cfg, feats, data_pull_day, anchor)
    
    #Get changes between treatment sessions
    df = get_change_since_prev_session(df)
    
    # Get missingness features
    # NOTE: we filter out unused features later on in inference.py
    # so easier to just calculate missingness for all columns
    df[df.columns + "_is_missing"] = df.isnull()
    
    # Encode Regimens and Intent
    df = encode_regimens(df, config.gi_regimens)
    df = encode_intent(df)
    
    # Remove / reorganize features for symptoms' models
    if model_name == 'symp':
        df = prep_symp_data(df)
        
    # Transofrm Data: Impute, Normalize and Clip
    clip_threshold = model.prep.clip_thresh
    normalize_scaler = model.prep.scaler
    data_imputer = model.prep.imp.imputer
    
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