"""
Final processing script 
"""
import pandas as pd
from datetime import timedelta 

from ml_common.engineer import get_change_since_prev_session
from ml_common.prep import fill_missing_data_heuristically

from data_prep.build import build_features
from data_prep.combine import combine_features
from data_prep.prep import encode_regimens, encode_intent, prep_symp_data
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

    # Recreate any missing columns
    missing_cols = [col for col in model.model_features if col not in df.columns]
    df[missing_cols] = 0

    # Remove columns not used in training (keep the mrn and dates though)
    cols = df.columns
    cols = cols[cols.str.contains('mrn|date') | cols.isin(model.model_features)].tolist()
    df = df[cols]
    
    # Transform Data: Impute, Normalize, and Clip
    df.loc[0, df.columns[df.isna().all()]] = 0
    df = model.prep.transform_data(df, one_hot_encode=False)
    
    # Fill remaining nan's
    df = fill_missing_data_heuristically(df)
    
    if anchor == 'treatment':
        # keep treatments scheduled for the next day
        mask = df['treatment_date'] == pd.to_datetime(data_pull_day) + timedelta(days=1) 
        df = df[mask]
    
    return df