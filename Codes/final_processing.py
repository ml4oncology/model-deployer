"""
Processing Script 
Using the current directory

"""

# import numpy as np
import pickle
# import pandas as pd
# import glob
# import os
# from datetime import date

from data_prep.build_features import build_features
from data_prep.combine_features import combine_features
from data_prep.engineer import get_change_since_prev_session, get_missingness_features
from data_prep.prep import encode_regimens, encode_intent, fill_missing_data, PrepData #Imputer 
from data_prep.prepData_Symps import prep_symp_data


def final_process(data_root_dir, info_data_dir, train_param_dir, code_dir, model_dir, proj_name, model_name, dataPull_day):
    
    
    # Build Features
    dart, canc_reg, opis, lab, er_visit = build_features(data_root_dir, info_data_dir, proj_name, dataPull_day)
    
    # Combine Features
    df = combine_features(lab, opis, canc_reg, dart, er_visit, code_dir)
    
    #Get changes between treatment sessions
    df =  get_change_since_prev_session(df)
    
    #Get missingness features
    df = get_missingness_features(df)
    
    #Encode Regimens and Intent
    df = encode_regimens(df, info_data_dir)
    df = encode_intent(df)
    
    
    # Drop columns not currently used by model
    drop_cols = ['basophil', 'basophil_change', 'bicarbonate_change', #'bicarbonate',
                 'basophil_is_missing', #'bicarbonate_is_missing', 
                 'basophil_change_is_missing', 'bicarbonate_change_is_missing',
                 'date_of_birth', 'first_treatment_date']
                 # 'treatment_date','mrn']
    df = df.drop(columns=drop_cols)
    
    
    # Remove / reorganize features for symptoms' models
    if model_name == 'symp':
        df = prep_symp_data(df)
    
    
    #Transofrm Data: Impute, Normalize and Clip
    clip_threshold = pickle.load(open(train_param_dir+'/'+'clip_thresh_'+model_name+'.pkl', 'rb'))
    normalize_scaler = pickle.load(open(train_param_dir+'/'+'scaler_'+model_name+'.pkl', 'rb'))
    data_imputer = pickle.load(open(train_param_dir+'/'+'imputer_'+model_name+'.pkl', 'rb'))
    
    # extraxt all the columns with order in which they were used; then reorder the pandas dataframe
    # also extract all thresholds from training data for data transformation
    clip_threshold_cols = list(clip_threshold.columns)
    normalize_scaler_cols = list(normalize_scaler.feature_names_in_)
    data_imputer_mean_cols = list(data_imputer['mean'].feature_names_in_)
    data_imputer_mfreq_cols = list(data_imputer['most_frequent'].feature_names_in_)
    
    ImpTrData = PrepData()
    ImpTrData.scaler = normalize_scaler
    ImpTrData.clip_thresh = clip_threshold
    ImpTrData.imp.imputer = data_imputer
    
    ImpTrData.clip_cols = clip_threshold_cols
    ImpTrData.norm_cols = normalize_scaler_cols    
    ImpTrData.imp.impute_cols = {
        'mean': data_imputer_mean_cols, 
        'most_frequent': data_imputer_mfreq_cols,
        'median': []
    }
    
    df = ImpTrData.transform_data(df) #clip=False
    
    
    # Fill remaining nan's
    df = fill_missing_data(df)
    
    
    return df
    

