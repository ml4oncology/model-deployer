"""
Script to combine the features into one unified dataset
"""
from itertools import product

import pandas as pd
import numpy as np
import yaml

from data_prep.src.combine import (
    add_engineered_features,
    combine_demographic_to_main_data, 
    combine_event_to_main_data,
    combine_feat_to_main_data, 
    combine_treatment_to_main_data
)

def combine_features(lab, trt, dmg, sym, erv, code_dir):
    
    align_on = 'treatment-dates' 
    main_date_col = 'treatment_date'
     
    with open(code_dir+'/config.yaml') as file:
        cfg = yaml.safe_load(file)

    if align_on == 'treatment-dates':
        df = trt
    elif align_on == 'weekly-mondays':
        #TODO: make mrn and start and end date selection robust (i.e. make them into command line arguments)
        mrns = trt['mrn'].unique()
        dates = pd.date_range(start='2018-01-01', end='2018-12-31', freq='W-MON')
        df = pd.DataFrame(product(mrns, dates), columns=['mrn', main_date_col])
    elif align_on.endswith('.parquet.gzip') or align_on.endswith('.parquet'):
        df = pd.read_parquet(align_on)
    elif align_on.endswith('.csv'):
        df = pd.read_csv(align_on)
    else:
        raise ValueError(f'Sorry, aligning features on {align_on} is not supported yet')

    if align_on != 'treatment-dates':
        # logger.info('Combining treatment features...')
        df = combine_treatment_to_main_data(df, trt, main_date_col=main_date_col, time_window=(-7,0))
    
    # Convert to date format
    df[main_date_col] = pd.to_datetime(df[main_date_col])
    df[main_date_col] = df[main_date_col].dt.strftime('%Y-%m-%d')
    df[main_date_col] = pd.to_datetime(df[main_date_col])
    
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])
    df['first_treatment_date'] = df['first_treatment_date'].dt.strftime('%Y-%m-%d')
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])
    
    sym['survey_date'] = pd.to_datetime(sym['survey_date'])
    sym['survey_date'] = sym['survey_date'].dt.strftime('%Y-%m-%d')
    sym['survey_date'] = pd.to_datetime(sym['survey_date'])
    
    # logger.info('Combining demographic features...')
    df = combine_demographic_to_main_data(main=df, demographic=dmg, main_date_col=main_date_col)
    
    # logger.info('Combining symptom features...')
    df = combine_feat_to_main_data(
        main=df, feat=sym, main_date_col=main_date_col, feat_date_col='survey_date', 
        time_window=(-cfg['symp_lookback_window'],0)
    )
    
    # logger.info('Combining lab features...')
    df = combine_feat_to_main_data(
        main=df, feat=lab, main_date_col=main_date_col, feat_date_col='obs_date', 
        time_window=(-cfg['lab_lookback_window'],0)
    )

    # logger.info('Combining ED visit features...')
    df = combine_event_to_main_data(
        main=df, event=erv, main_date_col='treatment_date', event_date_col='event_date', event_name='ED_visit',
        lookback_window=cfg['ed_visit_lookback_window']
    )
    
    # logger.info('Combining engineered features...')
    # df = combine_perc_dose_to_main_data(main=df, included_drugs=included_drugs)
    df = add_engineered_features(df, date_col=main_date_col)
    
    # df.to_parquet(f'{output_dir}/{output_filename}.parquet.gzip', compression='gzip', index=False)
    
    # Add missing feature
    df['hematocrit']=np.nan
    # df['red_cell_distribution_width']=np.nan
    
    #Drop columns
    drop_cols = ['esas_constipation', 'esas_diarrhea', 'esas_sleep', 'activated_partial_thromboplastin_time',
                 'carbohydrate_antigen_19-9', 'prothrombin_time_international_normalized_ratio']
    df = df.drop(columns=drop_cols)
    
    return df
    