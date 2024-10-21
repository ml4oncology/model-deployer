"""
Module to combine features
"""
from functools import partial
from itertools import product

from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Tuple

import yaml

from make_clinical_dataset.feat_eng import (
    get_days_since_last_event, 
    get_line_of_therapy, 
    get_visit_month_feature,
    get_years_diff, 
)
from ml_common.anchor import combine_feat_to_main_data


def combine_demographic_to_main_data(
    main: pd.DataFrame, 
    demographic: pd.DataFrame, 
    main_date_col: str
) -> pd.DataFrame:
    """
    Args:
        main_date_col: The column name of the main asessment date
    """
    df = pd.merge(main, demographic, on='mrn', how='left')

    # exclude patients with missing birth date
    mask = df['date_of_birth'].notnull()
    # get_excluded_numbers(df, mask, context=' with missing birth date')
    df = df[mask]
    
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])

    # exclude patients under 18 years of age
    df['age'] = get_years_diff(df, main_date_col, 'date_of_birth')
    mask = df['age'] >= 18
    # get_excluded_numbers(df, mask, context=' under 18 years of age')
    df = df[mask]

    return df


def combine_treatment_to_main_data(
    main: pd.DataFrame, 
    treatment: pd.DataFrame, 
    main_date_col: str, 
    **kwargs
) -> pd.DataFrame:
    """Combine treatment information to the main data
    For drug dosage features, add up the treatment drug dosages of the past x days for each drug
    For other treatment features, forward fill the features available in the past x days 

    Args:
        main_date_col: The column name of the main asessment date
    """
    cols = treatment.columns
    drug_cols = cols[cols.str.startswith('drug_')].tolist()
    # treatment_drugs = treatment[drug_cols + ['mrn', 'treatment_date']] # treatment drug dosage features
    treatment_feats = treatment.drop(columns=drug_cols) # other treatment features
    treatment_feats['trt_date'] = treatment_feats['treatment_date'] # include treatment date as a feature
    treatment_feats['treatment_date'] = pd.to_datetime(treatment_feats['treatment_date']).dt.date
    df = combine_feat_to_main_data(main, treatment_feats, main_date_col, 'treatment_date', parallelize=False, **kwargs)
    # df = combine_feat_to_main_data(df, treatment_drugs, main_date_col, 'treatment_date', keep='sum', parallelize=False, **kwargs)
    df = df.rename(columns={'trt_date': 'treatment_date'})
    return df


def combine_event_to_main_data(
    main: pd.DataFrame, 
    event: pd.DataFrame, 
    main_date_col: str, 
    event_date_col: str, 
    event_name: str,
    lookback_window: int = 5,
    **kwargs
) -> pd.DataFrame:
    """Combine features extracted from event data (emergency department visits, hospitalization, etc) to the main 
    dataset

    Args:
        main_date_col: The column name of the main visit date
        event_date_col: The column name of the event date
        lookback_window: The lookback window in terms of number of years from treatment date to extract event features
    """
    result = event_feature_extractor(main, event, main_date_col, event_date_col, lookback_window)
    cols = ['index', f'num_prior_{event_name}s_within_{lookback_window}_years', f'days_since_prev_{event_name}']
    result = pd.DataFrame(result, columns=cols).set_index('index')
    df = main.join(result)
    return df


def event_feature_extractor(
    main_df, 
    event_df, 
    main_date_col: str,
    event_date_col: str,
    lookback_window: int = 5,
) -> list:
    """Extract features from the event data, namely
    1. Number of days since the most recent event
    2. Number of prior events in the past X years

    Args:
        main_date_col: The column name of the main visit date
        event_date_col: The column name of the event date
        lookback_window: The lookback window in terms of number of years from treatment date to extract event features
    """

    result = []
    for mrn, main_group in tqdm(main_df.groupby('mrn')):
        event_group = event_df.query('mrn == @mrn')
        event_dates = event_group[event_date_col]
        
        for idx, date in main_group[main_date_col].items():
            # get feature
            # 1. number of days since closest event prior to main visit date
            # 2. number of events within the lookback window 
            earliest_date = date - pd.Timedelta(days=lookback_window*365)
            mask = event_dates.between(earliest_date, date, inclusive='left')
            if mask.any():
                N_prior_events = mask.sum()
                N_days = (date - event_dates[mask].iloc[-1]).days
                result.append([idx, N_prior_events, N_days])
    return result


def add_engineered_features(df, date_col: str = 'treatment_date') -> pd.DataFrame:
    df = get_visit_month_feature(df, col=date_col)
    df['line_of_therapy'] = df.groupby('mrn', group_keys=False).apply(get_line_of_therapy)
    df['days_since_starting_treatment'] = (df[date_col] - df['first_treatment_date']).dt.days
    get_days_since_last_treatment = partial(
        get_days_since_last_event, main_date_col=date_col, event_date_col='treatment_date'
    )
    df['days_since_last_treatment'] = df.groupby('mrn', group_keys=False).apply(get_days_since_last_treatment)
    return df


"""
Combine the features into one unified dataset
"""

def combine_features(lab, trt, dmg, sym, erv, code_dir, data_pull_date, anchored):
     
    with open(code_dir+'/data_prep/config.yaml') as file:
        cfg = yaml.safe_load(file)
        
    align_on = cfg['align_on'] 
    main_date_col = cfg['main_date_col'] 
    
    if anchored == '':
       align_on = align_on[0]
       main_date_col = main_date_col[0]
    else:
       align_on = align_on[1]
       main_date_col = main_date_col[1]

    if align_on == 'treatment-dates':
        df = trt
    elif align_on == 'clinic_date':
        df = pd.DataFrame({'mrn': trt['mrn'].unique(), 'clinic_date': data_pull_date})
        df['clinic_date'] = pd.to_datetime(df['clinic_date']).dt.date
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
        df = combine_treatment_to_main_data(main=df, treatment=trt, main_date_col=main_date_col, time_window=[-28, -1])
            
    # Convert to date format
    df[main_date_col] = pd.to_datetime(df[main_date_col])
    df[main_date_col] = df[main_date_col].dt.strftime('%Y-%m-%d')
    df[main_date_col] = pd.to_datetime(df[main_date_col])
    
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])
    df['first_treatment_date'] = df['first_treatment_date'].dt.strftime('%Y-%m-%d')
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])
    
    if anchored != '':
        df['treatment_date'] = pd.to_datetime(df['treatment_date'])
    
    sym['survey_date'] = pd.to_datetime(sym['survey_date'])
    sym['survey_date'] = sym['survey_date'].dt.strftime('%Y-%m-%d')
    sym['survey_date'] = pd.to_datetime(sym['survey_date'])
    
    df = combine_demographic_to_main_data(main=df, demographic=dmg, main_date_col=main_date_col)
    
    df = combine_feat_to_main_data(
        main=df, feat=sym, main_date_col=main_date_col, feat_date_col='survey_date', 
        parallelize=False, time_window=[-30, -1]
    )
    
    df = combine_feat_to_main_data(
        main=df, feat=lab, main_date_col=main_date_col, feat_date_col='obs_date', 
        parallelize=False, time_window=[-7, -1]
    )

    df = combine_event_to_main_data(
        main=df, event=erv, main_date_col=main_date_col, event_date_col='event_date', event_name='ED_visit',
        lookback_window=cfg['ed_visit_lookback_window']
    )
    
    df = add_engineered_features(df, date_col=main_date_col)
    
    # Add missing feature
    df['hematocrit']=np.nan
    # df['red_cell_distribution_width']=np.nan
    
    #Drop columns
    drop_cols = [
        'esas_constipation', 
        'esas_diarrhea', 
        'esas_sleep', 
        'activated_partial_thromboplastin_time',
        'carbohydrate_antigen_19-9', 
        'prothrombin_time_international_normalized_ratio'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    return df
    