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

from make_clinical_dataset.combine import combine_event_to_main_data
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
    df = combine_feat_to_main_data(main, treatment_feats, main_date_col, 'treatment_date', parallelize=False, **kwargs)
    # df = combine_feat_to_main_data(df, treatment_drugs, main_date_col, 'treatment_date', keep='sum', parallelize=False, **kwargs)
    df = df.rename(columns={'trt_date': 'treatment_date'})
    return df


def add_engineered_features(df, date_col: str = 'treatment_date') -> pd.DataFrame:
    df = get_visit_month_feature(df, col=date_col)
    df['line_of_therapy'] = df.groupby('mrn', group_keys=False).apply(get_line_of_therapy)
    df['days_since_starting_treatment'] = (df[date_col] - df['first_treatment_date']).dt.days
    get_days_since_last_treatment = partial(
        get_days_since_last_event, main_date_col=date_col, event_date_col='treatment_date'
    )
    df['days_since_last_treatment'] = df.groupby('mrn', group_keys=False).apply(get_days_since_last_treatment)
    return df


def combine_features(lab, trt, dmg, sym, erv, code_dir, data_pull_date, anchor):
    """Combine the features into one unified dataset
    """
    with open(code_dir+'/data_prep/config.yaml') as file:
        cfg = yaml.safe_load(file)

    if anchor == 'treatment':
        df = trt
        main_date_col = 'treatment_date'
    elif anchor == 'clinic':
        df = pd.DataFrame({'mrn': trt['mrn'].unique(), 'clinic_date': data_pull_date})
        df['clinic_date'] = pd.to_datetime(df['clinic_date'])
        main_date_col = 'clinic_date'
    else:
        raise ValueError(f'Sorry, aligning features on {anchor} is not supported yet')

    if anchor != 'treatment':
        df = combine_treatment_to_main_data(main=df, treatment=trt, main_date_col=main_date_col, time_window=[-28, -1])

    # Convert to date format
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'].dt.strftime('%Y-%m-%d'))
    sym['survey_date'] = pd.to_datetime(sym['survey_date'].dt.strftime('%Y-%m-%d'))
    
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
        parallelize=False, lookback_window=cfg['ed_visit_lookback_window']
    )
    
    df = add_engineered_features(df, date_col=main_date_col)
    
    # Add missing feature
    df['hematocrit'] = np.nan
    # df['red_cell_distribution_width'] =n p.nan
    
    # Drop columns
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
    