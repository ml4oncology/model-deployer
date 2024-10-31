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

from data_prep.engineer import ( 
    get_days_since_last_event, 
    get_line_of_therapy, 
    get_visit_month_feature,
    get_years_diff, 
)


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
    treatment_drugs = treatment[drug_cols + ['mrn', 'treatment_date']] # treatment drug dosage features
    treatment_feats = treatment.drop(columns=drug_cols) # other treatment features
    treatment_feats['trt_date'] = treatment_feats['treatment_date'] # include treatment date as a feature
    df = combine_feat_to_main_data(main, treatment_feats, main_date_col, 'treatment_date', **kwargs)
    df = combine_feat_to_main_data(df, treatment_drugs, main_date_col, 'treatment_date', keep='sum', **kwargs)
    df = df.rename(columns={'trt_date': 'treatment_date'})
    return df


def combine_feat_to_main_data(
    main: pd.DataFrame, 
    feat: pd.DataFrame, 
    main_date_col: str, 
    feat_date_col: str, 
    time_window
) -> pd.DataFrame:
    """Combine feature(s) to the main dataset

    Both main and feat should have mrn and date columns
    """
    result = feature_extractor(main, feat, main_date_col, feat_date_col, 'last', time_window)
    cols = ['index'] + feat.columns.drop(['mrn', feat_date_col]).tolist()
    result = pd.DataFrame(result, columns=cols).set_index('index')
    df = main.join(result)
    return df


def feature_extractor(
    main_df, 
    feat_df, 
    main_date_col: str,
    feat_date_col: str,
    keep: str = 'last', 
    time_window: Tuple[int, int] = (-5,0)
) -> list:
    """Extract either the sum, first, or last forward filled feature measurements (lab tests, symptom scores, etc) 
    taken within the time window (centered on each main visit date)

    Args:
        main_date_col: The column name of the main visit date
        feat_date_col: The column name of the feature measurement date
        time_window: The start and end of the window in terms of number of days after(+)/before(-) each visit date
        keep: Which measurements taken within the time window to keep, either `sum`, `first`, `last`
    """
    if keep not in ['first', 'last', 'sum']:
        raise ValueError('keep must be either first, last, or sum')
    
    lower_limit, upper_limit = time_window
    keep_cols = feat_df.columns.drop(['mrn', feat_date_col])

    results = []
    for mrn, main_group in tqdm(main_df.groupby('mrn')):
        feat_group = feat_df.query('mrn == @mrn')

        for idx, date in main_group[main_date_col].items():
            earliest_date = date + pd.Timedelta(days=lower_limit)
            latest_date = date + pd.Timedelta(days=upper_limit)

            mask = feat_group[feat_date_col].between(earliest_date, latest_date)
            if not mask.any(): 
                continue

            feats = feat_group.loc[mask, keep_cols]
            if keep == 'sum':
                result = feats.sum()
            elif keep == 'first':
                result = feats.iloc[0]
            elif keep == 'last':
                result = feats.ffill().iloc[-1]

            results.append([idx]+result.tolist())
    
    return results


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

def combine_features(lab, trt, dmg, sym, erv, code_dir, clinic_anchored):
     
    with open(code_dir+'/data_prep/config.yaml') as file:
        cfg = yaml.safe_load(file)
        
    align_on = cfg['align_on'] 
    main_date_col = cfg['main_date_col'] 
    
    if clinic_anchored == '':
       align_on = align_on[0]
       main_date_col = main_date_col[0]
    else:
       align_on = align_on[1]
       main_date_col = main_date_col[1]

    if align_on == 'treatment-dates' or align_on == 'clinic_date':
        df = trt
    # elif align_on == 'clinic_date':
    #     df = trt
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

    if align_on != 'treatment-dates' and align_on != 'clinic_date':
        # logger.info('Combining treatment features...')
        # df = combine_treatment_to_main_data(df, trt, main_date_col=main_date_col, time_window=(-7,0))
        df = combine_treatment_to_main_data(main=df, treatment=trt, main_date_col=main_date_col, time_window=[-28, -1])
    
    # Convert to date format
    df[main_date_col] = pd.to_datetime(df[main_date_col])
    df[main_date_col] = df[main_date_col].dt.strftime('%Y-%m-%d')
    df[main_date_col] = pd.to_datetime(df[main_date_col])
    
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])
    df['first_treatment_date'] = df['first_treatment_date'].dt.strftime('%Y-%m-%d')
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])
    
    if clinic_anchored != '':
        df['treatment_date'] = pd.to_datetime(df['treatment_date'])
    
    sym['survey_date'] = pd.to_datetime(sym['survey_date'])
    sym['survey_date'] = sym['survey_date'].dt.strftime('%Y-%m-%d')
    sym['survey_date'] = pd.to_datetime(sym['survey_date'])
    
    # logger.info('Combining demographic features...')
    df = combine_demographic_to_main_data(main=df, demographic=dmg, main_date_col=main_date_col)
    
    # logger.info('Combining symptom features...')
    # df = combine_feat_to_main_data(
    #     main=df, feat=sym, main_date_col=main_date_col, feat_date_col='survey_date', 
    #     time_window=(-cfg['symp_lookback_window'],0)
    # )
    df = combine_feat_to_main_data(
        main=df, feat=sym, main_date_col=main_date_col, feat_date_col='survey_date', 
        time_window=[-30, -1]
    )
    
    # logger.info('Combining lab features...')
    # df = combine_feat_to_main_data(
    #     main=df, feat=lab, main_date_col=main_date_col, feat_date_col='obs_date', 
    #     time_window=(-cfg['lab_lookback_window'],0)
    # )
    df = combine_feat_to_main_data(
        main=df, feat=lab, main_date_col=main_date_col, feat_date_col='obs_date', 
        time_window=[-7, -1]
    )

    # logger.info('Combining ED visit features...')
    # df = combine_event_to_main_data(
    #     main=df, event=erv, main_date_col='treatment_date', event_date_col='event_date', event_name='ED_visit',
    #     lookback_window=cfg['ed_visit_lookback_window']
    # )
    df = combine_event_to_main_data(
        main=df, event=erv, main_date_col=main_date_col, event_date_col='event_date', event_name='ED_visit',
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
    