"""
Module to preprocess OPIS (systemic therapy treatment data) - CHEMO
"""
from typing import Optional

import numpy as np
import pandas as pd
from datetime import timedelta 

from data_prep.constants import DROP_CLINIC_COLUMNS
from make_clinical_dataset.preprocess.opis import merge_same_day_treatments

def get_treatment_data(
    chemo_data_file,
    EPR_regimens: pd.DataFrame,
    EPR_to_EPIC_regimen_map: dict,
    data_pull_day: Optional[str] = None, 
    anchor: str = 'treatment'    
) -> pd.DataFrame:
    """
    Args:
        EPR_regimens: selected EPR regimens during model training
        EPR_EPIC_regimen_map: map between the old EPR regimens and the new EPIC regimens
    """
    df = pd.read_csv(chemo_data_file)
    if anchor == 'clinic' and data_pull_day is not None:
        df = df.drop(columns=DROP_CLINIC_COLUMNS)
    df = clean_treatment_data(df, EPR_regimens, EPR_to_EPIC_regimen_map, data_pull_day)
    df = process_treatment_data(df, data_pull_day, anchor)
    return df


def clean_treatment_data(
    df,
    EPR_regimens: pd.DataFrame, 
    EPR_to_EPIC_regimen_map: dict, 
    data_pull_day: str, 
):
    # clean up columns
    df.columns = df.columns.str.lower()
    col_map = {
        'research_id': 'mrn', 
        'first_trt_date_utc': 'first_treatment_date',
        'dose_ord_or_min_dose_ord': 'dose_ordered',
        'dose_given': 'given_dose'
    }
    df = df.rename(columns=col_map) 
    df['tx_sched_date'] = pd.to_datetime(df['tx_sched_date'])
    df['trt_date_utc'] = pd.to_datetime(df['trt_date_utc'])
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])

    # clean intent feature
    df['intent'] = df['intent'].replace('U', np.nan)

    df = clean_regimens(df, EPR_regimens, EPR_to_EPIC_regimen_map)
    return df


def process_treatment_data(df, data_pull_day: str, anchor: str) -> pd.DataFrame:

    # order by mrn, treatment date, scheduled treatment date and regimen
    df = df.sort_values(by=['mrn', 'tx_sched_date', 'trt_date_utc', 'regimen'])

    # forward and backward fill height, weight and body_surface_area
    for col in ['height', 'weight', 'body_surface_area']: 
        df[col] = df.groupby('mrn')[col].ffill().bfill()
    
    # Get scheduled treatments within 5 days of clinic visit
    if anchor == 'clinic' and data_pull_day is not None:
        scheduled_treatment = get_scheduled_treatments(df, data_pull_day)
    
    # Filter treatment sessions before data pull date
    df = filter_chemo_treatments(df, data_pull_day, anchor)

    df['treatment_date'] = df['trt_date_utc']
    df = merge_same_day_treatments(df)
    
    # forward and backward fill first treatment date
    df['first_treatment_date'] = df.groupby('mrn')['first_treatment_date'].ffill().bfill()
    
    # remove one-off duplicate rows (all values are same except for one, most likely due to human error)
    for col in ['first_treatment_date', 'cycle_number']: 
        cols = df.columns.drop(col)
        mask = ~df.duplicated(subset=cols, keep='first')
        df = df[mask]
     
    # Merge scheduled teratment dates to clinic visit data; for sanity check and reporting    
    if anchor == 'clinic' and data_pull_day is not None:
        df = pd.merge(df, scheduled_treatment, how='left', on='mrn')

    return df


def clean_regimens(df, EPR_regimens: pd.DataFrame, EPR_to_EPIC_regimen_map: dict) -> pd.DataFrame:
    # filter out rows with missing regimen info
    mask = df['regimen'].notnull()
    df = df[mask].copy()
    
    # map regimens from EPR to EPIC
    df['regimen_EPIC'] = df['regimen']
    df['regimen'] = df['regimen'].replace(EPR_to_EPIC_regimen_map)

    # rename some of the regimens
    regimen_map = dict(EPR_regimens.query('rename.notnull()')[['regimen', 'rename']].to_numpy())
    df['regimen'] = df['regimen'].replace(regimen_map)
    return df


def filter_chemo_treatments(df, data_pull_day: str, anchor: str):
    
    if data_pull_day is None:
        # keep rows with 'Completed' day_status
        mask = df['day_status'] == 'Completed'
        df = df[mask]
    else:
        pull_date = pd.to_datetime(data_pull_day)
        
        if anchor == 'treatment':
            mask = (df["trt_date_utc"] < pull_date) | (df['tx_sched_date'] == pull_date + timedelta(days=1))
        elif anchor == 'clinic':
            mask = (df["trt_date_utc"] < pull_date)
            
        df = df[mask]
        
        # For anchor=='treatment', replace NaN in treatment date with treatment scheduled date
        df.loc[df['trt_date_utc'].isnull(),'trt_date_utc'] = df['tx_sched_date']
    
    return df


def get_scheduled_treatments(df, data_pull_day: str):
    
    pull_date = pd.to_datetime(data_pull_day)
    
    # filter out rows where next scheduled treatment session does not occur within 5 days of pull date (clinic visit)
    filtered_df = df[df["tx_sched_date"].between(pull_date, pull_date + timedelta(days=5))]
    filtered_df = filtered_df[['mrn','tx_sched_date']]
    
    unique_dates_df = filtered_df.sort_values('tx_sched_date').drop_duplicates(subset='mrn', keep='first')
    
    return unique_dates_df
