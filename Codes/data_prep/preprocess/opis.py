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
    if anchor == 'clinic':
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
    df['tx_sched_date'] = pd.to_datetime(df['tx_sched_date']).dt.date
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])

    # clean intent feature
    df['intent'] = df['intent'].replace('U', np.nan)

    df = clean_regimens(df, EPR_regimens, EPR_to_EPIC_regimen_map)
    return df


def process_treatment_data(df, data_pull_day: str, anchor: str) -> pd.DataFrame:
    trt_date = 'trt_date_utc' if data_pull_day is None or anchor == 'clinic' else 'tx_sched_date'

    # order by mrn, treatment date, and regimen
    df = df.sort_values(by=['mrn', trt_date, 'regimen'])

    # forward and backward fill height, weight and body_surface_area
    for col in ['height', 'weight', 'body_surface_area']: 
        df[col] = df.groupby('mrn')[col].ffill().bfill()

    # forward fill treatment dates (showing as 'nan') that are scheduled but not completed
    df['trt_date_utc'] = df.groupby('mrn')['trt_date_utc'].ffill()
    
    # Keep only treatments scheduled the following day (i.e. one day after data pull)
    df['treatment_date'] = df[trt_date]
    if anchor == 'treatment':
        df = filter_chemo_treatments(df, data_pull_day)
    elif anchor == 'clinic':
        df = filter_clinic_treatments(df, data_pull_day)

    df['treatment_date'] = df[trt_date]
    if anchor == 'treatment':
        df = merge_same_day_treatments(df)
    
    # forward and backward fill first treatment date
    df['first_treatment_date'] = df.groupby('mrn')['first_treatment_date'].ffill().bfill()
    
    # remove one-off duplicate rows (all values are same except for one, most likely due to human error)
    for col in ['first_treatment_date', 'cycle_number']: 
        cols = df.columns.drop(col)
        mask = ~df.duplicated(subset=cols, keep='first')
        df = df[mask]

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


def filter_chemo_treatments(df, data_pull_day: str):
    if data_pull_day is None:
        # keep rows with 'Completed' day_status
        mask = df['day_status'] == 'Completed'
        df = df[mask]
    else:     
        # keep treatments scheduled for the next day
        mask = df['tx_sched_date'] == pd.to_datetime(data_pull_day).date() + timedelta(days=1) 
        df = df[mask]

        # filter out patients in which no treatment is scheduled for the next day
        # mask = df['tx_sched_date'] == pd.to_datetime(data_pull_day).date() + timedelta(days=1) 
        # keep_mrns = df.loc[mask, 'mrn'].unique()
        # df = df[df['mrn'].isin(keep_mrns)]
    
    return df


def filter_clinic_treatments(df, data_pull_day: str):
    # filter out rows where next scheduled treatment session does not occur within 5 days of clinic visit
    clinic_date = pd.to_datetime(data_pull_day).date()
    mask = df["tx_sched_date"].between(clinic_date, clinic_date + timedelta(days=5))
    df = df[mask]

    # filter out patients in which no treatment is scheduled within 5 days of clinic date
    # clinic_date = pd.to_datetime(data_pull_day).date()
    # mask = df["tx_sched_date"].between(clinic_date, clinic_date + timedelta(days=5))
    # keep_mrns = df.loc[mask, 'mrn'].unique()
    # df = df[df['mrn'].isin(keep_mrns)]
    
    return df
