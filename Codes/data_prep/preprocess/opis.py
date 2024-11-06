"""
Module to preprocess OPIS (systemic therapy treatment data) - CHEMO
"""


import numpy as np
import pandas as pd
from datetime import timedelta 
from data_prep.constants import DROP_CLINIC_COLUMNS

def get_treatment_data(
    chemo_data_file,
    included_regimens: pd.DataFrame,
    A2R_EPIC_GI_regimen_map,
    data_pull_day = None, 
    anchored = ''    
) -> pd.DataFrame:
    
    """
    Args:
    included_regimens: selected EPR regimens during model training
    A2R_EPIC_GI_regimen_map: map between the old EPR regimens and the new EPIC regimens
    
    anchored = '': treatment anchored files
    anchored = 'weekly_': clinic anchored files
    """

    # A2R_EPIC_GI_regimen_map = A2R_EPIC_GI_regimen_map[{'PROTOCOL_DISPLAY_NAME','Mapped_Name_All'}]
    A2R_EPIC_GI_regimen_map = A2R_EPIC_GI_regimen_map[['PROTOCOL_DISPLAY_NAME','Mapped_Name_All']]
    A2R_EPIC_GI_regimen_map = A2R_EPIC_GI_regimen_map.set_index('PROTOCOL_DISPLAY_NAME')['Mapped_Name_All'].to_dict()
    
    df = pd.read_csv(chemo_data_file)
    if anchored == 'weekly_':
        df = df.drop(columns=DROP_CLINIC_COLUMNS)
    df = process_treatment_data(df, data_pull_day, anchored)
    df = filter_treatment_data(df, included_regimens, A2R_EPIC_GI_regimen_map, data_pull_day)
    return df
    

def process_treatment_data(df, data_pull_day, anchored):
    
    trt_date = 'trt_date_utc' if data_pull_day is None or anchored=='weekly_' else 'tx_sched_date'
    
    # clean column names
    df.columns = df.columns.str.lower()
    
    # order by id, scheduled date and regimen
    df = df.sort_values(by=['research_id', trt_date, 'regimen']) #'tx_sched_date'
    
    # forward fill height, weight and body_surface_area
    for col in ['height', 'weight', 'body_surface_area']: df[col] = df.groupby('research_id')[col].ffill().bfill()
    
    # Keep only treatments scheduled the following day (i.e. one day after data pull)
    if anchored == '':
        df = filter_chemo_Trt(df, data_pull_day)
    else:
        df = filter_clinic_treatments(df, data_pull_day)
    
    col_map = {
        'research_id': 'mrn', 
        trt_date: 'treatment_date',  #'tx_sched_date'; 'trt_date_utc': 'treatment_date',
        'first_trt_date_utc': 'first_treatment_date',
        'dose_ord_or_min_dose_ord': 'dose_ordered',
        'dose_given': 'given_dose'
    }
    df = df.rename(columns=col_map) 

    df['first_treatment_date'] = df['first_treatment_date'].apply(str) # due to error in one instance
    if anchored == '':
        df = merge_same_day_treatments(df) #, dosage
    
    # forward and backward fill first treatment date
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])
    df['first_treatment_date'] = df.groupby('mrn')['first_treatment_date'].ffill().bfill()

    return df


def filter_treatment_data(df, regimens: pd.DataFrame, A2R_EPIC_GI_regimen_map, data_pull_day) -> pd.DataFrame: #drugs: pd.DataFrame, 
    
    # clean column names
    regimens.columns = regimens.columns.str.lower()
    
    # clean intent feature
    df['intent'] = df['intent'].replace('U', np.nan)
    
    # df = filter_chemo_Trt(df, dataPull_day)
    df = filter_regimens(df, regimens, A2R_EPIC_GI_regimen_map)

    # remove one-off duplicate rows (all values are same except for one, most likely due to human error)
    for col in ['first_treatment_date', 'cycle_number']: 
        cols = df.columns.drop(col)
        mask = ~df.duplicated(subset=cols, keep='first')
        # get_excluded_numbers(df, mask, context=f' that are duplicate rows except for {col}')
        df = df[mask]
    
    return df


def filter_regimens(df, regimens: pd.DataFrame, A2R_EPIC_GI_regimen_map) -> pd.DataFrame:
    # filter out rows with missing regimen info
    mask = df['regimen'].notnull()
    df = df[mask].copy()
    
    # Map Regimen from A2R to EPIC
    df['regimen_EPIC'] = df['regimen']
    df['regimen'] = df['regimen'].replace(A2R_EPIC_GI_regimen_map) # map mrn to patientid

    # rename some of the regimens
    regimen_map = dict(regimens.query('rename.notnull()')[['regimen', 'rename']].to_numpy())
    df['regimen'] = df['regimen'].replace(regimen_map)
    return df


def filter_chemo_Trt(df, data_pull_day):
    
    if data_pull_day is None:
        # keep rows with 'Completed' day_status
        mask = df['day_status'] == 'Completed'
        df = df[mask]
        
        # # keep rows with 'Completed' cycle_status
        # mask = df['cycle_status'] == 'Completed'
        # df = df[mask]
        
    else:     
        # Keep treatments scheduled for the next day
        df['tx_sched_date'] = pd.to_datetime(df['tx_sched_date']).dt.date
        
        #Treatment scheduled one day after data pull
        following_treatment_date = pd.to_datetime(data_pull_day).date() + timedelta(days=1) 
        mask = df['tx_sched_date']==following_treatment_date
        df = df[mask]
    
    return df


def filter_clinic_treatments(df, data_pull_day):
    
    df = df.set_index(['research_id','trt_date_utc']).sort_index()
    df = df.reset_index()
    
    # Forward fill treatment dates (showing as 'nan') that are scheduled but not completed
    df['trt_date_utc'] = df.groupby('research_id')['trt_date_utc'].ffill()
    
    # mask = df['day_status'] == 'Planned'
    # df = df[mask]
    
    df['tx_sched_date'] = pd.to_datetime(df['tx_sched_date']).dt.date
    
    # filter out rows where next scheduled treatment session does not occur within 5 days of clinic visit
    clinic_date = pd.to_datetime(data_pull_day).date()
    fiveday_treatment_date = clinic_date + timedelta(days=5) 
    mask = df["tx_sched_date"].between(
        clinic_date, fiveday_treatment_date
    )
    df = df[mask]
    
    return df

###############################################################################
# Mergers
###############################################################################
def merge_same_day_treatments(df): # dosage: pd.DataFrame
    """
    Collapse multiples rows with the same treatment day into one

    Essential for aggregating the different drugs administered on the same day
    """
    
    format_regimens = lambda regs: ' && '.join(sorted(set(regs)))
    df = (
        df
        .groupby(['mrn', 'treatment_date'])
        .agg({
            # handle conflicting data by 
            # 1. join them togehter
            'regimen': format_regimens,
            # 2. take the mean 
            'height': 'mean',
            'weight': 'mean',
            'body_surface_area': 'mean',
            # 3. output True if any are True
            
            # if two treatments (the old regimen and new regimen) overlap on same day, use data associated with the 
            # most recent regimen 
            # NOTE: examples found thru df.groupby(['mrn', 'treatment_date'])['first_treatment_date'].nunique() > 1
            'cycle_number': 'min',
            'first_treatment_date': 'max',
            
            # TODO: come up with robust way to handle the following conflicts
            'intent': 'first'
        })
    )
    df = df.reset_index()
    return df
