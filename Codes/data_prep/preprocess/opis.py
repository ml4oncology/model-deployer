"""
Module to preprocess OPIS (systemic therapy treatment data) - CHEMO
"""


import numpy as np
import pandas as pd
from datetime import timedelta 


def get_treatment_data(
    chemo_data_file,
    included_regimens: pd.DataFrame,
    A2R_EPIC_GI_regimen_map,
    data_pull_day = None    
) -> pd.DataFrame:
    
    """
    Args:
    included_regimens: selected EPR regimens during model training
    A2R_EPIC_GI_regimen_map: map between the old EPR regimens and the new EPIC regimens

    ** Use 'data_pull_day = None' for inference, to get patients with completed treatment sessions
    """

    # A2R_EPIC_GI_regimen_map = A2R_EPIC_GI_regimen_map[{'PROTOCOL_DISPLAY_NAME','Mapped_Name_All'}]
    A2R_EPIC_GI_regimen_map = A2R_EPIC_GI_regimen_map[['PROTOCOL_DISPLAY_NAME','Mapped_Name_All']]
    A2R_EPIC_GI_regimen_map = A2R_EPIC_GI_regimen_map.set_index('PROTOCOL_DISPLAY_NAME')['Mapped_Name_All'].to_dict()
    
    df = pd.read_csv(chemo_data_file)
    df = process_treatment_data(df, data_pull_day)
    df = filter_treatment_data(df, included_regimens, A2R_EPIC_GI_regimen_map, data_pull_day)
    return df
    

def process_treatment_data(df, data_pull_day):
    
    trt_date = 'trt_date_utc' if data_pull_day is None else 'tx_sched_date'
    
    # clean column names
    df.columns = df.columns.str.lower()
    
    # order by id, scheduled date and regimen
    df = df.sort_values(by=['research_id', trt_date, 'regimen']) #'tx_sched_date'
    
    # forward fill height, weight and body_surface_area
    for col in ['height', 'weight', 'body_surface_area']: df[col] = df.groupby('research_id')[col].ffill()
    
    # Keep only treatments scheduled the following day (i.e. one day after data pull)
    df = filter_chemo_Trt(df, data_pull_day)
    
    col_map = {
        'research_id': 'mrn', 
        trt_date: 'treatment_date',  #'tx_sched_date'; 'trt_date_utc': 'treatment_date',
        'first_trt_date_utc': 'first_treatment_date',
        'dose_ord_or_min_dose_ord': 'dose_ordered',
        'dose_given': 'given_dose'
    }
    df = df.rename(columns=col_map) 

    # merge rows with same treatment days
    df['first_treatment_date'] = df['first_treatment_date'].apply(str) # due to error in one instance
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
        
    else:     
        # Keep treatments scheduled for the next day
        df['tx_sched_date'] = pd.to_datetime(df['tx_sched_date']).dt.date
        
        #Treatment scheduled one day after data pull
        following_treatment_date = pd.to_datetime(data_pull_day).date() + timedelta(days=1) 
        mask = df['tx_sched_date']==following_treatment_date
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
