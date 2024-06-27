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
    dataPull_day    
) -> pd.DataFrame:
    
    """
    Args:
    included_regimens: selected EPR regimens during model training
    A2R_EPIC_GI_regimen_map: map between the old EPR regimens and the new EPIC regimens
    """

    # A2R_EPIC_GI_regimen_map = A2R_EPIC_GI_regimen_map[{'PROTOCOL_DISPLAY_NAME','Mapped_Name_All'}]
    A2R_EPIC_GI_regimen_map = A2R_EPIC_GI_regimen_map[['PROTOCOL_DISPLAY_NAME','Mapped_Name_All']]
    A2R_EPIC_GI_regimen_map = A2R_EPIC_GI_regimen_map.set_index('PROTOCOL_DISPLAY_NAME')['Mapped_Name_All'].to_dict()
    
    df = pd.read_csv(chemo_data_file)
    df = process_treatment_data(df, dataPull_day)
    df = filter_treatment_data(df, included_regimens, A2R_EPIC_GI_regimen_map, dataPull_day)
    return df
    

def process_treatment_data(df, dataPull_day):
    
    # clean column names
    df.columns = df.columns.str.lower()
    
    # order by id, scheduled date and regimen
    df = df.sort_values(by=['research_id', 'tx_sched_date', 'regimen'])
    
    # Forwardfill height, weight and bsa
    df = fwdfill(df)
    
    # Keep only treatments scheduled the following day (i.e. one day after data pull)
    df = filter_chemo_Trt(df, dataPull_day)
    
    col_map = {
        'research_id': 'mrn', 
        'tx_sched_date': 'treatment_date',  #'trt_date_utc': 'treatment_date',
        'first_trt_date_utc': 'first_treatment_date',
        'dose_ord_or_min_dose_ord': 'dose_ordered',
        'dose_given': 'given_dose'
    }
    df = df.rename(columns=col_map) 

    # merge rows with same treatment days
    df['first_treatment_date'] = df['first_treatment_date'].apply(str) # due to error in one instance
    df = merge_same_day_treatments(df) #, dosage

    # forward fill height, weight and body_surface_area
    for col in ['height', 'weight', 'body_surface_area']: df[col] = df.groupby('mrn')[col].ffill()

    # forward and backward fill first treatment date
    df['first_treatment_date'] = pd.to_datetime(df['first_treatment_date'])
    df['first_treatment_date'] = df.groupby('mrn')['first_treatment_date'].ffill().bfill()

    return df


def filter_treatment_data(df, regimens: pd.DataFrame, A2R_EPIC_GI_regimen_map, dataPull_day) -> pd.DataFrame: #drugs: pd.DataFrame, 
       
    # # Split dose and units
    # df[['given_dose','given_dose_unit']] = df["given_dose"].str.split(" ", n=1, expand=True)
    # df[['dose_ordered','dose_ordered_unit']] = df["dose_ordered"].str.split(" ", n=1, expand=True)
    
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


def fwdfill(df):
    
    df['height'] = df.groupby('research_id')['height'].ffill()

    df['weight'] = df.groupby('research_id')['weight'].ffill()

    df['body_surface_area'] = df.groupby('research_id')['body_surface_area'].ffill()
    
    return df

# def filter_drugs(df): #, drugs: pd.DataFrame
    
#     # filter out rows where no dosage is given (dose = nan or 0)
#     # e.g. patients get vital sign examination but don't receive treatment
#     mask=df['given_dose'].notnull()
#     df = df[mask]
    
#     df["given_dose"] = pd.to_numeric(df["given_dose"])
#     mask = df['given_dose'] > 0 
#     df = df[mask]
    
#     return df


def filter_chemo_Trt(df, dataPull_day):
    
    # Keep treatments scheduled for the next day
    df['tx_sched_date'] = pd.to_datetime(df['tx_sched_date']).dt.date
    
    #Treatment scheduled one day after data pull
    following_treatment_date = pd.to_datetime(dataPull_day).date() + timedelta(days=1) 
    mask = df['tx_sched_date']==following_treatment_date
    df = df[mask]
    
    # df['treatment_date'] = df['tx_sched_date']
    
    return df


# def fill_phys_char(df, regimens):
    
#     regimens.columns = regimens.columns.str.lower()

#     # clean column names
#     df.columns = df.columns.str.lower()
#     col_map = {
#         'research_id': 'mrn', 
#         'trt_date_utc': 'treatment_date', 
#         'first_trt_date_utc': 'first_treatment_date',
#         'dose_ord_or_min_dose_ord': 'dose_ordered',
#         'dose_given': 'given_dose'
#     }
#     df = df.rename(columns=col_map)
    
#     # order by date and regimen
#     df = df.sort_values(by=['treatment_date', 'regimen'])
    
#     # forward fill height, weight and body_surface_area
#     for col in ['height', 'weight','body_surface_area']: df[col] = df.groupby('mrn')[col].ffill()
    
#     # forward/backward fill first treatment date
#     df['first_treatment_date']=pd.to_datetime(df['first_treatment_date'])
#     for col in ['first_treatment_date']: df[col] = df.groupby('mrn')[col].ffill()
#     for col in ['first_treatment_date']: df[col] = df.groupby('mrn')[col].bfill()
    
#     return df


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
