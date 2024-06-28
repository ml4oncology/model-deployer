"""
Module to engineer features
"""

from tqdm import tqdm
import numpy as np
import pandas as pd

from data_prep.constants import lab_cols, lab_change_cols, symp_cols, symp_change_cols

###############################################################################
# Engineering Features
###############################################################################
def get_change_since_prev_session(df: pd.DataFrame) -> pd.DataFrame:
    """Get change since last session"""
    cols = symp_cols + lab_cols
    change_cols = symp_change_cols + lab_change_cols
    result = []
    for mrn, group in tqdm(df.groupby('mrn')):
        group[cols] = group[cols].apply(pd.to_numeric) # convert all columns of DataFrame # pd.to_numeric(s, errors='coerce')
        change = group[cols] - group[cols].shift()
        result.append(change.reset_index().to_numpy())
    result = np.concatenate(result)

    result = pd.DataFrame(result, columns=['index']+change_cols).set_index('index')
    result.index = result.index.astype(int)
    df = pd.concat([df, result], axis=1)

    return df

def get_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    
    target_cols = symp_cols + lab_cols + lab_change_cols + symp_change_cols
    
    for col in target_cols: df[f'{col}_is_missing'] = df[col].isnull()
        
    return df

###############################################################################
# Time
###############################################################################
def get_visit_month_feature(df, col: str = 'treatment_date'):
    # convert to cyclical features
    month = df[col].dt.month - 1
    df['visit_month_sin'] = np.sin(2*np.pi*month/12)
    df['visit_month_cos'] = np.cos(2*np.pi*month/12)
    return df

def get_days_since_last_event(df, main_date_col: str = 'treatment_date', event_date_col: str = 'treatment_date'):
    if main_date_col == event_date_col:
        return (df[main_date_col] - df[event_date_col].shift()).dt.days
    else:
        return (df[main_date_col] - df[event_date_col]).dt.days

def get_years_diff(df, col1: str, col2: str):
    return df[col1].dt.year - df[col2].dt.year

###############################################################################
# Treatment
###############################################################################
def get_line_of_therapy(df):
    # identify line of therapy (the nth different palliative intent treatment taken)
    # NOTE: all other intent treatment are given line of therapy of 0. Usually (not always but oh well) once the first
    # palliative treatment appears, the rest of the treatments remain palliative
    new_regimen = (df['first_treatment_date'] != df['first_treatment_date'].shift())
    palliative_intent = df['intent'] == 'PALLIATIVE'
    return (new_regimen & palliative_intent).cumsum()
