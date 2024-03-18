"""
Module to preprocess laboratory test data, which includes hematology and biochemistry data
"""
from typing import Optional

import pandas as pd

from ..constants import obs_map

def get_lab_data(hema_data_file, biochem_data_file):

    hema = pd.read_csv(hema_data_file)
    hema = filter_lab_data(hema, obs_name_map=obs_map['Hematology'])

    biochem = pd.read_csv(biochem_data_file)
    biochem = filter_lab_data(biochem, obs_name_map=obs_map['Biochemistry'])

    lab = pd.concat([hema, biochem])
    lab = process_lab_data(lab)
    lab = lab.rename(columns={'patientId': 'mrn'})
    return lab

def process_lab_data(df):
    df['obs_datetime'] = pd.to_datetime(df['obs_datetime'], utc=True)
    df['obs_date'] = pd.to_datetime(df['obs_datetime'].dt.date)
    df = df.sort_values(by='obs_datetime')

    # take the most recent value if multiple lab tests taken in the same day
    # NOTE: dataframe already sorted by obs_datetime
    df = df.groupby(['patientId', 'obs_date', 'obs_name']).agg({'obs_value': 'last'}).reset_index()

    # make each observation name into a new column
    df = df.pivot(index=['patientId', 'obs_date'], columns='obs_name', values='obs_value').reset_index()
    
    for i1 in range(len(df.columns)):
        df[df.columns[i1]] = df[df.columns[i1]].replace(['<5'], '2.5')

    df.columns.name = None
    return df

def filter_lab_data(df, obs_name_map: Optional[dict] = None):
    df = clean_lab_data(df)
    
    # exclude rows where observation value is missing
    df = df[df['obs_value'].notnull()]

    if obs_name_map is not None:
        df['obs_name'] = df['obs_name'].map(obs_name_map)
        # exclude observations not in the name map
        df = df[df['obs_name'].notnull()]

    df = filter_units(df)
    df = df.drop_duplicates(subset=['patientId', 'obs_value', 'obs_name', 'obs_unit', 'obs_datetime'])
    return df

def filter_units(df):
    # clean the units
    df['obs_unit'] = df['obs_unit'].replace({'bil/L': 'x10e9/L', 'fl': 'fL'})

    # some observations have measurements in different units (e.g. neutrophil observations contain measurements in 
    # x10e9/L (the majority) and % (the minority))
    # only keep one measurement unit for simplicity
    exclude_unit_map = {
        'creatinine': ['mmol/d', 'mmol/CP'],
        'eosinophil': ['%'], 
        'lymphocyte': ['%'], 
        'monocyte': ['%'], 
        'neutrophil': ['%'],
        'red_blood_cell': ['x10e6/L'],
        'white_blood_cell': ['x10e6/L'],
    }
    mask = False
    for obs_name, exclude_units in exclude_unit_map.items():
        mask |= (df['obs_name'] == obs_name) & df['obs_unit'].isin(exclude_units)
    df = df[~mask]

    return df

def clean_lab_data(df):
    # clean column names
    col_map = {
        # assign obs_ prefix to ensure no conflict with preexisting columns
        'component-code-coding-0-display': 'obs_name', 
        'component-valueQuantity-unit': 'obs_unit',
        'component-valueQuantity-value': 'obs_value',
        'lastUpdated': 'obs_datetime',
    }
    df = df.rename(columns=col_map)
    
    return df
