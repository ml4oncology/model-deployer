"""
Module to preprocess DART (symptom data)
"""
from typing import Optional
import pandas as pd
import numpy as np

# from ..util import get_excluded_numbers


def get_symptoms_data(esas_data_file):

    df = pd.read_csv(esas_data_file)
    df = df.rename(columns={'RESEARCH_ID': 'MRN'})
    df = filter_symptoms_data(df)
    symp = process_symptoms_data(df)
    return symp


def process_symptoms_data(df):
    
    # get indices of ecog scores containing values with descriptions
    # ecog_locs = df[df['patient_ecog'].str.len() > 1].index.values
    ecog_locs = np.where(df['patient_ecog'].apply(lambda x: len(str(x)) > 3))[0]
    if len(ecog_locs)>0:
        for i1 in range(len(ecog_locs)):
            df['patient_ecog'].iloc[ecog_locs[i1]] = df['patient_ecog'].iloc[ecog_locs[i1]][0]
     
    df['patient_ecog'] = pd.to_numeric(df['patient_ecog'])
    
    # order by survey date
    df = df.sort_values(by='survey_date')
        
    # get columns of interest
    cols = df.columns
    cols = cols[cols.str.contains('esas_|_ecog')]

    # merge rows with same survey dates
    df = (
        df
        .groupby(['mrn', 'survey_date'])
        .agg({col: 'mean' for col in cols}) # handle conflicting data by taking the mean
    )
    df = df.reset_index()

    return df


def filter_symptoms_data(df):
    # clean column names
    df.columns = df.columns.str.lower()
    # clean data types
    for col in ['date_of_birth', 'survey_date']: df[col] = pd.to_datetime(df[col])
    
    # # filter out patients who consented out of research
    # # WARNING: Beware of trailing spaces
    # # e.g. >> df['research_consent'].unique()
    # #      array(['Y                    ', '                     ', 'N                    '])
    # df['research_consent'] = df['research_consent'].str.strip()
    # mask = df['research_consent'] != 'N'
    # get_excluded_numbers(df, mask, context=' in which consent to research was declined')
    # df = df[mask]

    # filter out patients whose sex is not Male/Female
    mask = df['gender'] != 'Unknown'
    # get_excluded_numbers(df, mask, context=' in which sex is Unknown')
    df = df[mask].copy()
    df['female'] = df.pop('gender') == 'Female'

    # exclude rows where symptoms scores are all missing
    cols = df.columns
    cols = cols[cols.str.contains('esas_|_ecog')]
    mask = df[cols].isnull().all(axis=1)
    # get_excluded_numbers(df, ~mask, context=' without any symptom scores')
    df = df[~mask]

    return df