"""
Module to preprocess ESAS symptom data
"""
import pandas as pd
import numpy as np
from data_prep.constants import DROP_CLINIC_COLUMNS

def get_symptoms_data(esas_data_file, anchor):
    df = pd.read_csv(esas_data_file)
    if anchor == 'clinic':
        df = df.drop(columns=DROP_CLINIC_COLUMNS)
    df = df.rename(columns={'RESEARCH_ID': 'MRN'})
    df = filter_symptoms_data(df)
    symp = process_symptoms_data(df)
    return symp


def process_symptoms_data(df):    
    # Found one instance of df['patient_ecog']= 'Not Applicable'
    mask = df['patient_ecog'].isin(['Not Applicable'])  
    df.loc[mask, 'patient_ecog'] = np.nan
    # ecog_locs = np.where(df['patient_ecog'].apply(lambda x: (x!='nan') & (~x.isdigit())))[0]
    # df['patient_ecog'][ecog_locs]='nan'
    
    # some patient_ecog entries have the following format: score-description
    # remove the descriptions and convert the scores to int
    df['patient_ecog'] = df['patient_ecog'].astype(str).str.split('-').str[0]
    df['patient_ecog'] = df['patient_ecog'].astype(float)
    
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