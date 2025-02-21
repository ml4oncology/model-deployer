"""
Module to preprocess the demographic and diagnosis data
"""

import pandas as pd
from data_prep.constants import DROP_CLINIC_COLUMNS


def get_demographic_data(diagnosis_data_file, info_data_dir, anchor):
    df = pd.read_csv(diagnosis_data_file)
    if anchor == 'clinic':
        df = df.drop(columns=DROP_CLINIC_COLUMNS)
    df = filter_demographic_data(df)
    df = process_demographic_data(df, info_data_dir)
    return df

def process_demographic_data(df, info_data_dir):
    all_cancer_site_list = pd.read_excel(info_data_dir + '/Cancer_Site_List.xlsx')
    all_cancer_site_list = list(all_cancer_site_list['Cancer_Site'])
    
    cancer = df['primary_site'].str.get_dummies(', ')
    cancer = cancer.add_prefix('cancer_site_')
    
    # assign cancer sites not seen during model training as cancer_site_other
    other_sites =  [site for site in cancer.columns if site not in all_cancer_site_list]
    cancer['cancer_site_other'] = cancer[other_sites].any(axis=1)
    cancer.drop(columns=other_sites)
    
    # create missing cancer sites required by the model
    missing_sites = [site for site in all_cancer_site_list if site not in cancer.columns]
    cancer[missing_sites] = False
    
    df = df.join(cancer)

    # combine patients with mutliple diagnoses into one row
    df = (
        df
        .groupby(['mrn'])
        .agg({
            # handle conflicting data by taking the most recent entries
            'date_of_birth': 'last',
            'female': 'last',
            # if two diagnoses dates for same cancer site/morphology (e.g. first diagnoses in 2005, cancer returns in 
            # 2013) take the first date (e.g. 2005)
            **{col: 'min' for col in cancer.columns}
        })
    )
    df = df.reset_index()

    return df
    
def filter_demographic_data(df):
    # clean column names
    df.columns = df.columns.str.lower()
    df = df.rename(columns= {'research_id': 'mrn'})
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])

    # filter out patients without medical record numbers
    mask = df['mrn'].notnull()
    # get_excluded_numbers(df, mask, context=' with no MRN')
    df = df[mask]

    # clean data types
    df['mrn'] = df['mrn'].astype(int)

    # sanity check - ensure vital status and death date matches and makes sense
    mask = df['vital_status'].map({'Deceased': False, 'Alive': True}) == df['date_of_death'].isnull()
    try:
        assert mask.all()
    except AssertionError:
        df['vital_status'].loc[~mask] = 'Deceased'
    

    # filter out patients whose sex is not Male/Female
    mask = df['sex'].isin(['Male', 'Female'])
    df = df[mask].copy()
    df['female'] = df.pop('sex') == 'Female'

    # Separate multiple cancer sites
    # Check if any primary_site entries have more than one site
    if df['primary_site'].str.contains(', ').any():
        # Split primary_site into two new columns
        df[['primary_site_1', 'primary_site_2']] = df["primary_site"].str.split(", ", n=1, expand=True)
    
        # Update the original rows with the first site
        mask = df['primary_site_2'].notnull()
        df.loc[mask, "primary_site"] = df['primary_site_1']
    
        # Create new rows for the second site
        new_rows = df[mask].copy()
        new_rows["primary_site"] = new_rows["primary_site_2"]
        df = pd.concat([df, new_rows], ignore_index=True) # DONT USE APPEND, ITS NOW DEPRECATED
    
        # Drop the temporary columns
        df = df.drop(['primary_site_1', 'primary_site_2'], axis=1)
        
    # clean cancer site and morphology feature
    for col in ['primary_site']: #, 'morphology'
        # only keep first three characters - the rest are for specifics
        # e.g. C50 Breast: C501 Central portion, C504 Upper-outer quadrant, etc
        df[col] = df[col].str[:3]
           

    return df