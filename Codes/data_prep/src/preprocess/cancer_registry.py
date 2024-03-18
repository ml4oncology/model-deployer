"""
Module to preprocess the cancer registry (cancer patient demographic data) - DIAGNOSIS
"""

import pandas as pd
import numpy as np


def get_demographic_data(diagnosis_data_file, info_data_dir):

    df = pd.read_csv(diagnosis_data_file)
    df = filter_demographic_data(df)
    df = process_demographic_data(df, info_data_dir)
    return df

def process_demographic_data(df, info_data_dir):
    
    All_Cancer_Site_List = pd.read_excel(info_data_dir + '/Cancer_Site_List.xlsx')
    All_Cancer_Site_List = list(All_Cancer_Site_List['Cancer_Site'])
    
    # order by diagnosis date
    df = df.sort_values(by='last_contact_date')

    # make each cancer site into a new column with diagnosis date as entry
    cancer_site = df.pivot(columns='primary_site', values='last_contact_date').loc[df.index]
    cancer_site.columns = 'cancer_site_' + cancer_site.columns
    
    current_cancer_sites = list(cancer_site.columns.values)
    
    common_sites = list(set(current_cancer_sites).intersection(All_Cancer_Site_List))
    
    other_sites = list(set(current_cancer_sites) - set(common_sites))
    
    cancer_site['cancer_site_other']=np.nan
    for iC in range(len(other_sites)):
        cancer_site['cancer_site_other'] = cancer_site['cancer_site_other'] + cancer_site[other_sites[iC]]
        cancer_site = cancer_site.drop(columns=[other_sites[iC]])
        
    rem_sites = list(set(All_Cancer_Site_List) - set(common_sites))
    for iC in range(len(rem_sites)):
        cancer_site[rem_sites[iC]]=np.nan
    
    cancer = pd.concat([cancer_site], axis=1) 
    cancer = cancer.astype(str)
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

    # filter out patients without medical record numbers
    mask = df['mrn'].notnull()
    # get_excluded_numbers(df, mask, context=' with no MRN')
    df = df[mask]

    # clean data types
    df['mrn'] = df['mrn'].astype(int)

    # sanity check - ensure vital status and death date matches and makes sense
    # mask = df['vital_status'].map({'Dead': False, 'Alive': True}) == df['date_of_death'].isnull()
    mask = df['vital_status'].map({'Deceased': False, 'Alive': True}) == df['date_of_death'].isnull()
    assert mask.all()

    # filter out patients whose sex is not Male/Female
    mask = df['sex'].isin(['Male', 'Female'])
    # get_excluded_numbers(df, mask, context=' in which sex is other than Male/Female')
    df = df[mask].copy()
    df['female'] = df.pop('sex') == 'Female'

    # Separate multiple cancer sites
    col_len = df['primary_site'].apply(lambda x: len(str(x)) > 6)
    if any(col_len):
        df[['primary_site_1','primary_site_2']] = df["primary_site"].str.split(", ", n=1, expand=True)
        mul_sites_loc = np.where(df['primary_site_2'].notnull())[0]
        
        for i1 in range(len(mul_sites_loc)):
            new_row = df.loc[mul_sites_loc[i1]].copy()
            new_row["primary_site"]=new_row["primary_site_2"]
            df = df._append([new_row], ignore_index=True)
            df["primary_site"][mul_sites_loc[i1]]=df['primary_site_1'][mul_sites_loc[i1]]
            
        df = df.drop('primary_site_1', axis=1)
        df = df.drop('primary_site_2', axis=1)
        
    # clean cancer site and morphology feature
    for col in ['primary_site']: #, 'morphology'
        # only keep first three characters - the rest are for specifics
        # e.g. C50 Breast: C501 Central portion, C504 Upper-outer quadrant, etc
        df[col] = df[col].str[:3]
           

    return df