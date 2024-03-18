"""
Module to check if treatment/patient data is empty
"""

import pandas as pd

def check_data(data_root_dir, proj_name, dataPull_day):
    
    
    chemo_file = f"{data_root_dir}/{proj_name+'_chemo_'+dataPull_day}.csv"
    diagnosis_file = f"{data_root_dir}/{proj_name+'_diagnosis_'+dataPull_day}.csv"
    
    empty_data=0
    
    df_chemo = pd.read_csv(chemo_file)
    df_diag = pd.read_csv(diagnosis_file)
    
    if len(df_chemo)==0 and len(df_diag)==0:       
        empty_data=1
    
    return empty_data
