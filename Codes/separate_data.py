"""
Processing Script 
Using the current directory

"""

# import numpy as np

def separate_ptInfo_features(df):
    
    patient_info = df[{'mrn','treatment_date'}]
    model_features = df.copy()
    
    drop_cols = ['treatment_date','mrn']
    model_features = model_features.drop(columns=drop_cols)
    
    return patient_info, model_features