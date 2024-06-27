"""
Main Script 
Using the current directory

"""

# import pickle
import pandas as pd
# import numpy as np

#from checkData import check_data
from data_prep.final_processing import final_process
from model_eval.getModelPredictions import get_model_output

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


############################ Make changes #################################
ROOT_DIR = "C:/Users/Muammar/Desktop/MIRA_Test" # Select Root Directory
data_root_dir = f'{ROOT_DIR}/Data'
dataStart_day = '20240229' #date.today().strftime("%Y%m%d")
dataEnd_day = '20240620' #date.today().strftime("%Y%m%d")

# dataPull_day = '20240318' #date.today().strftime("%Y%m%d")

if __name__ == "__main__":
    
    info_data_dir= f'{ROOT_DIR}/Infos'
    train_param_dir = f'{ROOT_DIR}/Infos/Train_Data_parameters'
    code_dir = f'{ROOT_DIR}/Codes'
    model_dir = f'{ROOT_DIR}/Models' 
    proj_name = 'AIM2REDUCE'
    model_name = ['ED','symp'] #ED or symp
    
    sdate = pd.to_datetime(dataStart_day).date()   # start date
    edate = pd.to_datetime(dataEnd_day).date()   # end date
    dataPull_range = pd.date_range(sdate,edate,freq='d').strftime("%Y%m%d")
    
    fullData_pred = []
    
    for iD in tqdm(range(0,len(dataPull_range))): 
        
        dataPull_day = dataPull_range[iD]
    
        chemo_file = f"{data_root_dir}/{proj_name}_chemo_{dataPull_day}.csv"
        diagnosis_file = f"{data_root_dir}/{proj_name}_diagnosis_{dataPull_day}.csv"
        
        if pd.read_csv(chemo_file).empty and pd.read_csv(diagnosis_file).empty:
            print(f"No Patient Data for: {dataPull_day}")
            continue

    
        ######################### Data Processing ################################
        
        ##******************** ED **********************##
        # Process and prepare data
        prepared_data_ED = final_process(data_root_dir, info_data_dir, train_param_dir, code_dir, model_dir, proj_name, model_name[0], dataPull_day)
        
        # Separate patient mrn and trt_date from model features
        # patient_info, model_features = separate_ptInfo_features(prepared_data)
        patient_info_ED = prepared_data_ED[['mrn', 'treatment_date']].copy()
        model_features_ED = prepared_data_ED.drop(columns=['mrn', 'treatment_date'])
        
        ##******************** Symptoms **********************##
        # Process and prepare data
        prepared_data_symp = final_process(data_root_dir, info_data_dir, train_param_dir, code_dir, model_dir, proj_name, model_name[1], dataPull_day)
        
        # Separate patient mrn and trt_date from model features
        # patient_info, model_features = separate_ptInfo_features(prepared_data)
        patient_info_symp = prepared_data_symp[['mrn', 'treatment_date']].copy()
        model_features_symp = prepared_data_symp.drop(columns=['mrn', 'treatment_date'])
                
        
        ######################### Model Evaluation ################################
        # load the model from disk
        # NOTE: XGBoost version 2.0.3 (pip install xgboost==2.0.3 --user)
        
        comb_ptInfo_pred_ed, comb_ptInfo_pred_symp = get_model_output(model_dir, info_data_dir,
                                                                      patient_info_ED, model_features_ED,
                                                                      patient_info_symp, model_features_symp)
        
        comb_ptInfo_pred = comb_ptInfo_pred_ed.merge(comb_ptInfo_pred_symp, on=['mrn', 'treatment_date'])
        
        if iD==0:
            fullData_pred = comb_ptInfo_pred
        else:
            fullData_pred = pd.concat([fullData_pred, comb_ptInfo_pred], ignore_index=True, axis=0)
            
