"""
Main Script 
Using the current directory

"""

import pickle
import pandas as pd
# import numpy as np

#from checkData import check_data
from final_processing import final_process
from separate_data import separate_ptInfo_features

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


############################ Make changes #################################
ROOT_DIR = "C:/Users/..." # Select Root Directory
data_root_dir = f'{ROOT_DIR}/Data'
dataStart_day = '20240229' #date.today().strftime("%Y%m%d")
dataEnd_day = '20240530' #date.today().strftime("%Y%m%d")

dataPull_day = '20240318' #date.today().strftime("%Y%m%d")



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
        # Process and prepare data
        prepared_data = final_process(data_root_dir, info_data_dir, train_param_dir, code_dir, model_dir, proj_name, model_name, dataPull_day)
        
        # Separate patient mrn and trt_date from model features
        patient_info, model_features = separate_ptInfo_features(prepared_data)
                
        ######################### Model Evaluation ################################
        # load the model from disk
        # NOTE: XGBoost version 2.0.3 (pip install xgboost==2.0.3 --user)
        UCE = 'ED_visit'
        filename = 'XGB_' + UCE + '.pkl'
        loaded_model = pickle.load(open(model_dir+'/'+filename, 'rb'))
        
        # extraxt all the columns with order in which they were used; then reorder the pandas dataframe
        cols_when_model_builds = loaded_model.get_booster().feature_names
        model_features = model_features[cols_when_model_builds]
        
        # Generate predictions
        xgb_preds_labels = loaded_model.predict(model_features)
        xgb_preds_probabilities = loaded_model.predict_proba(model_features)[:,1]
        
        # Combine patient info with predictions
        comb_ptInfo_pred = patient_info.copy()
        comb_ptInfo_pred['ed_pred_labels'] = xgb_preds_labels
        comb_ptInfo_pred['ed_pred_probabilities_1'] = xgb_preds_probabilities
        
        # # Get most recent treatment dates for each patient
        # pred_latestTrtDate = comb_ptInfo_pred.copy()
        # pred_latestTrtDate = pred_latestTrtDate.loc[pred_latestTrtDate.groupby('mrn').treatment_date.idxmax()]
        
        # # Get treatment dates in the last 30 days from dataPull day
        # pred_last30days = comb_ptInfo_pred.copy()
        # pred_last30days = pred_last30days[pred_last30days["treatment_date"] >= (pd.to_datetime(dataPull_day) - pd.Timedelta(days=30))]
        # # pred_last30days.sort_values(by=['mrn','treatment_date'])
