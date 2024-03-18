"""
Main Script 
Using the current directory

"""

import pickle
# import numpy as np

from checkData import check_data
from final_processing import final_process
from separate_data import separate_ptInfo_features

import warnings
warnings.filterwarnings("ignore")


############################ Make changes #################################
ROOT_DIR = "C:/Users/..." # Select Root Directory
data_root_dir = f'{ROOT_DIR}/Data'
dataPull_day = '20240318' #date.today().strftime("%Y%m%d")



if __name__ == "__main__":
        
    info_data_dir= f'{ROOT_DIR}/Infos'
    train_param_dir = f'{ROOT_DIR}/Infos/Train_Data_parameters'
    code_dir = f'{ROOT_DIR}/Codes'
    model_dir = f'{ROOT_DIR}/Models' 
    proj_name = 'XXX' #Select project name required to access data file 
    model_name = 'ED' #ED or symp
    
    
    emptyData = check_data(data_root_dir, proj_name, dataPull_day)
    
    if not emptyData: 
    
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
        
    else:
        
        print("No Patient Data for: " + dataPull_day)


