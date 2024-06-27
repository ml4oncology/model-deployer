"""
Script to load the models and generate predictions

"""

import pickle
import pandas as pd
import numpy as np

def get_model_output(model_dir, info_data_dir, patient_info_ED, model_features_ED, patient_info_symp, model_features_symp):
    
    # load the model from disk
    # NOTE: XGBoost version 2.0.3 (pip install xgboost==2.0.3 --user)
    UCE = 'ED_visit'
    filename = 'XGB_' + UCE + '.pkl'
    loaded_model = pickle.load(open(model_dir+'/'+filename, 'rb'))
    
    ED_Pred_Threshold = pd.read_excel(info_data_dir + '/ED_Prediction_Threshold.xlsx')
    ED_Thres = ED_Pred_Threshold['Prediction_threshold'][0]
    
    # extraxt all the columns with order in which they were used; then reorder the pandas dataframe
    saved_model_features_ED = loaded_model.get_booster().feature_names
    model_features_ED = model_features_ED[saved_model_features_ED]
    
    # Generate predictions
    xgb_preds_labels = loaded_model.predict(model_features_ED)
    xgb_preds_probabilities = loaded_model.predict_proba(model_features_ED)[:,1]
    
    # Combine patient info with predictions
    comb_ptInfo_pred_ed = patient_info_ED.copy()
    comb_ptInfo_pred_ed['ed_pred_labels'] = xgb_preds_labels
    comb_ptInfo_pred_ed['ed_pred_probabilities_1'] = xgb_preds_probabilities
    comb_ptInfo_pred_ed['ed_pred_labels_thresh'] = comb_ptInfo_pred_ed['ed_pred_probabilities_1'].apply(lambda x: 1 if x > ED_Thres else 0)
    
    # # Get most recent treatment dates for each patient
    # pred_latestTrtDate = comb_ptInfo_pred.copy()
    # pred_latestTrtDate = pred_latestTrtDate.loc[pred_latestTrtDate.groupby('mrn').treatment_date.idxmax()]
    
    # # Get treatment dates in the last 30 days from dataPull day
    # pred_last30days = comb_ptInfo_pred.copy()
    # pred_last30days = pred_last30days[pred_last30days["treatment_date"] >= (pd.to_datetime(dataPull_day) - pd.Timedelta(days=30))]
    # # pred_last30days.sort_values(by=['mrn','treatment_date'])
    
    
    ################ Symptoms Model Evaluation #####################
    # load the model from disk
    # with open("C:/UHN CDI UoT/Projects/Aim2Reduce/Model Silent Deployment/Noke LGBM Models/train_results_2024_kevin_3pt_epic.pkl", "rb") as f:
    #     train_results = pickle.load(f)
    
    filename = 'LGBM_symp.pkl'
    loaded_models_symp = pickle.load(open(model_dir+'/'+filename, 'rb'))
    
    Symp_Pred_Thresholds = pd.read_excel(info_data_dir + '/Symptoms_Prediction_Thresholds.xlsx')
        
    symp_labels = ['Pain', 'Tired', 'Nausea', 'Depress', 'Anxious', 'Drowsy', 'Appetite', 'WellBeing', 'SOB']
    
    # Convert results to DataFrame, then set the index to the model and label
    results_df = pd.DataFrame.from_dict(loaded_models_symp, orient='index')
    results_df.index.names = ['model', 'label']
    results_df.reset_index(inplace=True)
    
    # Combine patient info with predictions
    comb_ptInfo_pred_symp = patient_info_symp.copy()
    
    for iL in range(len(symp_labels)):
         
        # pick label
        label = 'Label_'+symp_labels[iL]+'_3pt_change'
        
        # Get the corresponding threshold
        # label_thres_loc = np.where(Symp_Pred_Thresholds['Labels']==label)[0][0]
        # label_thres = Symp_Pred_Thresholds['Prediction_threshold'][label_thres_loc]
        label_thres = Symp_Pred_Thresholds.loc[Symp_Pred_Thresholds['Labels'] == label, 'Prediction_threshold'].iloc[0]
        # print(label + ':' + str(label_thres))
        
        # fetch lgbm model
        best_lgbm_model = results_df[results_df['label'] == label]['best_model'].iloc[0]
        
        # extraxt all the columns with order in which they were used; then reorder the pandas dataframe
        saved_model_features_symp = best_lgbm_model.feature_name_
        model_features_symp = model_features_symp[saved_model_features_symp]
    
        # fetch ir model
        best_lgbm_ir_model = results_df[results_df['label'] == label]['best_ir_model'].iloc[0]
        
        # Generate predictions
        # using best pain and ir model for inference
        test_predictions = best_lgbm_model.predict_proba(model_features_symp)[:, 1]
        y_pred_proba = best_lgbm_ir_model.transform(test_predictions)
       
        # Combine patient info with predictions
        pred_label = symp_labels[iL]+'_labels'
        probly_label = symp_labels[iL]+'__pred_probabilities_1'
        comb_ptInfo_pred_symp[probly_label] = y_pred_proba
        comb_ptInfo_pred_symp[pred_label] = comb_ptInfo_pred_symp[probly_label].apply(lambda x: 1 if x > label_thres else 0)
    
    
    return comb_ptInfo_pred_ed, comb_ptInfo_pred_symp