"""
Main Script 
"""
import argparse

import pandas as pd
from tqdm import tqdm

from data_prep.final_processing import final_process
from model_eval.inference import get_ED_visit_model_output #, get_symp_model_output

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='20240904') # date.today().strftime("%Y%m%d")
    parser.add_argument('--end-date', type=str, default='20241031') # date.today().strftime("%Y%m%d")
    parser.add_argument('--project-name', type=str, default='AIM2REDUCE')
    parser.add_argument('--treatment-output', type=str, default='./treatment_fullMonth_pred_Sep_Oct.csv') #pred.csv; ED_full_data.csv
    parser.add_argument('--clinic-output', type=str, default='./clinic_fullMonth_pred_Sep_Oct.csv') #pred.csv; ED_full_data.csv
    parser.add_argument('--root-dir', type=str, default='C:/UHN CDI UoT/Github/model-deployer') # "C:/Users/Muammar/Desktop/MIRA_Test"
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    proj_name = args.project_name
    treatment_output = args.treatment_output
    clinic_output = args.clinic_output
    ROOT_DIR = args.root_dir

    # TODO: maybe we should only make data-dir and model-dir and info-dir into CLI arguments and remove root-dir? 
    #       for better generalizability for end-users
    data_dir = f'{ROOT_DIR}/Data' #Data
    info_dir= f'{ROOT_DIR}/Infos'
    train_param_dir = f'{ROOT_DIR}/Infos/Train_Data_parameters'
    code_dir = f'{ROOT_DIR}/Archive/Codes' # TODO: load config.yaml here (the only time code_dir is used)
    model_dir = f'{ROOT_DIR}/Models' 
    fig_dir = f'{ROOT_DIR}/Archive/Figures'
    clinic_anchored_files = ['','weekly_']
    
    date_range = pd.date_range(start_date, end_date, freq='d').strftime("%Y%m%d")
    
    thresholds = pd.read_excel(f'{info_dir}/ED_Prediction_Threshold_Updated.xlsx')
    
    results_treatment = []
    results_clinic = []
    # prepared_data_ED_dfs = []
    for i, data_pull_date in tqdm(enumerate(date_range)): 
    
        print(f'**** Processing #{i}: {data_pull_date} *****')
        
        ######################### Treatment anchored ################################
        
        chemo_file_treatment = f"{data_dir}/{proj_name}_chemo_{clinic_anchored_files[0]}{data_pull_date}.csv"
        diagnosis_file_treatment = f"{data_dir}/{proj_name}_diagnosis_{clinic_anchored_files[0]}{data_pull_date}.csv"
        if not pd.read_csv(chemo_file_treatment).empty and not pd.read_csv(diagnosis_file_treatment).empty:
            
            ######################### Data Processing ################################
            ##******************** ED **********************##
            # Process and prepare data
            prepared_treatment_data_ED = final_process(data_dir, info_dir, train_param_dir, code_dir, proj_name, 'ED', data_pull_date, clinic_anchored_files[0])

            prepared_treatment_data_ED['regimen_GI_IRINO Q3W']=False
            prepared_treatment_data_ED['regimen_GI_PACLITAXEL']=False
            # prepared_treatment_data_ED['regimen_GI_CISPFU _ TRAS_MAIN_']=False
            # prepared_treatment_data_ED['regimen_GI_GEMCAP']=False
            
            # ##******************** Symptoms **********************##
            # # Process and prepare data
            # prepared_treatment_data_symp = final_process(data_dir, info_dir, train_param_dir, code_dir, proj_name, 'symp', data_pull_date, clinic_anchored_files[0])
            
            ######################### Model Evaluation ################################        
            ##******************** ED **********************##
            # Load pre-defined prediction thresholds
            thresholds_treatment = thresholds[thresholds['Model_anchor']=='Treatment-anchored']
            thresholds_treatment = thresholds_treatment.set_index('Labels')['Prediction_threshold']
            ED_treatment_result = get_ED_visit_model_output(prepared_treatment_data_ED, thresholds_treatment, model_dir, fig_dir, clinic_anchored_files[0])
            
            # ##******************** Symptoms **********************##
            #  # Load pre-defined prediction thresholds
            # thresholds = pd.read_excel(f'{info_dir}/Symptoms_Prediction_Thresholds.xlsx')
            # thresholds = thresholds.set_index('Labels')['Prediction_threshold']
            # symp_treatment_result = get_symp_model_output(prepared_treatment_data_symp, thresholds, model_dir, clinic_anchored_files[0])

            # output_treatment = ED_treatment_result.merge(symp_treatment_result, on=['mrn', 'treatment_date'])
            # assert len(ED_treatment_result) == len(symp_treatment_result) == len(output_treatment)
            results_treatment.append(ED_treatment_result) #output_treatment
            
        else:
            print(f"No Patient Treatment Data for: {data_pull_date}")
            
        ########################### Clinic anchored #################################
        
        chemo_file_clinic = f"{data_dir}/{proj_name}_chemo_{clinic_anchored_files[1]}{data_pull_date}.csv"
        diagnosis_file_clinic = f"{data_dir}/{proj_name}_diagnosis_{clinic_anchored_files[1]}{data_pull_date}.csv"
        if not pd.read_csv(chemo_file_clinic).empty and not pd.read_csv(diagnosis_file_clinic).empty:
            
            ######################### Data Processing ################################
            ##******************** ED **********************##
            # Process and prepare data
            prepared_clinic_data_ED = final_process(data_dir, info_dir, train_param_dir, code_dir, proj_name, 'ED', data_pull_date, clinic_anchored_files[1])
            
            prepared_clinic_data_ED['regimen_GI_IRINO Q3W']=False
            prepared_clinic_data_ED['regimen_GI_PACLITAXEL']=False
            # prepared_clinic_data_ED['regimen_GI_CISPFU _ TRAS_MAIN_']=False
            # prepared_clinic_data_ED['regimen_GI_GEMCAP']=False
            
            # ##******************** Symptoms **********************##
            # # Process and prepare data
            # prepared_clinic_data_symp = final_process(data_dir, info_dir, train_param_dir, code_dir, proj_name, 'symp', data_pull_date, clinic_anchored_files[1])
            
            ######################### Model Evaluation ################################        
            ##******************** ED **********************##
            # Load pre-defined prediction thresholds
            thresholds_clinic = thresholds[thresholds['Model_anchor']=='Clinic-anchored']
            thresholds_clinic = thresholds_clinic.set_index('Labels')['Prediction_threshold']
            ED_clinic_result = get_ED_visit_model_output(prepared_clinic_data_ED, thresholds_clinic, model_dir, fig_dir, clinic_anchored_files[1])
            
            # ##******************** Symptoms **********************##
            #  # Load pre-defined prediction thresholds
            # thresholds = pd.read_excel(f'{info_dir}/Symptoms_Prediction_Thresholds.xlsx')
            # thresholds = thresholds.set_index('Labels')['Prediction_threshold']
            # symp_clinic_result = get_symp_model_output(prepared_clinic_data_symp, thresholds, model_dir, clinic_anchored_files[1])

            # output_clinic = ED_treatment_result.merge(symp_clinic_result, on=['mrn', 'clinic_date'])
            # assert len(ED_treatment_result) == len(symp_clinic_result) == len(output_clinic)
            results_clinic.append(ED_clinic_result) # output_clinic
            
        else:
            print(f"No Patient Clinic Data for: {data_pull_date}")
       
    
    results_treatment = pd.concat(results_treatment, ignore_index=True, axis=0)
    results_treatment.to_csv(treatment_output)
    
    results_clinic = pd.concat(results_clinic, ignore_index=True, axis=0)
    results_clinic.to_csv(clinic_output)
