"""
Main Script 
"""
import argparse

import pandas as pd
from tqdm import tqdm

from data_prep.final_processing import final_process
from model_eval.evaluate import get_model_output

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='20240229') # date.today().strftime("%Y%m%d")
    parser.add_argument('--end-date', type=str, default='20240319') # date.today().strftime("%Y%m%d")
    parser.add_argument('--project-name', type=str, default='AIM2REDUCE')
    parser.add_argument('--output-path', type=str, default='./pred.csv')
    parser.add_argument('--root-dir', type=str, default='.') # "C:/Users/Muammar/Desktop/MIRA_Test"
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    proj_name = args.project_name
    output_path = args.output_path
    ROOT_DIR = args.root_dir

    # TODO: maybe we should only make data-dir and model-dir and info-dir into CLI arguments and remove root-dir? 
    #       for better generalizability for end-users
    data_dir = f'{ROOT_DIR}/Data'
    info_dir= f'{ROOT_DIR}/Infos'
    train_param_dir = f'{ROOT_DIR}/Infos/Train_Data_parameters'
    code_dir = f'{ROOT_DIR}/Codes' # TODO: load config.yaml here (the only time code_dir is used)
    model_dir = f'{ROOT_DIR}/Models' 
    
    date_range = pd.date_range(start_date, end_date, freq='d').strftime("%Y%m%d")
    
    output = []
    for i, data_pull_date in tqdm(enumerate(date_range)): 
    
        # TODO: maybe this should be if os.path.exists? Do we really need to check both files?
        chemo_file = f"{data_dir}/{proj_name}_chemo_{data_pull_date}.csv"
        diagnosis_file = f"{data_dir}/{proj_name}_diagnosis_{data_pull_date}.csv"
        if pd.read_csv(chemo_file).empty and pd.read_csv(diagnosis_file).empty:
            print(f"No Patient Data for: {data_pull_date}")
            continue

    
        # TODO: keep ED and symptoms separate throughout, for model evaluation as well
        #       process ED data and evaluate ED visit, then process Symptoms data and evaluate symptoms deterioration
        #       then combine all the predictions

        ######################### Data Processing ################################
        
        ##******************** ED **********************##
        # Process and prepare data
        prepared_data_ED = final_process(data_dir, info_dir, train_param_dir, code_dir, model_dir, proj_name, 'ED', data_pull_date)
        
        # Separate patient mrn and trt_date from model features
        metadata_ED = prepared_data_ED[['mrn', 'treatment_date']].copy()
        model_features_ED = prepared_data_ED.drop(columns=['mrn', 'treatment_date'])
        
        ##******************** Symptoms **********************##
        # Process and prepare data
        prepared_data_symp = final_process(data_dir, info_dir, train_param_dir, code_dir, model_dir, proj_name, 'symp', data_pull_date)
        
        # Separate patient mrn and trt_date from model features
        metadata_symp = prepared_data_symp[['mrn', 'treatment_date']].copy()
        model_features_symp = prepared_data_symp.drop(columns=['mrn', 'treatment_date'])
                
        
        ######################### Model Evaluation ################################        
        comb_ptInfo_pred_ed, comb_ptInfo_pred_symp = get_model_output(
            model_dir, info_dir, metadata_ED, model_features_ED, metadata_symp, model_features_symp
        )
        comb_ptInfo_pred = comb_ptInfo_pred_ed.merge(comb_ptInfo_pred_symp, on=['mrn', 'treatment_date'])

        output.append(comb_ptInfo_pred)
    
    output = pd.concat(output, ignore_index=True, axis=0)
    output.to_csv(output_path)
