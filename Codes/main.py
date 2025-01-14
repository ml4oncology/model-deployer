"""
Main Script 
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from data_prep.final_processing import final_process
from model_eval.inference import get_ED_visit_model_output

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='20240904') 
    parser.add_argument('--end-date', type=str, default='20250101') 
    parser.add_argument('--project-name', type=str, default='AIM2REDUCE')
    parser.add_argument('--output-folder', type=str, default='./Outputs')
    parser.add_argument('--model-anchor', type=str, default='clinic') # treatment clinic
    parser.add_argument('--root-dir', type=str, default='.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    proj_name = args.project_name
    output_folder = args.output_folder
    anchor = args.model_anchor
    ROOT_DIR = args.root_dir

    # TODO: maybe we should only make data-dir and model-dir and info-dir into CLI arguments and remove root-dir? 
    #       for better generalizability for end-users
    data_dir = f'{ROOT_DIR}/Data' #Data
    info_dir= f'{ROOT_DIR}/Infos'
    train_param_dir = f'{ROOT_DIR}/Infos/Train_Data_parameters'
    code_dir = f'{ROOT_DIR}/Codes' # TODO: load config.yaml here (the only time code_dir is used)
    model_dir = f'{ROOT_DIR}/Models' 
    fig_dir = f'{ROOT_DIR}/Figures'
    results_output = f"{anchor}_output.csv" 
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    date_range = pd.date_range(start_date, end_date, freq='d').strftime("%Y%m%d")
    
    thresholds = pd.read_excel(f'{info_dir}/ED_Prediction_Threshold.xlsx')
    
    # treatment anchored files named as eg. AIM2REDUCE_hematology_20241104
    # clinic anchored files named as eg. AIM2REDUCE_hematology_weekly_20241104
    postfix_map = {'treatment': '', 'clinic': 'weekly_'}
    results = defaultdict(list)

    for i, data_pull_date in tqdm(enumerate(date_range)): 

        print(f'**** Processing #{i}: {data_pull_date} *****')
    
        postfix = postfix_map[anchor]
        chemo_file_treatment = f"{data_dir}/{proj_name}_chemo_{postfix}{data_pull_date}.csv"
        diagnosis_file_treatment = f"{data_dir}/{proj_name}_diagnosis_{postfix}{data_pull_date}.csv"
        if not pd.read_csv(chemo_file_treatment).empty and not pd.read_csv(diagnosis_file_treatment).empty:
            ######################### Data Processing ################################
            ##******************** ED **********************##
            # Process and prepare data
            prepared_data_ED = final_process(data_dir, info_dir, train_param_dir, code_dir, proj_name, 'ED_visit', data_pull_date, anchor)
            prepared_data_ED['regimen_GI_IRINO Q3W'] = False
            prepared_data_ED['regimen_GI_PACLITAXEL'] = False
            prepared_data_ED['regimen_GI_CISPFU _ TRAS_MAIN_'] = False
            prepared_data_ED['regimen_GI_GEMCAP'] = False

            ######################### Model Evaluation ################################        
            ##******************** ED **********************##
            # Load pre-defined prediction thresholds
            thresholds_treatment = thresholds[thresholds['Model_anchor']==f'{anchor.title()}-anchored']
            thresholds_treatment = thresholds_treatment.set_index('Labels')['Prediction_threshold']
            ED_treatment_result = get_ED_visit_model_output(prepared_data_ED, thresholds_treatment, model_dir, fig_dir, anchor)

            results[anchor].append(ED_treatment_result)
        else:
            print(f"No Patient {anchor.title()} Data for: {data_pull_date}")
    
    results[anchor] = pd.concat(results[anchor], ignore_index=True, axis=0)
    results[anchor].to_csv(f"{output_folder}/{results_output}")