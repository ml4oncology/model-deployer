"""
Main Script 
"""
import argparse

import pandas as pd
from tqdm import tqdm

from data_prep.final_processing import final_process
from model_eval.inference import get_ED_visit_model_output, get_symp_model_output

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
    
    outputs = []
    for i, data_pull_date in tqdm(enumerate(date_range)): 
    
        # TODO: maybe this should be if os.path.exists? Do we really need to check both files?
        chemo_file = f"{data_dir}/{proj_name}_chemo_{data_pull_date}.csv"
        diagnosis_file = f"{data_dir}/{proj_name}_diagnosis_{data_pull_date}.csv"
        if pd.read_csv(chemo_file).empty and pd.read_csv(diagnosis_file).empty:
            print(f"No Patient Data for: {data_pull_date}")
            continue

        ######################### Data Processing ################################
        
        ##******************** ED **********************##
        # Process and prepare data
        prepared_data_ED = final_process(data_dir, info_dir, train_param_dir, code_dir, proj_name, 'ED', data_pull_date)

        ##******************** Symptoms **********************##
        # Process and prepare data
        prepared_data_symp = final_process(data_dir, info_dir, train_param_dir, code_dir, proj_name, 'symp', data_pull_date)
                
        
        ######################### Model Evaluation ################################        
        ##******************** ED **********************##
        ED_visit_result = get_ED_visit_model_output(model_dir, info_dir, prepared_data_ED)
        
        ##******************** Symptoms **********************##
        symp_result = get_symp_model_output(model_dir, info_dir, prepared_data_symp)

        # TODO: figure out do we really want to merge by inner?
        output = ED_visit_result.merge(symp_result, on=['mrn', 'treatment_date'])
        assert len(ED_visit_result) == len(symp_result) == len(output)
        outputs.append(output)

    outputs = pd.concat(outputs, ignore_index=True, axis=0)
    outputs.to_csv(output_path)
