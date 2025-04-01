import argparse
import os
import warnings

import pandas as pd
from data_prep.final_processing import final_process
from loader import Config, Model
from model_eval.inference import get_ED_visit_model_output
from tqdm import tqdm

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='20240904') 
    parser.add_argument('--end-date', type=str, default='20250101') 
    parser.add_argument('--project-name', type=str, default='AIM2REDUCE')
    parser.add_argument('--output-folder', type=str, default='./Outputs')
    parser.add_argument('--model-anchor', type=str, choices=['clinic', 'treatment'], default='clinic')
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

    data_dir = f'{ROOT_DIR}/Data'
    fig_dir = f'{ROOT_DIR}/Figures'
    results_output = f"{anchor}_output.csv" 
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    config = Config(info_dir=f'{ROOT_DIR}/Infos')
    model = Model(
        model_dir=f'{ROOT_DIR}/Models', 
        prep_dir=f'{ROOT_DIR}/Infos/Train_Data_parameters', 
        anchor=anchor
    )

    # Get pre-defined prediction thresholds
    thresholds = config.thresholds
    thresholds = thresholds.query(f'Model_anchor == "{anchor.title()}-anchored"')
    thresholds = thresholds.set_index('Labels')['Prediction_threshold']
    
    # treatment anchored files named as eg. AIM2REDUCE_hematology_20241104
    # clinic anchored files named as eg. AIM2REDUCE_hematology_weekly_20241104
    postfix_map = {'treatment': '', 'clinic': 'weekly_'}

    date_range = pd.date_range(start_date, end_date, freq='d').strftime("%Y%m%d")
    results, input_data = [], []
    for i, data_pull_date in tqdm(enumerate(date_range)): 

        print(f'**** Processing #{i}: {data_pull_date} *****')
    
        postfix = postfix_map[anchor]
        chemo_file = f"{data_dir}/{proj_name}_chemo_{postfix}{data_pull_date}.csv"
        diagnosis_file = f"{data_dir}/{proj_name}_diagnosis_{postfix}{data_pull_date}.csv"
        if not pd.read_csv(chemo_file).empty and not pd.read_csv(diagnosis_file).empty:
            ######################### Data Processing ################################
            ##******************** ED **********************##
            # Process and prepare data
            prepared_data_ED = final_process(config, model, data_dir, proj_name, 'ED_visit', data_pull_date, anchor)

            ######################### Model Evaluation ################################        
            ##******************** ED **********************##
            ED_result = get_ED_visit_model_output(model, prepared_data_ED, thresholds, fig_dir, anchor)

            input_data.append(prepared_data_ED)
            results.append(ED_result)
        else:
            print(f"No Patient {anchor.title()} Data for: {data_pull_date}")
    
    results = pd.concat(results, ignore_index=True, axis=0)
    results.to_csv(f"{output_folder}/{results_output}", index=False)

    input_data = pd.concat(input_data, ignore_index=True, axis=0)
    input_data.to_parquet(f"{output_folder}/input_{data_pull_date}_{anchor}.parquet")