"""
Main Script 
for Model Inference

"""

import argparse
import os

import pandas as pd

from model_eval.get_ED_visit_label import merge_ed_pred_label
from model_eval.get_patients_trt_complete import get_patients_with_completed_trt
from model_eval.save_model_output import save_and_display_model_results
from ml_common.eval import get_model_performance

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='20240904') 
    parser.add_argument('--end-date', type=str, default='20241130') 
    parser.add_argument('--ED-visit-end-date', type=str, default='20241230') 
    parser.add_argument('--monthly-pull-date', type=str, default='20250103')
    parser.add_argument('--output-folder', type=str, default='./Outputs')
    parser.add_argument('--model-anchor', type=str, default='clinic') # treatment clinic 
    parser.add_argument('--project-name', type=str, default='AIM2REDUCE')
    parser.add_argument('--root-dir', type=str, default='.') 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    ED_visit_end_date = args.ED_visit_end_date
    monthly_pull_date = args.monthly_pull_date
    output_folder = args.output_folder
    anchor = args.model_anchor
    proj_name = args.project_name
    ROOT_DIR = args.root_dir

    # TODO: maybe we should only make data-dir and model-dir and info-dir into CLI arguments and remove root-dir? 
    #       for better generalizability for end-users
    data_dir = f'{ROOT_DIR}/Data' #Data
    info_dir= f'{ROOT_DIR}/Infos'
    code_dir = f'{ROOT_DIR}/Codes' # TODO: load config.yaml here (the only time code_dir is used)
    pred_file = f"{anchor}_output.csv"
    perf_file = f"{anchor}_model_perf.csv"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    thresholds = pd.read_excel(f'{info_dir}/ED_Prediction_Threshold.xlsx')
    postfix_map = {'treatment': 'monthly_', 'clinic': 'weekly_monthly_'}
    date_col_map = {'treatment': 'treatment_date', 'clinic': 'clinic_date'}
    
    model_results = []
    
    postfix = postfix_map[anchor]
    date_col = date_col_map[anchor]
    # pred_file = pred_file_map[anchor]
    chemo_file = f"{data_dir}/{proj_name}_chemo_{postfix}{monthly_pull_date}.csv"
    ED_visits_file = f"{data_dir}/{proj_name}_ED_visits_{postfix}{monthly_pull_date}.csv"
    
    ############################ Analyze data #################################

    # Model Prediction file
    df = pd.read_csv(f'{output_folder}/{pred_file}', parse_dates=[date_col])
    df = df[df[date_col].between(start_date, end_date)]

    #Merge ED visit dates and true labels to Model prediction file
    df_ED_visit = merge_ed_pred_label(ED_visits_file, start_date, ED_visit_end_date, df, date_col)
    
    # Sort patients only with completed treatments during the month
    df_ED_visit = get_patients_with_completed_trt(info_dir, chemo_file, start_date, end_date, df_ED_visit, anchor)
    
    ######################  Check model Performance ###########################
    
    print(f"=========== Anchored on {date_col} ====================")
    
    event_col = 'ed_visit_dates_30days'
    label_col = 'ed_visit_labels_30days'
    pred_col = 'ed_pred_prob'
    
    # Load pre-defined prediction thresholds
    thresholds_treatment = thresholds[thresholds['Model_anchor']==f'{anchor.title()}-anchored']
    
    for idx, row in thresholds_treatment.iterrows(): 
          assert row['Labels'] == 'ED_visit'
          performance_metrics = get_model_performance(df_ED_visit, label_col, pred_col, 
                                                      pred_thresh=row['Prediction_threshold'], 
                                                      main_date_col=date_col, event_date_col=event_col)
          performance_metrics['Anchor'] = anchor
          performance_metrics['Alarm rate'] = row['Alarm_rate']     
          model_results.append(performance_metrics)
            
    ######################  Model Performance using seismometer ###########################
    
    from seismometer.data.performance import calculate_bin_stats, calculate_eval_ci
    from seismometer.plot.mpl.binary_classifier import evaluation
    
    y_true = df_ED_visit[label_col].to_numpy()
    y_pred = df_ED_visit[pred_col].to_numpy()
    
    stats = calculate_bin_stats(y_true, y_pred)
    ci_data = calculate_eval_ci(stats,y_true,y_pred)
    fig = evaluation(
        stats=stats,
        ci_data=ci_data,
        truth=y_true,
        output=y_pred
    )
    
    ######################  Save Output ###########################
    # Save and display model performance results
    save_and_display_model_results(model_results, output_folder, perf_file)
                 