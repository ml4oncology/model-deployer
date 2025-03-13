import argparse
import os

import pandas as pd

from data_prep.preprocess.chemo import get_treatment_data
from data_prep.preprocess.emergency import get_emergency_room_data
from make_clinical_dataset.label import get_ED_labels
from loader import Config

from ml_common.eval import get_model_performance

import warnings
warnings.filterwarnings("ignore")


def get_patients_with_completed_trt(config, chemo_file, start_date, end_date, df, anchor):    
    # Get treatment data
    data_pull_day = None
    chemo_data = get_treatment_data(chemo_file, config.epr_regimens, config.epr2epic_regimen_map, data_pull_day, anchor)
    
    # Filter chemo_data by date range
    treatment_date_mask = (pd.to_datetime(chemo_data['treatment_date']) >= pd.to_datetime(start_date)) & \
                          (pd.to_datetime(chemo_data['treatment_date']) <= pd.to_datetime(end_date))
    filtered_chemo_data = chemo_data.loc[treatment_date_mask]
    
    # Only keep patients who completed treatments
    mask = df['mrn'].isin(filtered_chemo_data['mrn'])
    df = df[mask]
            
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='20240904') 
    parser.add_argument('--end-date', type=str, default='20241130') 
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
    monthly_pull_date = args.monthly_pull_date
    output_folder = args.output_folder
    anchor = args.model_anchor
    proj_name = args.project_name
    ROOT_DIR = args.root_dir

    # TODO: maybe we should only make data-dir and model-dir and info-dir into CLI arguments and remove root-dir? 
    #       for better generalizability for end-users
    data_dir = f'{ROOT_DIR}/Data' #Data
    pred_file = f"{anchor}_output.csv"
    perf_file = f"{anchor}_model_perf.csv"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    postfix_map = {'treatment': 'monthly_', 'clinic': 'weekly_monthly_'}
    date_col_map = {'treatment': 'treatment_date', 'clinic': 'clinic_date'}
    
    postfix = postfix_map[anchor]
    date_col = date_col_map[anchor]
    chemo_file = f"{data_dir}/{proj_name}_chemo_{postfix}{monthly_pull_date}.csv"
    ED_visits_file = f"{data_dir}/{proj_name}_ED_visits_{postfix}{monthly_pull_date}.csv"

    config = Config(info_dir=f'{ROOT_DIR}/Infos')
    
    ############################ Analyze data #################################

    # Model Prediction file
    df = pd.read_csv(f'{output_folder}/{pred_file}', parse_dates=[date_col])
    df = df[df[date_col].between(start_date, end_date)]
    df['assessment_date'] = df[date_col]

    # Merge ED visit dates and true labels to Model prediction file
    ed_visit = get_emergency_room_data(ED_visits_file, anchor='')
    df_ED_visit = get_ED_labels(df, ed_visit, lookahead_window=31)
    # filter out cases where ED visit occurred on the same day
    df_ED_visit = df_ED_visit[(df_ED_visit['target_ED_date'] - df_ED_visit['assessment_date']).dt.days != 0]
    
    # Sort patients only with completed treatments during the month
    df_ED_visit = get_patients_with_completed_trt(config, chemo_file, start_date, end_date, df_ED_visit, anchor)
    
    ######################  Check model Performance ###########################
    
    print(f"=========== Anchored on {date_col} ====================")
    
    event_col = 'target_ED_date'
    label_col = 'target_ED_31d'
    pred_col = 'ed_pred_prob'
    
    # Get pre-defined prediction thresholds
    thresholds = config.thresholds
    thresholds = thresholds.query(f'Model_anchor == "{anchor.title()}-anchored"')

    model_results = []
    for idx, row in thresholds.iterrows(): 
        assert row['Labels'] == 'ED_visit'
        performance_metrics = get_model_performance(
            df_ED_visit, label_col, pred_col, 
                pred_thresh=row['Prediction_threshold'], 
                main_date_col=date_col, event_date_col=event_col
            )
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
    pd.DataFrame(model_results).to_csv(f"{output_folder}/{perf_file}", index=False)
    print(f"Performance metrics saved to {perf_file}.")
                 