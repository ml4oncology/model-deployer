import argparse
import os
import warnings

import pandas as pd
from deployer.data_prep.constants import DAILY_POSTFIX_MAP
from deployer.data_prep.final_processing import final_process
from deployer.loader import Config, Model
from deployer.model_eval.inference import get_ED_visit_model_output
from tqdm import tqdm

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="20240904")
    parser.add_argument("--end-date", type=str, default="20250101")
    parser.add_argument("--project-name", type=str, default="AIM2REDUCE")
    parser.add_argument("--model-anchor", type=str, choices=["clinic", "treatment"], default="clinic")

    parser.add_argument("--output-dir", type=str, default="./Outputs")
    parser.add_argument("--data-dir", type=str, default="./Data")
    parser.add_argument("--info-dir", type=str, default="./Infos")
    parser.add_argument("--model-dir", type=str, default="./Models")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    proj_name = args.project_name
    anchor = args.model_anchor
    output_dir = args.output_dir
    data_dir = args.data_dir
    info_dir = args.info_dir
    model_dir = args.model_dir

    results_output = f"{anchor}_output.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = Config(info_dir=info_dir)
    model = Model(model_dir=model_dir, prep_dir=f"{info_dir}/Train_Data_parameters", anchor=anchor, name="ED_visit")

    # Get pre-defined prediction thresholds
    thresholds = config.thresholds
    thresholds = thresholds.query(f'Model_anchor == "{anchor.title()}-anchored"')
    thresholds.columns = thresholds.columns.str.lower()

    date_range = pd.date_range(start_date, end_date, freq="d").strftime("%Y%m%d")
    results, input_data = [], []
    for i, data_pull_date in tqdm(enumerate(date_range)):
        print(f"**** Processing #{i}: {data_pull_date} *****")

        postfix = DAILY_POSTFIX_MAP[anchor]
        chemo_file = f"{data_dir}/{proj_name}_chemo_{postfix}{data_pull_date}.csv"
        diagnosis_file = f"{data_dir}/{proj_name}_diagnosis_{postfix}{data_pull_date}.csv"
        if not pd.read_csv(chemo_file).empty and not pd.read_csv(diagnosis_file).empty:
            ######################### Data Processing ################################
            ##******************** ED **********************##
            # Process and prepare data
            prepared_data = final_process(config, model, data_dir, proj_name, data_pull_date)

            ######################### Model Evaluation ################################
            ##******************** ED **********************##
            ED_result = get_ED_visit_model_output(model, prepared_data, thresholds, f"{output_dir}/Figures")

            input_data.append(prepared_data)
            results.append(ED_result)
        else:
            print(f"No Patient {anchor.title()} Data for: {data_pull_date}")

    res = pd.concat(results, ignore_index=True, axis=0)
    res.to_csv(f"{output_dir}/{results_output}", index=False)

    inp = pd.concat(input_data, ignore_index=True, axis=0)
    inp.to_parquet(f"{output_dir}/input_{data_pull_date}_{anchor}.parquet")
