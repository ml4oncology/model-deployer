"""
Script to turn raw data into features for modelling
"""
# from pathlib import Path
# import argparse
# import os
# import sys
# ROOT_DIR = Path(__file__).parent.parent.as_posix()
# sys.path.append(ROOT_DIR)

import pandas as pd

from data_prep.src.preprocess.cancer_registry import get_demographic_data
from data_prep.src.preprocess.dart import get_symptoms_data
from data_prep.src.preprocess.emergency import get_emergency_room_data
from data_prep.src.preprocess.lab import get_lab_data
from data_prep.src.preprocess.opis import get_treatment_data
# from src.util import load_included_drugs, load_included_regimens

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-dir', type=str, default=f'{ROOT_DIR}/data')
#     args = parser.parse_args()
#     return args

def  build_features(data_root_dir, info_data_dir, proj_name, dataPull_day):
    # args = parse_args()
    # data_dir = args.data_dir
    # if not os.path.exists(f'{data_dir}/interim'): os.makedirs(f'{data_dir}/interim')

    # included_drugs = load_included_drugs(data_dir=f'{data_dir}/external')
    # included_regimens = load_included_regimens(data_dir=f'{data_dir}/external')
    # mrn_map = pd.read_csv(f'{data_dir}/external/MRN_map.csv')
    # mrn_map = mrn_map.set_index('RESEARCH_ID')['PATIENT_MRN'].to_dict()
    
    biochem_file = f"{data_root_dir}/{proj_name+'_biochemistry_'+dataPull_day}.csv"
    hema_file = f"{data_root_dir}/{proj_name+'_hematology_'+dataPull_day}.csv"
    esas_file = f"{data_root_dir}/{proj_name+'_ESAS_'+dataPull_day}.csv"
    chemo_file = f"{data_root_dir}/{proj_name+'_chemo_'+dataPull_day}.csv"
    ed_file = f"{data_root_dir}/{proj_name+'_ED_visits_'+dataPull_day}.csv"
    diagnosis_file = f"{data_root_dir}/{proj_name+'_diagnosis_'+dataPull_day}.csv"
    
    A2R_EPIC_GI_regimen_map = pd.read_excel(info_data_dir + '/A2R_EPIC_GI_regimen_map.xlsx')
    included_regimens = pd.read_csv(info_data_dir + '/opis_regimen_list.csv')

    # symptoms
    dart = get_symptoms_data(esas_file)

    # demographics
    canc_reg = get_demographic_data(diagnosis_file, info_data_dir)

    # treatment
    opis = get_treatment_data(chemo_file, included_regimens, A2R_EPIC_GI_regimen_map, dataPull_day)

    # laboratory tests
    lab = get_lab_data(hema_file, biochem_file)

    # emergency room visits
    er_visit = get_emergency_room_data(ed_file)
    
    return dart, canc_reg, opis, lab, er_visit
    
# if __name__ == '__main__':
#     build_features()