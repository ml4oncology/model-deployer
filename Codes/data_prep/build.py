"""
Script to turn raw data into features for modelling
"""

import pandas as pd

from data_prep.preprocess.cancer_registry import get_demographic_data
from data_prep.preprocess.dart import get_symptoms_data
from data_prep.preprocess.emergency import get_emergency_room_data
from data_prep.preprocess.lab import get_lab_data
from data_prep.preprocess.opis import get_treatment_data


def build_features(data_dir, info_dir, proj_name, data_pull_day, anchor):
    postfix_map = {'treatment': '', 'clinic': 'weekly_'}
    postfix = postfix_map[anchor]
    biochem_file = f"{data_dir}/{proj_name}_biochemistry_{postfix}{data_pull_day}.csv"
    hema_file = f"{data_dir}/{proj_name}_hematology_{postfix}{data_pull_day}.csv"
    esas_file = f"{data_dir}/{proj_name}_ESAS_{postfix}{data_pull_day}.csv"
    chemo_file = f"{data_dir}/{proj_name}_chemo_{postfix}{data_pull_day}.csv"
    ed_file = f"{data_dir}/{proj_name}_ED_visits_{postfix}{data_pull_day}.csv"
    diagnosis_file = f"{data_dir}/{proj_name}_diagnosis_{postfix}{data_pull_day}.csv"
    
    EPR_to_EPIC_regimen_map = pd.read_excel(f'{info_dir}/A2R_EPIC_GI_regimen_map.xlsx')
    EPR_to_EPIC_regimen_map = dict(EPR_to_EPIC_regimen_map[['PROTOCOL_DISPLAY_NAME','Mapped_Name_All']].to_numpy())
    EPR_regimens = pd.read_csv(f'{info_dir}/opis_regimen_list.csv')
    EPR_regimens.columns = EPR_regimens.columns.str.lower()

    # symptoms
    dart = get_symptoms_data(esas_file, anchor)

    # demographics
    canc_reg = get_demographic_data(diagnosis_file, info_dir, anchor)

    # treatment
    opis = get_treatment_data(chemo_file, EPR_regimens, EPR_to_EPIC_regimen_map, data_pull_day, anchor)

    # laboratory tests
    lab = get_lab_data(hema_file, biochem_file, anchor)

    # emergency room visits
    er_visit = get_emergency_room_data(ed_file, anchor)
    
    return dart, canc_reg, opis, lab, er_visit
    
