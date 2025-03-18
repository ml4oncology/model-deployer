"""
Script to turn raw data into features for modelling
"""

import pandas as pd

from data_prep.preprocess.diagnosis import get_demographic_data
from data_prep.preprocess.esas import get_symptoms_data
from data_prep.preprocess.emergency import get_emergency_room_data
from data_prep.preprocess.lab import get_lab_data
from data_prep.preprocess.chemo import get_treatment_data
from loader import Config

def build_features(
    config: Config, 
    data_dir: str, 
    proj_name: str, 
    data_pull_day: str, 
    anchor: str
):
    postfix_map = {'treatment': '', 'clinic': 'weekly_'}
    postfix = postfix_map[anchor]
    biochem_file = f"{data_dir}/{proj_name}_biochemistry_{postfix}{data_pull_day}.csv"
    hema_file = f"{data_dir}/{proj_name}_hematology_{postfix}{data_pull_day}.csv"
    esas_file = f"{data_dir}/{proj_name}_ESAS_{postfix}{data_pull_day}.csv"
    chemo_file = f"{data_dir}/{proj_name}_chemo_{postfix}{data_pull_day}.csv"
    ed_file = f"{data_dir}/{proj_name}_ED_visits_{postfix}{data_pull_day}.csv"
    diagnosis_file = f"{data_dir}/{proj_name}_diagnosis_{postfix}{data_pull_day}.csv"

    # symptoms
    symp = get_symptoms_data(esas_file, anchor)

    # demographics
    demog = get_demographic_data(diagnosis_file, config.cancer_site_list, anchor)

    # treatment
    chemo = get_treatment_data(chemo_file, config.epr_regimens, config.epr2epic_regimen_map, data_pull_day, anchor)

    # laboratory tests
    lab = get_lab_data(hema_file, biochem_file, anchor)

    # emergency room visits
    er_visit = get_emergency_room_data(ed_file, anchor)
    
    return symp, demog, chemo, lab, er_visit
    
