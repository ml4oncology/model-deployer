import yaml
from pathlib import Path

import pandas as pd

from ml_common.util import load_pickle

class Config:
    """Loads configuration files like thresholds, mappings, etc."""
    def __init__(self, info_dir: str):
        self.thresholds = pd.read_excel(f'{info_dir}/ED_Prediction_Threshold.xlsx')

        self.epr2epic_regimen = pd.read_excel(f'{info_dir}/A2R_EPIC_GI_regimen_map.xlsx')
        self.epr2epic_regimen_map = dict(self.epr2epic_regimen[['PROTOCOL_DISPLAY_NAME','Mapped_Name_All']].to_numpy())
        
        self.epr_regimens = pd.read_csv(f'{info_dir}/opis_regimen_list.csv')
        self.epr_regimens.columns = self.epr_regimens.columns.str.lower()

        self.gi_regimens = pd.read_excel(f'{info_dir}/GI_regimen_feature_list.xlsx')


class Model:
    """Loads ML models and pipeline parameters"""
    def __init__(self, model_dir: str, prep_dir: str, anchor: str):
        with open(f'{Path(__file__).parent}/data_prep/config.yaml') as file:
            self.prep_cfg = yaml.safe_load(file)

        # Emergency Department Visit
        if anchor == 'treatment':
            self.prep = load_pickle(prep_dir, 'prep_ED_visit_trt_anchored')
            self.model = load_pickle(model_dir, 'RF_ED_visit_trt_anchored')
        elif anchor == 'clinic':
            self.prep = load_pickle(prep_dir, 'prep_ED_visit_clinic_anchored')
            self.model = load_pickle(model_dir, 'XGB_ED_visit_clinic_anchored')