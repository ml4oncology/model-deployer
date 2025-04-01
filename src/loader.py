from pathlib import Path

import pandas as pd
import yaml
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

        self.cancer_sites = pd.read_excel(f'{info_dir}/Cancer_Site_List.xlsx')
        self.cancer_site_list = self.cancer_sites['Cancer_Site'].tolist()


class Model:
    """Loads ML models and pipeline parameters

    #TODO: support multiple models / targets
    """
    # TODO: automatically store this info in ml-common.prep
    FILL_VALS = {
        'treatment': {'days_since_last_treatment': 4746, 'days_since_prev_ED_visit': 1822}, 
        'clinic': {'days_since_last_treatment': 28, 'days_since_prev_ED_visit': 1821}
    }
    
    def __init__(self, model_dir: str, prep_dir: str, anchor: str):
        self.anchor = anchor

        config_path = f'{Path(__file__).parent}/data_prep/config.yaml'
        with open(config_path) as file:
            self.prep_cfg = yaml.safe_load(file)

        # Emergency Department Visit
        if self.anchor == 'treatment':
            self.prep = load_pickle(prep_dir, 'prep_ED_visit_trt_anchored')
            self.model = load_pickle(model_dir, 'RF_ED_visit_trt_anchored')
        elif self.anchor == 'clinic':
            self.prep = load_pickle(prep_dir, 'prep_ED_visit_clinic_anchored')
            self.model = load_pickle(model_dir, 'XGB_ED_visit_clinic_anchored')
        self.model_features = self.model[0].feature_names_in_

        # column ordering needs to match
        # TODO: use the scaler, imputer, etc's pre-existing columns in ml-common.prep
        self.prep.norm_cols = self.prep.scaler.feature_names_in_
        self.prep.imp.impute_cols['mean'] = self.prep.imp.imputer['mean'].feature_names_in_
        self.prep.imp.impute_cols['most_frequent'] = self.prep.imp.imputer['most_frequent'].feature_names_in_