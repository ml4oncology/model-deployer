"""
Module to prepare data for model consumption
"""
from typing import Optional

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from data_prep.constants import lab_cols, lab_change_cols, symp_cols, symp_change_cols  

class Imputer:
    """Impute missing data by mean, mode, or median
    """
    def __init__(self):
        self.impute_cols = {
            'mean': lab_cols.copy() + lab_change_cols.copy(), 
            'most_frequent': symp_cols.copy() + symp_change_cols.copy(),
            'median': []
        }
        self.imputer = {'mean': None, 'most_frequent': None, 'median': None}

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        # loop through the mean, mode, and median imputer
        for strategy, imputer in self.imputer.items():
            cols = self.impute_cols[strategy]
            if len(cols)==0:
                continue
            
            data[cols] = data[cols].apply(pd.to_numeric)
            
            if imputer is None:
                # create the imputer and impute the data
                imputer = SimpleImputer(strategy=strategy) 
                data[cols] = imputer.fit_transform(data[cols])
                self.imputer[strategy] = imputer # save the imputer
            else:
                # use existing imputer to impute the data
                # print('Imputer Working!')
                data[cols] = imputer.transform(data[cols])
        return data
    

def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing data that can be filled heuristically"""
    # fill the following missing data with 0
    col = 'num_prior_ED_visits_within_5_years'
    df[col] = df[col].fillna(0)

    # fill the following missing data with the maximum value
    for col in ['days_since_last_treatment', 'days_since_prev_ED_visit']:
        df[col] = df[col].fillna(df[col].max())

    return df


def encode_regimens(df, regimen_data):
    
    regimens_features = list(regimen_data['Regimen'])
    regimens_renamed = list(regimen_data['Regimen_Rename'])
    
    mask = ~df['regimen'].isin(regimens_features) # Get locations of new regimens not in original regimen list
    df.loc[mask, 'regimen'] = 'regimen_other' # if regimen not in the list, set it to regimen_other
    
    df1_set = set(np.ravel(df['regimen'].values))
    df2_set = set(regimens_features)
    missing_regimens = list(df2_set - df1_set)
    
    df = pd.get_dummies(df, columns=['regimen'], prefix='', prefix_sep='') # one-hot encode
    df[missing_regimens] = 0
        
    rename_map = dict(zip(regimens_features, regimens_renamed))
    df = df.rename(columns=rename_map)
    
    return df

def encode_intent(df):

    intent_list = ['Palliative', 'Neoadjuvant', 'Adjuvant', 'Curative']
    intent_renamed = ['intent_PALLIATIVE', 'intent_NEOADJUVANT', 'intent_ADJUVANT', 'intent_CURATIVE']
    # intent_renamed = ['intent_' + s for s in intent_list]
    
    df1_set = set(np.ravel(df['intent'].values))
    df2_set = set(intent_list)
    missing_intents = list(df2_set - df1_set)
    
    df = pd.get_dummies(df, columns=['intent'], prefix='', prefix_sep='') # one-hot encode
    df[missing_intents] = 0
        
    rename_map = dict(zip(intent_list, intent_renamed))
    df = df.rename(columns=rename_map)
    
    return df
            

class PrepData:
    """Prepare the data for model training"""
    def __init__(self):
        self.imp = Imputer() # imputer
        self.scaler = None # normalizer
        self.clip_thresh = None # outlier clippers

        self.norm_cols = [
            'height',
            'weight',
            'body_surface_area',
            'cycle_number',
            'age',
            'visit_month_sin',
            'visit_month_cos',
            'line_of_therapy',
            'days_since_starting_treatment',
            'days_since_last_treatment',
            'num_prior_EDs_within_5_years',
            'days_since_prev_ED',
        ] + symp_cols + lab_cols + lab_change_cols + symp_change_cols #drug_cols +
        self.clip_cols = [
            'height',
            'weight',
            'body_surface_area',
        ] + lab_cols + lab_change_cols
    
    def transform_data(
        self, 
        data,
        clip: bool = True, 
        impute: bool = True, 
        normalize: bool = True, 
        ohe_kwargs: Optional[dict] = None,
        data_name: Optional[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Transform (one-hot encode, clip, impute, normalize) the data.
        
        Args:
            ohe_kwargs (dict): a mapping of keyword arguments fed into 
                OneHotEncoder.encode
                
        IMPORTANT: always make sure train data is done first before valid
        or test data
        """
        if ohe_kwargs is None: ohe_kwargs = {}
        if data_name is None: data_name = 'the'
        
        if clip:
            # Clip the outliers based on the train data quantiles
            data = self.clip_outliers(data)

        if impute:
            # Impute missing data based on the train data mode/median/mean
            allNaN_col = data.columns[data.isna().all()].tolist()
            for iC in range(len(allNaN_col)):
                data[allNaN_col[iC]][0] = 0
            data = self.imp.impute(data)
            
        if normalize:
            # Scale the data based on the train data distribution
            data = self.normalize_data(data)
            
        return data
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # use only the columns that exist in the data
        norm_cols = [col for col in self.norm_cols if col in data.columns]
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            data[norm_cols] = self.scaler.fit_transform(data[norm_cols])
        else:
            data[norm_cols] = self.scaler.transform(data[norm_cols])
        return data
    
    def clip_outliers(
        self, 
        data: pd.DataFrame, 
        lower_percentile: float = 0.001, 
        upper_percentile: float = 0.999
    ) -> pd.DataFrame:
        """Clip the upper and lower percentiles for the columns indicated below
        """
        # use only the columns that exist in the data
        cols = [col for col in self.clip_cols if col in data.columns]
        
        data[cols] = data[cols].apply(pd.to_numeric)
        
        if self.clip_thresh is None:
            percentiles = [lower_percentile, upper_percentile]
            self.clip_thresh = data[cols].quantile(percentiles)
            
        data[cols] = data[cols].clip(
            lower=self.clip_thresh.loc[lower_percentile], 
            upper=self.clip_thresh.loc[upper_percentile], 
            axis=1
        )
        return data
   
    
"""
Prepare data for symptoms models
"""

def prep_symp_data(df):

    # Regimen Columns to delete 
    reg_cols = ['regimen_GI_FLOT _GASTRIC_', 'regimen_GI_FOLFNALIRI _COMP_', 
                    'regimen_GI_FUFA C3 _GASTRIC_','regimen_GI_FUFA WEEKLY',
                    'regimen_GI_GEM D1_8 _ CAPECIT', 'regimen_GI_PACLI WEEKLY']
    
      
    mask = df[reg_cols].any(axis=1)
    df.loc[mask, 'regimen_other'] = 1
    # # alternative way
    # df['regimen_other'] |= df[reg_cols].any(axis=1)    
    
    # Delete columns
    lab_Cols = ['bicarbonate', 'bicarbonate_is_missing']
    All_Del_Cols = reg_cols + lab_Cols

    df = df.drop(columns=All_Del_Cols)

    # clean column names; replacing space with an underscore
    # col_map = {
    #     'regimen_GI_CISPFU ANAL': 'regimen_GI_CISPFU_ANAL', 
    #     'regimen_GI_CISPFU ESOPHAGEAL': 'regimen_GI_CISPFU_ESOPHAGEAL', 
    #     'regimen_GI_FOLFOX _GASTRIC_': 'regimen_GI_FOLFOX__GASTRIC_',
    #     'regimen_GI_FOLFOX_6 MOD': 'regimen_GI_FOLFOX_6_MOD',
    #     'regimen_GI_FU CIV _ RT': 'regimen_GI_FU_CIV___RT',
    #     'regimen_GI_FUFA C1_4_5 GASTRIC': 'regimen_GI_FUFA_C1_4_5_GASTRIC',
    #     'regimen_GI_FUFA C2 _GASTRIC_': 'regimen_GI_FUFA_C2__GASTRIC_',
    #     'regimen_GI_GEM 40MG_M2 2X_WK': 'regimen_GI_GEM_40MG_M2_2X_WK',
    #     'regimen_GI_GEM 7_WEEKLY': 'regimen_GI_GEM_7_WEEKLY',
    #     'regimen_GI_GEM D1_8': 'regimen_GI_GEM_D1_8',
    #     'regimen_GI_GEM D1_8_15': 'regimen_GI_GEM_D1_8_15',
    #     'regimen_GI_GEMCISP _BILIARY_': 'regimen_GI_GEMCISP__BILIARY_',
    #     'regimen_GI_GEMCISP _PANCREAS_': 'regimen_GI_GEMCISP__PANCREAS_',
    #     'regimen_GI_PACLI_CARBO WEEKX5': 'regimen_GI_PACLI_CARBO_WEEKX5'
    # }
    # df = df.rename(columns=col_map)
    
    df.columns = df.columns.str.replace(' ', '_')
    

    return df