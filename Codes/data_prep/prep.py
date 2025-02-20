"""
Module to prepare data for model consumption
"""
from typing import Optional

import numpy as np
import pandas as pd

from ml_common.prep import PrepData


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
    df = pd.get_dummies(df, columns=['intent'])

    # TODO: centralize the creation of all missing columns
    for intent in ['PALLIATIVE', 'NEOADJUVANT', 'ADJUVANT', 'CURATIVE']:
        if f'intent_{intent}' not in df.columns:
            df[f'intent_{intent}'] = 0
            
    return df
            

class PrepData(PrepData):
    """Prepare the data for model training"""
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
   

def prep_symp_data(df):
    """Prepare data for symptoms models
    """
    # lab columns to delete
    lab_cols = ['bicarbonate', 'bicarbonate_is_missing']

    # regimen columns to delete 
    reg_cols = [
        'regimen_GI_FLOT _GASTRIC_', 'regimen_GI_FOLFNALIRI _COMP_', 
        'regimen_GI_FUFA C3 _GASTRIC_','regimen_GI_FUFA WEEKLY',
        'regimen_GI_GEM D1_8 _ CAPECIT', 'regimen_GI_PACLI WEEKLY'
    ]
    
    # reassign those regimens as other
    mask = df[reg_cols].any(axis=1)
    df.loc[mask, 'regimen_other'] = True
    # alternative way
    # df['regimen_other'] |= df[reg_cols].any(axis=1)    

    df = df.drop(columns=reg_cols+lab_cols)
    df.columns = df.columns.str.replace(' ', '_')
    return df