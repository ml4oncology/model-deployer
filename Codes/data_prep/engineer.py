"""
Module to engineer features
"""

from tqdm import tqdm
import numpy as np
import pandas as pd

from data_prep.constants_postprocess import lab_cols, lab_change_cols, symp_cols, symp_change_cols

###############################################################################
# Engineering Features
###############################################################################
def get_change_since_prev_session(df: pd.DataFrame) -> pd.DataFrame:
    """Get change since last session"""
    cols = symp_cols + lab_cols
    change_cols = symp_change_cols + lab_change_cols
    result = []
    for mrn, group in tqdm(df.groupby('mrn')):
        group[cols] = group[cols].apply(pd.to_numeric) # convert all columns of DataFrame # pd.to_numeric(s, errors='coerce')
        change = group[cols] - group[cols].shift()
        result.append(change.reset_index().to_numpy())
    result = np.concatenate(result)

    result = pd.DataFrame(result, columns=['index']+change_cols).set_index('index')
    result.index = result.index.astype(int)
    df = pd.concat([df, result], axis=1)

    return df

def get_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    
    target_cols = symp_cols + lab_cols + lab_change_cols + symp_change_cols
    
    for iT in range(len(target_cols)):
        df[target_cols[iT] + '_is_missing'] = df[target_cols[iT]].isnull()
        
    return df
