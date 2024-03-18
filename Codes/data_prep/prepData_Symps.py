"""
Module to prepare data for symptoms models
"""

# import pandas as pd
import numpy as np

def prep_symp_data(df):

    # Regimen Columns to delete 
    Reg_Cols = ['regimen_GI_FLOT _GASTRIC_', 'regimen_GI_FOLFNALIRI _COMP_', 
                    'regimen_GI_FUFA C3 _GASTRIC_','regimen_GI_FUFA WEEKLY',
                    'regimen_GI_GEM D1_8 _ CAPECIT', 'regimen_GI_PACLI WEEKLY']
    
    # Transfer any '1' in the regimen columns to be deleted to 'regimen_other' column
    for iR in range(len(Reg_Cols)):
        # df['regimen_other'] = np.where(df[Reg_Cols[iR]] == 1, max(df['regimen_other'],1), df['regimen_other'])
        
        cols_with_one = np.where(df.index[df[Reg_Cols[iR]]].tolist())[0]
        df['regimen_other'][cols_with_one] = 1
        
    
    # Delete columns
    lab_Cols = ['bicarbonate', 'bicarbonate_is_missing']
    All_Del_Cols = Reg_Cols + lab_Cols

    df = df.drop(columns=All_Del_Cols)

    # clean column names
    col_map = {
        'regimen_GI_CISPFU ANAL': 'regimen_GI_CISPFU_ANAL', 
        'regimen_GI_CISPFU ESOPHAGEAL': 'regimen_GI_CISPFU_ESOPHAGEAL', 
        'regimen_GI_FOLFOX _GASTRIC_': 'regimen_GI_FOLFOX__GASTRIC_',
        'regimen_GI_FOLFOX_6 MOD': 'regimen_GI_FOLFOX_6_MOD',
        'regimen_GI_FU CIV _ RT': 'regimen_GI_FU_CIV___RT',
        'regimen_GI_FUFA C1_4_5 GASTRIC': 'regimen_GI_FUFA_C1_4_5_GASTRIC',
        'regimen_GI_FUFA C2 _GASTRIC_': 'regimen_GI_FUFA_C2__GASTRIC_',
        'regimen_GI_GEM 40MG_M2 2X_WK': 'regimen_GI_GEM_40MG_M2_2X_WK',
        'regimen_GI_GEM 7_WEEKLY': 'regimen_GI_GEM_7_WEEKLY',
        'regimen_GI_GEM D1_8': 'regimen_GI_GEM_D1_8',
        'regimen_GI_GEM D1_8_15': 'regimen_GI_GEM_D1_8_15',
        'regimen_GI_GEMCISP _BILIARY_': 'regimen_GI_GEMCISP__BILIARY_',
        'regimen_GI_GEMCISP _PANCREAS_': 'regimen_GI_GEMCISP__PANCREAS_',
        'regimen_GI_PACLI_CARBO WEEKX5': 'regimen_GI_PACLI_CARBO_WEEKX5'
    }
    df = df.rename(columns=col_map)
    

    return df