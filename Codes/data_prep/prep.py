"""
Module to prepare data for model consumption
"""
import pandas as pd


def encode_regimens(df, regimen_data):
    regimen_map = dict(regimen_data[['Regimen', 'Regimen_Rename']].to_numpy())
    df['regimen'] = df['regimen'].map(regimen_map).fillna('regimen_other')

    missing_regimens = list(set(regimen_map.values()) - set(df['regimen']))
    df[missing_regimens] = 0

    df = pd.get_dummies(df, columns=['regimen'], prefix='', prefix_sep='')

    df['regimen_GI_IRINO Q3W'] = False
    df['regimen_GI_PACLITAXEL'] = False
    df['regimen_GI_CISPFU _ TRAS_MAIN_'] = False
    df['regimen_GI_GEMCAP'] = False
    
    return df

def encode_intent(df):
    df = pd.get_dummies(df, columns=['intent'])

    # TODO: centralize the creation of all missing columns
    for intent in ['PALLIATIVE', 'NEOADJUVANT', 'ADJUVANT', 'CURATIVE']:
        if f'intent_{intent}' not in df.columns:
            df[f'intent_{intent}'] = 0
            
    return df
            

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