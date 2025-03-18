"""
Module to prepare data for model consumption
"""
import pandas as pd


def encode_regimens(df, regimen_data):
    regimen_map = dict(regimen_data[['Regimen', 'Regimen_Rename']].to_numpy())
    df['regimen'] = df['regimen'].map(regimen_map).fillna('regimen_other')
    df = pd.get_dummies(df, columns=['regimen'], prefix='', prefix_sep='')
    return df


def encode_intent(df):
    df = pd.get_dummies(df, columns=['intent'])
    return df
            

def prep_symp_data(df):
    """Prepare data for symptoms models
    """
    # reassign these regimens as other
    reg_cols = [
        'regimen_GI_FLOT _GASTRIC_', 'regimen_GI_FOLFNALIRI _COMP_', 
        'regimen_GI_FUFA C3 _GASTRIC_','regimen_GI_FUFA WEEKLY',
        'regimen_GI_GEM D1_8 _ CAPECIT', 'regimen_GI_PACLI WEEKLY'
    ]
    mask = df[reg_cols].any(axis=1)
    df.loc[mask, 'regimen_other'] = True
    df.columns = df.columns.str.replace(' ', '_')
    return df