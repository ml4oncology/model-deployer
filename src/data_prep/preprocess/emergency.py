"""
Module to preprocess emergency department visit (old pull) / emergency room data (new pull)
"""

import pandas as pd

from .data_prep.constants import DROP_CLINIC_COLUMNS


###############################################################################
# ER (Emergency Room - EPIC)
###############################################################################
def get_emergency_room_data(ed_data_file, anchor) -> pd.DataFrame:
    df = pd.read_csv(ed_data_file)
    if anchor == 'clinic':
        df = df.drop(columns=DROP_CLINIC_COLUMNS)
    df = clean_emergency_data(df)
    df = process_emergency_room_data(df)
    return df


def process_emergency_room_data(ER: pd.DataFrame) -> pd.DataFrame:

    # order by patient and date
    ER = ER.sort_values(by=['mrn', 'event_date'])

    # remove partially duplicate entries
    # i.e. ER visits occuring within 30 minutes, 80% of the entries duplicate
    ER = remove_partially_duplicate_entries(ER)

    return ER


def remove_partially_duplicate_entries(ER: pd.DataFrame) -> pd.DataFrame:
    exclude = []
    for mrn, group in ER.groupby('mrn'):
        # if event date occurs within 30 minutes of each other, it's most likely duplicate entries
        # even though they are given different visit numbers
        time_since_next_visit = group['event_date'].shift(-1) - group['event_date']
        mask = time_since_next_visit < pd.Timedelta(seconds=60*30)
        if mask.any(): 
            # ensure the rows are really duplicates (80% of the entries are the same)
            # NOTE: able to assess multiple duplicate rows
            tmp1 = group[mask].fillna('NULL').reset_index(drop=True)
            tmp2 = group[mask.shift(1).fillna(False)].fillna('NULL').reset_index(drop=True)
            assert all((tmp1 == tmp2).mean(axis=1) > 0.80)

            # we take the most recent entry, excluding the previous entries
            exclude += group.index[mask].tolist()

    mask = ~ER.index.isin(exclude)
    ER = ER[mask]
    return ER

###############################################################################
# Helpers
###############################################################################
def clean_emergency_data(df: pd.DataFrame) -> pd.DataFrame:
    # clean column names
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'patient_id': 'mrn', 'emergency_admission_date': 'event_date'})
    df['event_date'] = pd.to_datetime(df['event_date'])
    return df
