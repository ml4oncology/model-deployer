"""
Module to preprocess emergency department visit data
"""

import pandas as pd
from make_clinical_dataset.epic.preprocess.acu import get_epic_arrival_dates


def get_emergency_room_data(ed_data_file) -> pd.DataFrame:
    df = get_epic_arrival_dates(ed_data_file)
    df = df.rename(columns={"ED_arrival_date": "event_date"})
    return df
