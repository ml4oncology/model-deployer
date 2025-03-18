"""
Module to combine features
"""
import logging

import pandas as pd
import numpy as np

from make_clinical_dataset.combine import (
    add_engineered_features, 
    combine_demographic_to_main_data,
    combine_event_to_main_data, 
    combine_treatment_to_main_data
)
from ml_common.util import logger
from ml_common.anchor import merge_closest_measurements

logger.setLevel(logging.WARNING)


def combine_features(cfg: dict, feats: tuple[pd.DataFrame], data_pull_date: str, anchor: str):
    """Combine the features into one unified dataset
    """
    sym, dmg, trt, lab, erv = feats

    if anchor == 'treatment':
        df = trt
        df['assessment_date'] = df['treatment_date']
    elif anchor == 'clinic':
        df = pd.DataFrame({'mrn': trt['mrn'].unique(), 'clinic_date': data_pull_date})
        df['assessment_date'] = pd.to_datetime(df['clinic_date'])
    else:
        raise ValueError(f'Sorry, aligning features on {anchor} is not supported yet')

    if anchor != 'treatment':
        df = combine_treatment_to_main_data(
            df, trt, 'assessment_date', time_window=cfg['trt_lookback_window'], parallelize=False
        )
    
    df = combine_demographic_to_main_data(df, dmg, 'assessment_date')
    df = merge_closest_measurements(df, sym, 'assessment_date', 'survey_date', time_window=cfg['symp_lookback_window'])
    df = merge_closest_measurements(df, lab, 'assessment_date', 'obs_date', time_window=cfg['lab_lookback_window'])
    df = combine_event_to_main_data(
        df, erv, 'assessment_date', 'event_date', event_name='ED_visit', lookback_window=cfg['ed_visit_lookback_window'], 
        parallelize=False
    )
    df = add_engineered_features(df, 'assessment_date')

    # Add missing feature
    df['hematocrit'] = np.nan
    
    return df
    