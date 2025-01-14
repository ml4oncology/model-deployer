"""
Process ED visits:
    1. Add labels
    2. Get ED visits within lookahead window
"""

import numpy as np
import pandas as pd
from datetime import timedelta 

from data_prep.preprocess.emergency import get_emergency_room_data


def merge_ed_pred_label(ed_file, ed_date_start, ed_date_end, df, date_column):
    
    # emergency room visits
    ed_visit = get_emergency_room_data(ed_file,'')
    ed_visit = ed_visit[(ed_visit['event_date'] >= ed_date_start) & (ed_visit['event_date']  <= ed_date_end)]

    #Merge ED visit dates and true labels to Model prediction file
    ed_visits_wlocs = ed_visits_lookahead(df, ed_visit, date_column) # returns list of indices and visit dates

    # initialize columns in the dataframe    
    df_ED_visit = df.assign(
        ed_visit_dates_30days=np.nan,
        ed_visit_labels_30days=0
    )
    
    # Convert ed_visits_wlocs to a DataFrame for efficient vectorized processing
    ed_visits_df = pd.DataFrame(ed_visits_wlocs, columns=['index', 'visit_date'])
    
    # Vectorized update of ED visit dates and labels
    df_ED_visit.loc[ed_visits_df['index'], ['ed_visit_dates_30days', 'ed_visit_labels_30days']] = ed_visits_df['visit_date'], 1
       
    # Calculate days between treatment and ED visit
    df_ED_visit['ed_visit_dates_30days'] = pd.to_datetime(df_ED_visit['ed_visit_dates_30days'])
    df_ED_visit['trt_to_edvisit'] = (df_ED_visit['ed_visit_dates_30days'] - df_ED_visit[date_column]).dt.days
    
    # Remove treatments with same day ED visits, 
    # i.e. Difference between treatment date and ED visit = 0 
    df_ED_visit = df_ED_visit[df_ED_visit['trt_to_edvisit'] != 0]
    
    return df_ED_visit


def ed_visits_lookahead(df, ed_visit, date_column):
    
    result = []
    
    for mrn, chemo_group in df.groupby('mrn'):
        event_group = ed_visit.query('mrn == @mrn')
        adm_dates = event_group['event_date']
    
        for chemo_idx, visit_date in chemo_group[date_column].items(): #treatment_date, clinic_date
            # get target - closest event from visit date
            lookahead_date = pd.to_datetime(pd.to_datetime(visit_date).date() + timedelta(days=30))
            
            mask = (adm_dates >= visit_date) & (adm_dates <= lookahead_date)
            if not mask.any():
                continue
    
            # assert(sum(arrival_dates == arrival_dates[mask].min()) == 1)
            tmp = event_group.loc[mask].iloc[0]
            adm_date = tmp['event_date']
            result.append([chemo_idx, adm_date])
            
    return result
