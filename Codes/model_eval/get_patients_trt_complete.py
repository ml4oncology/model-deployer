"""
Get patients chemo dates 
for specific mrns

"""

import pandas as pd
# from datetime import timedelta #datetime,
from data_prep.preprocess.chemo import get_treatment_data


def get_patients_with_completed_trt(info_dir, chemo_file, start_date, end_date, df, anchor):
    
    # Load data
    EPR_to_EPIC_regimen_map = pd.read_excel(f'{info_dir}/A2R_EPIC_GI_regimen_map.xlsx')
    EPR_to_EPIC_regimen_map = dict(EPR_to_EPIC_regimen_map[['PROTOCOL_DISPLAY_NAME','Mapped_Name_All']].to_numpy())
    EPR_regimens = pd.read_csv(f'{info_dir}/opis_regimen_list.csv')
    EPR_regimens.columns = EPR_regimens.columns.str.lower()
    
    # Get treatment data
    data_pull_day = None
    chemo_data = get_treatment_data(chemo_file, EPR_regimens, EPR_to_EPIC_regimen_map, data_pull_day, anchor)
    
    # Filter chemo_data by date range
    treatment_date_mask = (pd.to_datetime(chemo_data['treatment_date']) >= pd.to_datetime(start_date)) & \
                          (pd.to_datetime(chemo_data['treatment_date']) <= pd.to_datetime(end_date))
    filtered_chemo_data = chemo_data.loc[treatment_date_mask]
    
    # Only keep patients who completed treatments
    mask = df['mrn'].isin(filtered_chemo_data['mrn'])
    df = df[mask]
            
    return df