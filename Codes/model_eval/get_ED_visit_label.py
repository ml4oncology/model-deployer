"""
Process ED visits:
    1. Add labels
    2. Get ED visits within lookahead window
"""

from data_prep.preprocess.emergency import get_emergency_room_data


def merge_ed_pred_label(ed_file, ed_date_start, ed_date_end, df, date_column):
    
    # emergency room visits
    ed_visit = get_emergency_room_data(ed_file,'')
    ed_visit = ed_visit[(ed_visit['event_date'] >= ed_date_start) & (ed_visit['event_date']  <= ed_date_end)]

    # Merge on 'mrn' using a left join to keep all treatment rows
    merged_df = df.merge(ed_visit, on="mrn", how="left")

    # Compute the days difference between treatment and ED visit
    merged_df["trt_to_edvisit"] = (merged_df["event_date"] - merged_df[date_column]).dt.days

    # Keep only ED visits that fall within the 30-day window
    valid_visits = merged_df[merged_df["trt_to_edvisit"].between(0, 30)]

    # Select the earliest ED visit within the 30-day window for each (mrn, treatment_date)
    first_ed_visits = valid_visits.sort_values(by=["mrn", date_column, "event_date", "trt_to_edvisit"]) \
                                  .groupby(["mrn", date_column], as_index=False) \
                                  .first()

    # Merge back with the original treatment data to retain all treatment rows
    final_df = df.merge(first_ed_visits[["mrn", date_column, "event_date", "trt_to_edvisit"]], 
                              on=["mrn", date_column], 
                              how="left")

    # Create labels
    final_df = final_df.rename(columns={'event_date': 'ed_visit_dates_30days'})
    final_df["ed_visit_labels_30days"] = final_df["ed_visit_dates_30days"].notna().astype(int)
    
    # Remove treatments with same day ED visits, 
    # i.e. Difference between treatment date and ED visit = 0 
    final_df = final_df[final_df['trt_to_edvisit'] != 0]
    
    ##########################################################################
    # # Flag only the first treatment within multiple treatment sessions linked to an ED visit within 30 days
    
    # # Filter only rows where ED visits exist
    # ed_visit_rows = final_df[final_df["ed_visit_dates_30days"].notna()].copy()

    # # Sort by mrn, ed_visit_date, treatment_date
    # ed_visit_rows = ed_visit_rows.sort_values(by=["mrn", "ed_visit_dates_30days", "treatment_date"])

    # # Mark only the first treatment per (mrn, ed_visit_date) as 1
    # ed_visit_rows["first_treatment_flag"] = ed_visit_rows.groupby(["mrn", "ed_visit_dates_30days"])["treatment_date"].rank(method="first", ascending=True) == 1

    # # Convert boolean to integer (1 for first treatment, 0 otherwise)
    # ed_visit_rows["first_treatment_flag"] = ed_visit_rows["first_treatment_flag"].astype(int)

    # # Merge this flagged column back into the final dataframe
    # final_df = final_df.merge(ed_visit_rows[["mrn", "treatment_date", "ed_visit_dates_30days", "first_treatment_flag"]], 
    #                           on=["mrn", "treatment_date", "ed_visit_dates_30days"], 
    #                           how="left")

    # # Fill NaN values with 0 (for rows without an ED visit)
    # final_df["first_treatment_flag"] = final_df["first_treatment_flag"].fillna(0).astype(int)

    return final_df
