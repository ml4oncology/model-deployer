PROJ_NAME = "AIM2REDUCE"

DROP_CLINIC_COLUMNS = [
    "MRN",
    "Lab Type",
    "Collected Date",
    "Result Date",
    "Finalized Date",
    "Last Update",
    "Accession",
    "Order ID",
    "Specimen Source",
    "Specimen Type",
    "Test Type",
    "Lab Status",
    "Agency",
    "Organism",
    "Comment",
    "Narrative",
]


DAILY_POSTFIX_MAP = {
    "treatment": "",  # treatment anchored files named as eg. AIM2REDUCE_hematology_20241104
    "clinic": "weekly_",  # clinic anchored files named as eg. AIM2REDUCE_hematology_weekly_20241104
}


MONTHLY_POSTFIX_MAP = {
    "treatment": "monthly_",  # treatment anchored files named as eg. AIM2REDUCE_hematology_monthly_20241104
    "clinic": "monthly_",  # clinic anchored files named as eg. AIM2REDUCE_hematology_monthly_20241104
}


# Number of days used to cap days_since_prev_ED_visit / days_since_last_treatment,
# and the lookback window passed to combine_event_to_main_data, per ED prior-visits feature.
ED_VISIT_COUNT_LOOKBACK_DAYS = {
    "num_prior_ED_visits_within_5_years": 5 * 365,
    "num_prior_ED_visits_within_1_year": 1 * 365,
}
