from make_clinical_dataset.shared.constants import (LAB_COLS, SYMP_COLS)

META_COLS = ['mrn', 'assessment_date', 'target_ED_date', 'treatment_date', 'prev_ED_visit_data_source']

CHEMO_COLS = ['regimen',
            'line_of_therapy',
            'intent',
            'cycle_number',
            'height',
            'weight',
            'body_surface_area']

DEMOG_COLS = ['female', # old pull
              'sex', # new pull
              'age',
              'primary_site_code', # new pull
              ]
COLUMN_PATTERNS = ["cancer_site_C"]   # this is for old pull

LAB_BIOCHEM_COLS = LAB_COLS.copy()
cols_to_remove=['carbohydrate_antigen_19-9', # not in the v1 pull
                'activated_partial_thromboplastin_time', # the following are not in the v2 pull
                'calcium',
                'carcinoembryonic_antigen',
                'eGFR',
                'hematocrit',
                'eosinophil' # high missingness in deployment
                ] 
for col in cols_to_remove:
    LAB_BIOCHEM_COLS.remove(col)

ESAS_COLS = SYMP_COLS.copy()
ESAS_COLS.remove('ecog') # high missingness in deployment

ED_LOOKBACK_OPTIONS = {
    5: {"prior_visits_feature": "num_prior_ED_visits_within_5_years", "lookback_days": 5 * 365},
    1: {"prior_visits_feature": "num_prior_ED_visits_within_1_year", "lookback_days": 1 * 365},
}

ENGINEERED_COLS = ['visit_month_sin',
                   'visit_month_cos',
                   'days_since_starting_treatment',
                   'days_since_last_treatment']

TARGET_COLS = ["target_ED_30d"]

_META_COLS = META_COLS + CHEMO_COLS + DEMOG_COLS + \
    LAB_BIOCHEM_COLS + ESAS_COLS + ENGINEERED_COLS + TARGET_COLS


def build_keep_columns_explicit(ed_lookback_years: int = 5) -> list[str]:
    """Return the explicit columns to keep for the given ED lookback variant."""
    ed_cols = ['days_since_prev_ED_visit',
               ED_LOOKBACK_OPTIONS[ed_lookback_years]["prior_visits_feature"]]
    return _META_COLS + ed_cols