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

ED_COLS = ['days_since_prev_ED_visit',
           'num_prior_ED_visits_within_5_years']

ENGINEERED_COLS = ['visit_month_sin',
                   'visit_month_cos',
                   'days_since_starting_treatment',
                   'days_since_last_treatment']

TARGET_COLS = ["target_ED_30d"]

KEEP_COLUMNS_EXPLICIT = META_COLS + CHEMO_COLS + DEMOG_COLS + \
    LAB_BIOCHEM_COLS + ESAS_COLS + ED_COLS + ENGINEERED_COLS + TARGET_COLS