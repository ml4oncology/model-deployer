"""
Module to preprocess ESAS symptom data
"""

from make_clinical_dataset.epic.preprocess.esas import get_epic_symp_data
from make_clinical_dataset.shared.constants import SYMP_COLS


def get_symptoms_data(esas_data_file):
    df = get_epic_symp_data(esas_data_file)

    # exclude rows where symptoms scores are all missing
    mask = df[SYMP_COLS].isna().all(axis=1)
    df = df[~mask]

    # temporarily rename the columns back to the original state, so its compatible with the current model
    col_map = {c: f"esas_{c}" for c in SYMP_COLS}
    col_map["ecog"] = "patient_ecog"
    col_map["obs_date"] = "survey_date"
    df = df.rename(columns=col_map)

    return df
