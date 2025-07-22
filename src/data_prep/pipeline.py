"""
Final processing script
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from deployer.data_prep.constants import DAILY_POSTFIX_MAP, FILL_VALS, PROJ_NAME
from deployer.data_prep.preprocess.chemo import get_treatment_data
from deployer.data_prep.preprocess.diagnosis import get_demographic_data
from deployer.data_prep.preprocess.emergency import get_emergency_room_data
from deployer.data_prep.preprocess.esas import get_symptoms_data
from deployer.data_prep.preprocess.lab import get_lab_data
from deployer.loader import Config, Model
from make_clinical_dataset.combine import (
    add_engineered_features,
    combine_demographic_to_main_data,
    combine_event_to_main_data,
    combine_treatment_to_main_data,
)
from ml_common.anchor import merge_closest_measurements
from ml_common.engineer import get_change_since_prev_session
from ml_common.prep import fill_missing_data_heuristically
from ml_common.util import logger

logger.setLevel(logging.WARNING)


def build_features(config: Config, data_dir: str, data_pull_day: str, anchor: str) -> dict[str, pd.DataFrame]:
    postfix = DAILY_POSTFIX_MAP[anchor]
    biochem_file = f"{data_dir}/{PROJ_NAME}_biochemistry_{postfix}{data_pull_day}.csv"
    hema_file = f"{data_dir}/{PROJ_NAME}_hematology_{postfix}{data_pull_day}.csv"
    esas_file = f"{data_dir}/{PROJ_NAME}_ESAS_{postfix}{data_pull_day}.csv"
    chemo_file = f"{data_dir}/{PROJ_NAME}_chemo_{postfix}{data_pull_day}.csv"
    ed_file = f"{data_dir}/{PROJ_NAME}_ED_visits_{postfix}{data_pull_day}.csv"
    diagnosis_file = f"{data_dir}/{PROJ_NAME}_diagnosis_{postfix}{data_pull_day}.csv"

    if pd.read_csv(chemo_file).empty or pd.read_csv(diagnosis_file).empty:
        return {"error": f"No Patient {anchor.title()} Data for: {data_pull_day}"}

    feats = {}
    feats["symptom"] = get_symptoms_data(esas_file, anchor)
    feats["demographic"] = get_demographic_data(diagnosis_file, anchor)
    feats["treatment"] = get_treatment_data(chemo_file, config, data_pull_day, anchor)
    feats["laboratory"] = get_lab_data(hema_file, biochem_file, anchor)
    feats["emergency"] = get_emergency_room_data(ed_file, anchor)

    if anchor == "clinic":
        mrns = feats["treatment"]["mrn"].unique()
        feats["clinic"] = pd.DataFrame({"mrn": mrns, "clinic_date": data_pull_day})

    return feats


def get_data(
    config: Config,
    model: Model,
    feats: dict[str, pd.DataFrame],
    data_pull_day: str,
) -> pd.DataFrame:
    # Combine Features
    df = combine_features(model.prep_cfg, feats, model.anchor)

    # Get changes between treatment sessions
    df["hematocrit"] = None  # need to add this missing feature here. TODO: clean this up
    df = get_change_since_prev_session(df)

    # Fill missing data that can be filled heuristically (zeros, max values, etc)
    df = fill_missing_data_heuristically(df, max_fills=[], custom_fills=FILL_VALS[model.anchor])

    # Get missingness features
    # NOTE: we filter out unused features later on in inference.py
    # so easier to just calculate missingness for all columns
    df[df.columns + "_is_missing"] = df.isnull()

    # Encode Regimens and Intent
    df = encode_regimens(df, config.gi_regimens)
    df = encode_primary_sites(df, config.cancer_site_list)
    df = encode_intent(df)

    # Remove / reorganize features for symptoms' models
    if model.name == "symp":
        df = prep_symp_data(df)

    # Recreate any missing columns
    missing_cols = [col for col in model.model_features if col not in df.columns]
    df[missing_cols] = 0

    # Remove columns not used in training (keep the mrn and dates though)
    cols = df.columns
    cols = cols[cols.str.contains("mrn|date") | cols.isin(model.model_features)]
    df = df[cols]

    # Transform Data: Impute, Normalize, and Clip
    df.loc[0, df.columns[df.isna().all()]] = 0
    df = model.prep.transform_data(df, one_hot_encode=False)

    if model.anchor == "treatment":
        # Only keep treatments scheduled for the next day
        mask = df["treatment_date"] == pd.to_datetime(data_pull_day) + timedelta(days=1)
        df = df[mask]

    return df


def combine_features(cfg: dict, feats: dict[str, pd.DataFrame], anchor: str):
    """Combine the features into one unified dataset"""
    sym = feats["symptom"]
    dmg = feats["demographic"]
    trt = feats["treatment"]
    lab = feats["laboratory"]
    erv = feats["emergency"]

    if anchor == "treatment":
        df = feats["treatment"]
        df["assessment_date"] = pd.to_datetime(df["treatment_date"])
    elif anchor == "clinic":
        df = feats["clinic"]
        df["assessment_date"] = pd.to_datetime(df["clinic_date"])
    else:
        raise ValueError(f"Sorry, aligning features on {anchor} is not supported yet")

    if anchor != "treatment":
        df = combine_treatment_to_main_data(
            df, trt, "assessment_date", time_window=cfg["trt_lookback_window"], parallelize=False
        )

    df = combine_demographic_to_main_data(df, dmg, "assessment_date")
    df = merge_closest_measurements(df, sym, "assessment_date", "survey_date", time_window=cfg["symp_lookback_window"])
    df = merge_closest_measurements(df, lab, "assessment_date", "obs_date", time_window=cfg["lab_lookback_window"])
    df = combine_event_to_main_data(
        df,
        erv,
        "assessment_date",
        "event_date",
        event_name="ED_visit",
        lookback_window=cfg["ed_visit_lookback_window"],
        parallelize=False,
    )
    df = add_engineered_features(df, "assessment_date")

    return df


def encode_regimens(df, regimen_data):
    regimen_map = dict(regimen_data[["Regimen", "Regimen_Rename"]].to_numpy())
    df["regimen"] = df["regimen"].map(regimen_map).fillna("regimen_other")
    df = pd.get_dummies(df, columns=["regimen"], prefix="", prefix_sep="")
    return df


def encode_intent(df):
    df = pd.get_dummies(df, columns=["intent"])
    return df


def encode_primary_sites(df, cancer_sites):
    cancer = df["primary_site"].str.get_dummies(",")
    cancer = cancer.add_prefix("cancer_site_")

    # assign cancer sites not seen during model training as cancer_site_other
    other_sites = [site for site in cancer.columns if site not in cancer_sites]
    cancer["cancer_site_other"] = cancer[other_sites].any(axis=1).astype(int)
    cancer.drop(columns=other_sites)

    df = df.join(cancer)
    return df


def prep_symp_data(df):
    """Prepare data for symptoms models"""
    # reassign these regimens as other
    reg_cols = [
        "regimen_GI_FLOT _GASTRIC_",
        "regimen_GI_FOLFNALIRI _COMP_",
        "regimen_GI_FUFA C3 _GASTRIC_",
        "regimen_GI_FUFA WEEKLY",
        "regimen_GI_GEM D1_8 _ CAPECIT",
        "regimen_GI_PACLI WEEKLY",
    ]
    mask = df[reg_cols].any(axis=1)
    df.loc[mask, "regimen_other"] = True
    df.columns = df.columns.str.replace(" ", "_")
    return df
