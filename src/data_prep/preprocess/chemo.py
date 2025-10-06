"""
Module to preprocess chemotherapy treatment data
"""

import numpy as np
import pandas as pd
from deployer.data_prep.constants import DROP_CLINIC_COLUMNS
from deployer.loader import Config
from make_clinical_dataset.epr.engineer import get_line_of_therapy
from make_clinical_dataset.epr.preprocess.opis import merge_same_day_treatments


def get_treatment_data(
    chemo_data_file: str,
    config: Config,
    data_pull_day: pd.Timestamp | None = None,
    anchor: str = "treatment",
) -> pd.DataFrame:
    """
    Load and preprocess chemotherapy treatment data.

    Args:
        chemo_data_file (str): Path to the chemotherapy data CSV.
        config (Config): Configuration object containing EPR regimens and mapping.
        data_pull_day (str | None): Date of data pull.
        anchor (str): Anchor type, either 'treatment' or 'clinic'.

    Returns:
        pd.DataFrame: Preprocessed treatment data.
    """
    df = pd.read_csv(chemo_data_file)
    if anchor == "clinic" and data_pull_day is not None:
        df = df.drop(columns=DROP_CLINIC_COLUMNS)
    df = clean_treatment_data(df, config)
    df = process_treatment_data(df, anchor, data_pull_day)
    return df


def clean_treatment_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Clean and standardize treatment data columns and values.
    """
    # clean column names
    df.columns = df.columns.str.lower()
    col_map = {
        "research_id": "mrn",
        "trt_date_utc": "treatment_date",
        "first_trt_date_utc": "first_treatment_date",
        "dose_ord_or_min_dose_ord": "dose_ordered",
        "dose_given": "given_dose",
    }
    df = df.rename(columns=col_map)

    # convert columns to appropriate data types
    for date_col in ["treatment_date", "tx_sched_date", "first_treatment_date"]:
        df[date_col] = pd.to_datetime(df[date_col])

    # clean intent feature
    df["intent"] = df["intent"].replace("U", np.nan).str.upper()

    df = clean_regimens(df, config)
    return df


def process_treatment_data(df: pd.DataFrame, anchor: str, data_pull_day: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Process treatment data: sort, fill, filter, merge, and compute features.
    """
    df = df.sort_values(by=["mrn", "tx_sched_date", "treatment_date", "regimen"])
    df = fill_patient_columns(df, ["height", "weight", "body_surface_area"])

    clinic_eval = anchor == "clinic" and data_pull_day is not None
    if clinic_eval:
        # Get next scheduled treatment dates (within 5 days of clinic visit)
        mask = df["tx_sched_date"].between(data_pull_day, data_pull_day + pd.Timedelta(days=5))
        next_sched_trt_date = df[mask].groupby("mrn")["tx_sched_date"].first()

    # Filter treatment sessions before data pull date
    df = filter_chemo_treatments(df, anchor, data_pull_day)

    # Process treatment dates
    df = merge_same_day_treatments(df)
    df = fill_patient_columns(df, ["first_treatment_date"])

    # Remove one-off duplicate rows (all values are same except for one, most likely due to human error)
    for col in ["first_treatment_date", "cycle_number"]:
        cols = df.columns.drop(col)
        mask = ~df.duplicated(subset=cols, keep="first")
        df = df[mask]

    # Merge next scheduled treatment dates
    if clinic_eval:
        df["next_sched_trt_date"] = df["mrn"].map(next_sched_trt_date)

    # Compute line of therapy
    df["line_of_therapy"] = df.groupby("mrn", group_keys=False).apply(get_line_of_therapy, include_groups=False)

    return df


def clean_regimens(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    # filter out rows with missing regimen info
    mask = df["regimen"].notnull()
    df = df[mask].copy()

    # map regimens from EPIC to EPR
    df["regimen_EPIC"] = df["regimen"]
    df["regimen"] = df["regimen"].replace(config.epr2epic_regimen_map)

    # rename some of the regimens
    regimen_map = dict(config.epr_regimens.query("rename.notnull()")[["regimen", "rename"]].to_numpy())
    df["regimen"] = df["regimen"].replace(regimen_map)
    return df


def filter_chemo_treatments(df, anchor: str, data_pull_day: pd.Timestamp | None = None):
    if data_pull_day is None:
        return df[df["day_status"] == "Completed"]

    mask = df["treatment_date"] < data_pull_day
    if anchor == "treatment":
        mask |= df["tx_sched_date"] == data_pull_day + pd.Timedelta(days=1)
    df = df[mask].copy()

    # fill missing treatment dates with scheduled treatment date
    df["treatment_date"] = df["treatment_date"].fillna(df["tx_sched_date"])
    return df


def fill_patient_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Forward and backward fill specified columns grouped by 'mrn'.
    """
    for col in columns:
        df[col] = df.groupby("mrn")[col].ffill().bfill()
    return df
