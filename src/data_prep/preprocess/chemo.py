"""
Module to preprocess chemotherapy treatment data
"""

import numpy as np
import pandas as pd
from deployer.data_prep.constants import DROP_CLINIC_COLUMNS
from make_clinical_dataset.feat_eng import get_line_of_therapy
from make_clinical_dataset.preprocess.opis import merge_same_day_treatments


def get_treatment_data(
    chemo_data_file: str,
    epr_regimens: pd.DataFrame,
    epr_to_epic_regimen_map: dict,
    data_pull_day: str | None = None,
    anchor: str = "treatment",
) -> pd.DataFrame:
    """
    Load and preprocess chemotherapy treatment data.

    Args:
        chemo_data_file (str): Path to the chemotherapy data CSV.
        epr_regimens (pd.DataFrame): Selected EPR regimens during model training.
        epr_to_epic_regimen_map (dict): Map between old EPR and new EPIC regimens.
        data_pull_day (str | None): Date of data pull.
        anchor (str): Anchor type, either 'treatment' or 'clinic'.

    Returns:
        pd.DataFrame: Preprocessed treatment data.
    """
    df = pd.read_csv(chemo_data_file)
    if anchor == "clinic" and data_pull_day is not None:
        df = df.drop(columns=DROP_CLINIC_COLUMNS)
    df = clean_treatment_data(df, epr_regimens, epr_to_epic_regimen_map)
    df = process_treatment_data(df, anchor, data_pull_day)
    return df


def clean_treatment_data(
    df: pd.DataFrame,
    epr_regimens: pd.DataFrame,
    epr_to_epic_regimen_map: dict[str, str],
) -> pd.DataFrame:
    """
    Clean and standardize treatment data columns and values.
    """
    # clean column names
    df.columns = df.columns.str.lower()
    col_map = {
        "research_id": "mrn",
        "first_trt_date_utc": "first_treatment_date",
        "dose_ord_or_min_dose_ord": "dose_ordered",
        "dose_given": "given_dose",
    }
    df = df.rename(columns=col_map)

    # convert columns to appropriate data types
    for date_col in ["trt_date_utc", "tx_sched_date", "first_treatment_date"]:
        df[date_col] = pd.to_datetime(df[date_col])

    # clean intent feature
    df["intent"] = df["intent"].replace("U", np.nan)
    df["intent"] = df["intent"].str.upper()

    df = clean_regimens(df, epr_regimens, epr_to_epic_regimen_map)
    return df


def process_treatment_data(df: pd.DataFrame, anchor: str, data_pull_day: str | None = None) -> pd.DataFrame:
    """
    Process treatment data: sort, fill, filter, merge, and compute features.
    """
    df = df.sort_values(by=["mrn", "tx_sched_date", "trt_date_utc", "regimen"])
    df = fill_patient_columns(df, ["height", "weight", "body_surface_area"])

    # Get scheduled treatments within 5 days of clinic visit
    clinic_eval = anchor == "clinic" and data_pull_day is not None
    if clinic_eval:
        scheduled_treatment = get_scheduled_treatments(df, data_pull_day)

    # Filter treatment sessions before data pull date
    df = filter_chemo_treatments(df, anchor, data_pull_day)

    # Process treatment dates
    df["treatment_date"] = df["trt_date_utc"]
    df = merge_same_day_treatments(df)
    df = fill_patient_columns(df, ["first_treatment_date"])

    # remove one-off duplicate rows (all values are same except for one, most likely due to human error)
    for col in ["first_treatment_date", "cycle_number"]:
        cols = df.columns.drop(col)
        mask = ~df.duplicated(subset=cols, keep="first")
        df = df[mask]

    # Merge scheduled treatment dates to clinic visit dates; for sanity check and reporting
    if clinic_eval:
        df = pd.merge(df, scheduled_treatment, how="left", on="mrn")

    # compute line of therapy
    df["line_of_therapy"] = df.groupby("mrn", group_keys=False).apply(get_line_of_therapy, include_groups=False)

    return df


def clean_regimens(
    df: pd.DataFrame, epr_regimens: pd.DataFrame, epr_to_epic_regimen_map: dict[str, str]
) -> pd.DataFrame:
    # filter out rows with missing regimen info
    mask = df["regimen"].notnull()
    df = df[mask].copy()

    # map regimens from EPR to EPIC
    df["regimen_EPIC"] = df["regimen"]
    df["regimen"] = df["regimen"].replace(epr_to_epic_regimen_map)

    # rename some of the regimens
    regimen_map = dict(epr_regimens.query("rename.notnull()")[["regimen", "rename"]].to_numpy())
    df["regimen"] = df["regimen"].replace(regimen_map)
    return df


def filter_chemo_treatments(df, anchor: str, data_pull_day: str | None = None):
    if data_pull_day is None:
        return df[df["day_status"] == "Completed"]

    pull_date = pd.to_datetime(data_pull_day)
    mask = df["trt_date_utc"] < pull_date
    if anchor == "treatment":
        mask |= df["tx_sched_date"] == pull_date + pd.Timedelta(days=1)
    df = df[mask]

    # fill missing treatment dates with scheduled treatment date
    df["trt_date_utc"] = df["trt_date_utc"].fillna(df["tx_sched_date"])
    return df


def fill_patient_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Forward and backward fill specified columns grouped by 'mrn'.
    """
    for col in columns:
        df[col] = df.groupby("mrn")[col].ffill().bfill()
    return df


def get_scheduled_treatments(df: pd.DataFrame, data_pull_day: str | None):
    # filter out rows where next scheduled treatment session does not occur within 5 days of pull date (clinic visit)
    pull_date = pd.to_datetime(data_pull_day)
    mask = df["tx_sched_date"].between(pull_date, pull_date + pd.Timedelta(days=5))
    res = (
        df.loc[mask, ["mrn", "tx_sched_date"]].sort_values("tx_sched_date").drop_duplicates(subset="mrn", keep="first")
    )
    return res
