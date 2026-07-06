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
    mode: str = "prediction",
) -> pd.DataFrame:
    """
    Load and preprocess chemotherapy treatment data.

    Args:
        chemo_data_file (str): Path to the chemotherapy data CSV.
        config (Config): Configuration object containing EPR regimens and mapping.
        data_pull_day (str | None): Date of data pull.
        anchor (str): Anchor type, either 'treatment' or 'clinic'.
        mode (str): Mode of operation, either 'prediction' or 'evaluation'. 
    Returns:
        pd.DataFrame: Preprocessed treatment data.
    """
    df = pd.read_csv(chemo_data_file)
    if anchor == "clinic" and data_pull_day is not None:
        df = df.drop(columns=DROP_CLINIC_COLUMNS)
    df = clean_treatment_data(df, config)

    if mode == "prediction":
        df = process_treatment_data_pred(df, anchor, config, data_pull_day)
    elif mode == "evaluation":
        df = process_treatment_data_eval(df, anchor, config, data_pull_day)
  
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


def process_treatment_data_pred(df: pd.DataFrame, anchor: str, config: Config, data_pull_day: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Process treatment data: sort, fill, filter, merge, and compute features.
    """
    df = df.sort_values(by=["mrn", "tx_sched_date", "treatment_date", "regimen"])
    df = fill_patient_columns(df, ["height"])
   
    clinic_eval = anchor == "clinic" and data_pull_day is not None
    if clinic_eval:
        # Get next scheduled treatment dates (within 5 days of clinic visit)
        mask = df["tx_sched_date"].between(data_pull_day, data_pull_day + pd.Timedelta(days=config["trt_lookahead_window"]))
        next_sched_trt_date = df[mask].groupby("mrn")["tx_sched_date"].first()
    
    # drop rows with missing tx_sched_date
    df = df.dropna(subset=["tx_sched_date"])
    # if treatment_date value is missing, fill with tx_sched_date
    df["treatment_date"] = df["treatment_date"].fillna(df["tx_sched_date"])

    df = fill_patient_columns(df, ["weight", "body_surface_area"], lookback_days=30, date_column="treatment_date")

    # currently, drug dosage columns are being disregarded
    df = merge_same_day_treatments(df)

    # Process treatment dates
    # df = fill_patient_columns(df, ["first_treatment_date"])
    df = fill_first_treatment_date(df)

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
    
    # Compute days since
    df = compute_days_since(df, data_pull_day)
    
    # Only keep the first row with treatment_date on or after data_pull_day
    # this limits chemo to one row per patient
    if data_pull_day is not None:
        df = df[df["treatment_date"] >= data_pull_day].copy()
        df = df.sort_values(by=["treatment_date"], ascending=True)
        # keep only the first treatment row for each patient
        df = df.groupby("mrn", group_keys=False).apply(lambda group: group.head(1)).reset_index(drop=True) 
    else:
        raise ValueError("Not implemented yet")

    # Note: If we filter sessions before pull date, we will not get relevant treatment info
    # for the upcoming treatment session
    # # Filter treatment sessions before data pull date
    # df = filter_chemo_treatments(df, anchor, data_pull_day)

    return df

def process_treatment_data_eval(df: pd.DataFrame, anchor: str, config: Config, data_pull_day: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Process treatment data: sort, fill, filter, merge, and compute features.
    """
    df = df.sort_values(by=["mrn", "tx_sched_date", "treatment_date", "regimen"])
    df = fill_patient_columns(df, ["height", "weight", "body_surface_area"])

    clinic_eval = anchor == "clinic" and data_pull_day is not None
    if clinic_eval:
        # Get next scheduled treatment dates (within 5 days of clinic visit)
        mask = df["tx_sched_date"].between(data_pull_day, data_pull_day + pd.Timedelta(days=config["trt_lookahead_window"]))
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

    # normalize whitespace in both the data and the mapping keys/values
    df["regimen"] = df["regimen"].str.strip()
    clean_map = {k.strip(): v.strip() if isinstance(v, str) else v 
                 for k, v in config.epr2epic_regimen_map.items()}

    # map regimens from EPIC to EPR
    df["regimen_EPIC"] = df["regimen"]
    df["regimen"] = df["regimen"].replace(clean_map)

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


def fill_patient_columns(
    df: pd.DataFrame,
    columns: list,
    lookback_days: int = None,
    date_column: str = None,
) -> pd.DataFrame:
    """
    Forward and backward fill specified columns grouped by 'mrn'.

    Args:
        df:             Input dataframe.
        columns:        Columns to forward/backward fill.
        lookback_days:  Optional. Maximum number of days allowed between the
                        source (donor) row's date and the target (recipient)
                        row's date for a fill to be applied.
        date_column:    Optional. Column containing dates used to enforce the
                        lookback window. Must be specified together with
                        lookback_days.
    """
    use_window = (lookback_days is not None) and (date_column is not None)

    if not use_window:
        # Original behaviour
        for col in columns:
            df[col] = df.groupby("mrn")[col].ffill().bfill()
        return df

    # --- Windowed fill ---
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    window = pd.Timedelta(days=lookback_days)

    for col in columns:
        filled = []

        for _, group in df.groupby("mrn", sort=False):
            group = group.sort_index()
            dates = group[date_column]
            values = group[col].copy()

            # Forward fill: propagate last known value forward
            last_val = None
            last_date = None
            forward = values.copy()
            for idx in group.index:
                if pd.notna(values[idx]):
                    last_val = values[idx]
                    last_date = dates[idx]
                elif last_val is not None and (dates[idx] - last_date) <= window:
                    forward[idx] = last_val

            # Backward fill: propagate next known value backward
            next_val = None
            next_date = None
            backward = forward.copy()
            for idx in reversed(group.index):
                if pd.notna(forward[idx]):
                    next_val = forward[idx]
                    next_date = dates[idx]
                elif next_val is not None and (next_date - dates[idx]) <= window:
                    backward[idx] = next_val

            filled.append(backward)

        df[col] = pd.concat(filled)

    return df

def fill_first_treatment_date(df):
    """
    Fill missing values in 'first_treatment_date' column based on cycle_number logic.
    
    For each mrn:
    - cycle_number == 1: use treatment_date
    - cycle_number != 1: use most recent non-NaN first_treatment_date
    - cycle_number is NaN: if regimen changed vs most recent non-NaN regimen → use treatment_date,
                           else use first_treatment_date from that most recent row
    """
    df = df.copy()

    for mrn, group in df.groupby("mrn"):
        indices = group.index.tolist()

        for idx in indices:
            # Skip if already filled
            if pd.notna(df.at[idx, "first_treatment_date"]):
                continue

            cycle = df.at[idx, "cycle_number"]

            if pd.notna(cycle) and cycle == 1:
                # Rule 1: cycle == 1 → use treatment_date
                df.at[idx, "first_treatment_date"] = df.at[idx, "treatment_date"]

            elif pd.notna(cycle) and cycle != 1:
                # Rule 2: cycle != 1 → use most recent non-NaN first_treatment_date
                prior_indices = [i for i in indices if i < idx]
                non_nan_prior = [
                    i for i in prior_indices
                    if pd.notna(df.at[i, "first_treatment_date"])
                ]
                if non_nan_prior:
                    most_recent = max(non_nan_prior)
                    df.at[idx, "first_treatment_date"] = df.at[most_recent, "first_treatment_date"]

            else:
                # Rule 3: cycle is NaN → compare regimen with most recent non-NaN regimen
                prior_indices = [i for i in indices if i < idx]
                non_nan_regimen_prior = [
                    i for i in prior_indices
                    if pd.notna(df.at[i, "regimen"])
                ]
                if non_nan_regimen_prior:
                    most_recent = max(non_nan_regimen_prior)
                    current_regimen = df.at[idx, "regimen"]
                    recent_regimen = df.at[most_recent, "regimen"]

                    if current_regimen != recent_regimen:
                        df.at[idx, "first_treatment_date"] = df.at[idx, "treatment_date"]
                    else:
                        df.at[idx, "first_treatment_date"] = df.at[most_recent, "first_treatment_date"]

    return df

def compute_days_since(
    df: pd.DataFrame,
    data_pull_day: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute:
        1. days_since_starting_treatment
        2. days_since_last_treatment

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing:
            - mrn
            - first_treatment_date
            - treatment_date

    data_pull_day : pd.Timestamp
        Reference date used to compute elapsed days.

    Returns
    -------
    pd.DataFrame
        Original dataframe with two additional columns.
    """

    df = df.copy()

    # Ensure datetime
    df["first_treatment_date"] = pd.to_datetime(df["first_treatment_date"])
    df["treatment_date"] = pd.to_datetime(df["treatment_date"])
    data_pull_day = pd.to_datetime(data_pull_day)

    # ------------------------------------------------------------------
    # Days since starting treatment
    # ------------------------------------------------------------------
    df["days_since_starting_treatment"] = (
        data_pull_day - df["first_treatment_date"]
    ).dt.days

    df["days_since_starting_treatment"] = (
        df["days_since_starting_treatment"]
        .clip(lower=0)
    )

    # ------------------------------------------------------------------
    # Days since last treatment
    # ------------------------------------------------------------------

    # Preserve original order
    original_index = df.index

    # Sort within patient by treatment date
    df = df.sort_values(["mrn", "treatment_date"])

    # Previous treatment date within MRN
    prev_treatment_date = (
        df.groupby("mrn")["treatment_date"]
        .shift(1)
    )

    # Compute days from previous treatment to data pull day
    df["days_since_last_treatment"] = (
        data_pull_day - prev_treatment_date
    ).dt.days

    # First treatment per MRN -> NaN -> set to 0
    df["days_since_last_treatment"] = (
        df["days_since_last_treatment"]
        .fillna(0)
        .clip(lower=0)
        .astype(int)
    )

    # Restore original order
    df = df.loc[original_index]

    return df