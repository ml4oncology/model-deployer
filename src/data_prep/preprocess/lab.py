"""
Module to preprocess laboratory test data, which includes hematology and biochemistry data
"""

import re
from typing import Optional

import pandas as pd
from deployer.data_prep.constants import DROP_CLINIC_COLUMNS
from make_clinical_dataset.shared.constants import OBS_MAP


def get_lab_data(hema_data_file, biochem_data_file, anchor):
    hema = pd.read_csv(hema_data_file)
    if anchor == "clinic":
        hema = hema.drop(columns=DROP_CLINIC_COLUMNS)
    hema = filter_lab_data(hema, obs_name_map=OBS_MAP["Hematology"])

    biochem = pd.read_csv(biochem_data_file)
    if anchor == "clinic":
        biochem = biochem.drop(columns=DROP_CLINIC_COLUMNS)
    biochem = filter_lab_data(biochem, obs_name_map=OBS_MAP["Biochemistry"])

    lab = pd.concat([hema, biochem])
    lab = process_lab_data(lab)
    lab = lab.rename(columns={"patientId": "mrn"})
    return lab


def normalize_lab_value(value: str) -> float | str:
    """Normalize a lab value string to a float where possible.

    - '<number'  → number / 2  (e.g. '<5' → 2.5)
    - '>number'  → number      (e.g. '>1000000' → 1000000.0)
    - Anything else non-numeric → left unchanged (will be coerced to NaN later)
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("<"):
        numbers = re.findall(r"[\d.]+", stripped)
        if numbers:
            return float(numbers[0]) / 2
    elif stripped.startswith(">"):
        numbers = re.findall(r"[\d.]+", stripped)
        if numbers:
            return float(numbers[0])
    return value


def process_lab_data(df):
    df["obs_datetime"] = pd.to_datetime(df["obs_datetime"], utc=True)
    df["obs_date"] = pd.to_datetime(df["obs_datetime"].dt.date)
    df = df.sort_values(by="obs_datetime")

    # take the most recent value if multiple lab tests taken in the same day
    # NOTE: dataframe already sorted by obs_datetime
    df = df.groupby(["patientId", "obs_date", "obs_name"]).agg({"obs_value": "last"}).reset_index()

    # make each observation name into a new column
    df = df.pivot(index=["patientId", "obs_date"], columns="obs_name", values="obs_value")

    # normalize inequality entries (e.g. '<5' → 2.5, '>1000' → 1000.0) then
    # coerce any remaining non-numeric strings to NaN — handles unknown entries
    # automatically without requiring a hardcoded blocklist
    for col in df.columns:
        df[col] = df[col].apply(normalize_lab_value)
    df = df.apply(pd.to_numeric, errors="coerce")

    df.columns.name = None
    return df.reset_index()


def filter_lab_data(df, obs_name_map: Optional[dict] = None):
    df = clean_lab_data(df)

    # exclude rows where observation value is missing
    df = df[df["obs_value"].notnull()]

    if obs_name_map is not None:
        df["obs_name"] = df["obs_name"].map(obs_name_map)
        # exclude observations not in the name map
        df = df[df["obs_name"].notnull()]

    df = filter_units(df)
    df = df.drop_duplicates(subset=["patientId", "obs_value", "obs_name", "obs_unit", "obs_datetime"])
    return df


def filter_units(df):
    # clean the units
    df["obs_unit"] = df["obs_unit"].replace({"bil/L": "x10e9/L", "fl": "fL"})

    # some observations have measurements in different units (e.g. neutrophil observations contain measurements in
    # x10e9/L (the majority) and % (the minority))
    # only keep one measurement unit for simplicity
    exclude_unit_map = {
        "creatinine": ["mmol/d", "mmol/CP"],
        "eosinophil": ["%"],
        "lymphocyte": ["%"],
        "monocyte": ["%"],
        "neutrophil": ["%"],
        "red_blood_cell": ["x10e6/L"],
        "white_blood_cell": ["x10e6/L"],
    }
    mask = False
    for obs_name, exclude_units in exclude_unit_map.items():
        mask |= (df["obs_name"] == obs_name) & df["obs_unit"].isin(exclude_units)
    df = df[~mask]

    return df


def clean_lab_data(df):
    # clean column names
    col_map = {
        # assign obs_ prefix to ensure no conflict with preexisting columns
        "component-code-coding-0-display": "obs_name",
        "component-valueQuantity-unit": "obs_unit",
        "component-valueQuantity-value": "obs_value",
        "lastUpdated": "obs_datetime",
    }
    df = df.rename(columns=col_map)

    return df
