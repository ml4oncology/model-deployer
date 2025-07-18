"""
Module to preprocess the demographic and diagnosis data
"""

import pandas as pd
from deployer.data_prep.constants import DROP_CLINIC_COLUMNS


def get_demographic_data(diagnosis_data_file: str, anchor: str):
    df = pd.read_csv(diagnosis_data_file)
    if anchor == "clinic":
        df = df.drop(columns=DROP_CLINIC_COLUMNS)
    df = filter_demographic_data(df)
    df = process_demographic_data(df)
    return df


def process_demographic_data(df: pd.DataFrame) -> pd.DataFrame:
    # combine patients with mutliple diagnoses into one row
    dtypes = df.dtypes
    df = df.groupby("mrn").agg(lambda col: ",".join(col.astype(str).unique()))
    df = df.reset_index().astype(dtypes)
    return df


def filter_demographic_data(df: pd.DataFrame) -> pd.DataFrame:
    # clean column names
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"research_id": "mrn"})
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])

    # filter out patients without medical record numbers
    mask = df["mrn"].notnull()
    df = df[mask]

    # clean data types
    df["mrn"] = df["mrn"].astype(int)

    # filter out patients whose sex is not Male/Female
    mask = df["sex"].isin(["Male", "Female"])
    df = df[mask].copy()
    df["female"] = df.pop("sex") == "Female"

    # clean cancer site (some have multiple sites)
    # only keep first three characters of the ICD codes - the rest are for specifics
    # e.g. C50 Breast: C501 Central portion, C504 Upper-outer quadrant, etc
    df["primary_site"] = (
        df["primary_site"].str.split(",").apply(lambda items: ",".join([item.strip()[:3] for item in items]))
    )

    return df
