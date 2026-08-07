"""
Module for data preparation pipelines and feature summarization
"""

import logging
from collections.abc import Sequence
from typing import Optional
from warnings import simplefilter

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from make_clinical_dataset.epr.engineer import (
    collapse_rare_categories,
    get_change_since_prev_session,
    get_missingness_features,
    get_visit_month_feature
)
from make_clinical_dataset.shared.constants import DEFAULT_CONFIG_PATH
from make_clinical_dataset.epr.filter import (
    drop_highly_missing_features,
    drop_samples_outside_study_date,
    drop_unused_drug_features,
    keep_only_one_per_week,
)
from make_clinical_dataset.epr.prep import (
    PrepData,
    Splitter,
    fill_missing_data_heuristically,
)
from make_clinical_dataset.epr.util import get_excluded_numbers
from make_clinical_dataset.shared.constants import EPR_DRUG_COLS, LAB_COLS, UNIT_MAP
from ml_common.constants import CANCER_CODE_MAP
from sklearn.model_selection import StratifiedGroupKFold
from .constants import COLUMN_PATTERNS, ED_LOOKBACK_OPTIONS, build_keep_columns_explicit

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

###############################################################################
# Helper
###############################################################################
def _build_keep_columns(df: pd.DataFrame, ed_lookback_years: int = 5) -> list[str]:
    pattern_cols = df.columns[df.columns.str.contains("|".join(COLUMN_PATTERNS))].tolist()
    explicit_cols = [c for c in build_keep_columns_explicit(ed_lookback_years) if c in df.columns]
    prior_visits_feature = ED_LOOKBACK_OPTIONS[ed_lookback_years]["prior_visits_feature"]
    assert prior_visits_feature in df.columns, (
        f"Expected feature '{prior_visits_feature}' for ed_lookback_years={ed_lookback_years} "
        "but it is missing from the training data."
    )
    return explicit_cols + pattern_cols

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGIMEN_PATH = _REPO_ROOT / "Infos" / "master_regimen_map.csv"

with open(DEFAULT_CONFIG_PATH) as f:
    _DATA_PREP_CONFIG = yaml.safe_load(f)

def _clean_regimens(df):
    df_map = pd.read_csv(_REGIMEN_PATH)
    master_regimen_map = dict(df_map[["regimen", "mapped_regimen"]].to_numpy())
    regimens_to_exclude = df_map.loc[df_map["status"].isin(["Remove", "Oral"]), "regimen"].tolist()

    df["regimen"] = df["regimen"].str.strip()
    df["regimen"] = df["regimen"].replace(master_regimen_map)
    df = df[~df["regimen"].isin(regimens_to_exclude)]
    return df


###############################################################################
# Filtering
###############################################################################
def exclude_immediate_events(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["target_ED_date"].notna() & df["treatment_date"].notna() & (df["target_ED_date"] <= df["treatment_date"])
    get_excluded_numbers(
        df, ~mask, context=" in which patient had an ED visit on or before their treatment date."
    )
    df = df[~mask]
    return df

# TO DO: look at get_acu_labels

###############################################################################
# Data Preparation Pipeline
###############################################################################
class PrepACUData(PrepData):
    def __init__(self, ed_lookback_years: int = 5):
        super().__init__()
        if ed_lookback_years not in ED_LOOKBACK_OPTIONS:
            raise ValueError(f"ed_lookback_years must be one of {list(ED_LOOKBACK_OPTIONS)}")
        if ed_lookback_years == 1:
            deployment_lookback_window = _DATA_PREP_CONFIG["ed_visit_lookback_window_deployment"]
            assert deployment_lookback_window > 1, (
                "ed_lookback_years=1 is only valid when the deployment ED visit lookback window is > 1 "
                f"but got ed_visit_lookback_window_deployment={deployment_lookback_window}"
            )
        self.ed_lookback_years = ed_lookback_years
        self.ed_lookback = ED_LOOKBACK_OPTIONS[ed_lookback_years]

    def preprocess(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        drop_cols_missing_thresh: int = 80,
        drop_rows_missing_thresh: int = 80,
        anchor: str = "clinic",
    ) -> pd.DataFrame:
        """
        Args:
            drop_cols_missing_thresh: the percentage of missingness in which a column would be dropped.
                If set to -1, no columns will be dropped
            drop_rows_missing_thresh: the percentage of missingness in which a row would be dropped.
                If set to -1, no rows will be dropped
        """

        # keep relevant columns
        df = df[_build_keep_columns(df, ed_lookback_years=self.ed_lookback_years)]

        #-----------------------------------------------------------------------
        # filter rows based on various categories
        #-----------------------------------------------------------------------

        # filter based on output value
        # assert there is only 1 target column
        target_cols = df.columns[df.columns.str.startswith("target_") & ~df.columns.str.contains("date")]
        assert len(target_cols) == 1, f"Expected exactly 1 target column, found {len(target_cols)}: {target_cols.tolist()}"
        target_col = target_cols[0]
        # filter out rows where target is -1
        df = df[df[target_col] != -1]

        # filter based on regimen
        mask = (df["regimen"].str.startswith("GI")) & (df["regimen"].notnull())
        get_excluded_numbers(df, mask, context=" not from GI department")
        df = df[mask]
        df = _clean_regimens(df)

        # filter out dates before start date and after end date
        df = drop_samples_outside_study_date(
            df, start_date=start_date, end_date=end_date
        )

        # filter out immediate events
        df = exclude_immediate_events(df)

        #-----------------------------------------------------------------------
        # harmonize features since data source format may have changed over time
        #-----------------------------------------------------------------------

        # create cancer site columns if they don't exist
        cancer_site_cols = df.columns[df.columns.str.startswith("cancer_site_C")]
        if len(cancer_site_cols) == 0 and "primary_site_code" in df.columns:
            # one-hot encode primary_site_code into cancer_site_CXX columns
            dummies = pd.get_dummies(df["primary_site_code"], prefix="cancer_site")
            df = pd.concat([df.drop(columns=["primary_site_code"]), dummies], axis=1)

        # convert cancer site and morphology features to binary variables
        # by taking the most recent diagnosis prior to assessment date, represented as 2
        cols = df.columns[df.columns.str.contains("cancer_site_|morphology_")]
        if (df[cols] == 2).any().any():
            df[cols] = df[cols] == 2

        # harmonize sex at birth feature
        if "female" not in df.columns and "sex" in df.columns:
            df["female"] = (df["sex"] == "female").astype(int)
            df = df.drop(columns=["sex"])

        # compute visit month features if they don't exist
        visit_month_cols = df.columns[df.columns.str.startswith("visit_month_")]
        if len(visit_month_cols) == 0:
            df = get_visit_month_feature(df, col="assessment_date")

        # keep only the first treatment session of a given week
        df = keep_only_one_per_week(df)

        # get the change in measurement since previous assessment
        # WAYNE's notes: I am commenting this out. If you check the implementation
        # of this code, it assumes that the assessment date column is sorted
        # which is not ensured here. More importantly, there is no limit as to
        # how far apart the sessions are for the difference to be computed. It
        # also appears very arbitrary in that we do not have records of all
        # assessment dates so it's as if we are computing the change between the
        # 2nd and 8th visit, for example. This only works well for very clean
        # data sets which is not the case here.
        # df = get_change_since_prev_session(df)

        # drop drug features that were never used
        # WAYNE's notes: The original code dropped % dose given so this is irrelevant
        # df = drop_unused_drug_features(df)

        # fill missing data that can be filled heuristically (zeros, max values, etc)
        imputation_val = self.ed_lookback["lookback_days"]
        fill_vals = {
            "days_since_prev_ED_visit": imputation_val,
            "days_since_last_treatment": imputation_val,
        }
        df = fill_missing_data_heuristically(df, max_fills=[], custom_fills=fill_vals)
        for col in ("days_since_last_treatment", "days_since_prev_ED_visit"):
            if col in df.columns:
                df.loc[df[col] < 0, col] = imputation_val
                df[col] = df[col].clip(upper=imputation_val)

        if drop_cols_missing_thresh != -1:
            # drop features with high missingness
            keep_cols = df.columns[df.columns.str.contains("target_")]
            df = drop_highly_missing_features(
                df, missing_thresh=drop_cols_missing_thresh, keep_cols=keep_cols
            )

        if drop_rows_missing_thresh != -1:
            # drop samples with high missingness
            keep_cols = df.columns[~df.columns.str.contains("target_|date|mrn")]
            tmp = df[keep_cols].copy()
            # temporarily reverse the encoding for cancer-site
            cancer_site_cols = tmp.columns[tmp.columns.str.contains("cancer_site")]
            tmp["cancer_site"] = tmp[cancer_site_cols].apply(
                lambda mask: ", ".join(
                    cancer_site_cols[mask].str.removeprefix("cancer_site_")
                ),
                axis=1,
            )
            tmp["cancer_site"] = tmp["cancer_site"].replace("", None)
            tmp = tmp.drop(columns=cancer_site_cols)
            mask = tmp.isnull().mean(axis=1) * 100 < drop_rows_missing_thresh
            get_excluded_numbers(
                df,
                mask,
                context=f" with at least {drop_rows_missing_thresh} percent of features missing",
            )
            df = df[mask]

        # create missingness features
        df = get_missingness_features(df)

        # collapse rare morphology and cancer sites into 'Other' category
        df = collapse_rare_categories(df, catcols=["cancer_site", "morphology"])

        return df

    def prepare(
        self,
        train_test_split_date: str,
        df: pd.DataFrame,
        n_folds: int = 3,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # split the data - create training and testing set
        splitter = Splitter()
        train_data, test_data = splitter.temporal_split(
            df, split_date=train_test_split_date, visit_col="assessment_date"
        )

        # Remove sessions where event occured immediately afterwards on the train set ONLY
        # Wayne's Notes: I am commenting this out to make sure data for training model is 
        # consistent with deployment
        # train_data = exclude_immediate_events(train_data, date_cols=["target_ED_date"])

        # IMPORTANT: always make sure train data is done first for one-hot encoding, clipping, imputing, scaling
        train_data = self.transform_data(train_data, data_name="training")
        test_data = self.transform_data(test_data, data_name="testing")

        # split training data into folds for cross validation
        # NOTE: feel free to add more columns for different fold splits by looping through different random states
        kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        kf_splits = kf.split(
            X=train_data,
            y=train_data["target_ED_30d"],  # placeholder
            groups=train_data["mrn"],
        )
        cv_folds = np.zeros(len(train_data))
        for fold, (_, valid_idxs) in enumerate(kf_splits):
            cv_folds[valid_idxs] = fold
        train_data["cv_folds"] = cv_folds

        # create a split column and combine the data for convenience
        train_data["split"], test_data["split"] = "Train", "Test"
        data = pd.concat([train_data, test_data])

        # split into input features, output labels, and metainfo
        cols = data.columns
        meta_cols = ["mrn", "split", "cv_folds"] + cols[
            cols.str.contains("date")
        ].tolist()
        targ_cols = cols[
            cols.str.contains("target_") & ~cols.str.contains("date")
        ].tolist()
        feat_cols = cols.drop(meta_cols + targ_cols).tolist()
        X, Y, metainfo = (
            data[feat_cols].copy(),
            data[targ_cols].copy(),
            data[meta_cols].copy(),
        )

        # clean up Y
        Y.columns = Y.columns.str.replace("target_", "")

        return X, Y, metainfo


###############################################################################
# Feature Summary
###############################################################################
def feature_summary(
    X_train: pd.DataFrame,
    save_path: Optional[str] = None,
    keep_orig_names: bool = False,
    remove_missingness_feats: bool = True,
) -> pd.DataFrame:
    """
    Args:
        X_train (pd.DataFrame): table of the original data (not one-hot encoded, normalized, clipped, etc) for the
            training set
    """
    N = len(X_train)

    if remove_missingness_feats:
        # remove missingness features
        cols = X_train.columns
        drop_cols = cols[cols.str.contains("is_missing")]
        X_train = X_train.drop(columns=drop_cols)

    # get number of missing values, mean, and standard deviation for each feature in the training set
    summary = X_train.astype(float).describe()
    summary = summary.loc[["count", "mean", "std"]].T
    count = N - summary["count"]
    mean = summary["mean"].round(3).apply(lambda x: f"{x:.3f}")
    std = summary["std"].round(3).apply(lambda x: f"{x:.3f}")
    count[count.between(1, 5)] = 6  # mask small cells less than 6
    summary["Mean (SD)"] = mean + " (" + std + ")"
    summary["Missingness (%)"] = (count / N * 100).round(1)
    summary = summary.drop(columns=["count", "mean", "std"])
    # special case for drug features (percentage of dose given)
    for col in EPR_DRUG_COLS:
        if col not in X_train.columns:
            continue
        mask = (
            X_train[col] != 0
        )  # 0 indicates no drugs were given (not the percentage of the given dose)
        vals = X_train.loc[mask, col]
        summary.loc[col, "Mean (SD)"] = f"{vals.mean():.3f} ({vals.std():.3f})"

    # assign the groupings for each feature
    feature_groupings_by_keyword = {
        "Acute care use": "ED_visit",
        "Cancer": "cancer_site|morphology",
        "Demographic": "height|weight|body_surface_area|female|age",
        "Laboratory": "|".join(LAB_COLS),
        "Treatment": "visit_month|regimen|intent|treatment|dose|therapy|cycle",
        "Symptoms": "esas|ecog",
    }
    features = summary.index
    for group, keyword in feature_groupings_by_keyword.items():
        summary.loc[features.str.contains(keyword), "Group"] = group
    summary = summary[["Group", "Mean (SD)", "Missingness (%)"]]

    if keep_orig_names:
        summary["Features (original)"] = summary.index

    # insert units
    rename_map = {
        feat: f"{feat} ({unit})" for unit, feats in UNIT_MAP.items() for feat in feats
    }
    rename_map["female"] = "female (yes/no)"
    summary = summary.rename(index=rename_map)

    summary.index = [clean_feature_name(feat) for feat in summary.index]
    summary = summary.reset_index(names="Features")
    summary = summary.sort_values(by=["Group", "Features"])
    if save_path is not None:
        summary.to_csv(f"{save_path}", index=False)
    return summary


def clean_feature_name(name: str) -> str:
    if name == "patient_ecog":
        return "Eastern Cooperative Oncology Group (ECOG) Performance Status"

    mapping = {
        "prev": "previous",
        "num_": "number_of_",
        "%_ideal_dose": "percentage_of_ideal_dose",
        "intent": "intent_of_systemic_treatment",
        "cancer_site": "topography_ICD-0-3",
        "morphology": "morphology_ICD-0-3",
        "shortness_of_breath": "dyspnea",
        "tiredness": "fatigue",
        "patient_ecog": "eastern_cooperative_oncology_group_(ECOG)_performance_status",
        "cycle_number": "chemotherapy_cycle",
    }
    for orig, new in mapping.items():
        name = name.replace(orig, new)

    # title the name and replace underscores with space, but don't modify anything inside brackets at the end
    if name.endswith(")") and not name.startswith("regimen"):
        name, extra_info = name.split("(")
        name = "(".join([name.replace("_", " ").title(), extra_info])
    else:
        name = name.replace("_", " ").title()

    # capitalize certain substrings
    for substr in ["Ed V", "Icd", "Other", "Esas", "Ecog"]:
        name = name.replace(substr, substr.upper())
    # lowercase certain substrings
    for substr in [" Of "]:
        name = name.replace(substr, substr.lower())

    if name.startswith("Topography ") or name.startswith("Morphology "):
        # get full cancer description
        code = name.split(" ")[-1]
        if code in CANCER_CODE_MAP:
            name = f"{name}, {CANCER_CODE_MAP[code]}"
    elif name.startswith("ESAS "):
        # add 'score'
        if "Change" in name:
            name = name.replace("Change", "Score Change")
        else:
            name += " Score"

    for prefix in ["Regimen ", "Percentage of Ideal Dose Given "]:
        if name.startswith(prefix):
            # capitalize all regimen / drug names
            name = f"{prefix}{name.split(prefix)[-1].upper()}"

    return name
