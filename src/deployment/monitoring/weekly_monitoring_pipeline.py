#!/usr/bin/env python3
"""Maintain weekly live-deployment monitoring aggregates.

This is the weekly monitoring stage: it owns Friday-ending week selection,
aggregate creation and updates, CSV/metadata invariants, and serialized
execution. PSI reference counts and PSI reports are delegated to
:mod:`deployer.monitoring.weekly_psi_monitoring`, which remains a deliberately named
companion stage rather than a generic helper.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from deployer.monitoring import weekly_psi_monitoring
from deployer.monitoring.monitoring_common import (
    WEEKLY_INTERIM_DIR,
    MonitoringPipelineError,
    argparse_date,
    atomic_write_csv,
    atomic_write_text,
    consented_provider_mask,
    daily_prediction_path,
    discover_prediction_files,
    friday_on_or_after,
    last_eligible_friday,
    load_consented_physicians,
    load_prediction_window,
    metadata_text,
    print_files,
    read_metadata,
    run_serialized,
    to_date,
    utc_now,
    validate_metadata_identity,
    week_ending_range,
    week_window_start,
)

WEEKLY_SCHEMA_VERSION = 1
SILENT_DEPLOYMENT_SCHEMA_VERSION = 1
MISSINGNESS_SCHEMA_VERSION = 1
SILENT_MISSINGNESS_SCHEMA_VERSION = 1
UNIQUE_PATIENTS_SCHEMA_VERSION = 1
WEEKLY_ALERTS_METRIC = "weekly_alerts"
WEEKLY_RISK_SCORES_METRIC = "weekly_risk_scores"
WEEKLY_MISSINGNESS_METRIC = "weekly_missingness"
UNIQUE_PATIENTS_REGISTRY_METRIC = "unique_patients_registry"
WEEKLY_UNIQUE_PATIENTS_METRIC = "weekly_unique_patients"
WEEK_ENDING_WEEKDAY = 4  # Friday, with Monday == 0
COHORT_COLUMN = "first_trt"
COHORT_VALUE = 1
CLINICIAN_COLUMN = "VISIT_PROVIDER_NAME"
MRN_COLUMN = "mrn"
FIRST_SEEN_DATE_COLUMN = "first_seen_date"
NEW_PATIENTS_COLUMN = "new_patients"
CUMULATIVE_PATIENTS_COLUMN = "cumulative_patients"
SCORE_COLUMN = "ed_pred_prob"
REGIMEN_COLUMN = "regimen"
REGIMEN_PREFIX_COLUMN = "regimen_prefix"
REGIMEN_PREFIX_LENGTH = 2
NO_CLINICIAN_LABEL = "No clinician"
MISSINGNESS_DIR = "daily_missingness"
MISSINGNESS_SUFFIX = "_is_missing"
SILENT_MISSINGNESS_PREFIX = "silent_deployment_input_missingness"

first_friday_on_or_after = friday_on_or_after
_metadata_text = metadata_text


# Metadata schemas and readers


@dataclass(frozen=True)
class WeeklyAlertsMetadata:
    schema_version: int
    metric: str
    anchor: str
    deployment_start_date: date
    last_week_ending: date
    clinician_column: str
    cohort_column: str
    cohort_value: int
    updated_at: str

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "anchor": self.anchor,
            "deployment_start_date": self.deployment_start_date.isoformat(),
            "last_week_ending": self.last_week_ending.isoformat(),
            "alert_definition": {
                "cohort_column": self.cohort_column,
                "cohort_value": self.cohort_value,
            },
            "clinician_column": self.clinician_column,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            alert_definition = data["alert_definition"]
            return cls(
                schema_version=int(data["schema_version"]),
                metric=str(data["metric"]),
                anchor=str(data["anchor"]),
                deployment_start_date=to_date(data["deployment_start_date"]),
                last_week_ending=to_date(data["last_week_ending"]),
                clinician_column=str(data["clinician_column"]),
                cohort_column=str(alert_definition["cohort_column"]),
                cohort_value=int(alert_definition["cohort_value"]),
                updated_at=str(data["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid metadata: {exc}") from exc


@dataclass(frozen=True)
class UniquePatientsRegistryMetadata:
    schema_version: int
    metric: str
    anchor: str
    deployment_start_date: date
    last_week_ending: date
    mrn_column: str
    first_seen_date_column: str
    cohort_column: str
    cohort_value: int
    row_count: int
    updated_at: str

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "anchor": self.anchor,
            "deployment_start_date": self.deployment_start_date.isoformat(),
            "last_week_ending": self.last_week_ending.isoformat(),
            "mrn_column": self.mrn_column,
            "first_seen_date_column": self.first_seen_date_column,
            "cohort_definition": {
                "cohort_column": self.cohort_column,
                "cohort_value": self.cohort_value,
            },
            "row_count": self.row_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            cohort_definition = data["cohort_definition"]
            return cls(
                schema_version=int(data["schema_version"]),
                metric=str(data["metric"]),
                anchor=str(data["anchor"]),
                deployment_start_date=to_date(data["deployment_start_date"]),
                last_week_ending=to_date(data["last_week_ending"]),
                mrn_column=str(data["mrn_column"]),
                first_seen_date_column=str(data["first_seen_date_column"]),
                cohort_column=str(cohort_definition["cohort_column"]),
                cohort_value=int(cohort_definition["cohort_value"]),
                row_count=int(data["row_count"]),
                updated_at=str(data["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid unique patient registry metadata: {exc}") from exc


@dataclass(frozen=True)
class WeeklyUniquePatientsMetadata:
    schema_version: int
    metric: str
    anchor: str
    deployment_start_date: date
    last_week_ending: date
    cohort_column: str
    cohort_value: int
    new_patients_column: str
    cumulative_patients_column: str
    row_count: int
    week_count: int
    patient_count: int
    updated_at: str

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "anchor": self.anchor,
            "deployment_start_date": self.deployment_start_date.isoformat(),
            "last_week_ending": self.last_week_ending.isoformat(),
            "cohort_definition": {
                "cohort_column": self.cohort_column,
                "cohort_value": self.cohort_value,
            },
            "new_patients_column": self.new_patients_column,
            "cumulative_patients_column": self.cumulative_patients_column,
            "row_count": self.row_count,
            "week_count": self.week_count,
            "patient_count": self.patient_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            cohort_definition = data["cohort_definition"]
            return cls(
                schema_version=int(data["schema_version"]),
                metric=str(data["metric"]),
                anchor=str(data["anchor"]),
                deployment_start_date=to_date(data["deployment_start_date"]),
                last_week_ending=to_date(data["last_week_ending"]),
                cohort_column=str(cohort_definition["cohort_column"]),
                cohort_value=int(cohort_definition["cohort_value"]),
                new_patients_column=str(data["new_patients_column"]),
                cumulative_patients_column=str(data["cumulative_patients_column"]),
                row_count=int(data["row_count"]),
                week_count=int(data["week_count"]),
                patient_count=int(data["patient_count"]),
                updated_at=str(data["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid weekly unique patient metadata: {exc}") from exc


@dataclass(frozen=True)
class WeeklyRiskScoresMetadata:
    schema_version: int
    metric: str
    anchor: str
    deployment_start_date: date
    last_week_ending: date
    score_column: str
    regimen_column: str
    regimen_prefix_column: str
    cohort_column: str
    cohort_value: int
    row_count: int
    week_count: int
    updated_at: str

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "anchor": self.anchor,
            "deployment_start_date": self.deployment_start_date.isoformat(),
            "last_week_ending": self.last_week_ending.isoformat(),
            "cohort_definition": {
                "cohort_column": self.cohort_column,
                "cohort_value": self.cohort_value,
            },
            "score_column": self.score_column,
            "regimen_column": self.regimen_column,
            "regimen_prefix_column": self.regimen_prefix_column,
            "row_count": self.row_count,
            "week_count": self.week_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            cohort_definition = data["cohort_definition"]
            return cls(
                schema_version=int(data["schema_version"]),
                metric=str(data["metric"]),
                anchor=str(data["anchor"]),
                deployment_start_date=to_date(data["deployment_start_date"]),
                last_week_ending=to_date(data["last_week_ending"]),
                score_column=str(data["score_column"]),
                regimen_column=str(data["regimen_column"]),
                regimen_prefix_column=str(data["regimen_prefix_column"]),
                cohort_column=str(cohort_definition["cohort_column"]),
                cohort_value=int(cohort_definition["cohort_value"]),
                row_count=int(data["row_count"]),
                week_count=int(data["week_count"]),
                updated_at=str(data["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid metadata: {exc}") from exc


@dataclass(frozen=True)
class SilentDeploymentRiskScoresMetadata:
    schema_version: int
    anchor: str
    source_file: str
    score_column: str
    regimen_column: str
    regimen_prefix_column: str
    row_count: int
    source_modified_at: str
    source_mtime_ns: int
    created_at: str

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "anchor": self.anchor,
            "source_file": self.source_file,
            "score_column": self.score_column,
            "regimen_column": self.regimen_column,
            "regimen_prefix_column": self.regimen_prefix_column,
            "row_count": self.row_count,
            "source_modified_at": self.source_modified_at,
            "source_mtime_ns": self.source_mtime_ns,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                schema_version=int(data["schema_version"]),
                anchor=str(data["anchor"]),
                source_file=str(data["source_file"]),
                score_column=str(data["score_column"]),
                regimen_column=str(data["regimen_column"]),
                regimen_prefix_column=str(data["regimen_prefix_column"]),
                row_count=int(data["row_count"]),
                source_modified_at=str(data["source_modified_at"]),
                source_mtime_ns=int(data["source_mtime_ns"]),
                created_at=str(data["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid silent deployment metadata: {exc}") from exc


@dataclass(frozen=True)
class WeeklyMissingnessMetadata:
    schema_version: int
    metric: str
    anchor: str
    deployment_start_date: date
    last_week_ending: date
    cohort_column: str
    cohort_value: int
    missingness_suffix: str
    feature_columns: tuple[str, ...]
    row_count: int
    week_count: int
    updated_at: str

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "anchor": self.anchor,
            "deployment_start_date": self.deployment_start_date.isoformat(),
            "last_week_ending": self.last_week_ending.isoformat(),
            "cohort_definition": {
                "cohort_column": self.cohort_column,
                "cohort_value": self.cohort_value,
            },
            "missingness_suffix": self.missingness_suffix,
            "feature_columns": list(self.feature_columns),
            "row_count": self.row_count,
            "week_count": self.week_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            cohort_definition = data["cohort_definition"]
            return cls(
                schema_version=int(data["schema_version"]),
                metric=str(data["metric"]),
                anchor=str(data["anchor"]),
                deployment_start_date=to_date(data["deployment_start_date"]),
                last_week_ending=to_date(data["last_week_ending"]),
                cohort_column=str(cohort_definition["cohort_column"]),
                cohort_value=int(cohort_definition["cohort_value"]),
                missingness_suffix=str(data["missingness_suffix"]),
                feature_columns=tuple(str(col) for col in data["feature_columns"]),
                row_count=int(data["row_count"]),
                week_count=int(data["week_count"]),
                updated_at=str(data["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid metadata: {exc}") from exc


@dataclass(frozen=True)
class SilentDeploymentMissingnessMetadata:
    schema_version: int
    anchor: str
    source_file: str
    missingness_suffix: str
    feature_names: tuple[str, ...]
    row_count: int
    feature_count: int
    source_modified_at: str
    source_mtime_ns: int
    created_at: str

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "anchor": self.anchor,
            "source_file": self.source_file,
            "missingness_suffix": self.missingness_suffix,
            "feature_names": list(self.feature_names),
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "source_modified_at": self.source_modified_at,
            "source_mtime_ns": self.source_mtime_ns,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                schema_version=int(data["schema_version"]),
                anchor=str(data["anchor"]),
                source_file=str(data["source_file"]),
                missingness_suffix=str(data["missingness_suffix"]),
                feature_names=tuple(str(name) for name in data["feature_names"]),
                row_count=int(data["row_count"]),
                feature_count=int(data["feature_count"]),
                source_modified_at=str(data["source_modified_at"]),
                source_mtime_ns=int(data["source_mtime_ns"]),
                created_at=str(data["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid silent deployment missingness metadata: {exc}") from exc


# Row normalization, consistency checks, and path helpers


def clean_clinician_names(series):
    cleaned = series.astype("string").str.replace(r"\s*\[\s*\d+\s*\]\s*$", "", regex=True).str.strip()
    return cleaned.fillna(NO_CLINICIAN_LABEL).replace("", NO_CLINICIAN_LABEL)


def _order_clinician_columns(columns):
    names = sorted({str(col) for col in columns if str(col) != NO_CLINICIAN_LABEL})
    return [*names, NO_CLINICIAN_LABEL]


def _paths(output_dir, anchor):
    interim_dir = Path(output_dir) / WEEKLY_INTERIM_DIR
    return (
        interim_dir / f"weekly_alerts_{anchor}.csv",
        interim_dir / f"weekly_alerts_{anchor}_metadata.json",
        interim_dir / f"weekly_monitoring_{anchor}.lock",
    )


def _unique_patient_paths(output_dir, anchor):
    interim_dir = Path(output_dir) / WEEKLY_INTERIM_DIR
    return (
        interim_dir / f"unique_patients_{anchor}.csv",
        interim_dir / f"unique_patients_{anchor}_metadata.json",
        interim_dir / f"weekly_unique_patients_{anchor}.csv",
        interim_dir / f"weekly_unique_patients_{anchor}_metadata.json",
    )


def _risk_paths(output_dir, anchor):
    interim_dir = Path(output_dir) / WEEKLY_INTERIM_DIR
    return (
        interim_dir / f"weekly_risk_scores_{anchor}.csv",
        interim_dir / f"weekly_risk_scores_{anchor}_metadata.json",
        interim_dir / f"silent_deployment_risk_scores_{anchor}.csv",
        interim_dir / f"silent_deployment_risk_scores_{anchor}_metadata.json",
    )


def _missingness_paths(output_dir, anchor):
    interim_dir = Path(output_dir) / WEEKLY_INTERIM_DIR
    return (
        interim_dir / f"weekly_missingness_counts_{anchor}.csv",
        interim_dir / f"weekly_missingness_counts_{anchor}_metadata.json",
        interim_dir / f"silent_deployment_missingness_{anchor}.csv",
        interim_dir / f"silent_deployment_missingness_{anchor}_metadata.json",
    )


def _silent_source_path(output_dir, anchor):
    return Path(output_dir) / f"silent_deployment_output_{anchor}.csv"


def _missingness_source_dir(output_dir):
    return Path(output_dir) / MISSINGNESS_DIR


def _silent_missingness_source_path(output_dir, anchor):
    return _missingness_source_dir(output_dir) / f"{SILENT_MISSINGNESS_PREFIX}_{anchor}.csv"


def _missingness_file_pattern(anchor):
    return re.compile(rf"^missingness_(\d{{8}})_{re.escape(anchor)}\.csv$")


def _discover_missingness_files(output_dir, anchor, start_date, end_date):
    source_dir = _missingness_source_dir(output_dir)
    if not source_dir.is_dir():
        return []

    start_date = to_date(start_date)
    end_date = to_date(end_date)
    pattern = _missingness_file_pattern(anchor)
    found = []

    for path in source_dir.glob(f"missingness_*_{anchor}.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        file_date = to_date(match.group(1))
        if start_date <= file_date <= end_date:
            found.append((file_date, path))

    found.sort(key=lambda item: item[0])
    return found


def _read_metadata(path):
    return read_metadata(
        path,
        WeeklyAlertsMetadata,
        WEEKLY_SCHEMA_VERSION,
        label="metadata",
    )


def _validate_metadata(metadata, anchor, deployment_start_date):
    if metadata.metric != WEEKLY_ALERTS_METRIC:
        raise MonitoringPipelineError(f"Metadata metric {metadata.metric!r} does not match {WEEKLY_ALERTS_METRIC!r}")
    validate_metadata_identity(metadata, anchor, deployment_start_date)
    if metadata.clinician_column != CLINICIAN_COLUMN:
        raise MonitoringPipelineError(
            f"Metadata clinician column {metadata.clinician_column!r} does not match {CLINICIAN_COLUMN!r}"
        )
    if metadata.cohort_column != COHORT_COLUMN or metadata.cohort_value != COHORT_VALUE:
        raise MonitoringPipelineError(f"Metadata alert definition does not match {COHORT_COLUMN} == {COHORT_VALUE}")


def _read_unique_patient_registry_metadata(path):
    return read_metadata(
        path,
        UniquePatientsRegistryMetadata,
        UNIQUE_PATIENTS_SCHEMA_VERSION,
        label="unique patient registry metadata",
    )


def _validate_unique_patient_registry_metadata(metadata, anchor, deployment_start_date):
    if metadata.metric != UNIQUE_PATIENTS_REGISTRY_METRIC:
        raise MonitoringPipelineError(
            f"Metadata metric {metadata.metric!r} does not match {UNIQUE_PATIENTS_REGISTRY_METRIC!r}"
        )
    validate_metadata_identity(metadata, anchor, deployment_start_date)
    if metadata.mrn_column != MRN_COLUMN or metadata.first_seen_date_column != FIRST_SEEN_DATE_COLUMN:
        raise MonitoringPipelineError(
            f"Unique patient registry columns do not match {MRN_COLUMN!r}, {FIRST_SEEN_DATE_COLUMN!r}"
        )
    if metadata.cohort_column != COHORT_COLUMN or metadata.cohort_value != COHORT_VALUE:
        raise MonitoringPipelineError(f"Metadata cohort definition does not match {COHORT_COLUMN} == {COHORT_VALUE}")


def _read_weekly_unique_patients_metadata(path):
    return read_metadata(
        path,
        WeeklyUniquePatientsMetadata,
        UNIQUE_PATIENTS_SCHEMA_VERSION,
        label="weekly unique patient metadata",
    )


def _validate_weekly_unique_patients_metadata(metadata, anchor, deployment_start_date):
    if metadata.metric != WEEKLY_UNIQUE_PATIENTS_METRIC:
        raise MonitoringPipelineError(
            f"Metadata metric {metadata.metric!r} does not match {WEEKLY_UNIQUE_PATIENTS_METRIC!r}"
        )
    validate_metadata_identity(metadata, anchor, deployment_start_date)
    if metadata.cohort_column != COHORT_COLUMN or metadata.cohort_value != COHORT_VALUE:
        raise MonitoringPipelineError(f"Metadata cohort definition does not match {COHORT_COLUMN} == {COHORT_VALUE}")
    if metadata.new_patients_column != NEW_PATIENTS_COLUMN:
        raise MonitoringPipelineError(
            f"Metadata new patient column {metadata.new_patients_column!r} does not match {NEW_PATIENTS_COLUMN!r}"
        )
    if metadata.cumulative_patients_column != CUMULATIVE_PATIENTS_COLUMN:
        raise MonitoringPipelineError(
            "Metadata cumulative patient column "
            f"{metadata.cumulative_patients_column!r} does not match {CUMULATIVE_PATIENTS_COLUMN!r}"
        )


def _read_risk_metadata(path):
    return read_metadata(
        path,
        WeeklyRiskScoresMetadata,
        WEEKLY_SCHEMA_VERSION,
        label="metadata",
    )


def _validate_risk_metadata(metadata, anchor, deployment_start_date):
    if metadata.metric != WEEKLY_RISK_SCORES_METRIC:
        raise MonitoringPipelineError(
            f"Metadata metric {metadata.metric!r} does not match {WEEKLY_RISK_SCORES_METRIC!r}"
        )
    validate_metadata_identity(metadata, anchor, deployment_start_date)
    if metadata.score_column != SCORE_COLUMN:
        raise MonitoringPipelineError(
            f"Metadata score column {metadata.score_column!r} does not match {SCORE_COLUMN!r}"
        )
    if metadata.regimen_column != REGIMEN_COLUMN:
        raise MonitoringPipelineError(
            f"Metadata regimen column {metadata.regimen_column!r} does not match {REGIMEN_COLUMN!r}"
        )
    if metadata.regimen_prefix_column != REGIMEN_PREFIX_COLUMN:
        raise MonitoringPipelineError(
            "Metadata regimen prefix column "
            f"{metadata.regimen_prefix_column!r} does not match "
            f"{REGIMEN_PREFIX_COLUMN!r}"
        )
    if metadata.cohort_column != COHORT_COLUMN or metadata.cohort_value != COHORT_VALUE:
        raise MonitoringPipelineError(f"Metadata cohort definition does not match {COHORT_COLUMN} == {COHORT_VALUE}")


def _read_silent_metadata(path):
    return read_metadata(
        path,
        SilentDeploymentRiskScoresMetadata,
        SILENT_DEPLOYMENT_SCHEMA_VERSION,
        label="silent deployment metadata",
    )


def _read_missingness_metadata(path):
    return read_metadata(
        path,
        WeeklyMissingnessMetadata,
        MISSINGNESS_SCHEMA_VERSION,
        label="weekly missingness metadata",
    )


def _validate_missingness_metadata(metadata, anchor, deployment_start_date):
    if metadata.metric != WEEKLY_MISSINGNESS_METRIC:
        raise MonitoringPipelineError(
            f"Metadata metric {metadata.metric!r} does not match {WEEKLY_MISSINGNESS_METRIC!r}"
        )
    validate_metadata_identity(metadata, anchor, deployment_start_date)
    if metadata.cohort_column != COHORT_COLUMN or metadata.cohort_value != COHORT_VALUE:
        raise MonitoringPipelineError(f"Metadata cohort definition does not match {COHORT_COLUMN} == {COHORT_VALUE}")
    if metadata.missingness_suffix != MISSINGNESS_SUFFIX:
        raise MonitoringPipelineError(
            f"Metadata missingness suffix {metadata.missingness_suffix!r} does not match {MISSINGNESS_SUFFIX!r}"
        )


def _read_silent_missingness_metadata(path):
    return read_metadata(
        path,
        SilentDeploymentMissingnessMetadata,
        SILENT_MISSINGNESS_SCHEMA_VERSION,
        label="silent deployment missingness metadata",
    )


def _with_week_ending(df, date_col):
    table = df.copy()
    table[date_col] = pd.to_datetime(table[date_col])
    table = table[table[date_col].dt.weekday <= WEEK_ENDING_WEEKDAY].copy()
    table["week_ending"] = table[date_col] + pd.to_timedelta(
        (WEEK_ENDING_WEEKDAY - table[date_col].dt.weekday) % 7, unit="D"
    )
    return table


def _alerts_from_prediction_frame(df, consented_physicians, first_week_ending, last_week_ending, date_col):
    alerts = df.loc[(df[COHORT_COLUMN] == COHORT_VALUE) & consented_provider_mask(df, consented_physicians)].copy()
    alerts = _with_week_ending(alerts, date_col)

    if alerts.empty:
        counts = pd.DataFrame()
    else:
        alerts["_clinician"] = clean_clinician_names(alerts[CLINICIAN_COLUMN])
        counts = alerts.groupby(["week_ending", "_clinician"], dropna=False).size().unstack(fill_value=0)
        counts.columns.name = None

    fridays = week_ending_range(first_week_ending, last_week_ending)
    counts = counts.reindex(fridays, fill_value=0)
    if NO_CLINICIAN_LABEL not in counts.columns:
        counts[NO_CLINICIAN_LABEL] = 0
    counts = counts.reindex(columns=_order_clinician_columns(counts.columns), fill_value=0)
    counts = counts.astype(int)
    counts.index.name = "week_ending"
    return counts.reset_index()


def _regimen_prefixes(series):
    return series.astype("string").str.strip().str[:REGIMEN_PREFIX_LENGTH].fillna("")


def _risk_scores_from_prediction_frame(df, consented_physicians, first_week_ending, last_week_ending, date_col):
    scores = df.loc[(df[COHORT_COLUMN] == COHORT_VALUE) & consented_provider_mask(df, consented_physicians)].copy()
    scores = _with_week_ending(scores, date_col)

    missing_columns = [col for col in (SCORE_COLUMN, REGIMEN_COLUMN) if col not in scores.columns]
    if missing_columns:
        raise MonitoringPipelineError(f"Missing required weekly risk score column(s): {missing_columns}")

    numeric_scores = pd.to_numeric(scores[SCORE_COLUMN], errors="coerce")
    invalid = numeric_scores.isna() & scores[SCORE_COLUMN].notna()
    if invalid.any():
        raise MonitoringPipelineError(f"Non-numeric values found in weekly risk score column {SCORE_COLUMN!r}")

    missing = numeric_scores.isna()
    if missing.any():
        print(f"[WARN] Dropping {int(missing.sum())} weekly risk score(s) with missing values")

    scores = scores.loc[~missing].copy()
    scores[SCORE_COLUMN] = numeric_scores.loc[~missing]
    scores[REGIMEN_PREFIX_COLUMN] = _regimen_prefixes(scores[REGIMEN_COLUMN])
    scores = scores[scores["week_ending"].between(pd.Timestamp(first_week_ending), pd.Timestamp(last_week_ending))]
    scores = scores[["week_ending", SCORE_COLUMN, REGIMEN_PREFIX_COLUMN]].sort_values(
        ["week_ending", REGIMEN_PREFIX_COLUMN, SCORE_COLUMN],
        kind="mergesort",
    )
    return scores.reset_index(drop=True)


def _missingness_feature_columns(columns):
    suffix = MISSINGNESS_SUFFIX
    return sorted(str(col) for col in columns if str(col).endswith(suffix) and len(str(col)) > len(suffix))


def _missingness_feature_name(column):
    return str(column)[: -len(MISSINGNESS_SUFFIX)]


def _parse_missingness_flags(series, column, path):
    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise MonitoringPipelineError(f"Missing boolean value(s) in missingness column {column!r}: {path}")
        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
        }
    )
    if parsed.isna().any():
        raise MonitoringPipelineError(f"Invalid missingness flag(s) in column {column!r}: {path}")
    return parsed.astype(bool)


def _read_daily_missingness_pair(missingness_path, output_path, date_col, consented_physicians):
    missingness_path = Path(missingness_path)
    output_path = Path(output_path)
    if not output_path.exists():
        raise MonitoringPipelineError(f"Missing daily output file for weekly missingness: {output_path}")

    try:
        missingness_df = pd.read_csv(missingness_path)
        output_df = pd.read_csv(output_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise MonitoringPipelineError(
            f"Unable to read daily missingness/output pair {missingness_path}, {output_path}: {exc}"
        ) from exc

    if "idx" in missingness_df.columns:
        missingness_df = missingness_df.drop(columns=["idx"])
    missingness_columns = [col for col in (date_col, COHORT_COLUMN) if col not in missingness_df.columns]
    if missingness_columns:
        raise MonitoringPipelineError(
            f"Missing required missingness column(s) {missingness_columns}: {missingness_path}"
        )
    output_columns = [col for col in (COHORT_COLUMN, CLINICIAN_COLUMN) if col not in output_df.columns]
    if output_columns:
        raise MonitoringPipelineError(f"Missing required weekly output column(s): {output_columns}: {output_path}")
    if len(missingness_df) != len(output_df):
        raise MonitoringPipelineError(
            "Daily missingness/output row count mismatch for weekly missingness: "
            f"{missingness_path} ({len(missingness_df)}) vs "
            f"{output_path} ({len(output_df)})"
        )

    first_trt = pd.to_numeric(output_df[COHORT_COLUMN], errors="coerce").fillna(0).eq(1).to_numpy()
    consented = consented_provider_mask(output_df, consented_physicians).to_numpy()
    return missingness_df.loc[first_trt & consented].reset_index(drop=True)


def _load_missingness_window(pairs, anchor, output_dir, date_col, start_date, end_date, consented_physicians):
    frames = []
    feature_columns = set()

    for file_date, missingness_path in pairs:
        output_path = daily_prediction_path(output_dir, file_date, anchor)
        df = _read_daily_missingness_pair(missingness_path, output_path, date_col, consented_physicians)

        file_features = _missingness_feature_columns(df.columns)
        if not file_features:
            raise MonitoringPipelineError(
                f"No {MISSINGNESS_SUFFIX!r} columns found in missingness file: {missingness_path}"
            )

        feature_columns.update(file_features)
        for column in file_features:
            df[column] = _parse_missingness_flags(df[column], column, missingness_path)
        frames.append(df[[date_col, COHORT_COLUMN, *file_features]])

    combined = pd.concat(frames, ignore_index=True)
    features = sorted(feature_columns)
    for feature in features:
        if feature not in combined.columns:
            combined[feature] = False
    combined[features] = combined[features].fillna(False).astype(bool)

    try:
        combined[date_col] = pd.to_datetime(combined[date_col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise MonitoringPipelineError(f"Invalid {date_col!r} value in missingness files: {exc}") from exc

    combined = combined[combined[date_col].between(pd.Timestamp(start_date), pd.Timestamp(end_date))].copy()
    return combined, features


def _missingness_from_missingness_frame(df, first_week_ending, last_week_ending, date_col):
    feature_columns = _missingness_feature_columns(df.columns)
    if not feature_columns:
        raise MonitoringPipelineError(f"No {MISSINGNESS_SUFFIX!r} columns found in missingness data")

    feature_names = [_missingness_feature_name(column) for column in feature_columns]
    cohort = df[df[COHORT_COLUMN] == COHORT_VALUE].copy()
    cohort = _with_week_ending(cohort, date_col)

    fridays = week_ending_range(first_week_ending, last_week_ending)
    if cohort.empty:
        table = pd.DataFrame({"week_ending": fridays, "encounter_count": 0})
        for name in feature_names:
            table[name] = 0
        return table[["week_ending", "encounter_count", *feature_names]]

    grouped = cohort.groupby("week_ending", sort=True)
    counts = grouped[feature_columns].sum()
    counts.columns = feature_names
    denominator = grouped.size().rename("encounter_count")
    table = pd.concat([denominator, counts], axis=1).reset_index()

    table = table.set_index("week_ending").reindex(fridays)
    table["encounter_count"] = table["encounter_count"].fillna(0).astype(int)
    table[feature_names] = table[feature_names].fillna(0).astype(int)
    table = table.reset_index()
    return table[["week_ending", "encounter_count", *feature_names]]


def _build_missingness_table(
    deployment_start_date,
    first_week_ending,
    last_week_ending,
    anchor,
    output_dir,
    date_col,
    consented_physicians,
    dry_run,
):
    window_start = week_window_start(first_week_ending, deployment_start_date)
    files = _discover_missingness_files(
        output_dir,
        anchor,
        window_start,
        last_week_ending,
    )
    print(
        f"[INFO] Aggregating weekly missingness: "
        f"weeks={first_week_ending.isoformat()}..{last_week_ending.isoformat()} "
        f"daily_files={len(files)}"
    )
    print_files([path for _, path in files])

    if not files:
        print(
            "[WARN] No daily missingness files found for weekly missingness "
            f"window {window_start.isoformat()} to {last_week_ending.isoformat()}; "
            "skipping weekly missingness"
        )
        return None
    if dry_run:
        return None

    df, _ = _load_missingness_window(
        files,
        anchor,
        output_dir,
        date_col,
        window_start,
        last_week_ending,
        consented_physicians,
    )
    return _missingness_from_missingness_frame(df, first_week_ending, last_week_ending, date_col)


def _build_alerts_table(
    deployment_start_date,
    first_week_ending,
    last_week_ending,
    anchor,
    output_dir,
    date_col,
    consented_physicians,
    dry_run,
):
    window_start = week_window_start(first_week_ending, deployment_start_date)
    files = discover_prediction_files(
        output_dir,
        anchor,
        window_start,
        last_week_ending,
    )
    print(
        f"[INFO] Aggregating weekly alerts: "
        f"weeks={first_week_ending.isoformat()}..{last_week_ending.isoformat()} "
        f"daily_files={len(files)}"
    )
    print_files(files)

    if not files:
        raise MonitoringPipelineError(
            "No daily prediction files found for weekly alerts window "
            f"{window_start.isoformat()} to {last_week_ending.isoformat()}"
        )
    if dry_run:
        return None

    df = load_prediction_window(files, date_col, window_start, last_week_ending)
    if df.empty:
        raise MonitoringPipelineError(
            "No prediction rows found for weekly alerts window "
            f"{window_start.isoformat()} to {last_week_ending.isoformat()}"
        )

    missing = [col for col in (COHORT_COLUMN, CLINICIAN_COLUMN) if col not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"Missing required weekly alert column(s): {missing}")

    return _alerts_from_prediction_frame(df, consented_physicians, first_week_ending, last_week_ending, date_col)


def _unique_patients_from_prediction_frame(df, consented_physicians, date_col):
    missing = [col for col in (COHORT_COLUMN, CLINICIAN_COLUMN, MRN_COLUMN, date_col) if col not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"Missing required weekly unique patient column(s): {missing}")

    eligible = df.loc[(df[COHORT_COLUMN] == COHORT_VALUE) & consented_provider_mask(df, consented_physicians)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=[MRN_COLUMN, FIRST_SEEN_DATE_COLUMN])

    mrn = eligible[MRN_COLUMN].astype("string").str.strip()
    valid_mrn = mrn.notna() & mrn.ne("").fillna(False)
    eligible = eligible.loc[valid_mrn].copy()
    eligible[MRN_COLUMN] = mrn.loc[valid_mrn].astype(str)

    dates = pd.to_datetime(eligible[date_col], errors="coerce")
    if dates.isna().any():
        raise MonitoringPipelineError(f"Invalid {date_col!r} value(s) in weekly unique patient source rows")
    eligible = eligible.loc[dates.dt.weekday <= WEEK_ENDING_WEEKDAY].copy()
    dates = dates.loc[eligible.index]
    eligible[FIRST_SEEN_DATE_COLUMN] = dates

    return (
        eligible.groupby(MRN_COLUMN, as_index=False)[FIRST_SEEN_DATE_COLUMN]
        .min()
        .sort_values(MRN_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )


def _build_unique_patients_registry(
    deployment_start_date,
    first_week_ending,
    last_week_ending,
    anchor,
    output_dir,
    date_col,
    consented_physicians,
    dry_run,
):
    window_start = week_window_start(first_week_ending, deployment_start_date)
    files = discover_prediction_files(output_dir, anchor, window_start, last_week_ending)
    print(
        f"[INFO] Aggregating weekly unique patients: "
        f"weeks={first_week_ending.isoformat()}..{last_week_ending.isoformat()} "
        f"daily_files={len(files)}"
    )
    print_files(files)

    if not files:
        raise MonitoringPipelineError(
            "No daily prediction files found for weekly unique patients window "
            f"{window_start.isoformat()} to {last_week_ending.isoformat()}"
        )
    if dry_run:
        return None

    df = load_prediction_window(files, date_col, window_start, last_week_ending)
    if df.empty:
        raise MonitoringPipelineError(
            "No prediction rows found for weekly unique patients window "
            f"{window_start.isoformat()} to {last_week_ending.isoformat()}"
        )
    return _unique_patients_from_prediction_frame(df, consented_physicians, date_col)


def _build_risk_scores_table(
    deployment_start_date,
    first_week_ending,
    last_week_ending,
    anchor,
    output_dir,
    date_col,
    consented_physicians,
    dry_run,
):
    window_start = week_window_start(first_week_ending, deployment_start_date)
    files = discover_prediction_files(
        output_dir,
        anchor,
        window_start,
        last_week_ending,
    )
    print(
        f"[INFO] Aggregating weekly risk scores: "
        f"weeks={first_week_ending.isoformat()}..{last_week_ending.isoformat()} "
        f"daily_files={len(files)}"
    )
    print_files(files)

    if not files:
        raise MonitoringPipelineError(
            "No daily prediction files found for weekly risk scores window "
            f"{window_start.isoformat()} to {last_week_ending.isoformat()}"
        )
    if dry_run:
        return None

    df = load_prediction_window(files, date_col, window_start, last_week_ending)
    if df.empty:
        raise MonitoringPipelineError(
            "No prediction rows found for weekly risk scores window "
            f"{window_start.isoformat()} to {last_week_ending.isoformat()}"
        )

    missing = [col for col in (COHORT_COLUMN, SCORE_COLUMN, REGIMEN_COLUMN) if col not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"Missing required weekly risk score column(s): {missing}")

    return _risk_scores_from_prediction_frame(df, consented_physicians, first_week_ending, last_week_ending, date_col)


def _read_alerts_csv(path):
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise MonitoringPipelineError(f"Unable to read weekly alerts {path}: {exc}") from exc

    if "week_ending" not in df.columns:
        raise MonitoringPipelineError(f"Weekly alerts file is missing required column 'week_ending': {path}")
    try:
        df["week_ending"] = pd.to_datetime(df["week_ending"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise MonitoringPipelineError(f"Invalid week_ending value in weekly alerts {path}: {exc}") from exc
    if df["week_ending"].duplicated().any():
        raise MonitoringPipelineError(f"Weekly alerts file contains duplicate week_ending values: {path}")
    return df


def _merge_alerts(existing, new):
    if existing is None or existing.empty:
        combined = new.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)

    combined["week_ending"] = pd.to_datetime(combined["week_ending"])
    combined = combined.sort_values("week_ending", kind="mergesort")
    combined = combined.drop_duplicates(subset=["week_ending"], keep="last")

    clinician_columns = _order_clinician_columns([col for col in combined.columns if col != "week_ending"])
    combined = combined.set_index("week_ending")
    combined = combined.reindex(columns=clinician_columns, fill_value=0)
    combined = combined.fillna(0).astype(int)
    combined.index.name = "week_ending"
    return combined.reset_index()


def _reindex_week_range(df, deployment_start_date, last_week_ending):
    first_week_ending = first_friday_on_or_after(deployment_start_date)
    fridays = week_ending_range(first_week_ending, last_week_ending)
    clinician_columns = _order_clinician_columns([col for col in df.columns if col != "week_ending"])
    indexed = df.set_index(pd.to_datetime(df["week_ending"]))
    indexed = indexed.reindex(index=fridays, columns=clinician_columns, fill_value=0)
    indexed = indexed.fillna(0).astype(int)
    indexed.index.name = "week_ending"
    return indexed.reset_index()


def _write_alerts_csv(df, path):
    output = df.copy()
    output["week_ending"] = pd.to_datetime(output["week_ending"]).dt.strftime("%Y-%m-%d")
    atomic_write_csv(output, path)


def _week_count(df):
    return int(df["week_ending"].nunique()) if not df.empty else 0


def _read_unique_patients_registry_csv(path):
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise MonitoringPipelineError(f"Unable to read unique patient registry {path}: {exc}") from exc

    missing = [col for col in (MRN_COLUMN, FIRST_SEEN_DATE_COLUMN) if col not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"Unique patient registry is missing required column(s) {missing}: {path}")

    mrn = df[MRN_COLUMN].astype("string").str.strip()
    if not (mrn.notna() & mrn.ne("").fillna(False)).all():
        raise MonitoringPipelineError(f"Unique patient registry contains missing MRN values: {path}")
    if mrn.duplicated().any():
        raise MonitoringPipelineError(f"Unique patient registry contains duplicate MRN values: {path}")

    try:
        dates = pd.to_datetime(df[FIRST_SEEN_DATE_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise MonitoringPipelineError(f"Invalid first seen date in unique patient registry {path}: {exc}") from exc
    if dates.isna().any():
        raise MonitoringPipelineError(f"Unique patient registry contains missing first seen dates: {path}")

    table = pd.DataFrame({MRN_COLUMN: mrn.astype(str), FIRST_SEEN_DATE_COLUMN: dates})
    return table.sort_values(MRN_COLUMN, kind="mergesort").reset_index(drop=True)


def _merge_unique_patients_registries(existing, new):
    frames = [frame for frame in (existing, new) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=[MRN_COLUMN, FIRST_SEEN_DATE_COLUMN])

    combined = pd.concat(frames, ignore_index=True)
    combined[MRN_COLUMN] = combined[MRN_COLUMN].astype("string").str.strip()
    combined[FIRST_SEEN_DATE_COLUMN] = pd.to_datetime(combined[FIRST_SEEN_DATE_COLUMN])
    return (
        combined.groupby(MRN_COLUMN, as_index=False)[FIRST_SEEN_DATE_COLUMN]
        .min()
        .sort_values(MRN_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )


def _write_unique_patients_registry_csv(df, path):
    output = df[[MRN_COLUMN, FIRST_SEEN_DATE_COLUMN]].copy()
    if output.empty:
        output[FIRST_SEEN_DATE_COLUMN] = pd.Series(dtype="string")
    else:
        output[FIRST_SEEN_DATE_COLUMN] = pd.to_datetime(output[FIRST_SEEN_DATE_COLUMN]).dt.strftime("%Y-%m-%d")
    atomic_write_csv(output.sort_values(MRN_COLUMN, kind="mergesort"), path)


def _unique_patients_registry_consistent(df, metadata):
    if len(df) != metadata.row_count:
        return False
    if df.empty:
        return True
    if df[FIRST_SEEN_DATE_COLUMN].max() > pd.Timestamp(metadata.last_week_ending):
        return False
    return True


def _unique_patients_registry_metadata(anchor, deployment_start_date, last_week_ending, table, metadata):
    return UniquePatientsRegistryMetadata(
        schema_version=UNIQUE_PATIENTS_SCHEMA_VERSION,
        metric=UNIQUE_PATIENTS_REGISTRY_METRIC,
        anchor=anchor,
        deployment_start_date=deployment_start_date,
        last_week_ending=last_week_ending,
        mrn_column=MRN_COLUMN,
        first_seen_date_column=FIRST_SEEN_DATE_COLUMN,
        cohort_column=COHORT_COLUMN,
        cohort_value=COHORT_VALUE,
        row_count=len(table),
        updated_at=utc_now(),
    )


def _weekly_unique_patients_from_registry(registry, deployment_start_date, last_week_ending):
    first_week_ending = first_friday_on_or_after(deployment_start_date)
    fridays = week_ending_range(first_week_ending, last_week_ending)
    weekly_rows = _with_week_ending(registry, FIRST_SEEN_DATE_COLUMN)

    counts = weekly_rows.groupby("week_ending", sort=True).size().rename(NEW_PATIENTS_COLUMN)
    counts = counts.reindex(fridays, fill_value=0).astype(int)
    table = pd.DataFrame(
        {
            "week_ending": fridays,
            NEW_PATIENTS_COLUMN: counts.to_numpy(),
            CUMULATIVE_PATIENTS_COLUMN: counts.cumsum().to_numpy(dtype=int),
        }
    )
    table.index.name = "week_ending"
    return table.reset_index(drop=True)


def _read_weekly_unique_patients_csv(path):
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise MonitoringPipelineError(f"Unable to read weekly unique patients {path}: {exc}") from exc

    required = ["week_ending", NEW_PATIENTS_COLUMN, CUMULATIVE_PATIENTS_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"Weekly unique patients file is missing required column(s) {missing}: {path}")

    try:
        df["week_ending"] = pd.to_datetime(df["week_ending"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise MonitoringPipelineError(f"Invalid week_ending value in weekly unique patients {path}: {exc}") from exc
    if df["week_ending"].duplicated().any():
        raise MonitoringPipelineError(f"Weekly unique patients file contains duplicate week_ending values: {path}")

    for column in (NEW_PATIENTS_COLUMN, CUMULATIVE_PATIENTS_COLUMN):
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid {column} value in weekly unique patients {path}: {exc}") from exc
        if df[column].isna().any() or (df[column] < 0).any():
            raise MonitoringPipelineError(f"Invalid {column} value in weekly unique patients {path}")
        df[column] = df[column].astype(int)

    return df[required].sort_values("week_ending", kind="mergesort").reset_index(drop=True)


def _write_weekly_unique_patients_csv(df, path):
    output = df.copy()
    output["week_ending"] = pd.to_datetime(output["week_ending"]).dt.strftime("%Y-%m-%d")
    atomic_write_csv(output, path)


def _weekly_unique_patients_consistent(df, metadata):
    if df.empty or len(df) != metadata.row_count or _week_count(df) != metadata.week_count:
        return False
    if to_date(df["week_ending"].max()) != metadata.last_week_ending:
        return False
    if not df[CUMULATIVE_PATIENTS_COLUMN].eq(df[NEW_PATIENTS_COLUMN].cumsum()).all():
        return False
    if int(df[CUMULATIVE_PATIENTS_COLUMN].iloc[-1]) != metadata.patient_count:
        return False
    return True


def _weekly_unique_patients_metadata(anchor, deployment_start_date, last_week_ending, table, metadata):
    return WeeklyUniquePatientsMetadata(
        schema_version=UNIQUE_PATIENTS_SCHEMA_VERSION,
        metric=WEEKLY_UNIQUE_PATIENTS_METRIC,
        anchor=anchor,
        deployment_start_date=deployment_start_date,
        last_week_ending=last_week_ending,
        cohort_column=COHORT_COLUMN,
        cohort_value=COHORT_VALUE,
        new_patients_column=NEW_PATIENTS_COLUMN,
        cumulative_patients_column=CUMULATIVE_PATIENTS_COLUMN,
        row_count=len(table),
        week_count=_week_count(table),
        patient_count=int(table[CUMULATIVE_PATIENTS_COLUMN].iloc[-1]),
        updated_at=utc_now(),
    )


def _read_risk_scores_csv(path):
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise MonitoringPipelineError(f"Unable to read weekly risk scores {path}: {exc}") from exc

    missing = [col for col in ("week_ending", SCORE_COLUMN, REGIMEN_PREFIX_COLUMN) if col not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"Weekly risk scores file is missing required column(s) {missing}: {path}")

    try:
        df["week_ending"] = pd.to_datetime(df["week_ending"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise MonitoringPipelineError(f"Invalid week_ending value in weekly risk scores {path}: {exc}") from exc

    try:
        df[SCORE_COLUMN] = pd.to_numeric(df[SCORE_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise MonitoringPipelineError(f"Invalid risk score value in weekly risk scores {path}: {exc}") from exc

    if df[SCORE_COLUMN].isna().any():
        raise MonitoringPipelineError(f"Weekly risk scores file contains missing scores: {path}")
    df[REGIMEN_PREFIX_COLUMN] = df[REGIMEN_PREFIX_COLUMN].astype("string").fillna("")
    return df


def _risk_scores_consistent(df, metadata):
    if len(df) != metadata.row_count:
        return False
    if _week_count(df) != metadata.week_count:
        return False
    if not df.empty and df["week_ending"].max() > pd.Timestamp(metadata.last_week_ending):
        return False
    return True


def _write_risk_scores_csv(df, path):
    output = df[["week_ending", SCORE_COLUMN, REGIMEN_PREFIX_COLUMN]].copy()
    output["week_ending"] = pd.to_datetime(output["week_ending"]).dt.strftime("%Y-%m-%d")
    atomic_write_csv(output, path)


def _missingness_feature_names_from_table(df):
    return sorted(str(col) for col in df.columns if str(col) not in ("week_ending", "encounter_count"))


def _read_missingness_counts_csv(path):
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise MonitoringPipelineError(f"Unable to read weekly missingness counts {path}: {exc}") from exc

    required = ["week_ending", "encounter_count"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"Weekly missingness counts file is missing required column(s) {missing}: {path}")

    try:
        df["week_ending"] = pd.to_datetime(df["week_ending"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise MonitoringPipelineError(f"Invalid week_ending value in weekly missingness counts {path}: {exc}") from exc

    if df["week_ending"].duplicated().any():
        raise MonitoringPipelineError(f"Weekly missingness counts file contains duplicate week_ending values: {path}")

    feature_names = _missingness_feature_names_from_table(df)
    if not feature_names:
        raise MonitoringPipelineError(f"Weekly missingness counts file contains no feature columns: {path}")

    for column in ["encounter_count", *feature_names]:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise MonitoringPipelineError(
                f"Invalid missingness count value in column {column!r} in {path}: {exc}"
            ) from exc
        if df[column].isna().any():
            raise MonitoringPipelineError(f"Missing count value in column {column!r} in {path}")
        if (df[column] < 0).any():
            raise MonitoringPipelineError(f"Negative count value in column {column!r} in {path}")
        df[column] = df[column].astype(int)

    return df


def _merge_missingness_counts(existing, new):
    existing_features = (
        _missingness_feature_names_from_table(existing) if existing is not None and not existing.empty else []
    )
    new_features = _missingness_feature_names_from_table(new)
    feature_names = sorted({*existing_features, *new_features})

    if existing is None or existing.empty:
        combined = new.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)

    combined["week_ending"] = pd.to_datetime(combined["week_ending"])
    combined = combined.sort_values("week_ending", kind="mergesort")
    combined = combined.drop_duplicates(subset=["week_ending"], keep="last")

    for feature in feature_names:
        if feature not in combined.columns:
            combined[feature] = 0

    combined["encounter_count"] = pd.to_numeric(combined["encounter_count"], errors="raise")
    combined[feature_names] = combined[feature_names].apply(pd.to_numeric, errors="raise")
    combined["encounter_count"] = combined["encounter_count"].fillna(0).astype(int)
    combined[feature_names] = combined[feature_names].fillna(0).astype(int)
    return combined[["week_ending", "encounter_count", *feature_names]]


def _reindex_missingness_weeks(df, deployment_start_date, last_week_ending):
    first_week_ending = first_friday_on_or_after(deployment_start_date)
    fridays = week_ending_range(first_week_ending, last_week_ending)
    feature_names = _missingness_feature_names_from_table(df)
    indexed = df.set_index(pd.to_datetime(df["week_ending"]))
    columns = ["encounter_count", *feature_names]
    indexed = indexed.reindex(index=fridays, columns=columns)
    indexed["encounter_count"] = indexed["encounter_count"].fillna(0).astype(int)
    indexed[feature_names] = indexed[feature_names].fillna(0).astype(int)
    indexed.index.name = "week_ending"
    return indexed.reset_index()[["week_ending", "encounter_count", *feature_names]]


def _write_missingness_counts_csv(df, path):
    output = df.copy()
    output["week_ending"] = pd.to_datetime(output["week_ending"]).dt.strftime("%Y-%m-%d")
    atomic_write_csv(output, path)


def _missingness_counts_consistent(df, metadata):
    if len(df) != metadata.row_count:
        return False
    if _week_count(df) != metadata.week_count:
        return False
    if _missingness_feature_names_from_table(df) != list(metadata.feature_columns):
        return False
    if not df.empty and to_date(df["week_ending"].max()) != metadata.last_week_ending:
        return False
    return True


# Silent-source snapshots and baselines


def _read_cached_source_metadata(csv_path, metadata_path, reader):
    if csv_path.exists() and metadata_path.exists():
        try:
            return reader(metadata_path)
        except MonitoringPipelineError:
            return None
    return None


def _source_cache_needs_rebuild(csv_path, metadata, source_stat, anchor, *extra_checks):
    return (
        not csv_path.exists()
        or metadata is None
        or metadata.anchor != anchor
        or metadata.source_mtime_ns != source_stat.st_mtime_ns
        or any(check(metadata) for check in extra_checks)
    )


def _ensure_silent_snapshot(
    output_dir,
    anchor,
    silent_csv_path,
    silent_metadata_path,
    dry_run,
):
    silent_csv_path = Path(silent_csv_path)
    silent_metadata_path = Path(silent_metadata_path)
    source_path = _silent_source_path(output_dir, anchor)
    source_exists = source_path.exists()
    source_stat = source_path.stat() if source_exists else None

    metadata = _read_cached_source_metadata(silent_csv_path, silent_metadata_path, _read_silent_metadata)

    if not source_exists:
        if silent_csv_path.exists() and metadata is not None:
            print(
                f"[WARN] Silent deployment source file not found: {source_path}; keeping existing risk-score snapshot"
            )
            return
        if dry_run:
            print(f"[WARN] Silent deployment source file not found: {source_path}")
            return
        raise MonitoringPipelineError(f"Silent deployment source file not found: {source_path}")

    needs_rebuild = _source_cache_needs_rebuild(silent_csv_path, metadata, source_stat, anchor)
    if not needs_rebuild:
        return

    if dry_run:
        print(f"[INFO] Would rebuild silent deployment risk-score snapshot from {source_path}")
        return

    try:
        source = pd.read_csv(source_path, usecols=[SCORE_COLUMN, REGIMEN_COLUMN])
    except (FileNotFoundError, ValueError) as exc:
        raise MonitoringPipelineError(f"Unable to load silent deployment scores {source_path}: {exc}") from exc

    numeric_scores = pd.to_numeric(source[SCORE_COLUMN], errors="coerce")
    invalid = numeric_scores.isna() & source[SCORE_COLUMN].notna()
    if invalid.any():
        raise MonitoringPipelineError(f"Non-numeric silent deployment scores in {source_path}")
    if numeric_scores.isna().any():
        raise MonitoringPipelineError(f"Silent deployment scores contain missing values: {source_path}")

    snapshot = pd.DataFrame(
        {
            SCORE_COLUMN: numeric_scores,
            REGIMEN_PREFIX_COLUMN: _regimen_prefixes(source[REGIMEN_COLUMN]),
        }
    )
    atomic_write_csv(snapshot, silent_csv_path)

    metadata = SilentDeploymentRiskScoresMetadata(
        schema_version=SILENT_DEPLOYMENT_SCHEMA_VERSION,
        anchor=anchor,
        source_file=str(source_path),
        score_column=SCORE_COLUMN,
        regimen_column=REGIMEN_COLUMN,
        regimen_prefix_column=REGIMEN_PREFIX_COLUMN,
        row_count=len(snapshot),
        source_modified_at=datetime.fromtimestamp(source_stat.st_mtime, tz=timezone.utc).isoformat(),
        source_mtime_ns=source_stat.st_mtime_ns,
        created_at=utc_now(),
    )
    atomic_write_text(_metadata_text(metadata), silent_metadata_path)
    print(f"[INFO] Silent deployment risk-score snapshot written to {silent_csv_path}")


def _ensure_silent_missingness_baseline(
    output_dir,
    anchor,
    silent_csv_path,
    silent_metadata_path,
    dry_run,
):
    silent_csv_path = Path(silent_csv_path)
    silent_metadata_path = Path(silent_metadata_path)
    source_path = _silent_missingness_source_path(output_dir, anchor)
    source_exists = source_path.exists()
    source_stat = source_path.stat() if source_exists else None

    metadata = _read_cached_source_metadata(
        silent_csv_path,
        silent_metadata_path,
        _read_silent_missingness_metadata,
    )

    if not source_exists:
        if silent_csv_path.exists() and metadata is not None:
            print(
                f"[WARN] Silent deployment missingness source file not found: {source_path}; keeping existing baseline"
            )
        else:
            print(
                f"[WARN] Silent deployment missingness source file not found: "
                f"{source_path}; comparison baseline unavailable"
            )
        return

    needs_rebuild = _source_cache_needs_rebuild(
        silent_csv_path,
        metadata,
        source_stat,
        anchor,
        lambda cached: cached.missingness_suffix != MISSINGNESS_SUFFIX,
    )
    if not needs_rebuild:
        return

    if dry_run:
        print(f"[INFO] Would rebuild silent deployment missingness baseline from {source_path}")
        return

    try:
        source = pd.read_csv(source_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise MonitoringPipelineError(f"Unable to load silent deployment missingness {source_path}: {exc}") from exc

    if source.empty:
        raise MonitoringPipelineError(f"Silent deployment missingness source has no rows: {source_path}")

    feature_columns = _missingness_feature_columns(source.columns)
    if not feature_columns:
        raise MonitoringPipelineError(
            f"No {MISSINGNESS_SUFFIX!r} columns found in silent deployment missingness file: {source_path}"
        )

    rows = []
    for column in feature_columns:
        flags = _parse_missingness_flags(source[column], column, source_path)
        rows.append(
            {
                "feature": _missingness_feature_name(column),
                "missing_pct": float(flags.mean()),
            }
        )

    baseline = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    atomic_write_csv(baseline, silent_csv_path)

    metadata = SilentDeploymentMissingnessMetadata(
        schema_version=SILENT_MISSINGNESS_SCHEMA_VERSION,
        anchor=anchor,
        source_file=str(source_path),
        missingness_suffix=MISSINGNESS_SUFFIX,
        feature_names=tuple(str(name) for name in baseline["feature"]),
        row_count=len(source),
        feature_count=len(baseline),
        source_modified_at=datetime.fromtimestamp(source_stat.st_mtime, tz=timezone.utc).isoformat(),
        source_mtime_ns=source_stat.st_mtime_ns,
        created_at=utc_now(),
    )
    atomic_write_text(_metadata_text(metadata), silent_metadata_path)
    print(f"[INFO] Silent deployment missingness baseline written to {silent_csv_path}")


def _alerts_consistent(existing, metadata):
    if existing.empty:
        print("[WARN] Weekly alerts CSV is empty; rebuilding the baseline")
        return False
    existing_last = to_date(existing["week_ending"].max())
    if existing_last != metadata.last_week_ending:
        print("[WARN] Weekly alerts CSV and metadata disagree on the last week; rebuilding the baseline")
        return False
    return True


def _alerts_combine(existing, new_table, deployment_start_date, last_week_ending):
    return _merge_alerts(existing, new_table)


def _risk_scores_combine(existing, new_table, deployment_start_date, last_week_ending):
    if existing.empty:
        combined = new_table.copy()
    else:
        combined = pd.concat(
            [
                existing[["week_ending", SCORE_COLUMN, REGIMEN_PREFIX_COLUMN]],
                new_table,
            ],
            ignore_index=True,
        )
    return combined.sort_values(
        ["week_ending", REGIMEN_PREFIX_COLUMN, SCORE_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)


def _missingness_combine(existing, new_table, deployment_start_date, last_week_ending):
    return _merge_missingness_counts(existing, new_table)


def _alerts_metadata(anchor, deployment_start_date, last_week_ending, table, metadata):
    return WeeklyAlertsMetadata(
        schema_version=WEEKLY_SCHEMA_VERSION,
        metric=WEEKLY_ALERTS_METRIC,
        anchor=anchor,
        deployment_start_date=deployment_start_date,
        last_week_ending=last_week_ending,
        clinician_column=CLINICIAN_COLUMN,
        cohort_column=COHORT_COLUMN,
        cohort_value=COHORT_VALUE,
        updated_at=utc_now(),
    )


def _risk_scores_metadata(anchor, deployment_start_date, last_week_ending, table, metadata):
    return WeeklyRiskScoresMetadata(
        schema_version=WEEKLY_SCHEMA_VERSION,
        metric=WEEKLY_RISK_SCORES_METRIC,
        anchor=anchor,
        deployment_start_date=deployment_start_date,
        last_week_ending=last_week_ending,
        score_column=SCORE_COLUMN,
        regimen_column=REGIMEN_COLUMN,
        regimen_prefix_column=REGIMEN_PREFIX_COLUMN,
        cohort_column=COHORT_COLUMN,
        cohort_value=COHORT_VALUE,
        row_count=len(table),
        week_count=_week_count(table),
        updated_at=utc_now(),
    )


def _missingness_metadata(anchor, deployment_start_date, last_week_ending, table, metadata):
    return WeeklyMissingnessMetadata(
        schema_version=MISSINGNESS_SCHEMA_VERSION,
        metric=WEEKLY_MISSINGNESS_METRIC,
        anchor=anchor,
        deployment_start_date=deployment_start_date,
        last_week_ending=last_week_ending,
        cohort_column=COHORT_COLUMN,
        cohort_value=COHORT_VALUE,
        missingness_suffix=MISSINGNESS_SUFFIX,
        feature_columns=tuple(_missingness_feature_names_from_table(table)),
        row_count=len(table),
        week_count=_week_count(table),
        updated_at=utc_now(),
    )


def _create_weekly_baseline(
    spec,
    deployment_start_date,
    last_week_ending,
    anchor,
    output_dir,
    date_col,
    csv_path,
    metadata_path,
    consented_physicians,
    dry_run,
):
    first_week_ending = first_friday_on_or_after(deployment_start_date)
    if first_week_ending > last_week_ending:
        raise MonitoringPipelineError(
            f"No complete Friday-ending week between deployment start "
            f"{deployment_start_date.isoformat()} and last eligible Friday "
            f"{last_week_ending.isoformat()}"
        )

    print(
        f"[INFO] Creating weekly {spec['label']} baseline: "
        f"weeks={first_week_ending.isoformat()}..{last_week_ending.isoformat()}"
    )
    table = spec["build_table"](
        deployment_start_date,
        first_week_ending,
        last_week_ending,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )
    if table is None:
        return

    if spec["reindex"] is not None:
        table = spec["reindex"](table, deployment_start_date, last_week_ending)
    spec["write_csv"](table, csv_path)
    metadata = spec["metadata"](anchor, deployment_start_date, last_week_ending, table, None)
    atomic_write_text(_metadata_text(metadata), metadata_path)
    print(f"[INFO] Weekly {spec['label']} baseline written to {csv_path}")


def _update_weekly_baseline(
    spec,
    metadata,
    run_date,
    anchor,
    output_dir,
    date_col,
    csv_path,
    metadata_path,
    consented_physicians,
    dry_run,
):
    last_week_ending = last_eligible_friday(run_date)
    first_new_week = metadata.last_week_ending + timedelta(days=7)
    if first_new_week > last_week_ending:
        print(f"[INFO] No new complete {spec['no_new_week_label']} after {metadata.last_week_ending.isoformat()}")
        return

    print(f"[INFO] Updating weekly {spec['label']}: weeks={first_new_week.isoformat()}..{last_week_ending.isoformat()}")
    new_table = spec["build_table"](
        metadata.deployment_start_date,
        first_new_week,
        last_week_ending,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )
    if new_table is None:
        return

    existing = spec["read_csv"](csv_path)
    combined = spec["combine"](
        existing,
        new_table,
        metadata.deployment_start_date,
        last_week_ending,
    )
    if spec["reindex"] is not None:
        combined = spec["reindex"](combined, metadata.deployment_start_date, last_week_ending)
    spec["write_csv"](combined, csv_path)
    updated_metadata = spec["metadata"](
        anchor,
        metadata.deployment_start_date,
        last_week_ending,
        combined,
        metadata,
    )
    atomic_write_text(_metadata_text(updated_metadata), metadata_path)
    print(f"[INFO] Weekly {spec['label']} updated through {last_week_ending.isoformat()}; rows={len(combined)}")


# Generic weekly aggregate orchestration


def _run_weekly_aggregate(
    spec,
    deployment_start_date,
    run_date,
    anchor,
    output_dir,
    date_col,
    consented_physicians,
    dry_run,
):
    csv_path, metadata_path = spec["paths"](output_dir, anchor)[:2]
    last_week_ending = last_eligible_friday(run_date)
    create_kwargs = {
        "deployment_start_date": deployment_start_date,
        "last_week_ending": last_week_ending,
        "anchor": anchor,
        "output_dir": output_dir,
        "date_col": date_col,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "consented_physicians": consented_physicians,
        "dry_run": dry_run,
    }

    csv_exists = csv_path.exists()
    metadata_exists = metadata_path.exists()
    if not csv_exists or not metadata_exists:
        if csv_exists != metadata_exists:
            print(f"[WARN] Weekly {spec['label']} CSV and metadata are inconsistent; rebuilding the baseline")
        spec["create"](spec, **create_kwargs)
        return

    metadata = spec["read_metadata"](metadata_path)
    spec["validate_metadata"](metadata, anchor, deployment_start_date)
    existing = spec["read_csv"](csv_path)
    if not spec["consistent"](existing, metadata):
        if spec["warn_on_inconsistent"]:
            print(f"[WARN] Weekly {spec['label']} CSV and metadata are inconsistent; rebuilding the baseline")
        spec["create"](spec, **create_kwargs)
        return

    spec["update"](
        spec,
        metadata=metadata,
        run_date=run_date,
        anchor=anchor,
        output_dir=output_dir,
        date_col=date_col,
        csv_path=csv_path,
        metadata_path=metadata_path,
        consented_physicians=consented_physicians,
        dry_run=dry_run,
    )


def _create_unique_patients_baseline(
    deployment_start_date,
    last_week_ending,
    anchor,
    output_dir,
    date_col,
    registry_path,
    registry_metadata_path,
    weekly_path,
    weekly_metadata_path,
    consented_physicians,
    dry_run,
):
    first_week_ending = first_friday_on_or_after(deployment_start_date)
    if first_week_ending > last_week_ending:
        raise MonitoringPipelineError(
            f"No complete Friday-ending week between deployment start "
            f"{deployment_start_date.isoformat()} and last eligible Friday "
            f"{last_week_ending.isoformat()}"
        )

    print(
        f"[INFO] Creating weekly unique patient baseline: "
        f"weeks={first_week_ending.isoformat()}..{last_week_ending.isoformat()}"
    )
    registry = _build_unique_patients_registry(
        deployment_start_date,
        first_week_ending,
        last_week_ending,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )
    if registry is None:
        return

    weekly = _weekly_unique_patients_from_registry(registry, deployment_start_date, last_week_ending)
    _write_unique_patients_registry_csv(registry, registry_path)
    _write_weekly_unique_patients_csv(weekly, weekly_path)
    registry_metadata = _unique_patients_registry_metadata(
        anchor, deployment_start_date, last_week_ending, registry, None
    )
    weekly_metadata = _weekly_unique_patients_metadata(anchor, deployment_start_date, last_week_ending, weekly, None)
    atomic_write_text(_metadata_text(registry_metadata), registry_metadata_path)
    atomic_write_text(_metadata_text(weekly_metadata), weekly_metadata_path)
    print(
        f"[INFO] Weekly unique patient baseline written to {weekly_path}; "
        f"registry={registry_path}, patients={len(registry)}"
    )


def _update_unique_patients_baseline(
    registry_metadata,
    last_week_ending,
    deployment_start_date,
    anchor,
    output_dir,
    date_col,
    registry_path,
    registry_metadata_path,
    weekly_path,
    weekly_metadata_path,
    consented_physicians,
    dry_run,
):
    first_new_week = registry_metadata.last_week_ending + timedelta(days=7)
    print(f"[INFO] Updating weekly unique patients: weeks={first_new_week.isoformat()}..{last_week_ending.isoformat()}")
    new_registry = _build_unique_patients_registry(
        deployment_start_date,
        first_new_week,
        last_week_ending,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )
    if new_registry is None:
        return

    existing = _read_unique_patients_registry_csv(registry_path)
    combined = _merge_unique_patients_registries(existing, new_registry)
    weekly = _weekly_unique_patients_from_registry(combined, deployment_start_date, last_week_ending)

    _write_unique_patients_registry_csv(combined, registry_path)
    _write_weekly_unique_patients_csv(weekly, weekly_path)
    updated_registry_metadata = _unique_patients_registry_metadata(
        anchor, deployment_start_date, last_week_ending, combined, registry_metadata
    )
    updated_weekly_metadata = _weekly_unique_patients_metadata(
        anchor, deployment_start_date, last_week_ending, weekly, None
    )
    atomic_write_text(_metadata_text(updated_registry_metadata), registry_metadata_path)
    atomic_write_text(_metadata_text(updated_weekly_metadata), weekly_metadata_path)
    print(f"[INFO] Weekly unique patients updated through {last_week_ending.isoformat()}; patients={len(combined)}")


def _run_unique_patients_aggregate(
    deployment_start_date,
    run_date,
    anchor,
    output_dir,
    date_col,
    consented_physicians,
    dry_run,
):
    registry_path, registry_metadata_path, weekly_path, weekly_metadata_path = _unique_patient_paths(output_dir, anchor)
    last_week_ending = last_eligible_friday(run_date)
    paths = (registry_path, registry_metadata_path, weekly_path, weekly_metadata_path)
    create_kwargs = {
        "deployment_start_date": deployment_start_date,
        "last_week_ending": last_week_ending,
        "anchor": anchor,
        "output_dir": output_dir,
        "date_col": date_col,
        "registry_path": registry_path,
        "registry_metadata_path": registry_metadata_path,
        "weekly_path": weekly_path,
        "weekly_metadata_path": weekly_metadata_path,
        "consented_physicians": consented_physicians,
        "dry_run": dry_run,
    }

    if not all(path.exists() for path in paths):
        if any(path.exists() for path in paths):
            print("[WARN] Weekly unique patient files are inconsistent; rebuilding the baseline")
        _create_unique_patients_baseline(**create_kwargs)
        return

    registry_metadata = _read_unique_patient_registry_metadata(registry_metadata_path)
    _validate_unique_patient_registry_metadata(registry_metadata, anchor, deployment_start_date)
    weekly_metadata = _read_weekly_unique_patients_metadata(weekly_metadata_path)
    _validate_weekly_unique_patients_metadata(weekly_metadata, anchor, deployment_start_date)

    registry = _read_unique_patients_registry_csv(registry_path)
    weekly = _read_weekly_unique_patients_csv(weekly_path)
    consistent = (
        registry_metadata.last_week_ending == weekly_metadata.last_week_ending
        and _unique_patients_registry_consistent(registry, registry_metadata)
        and _weekly_unique_patients_consistent(weekly, weekly_metadata)
    )
    if not consistent:
        if dry_run:
            print("[WARN] Weekly unique patient files are inconsistent; would rebuild the baseline")
            return
        print("[WARN] Weekly unique patient files are inconsistent; rebuilding the baseline")
        _create_unique_patients_baseline(**create_kwargs)
        return

    if registry_metadata.last_week_ending >= last_week_ending:
        print(f"[INFO] No new complete unique-patient week after {registry_metadata.last_week_ending.isoformat()}")
        return

    _update_unique_patients_baseline(
        registry_metadata=registry_metadata,
        last_week_ending=last_week_ending,
        deployment_start_date=deployment_start_date,
        anchor=anchor,
        output_dir=output_dir,
        date_col=date_col,
        registry_path=registry_path,
        registry_metadata_path=registry_metadata_path,
        weekly_path=weekly_path,
        weekly_metadata_path=weekly_metadata_path,
        consented_physicians=consented_physicians,
        dry_run=dry_run,
    )


_ALERTS_AGGREGATE = {
    "label": "alerts",
    "no_new_week_label": "week",
    "paths": _paths,
    "read_metadata": _read_metadata,
    "validate_metadata": _validate_metadata,
    "read_csv": _read_alerts_csv,
    "consistent": _alerts_consistent,
    "warn_on_inconsistent": False,
    "build_table": _build_alerts_table,
    "reindex": _reindex_week_range,
    "write_csv": _write_alerts_csv,
    "combine": _alerts_combine,
    "metadata": _alerts_metadata,
    "create": _create_weekly_baseline,
    "update": _update_weekly_baseline,
}

_RISK_SCORES_AGGREGATE = {
    "label": "risk scores",
    "no_new_week_label": "risk-score week",
    "paths": _risk_paths,
    "read_metadata": _read_risk_metadata,
    "validate_metadata": _validate_risk_metadata,
    "read_csv": _read_risk_scores_csv,
    "consistent": _risk_scores_consistent,
    "warn_on_inconsistent": True,
    "build_table": _build_risk_scores_table,
    "reindex": None,
    "write_csv": _write_risk_scores_csv,
    "combine": _risk_scores_combine,
    "metadata": _risk_scores_metadata,
    "create": _create_weekly_baseline,
    "update": _update_weekly_baseline,
}

_MISSINGNESS_AGGREGATE = {
    "label": "missingness",
    "no_new_week_label": "missingness week",
    "paths": _missingness_paths,
    "read_metadata": _read_missingness_metadata,
    "validate_metadata": _validate_missingness_metadata,
    "read_csv": _read_missingness_counts_csv,
    "consistent": _missingness_counts_consistent,
    "warn_on_inconsistent": True,
    "build_table": _build_missingness_table,
    "reindex": _reindex_missingness_weeks,
    "write_csv": _write_missingness_counts_csv,
    "combine": _missingness_combine,
    "metadata": _missingness_metadata,
    "create": _create_weekly_baseline,
    "update": _update_weekly_baseline,
}


def _create_baseline(
    deployment_start_date,
    last_week_ending,
    anchor,
    output_dir,
    date_col,
    csv_path,
    metadata_path,
    consented_physicians,
    dry_run,
):
    _create_weekly_baseline(
        _ALERTS_AGGREGATE,
        deployment_start_date,
        last_week_ending,
        anchor,
        output_dir,
        date_col,
        csv_path,
        metadata_path,
        consented_physicians,
        dry_run,
    )


def _update_baseline(
    metadata,
    run_date,
    anchor,
    output_dir,
    date_col,
    csv_path,
    metadata_path,
    consented_physicians,
    dry_run,
):
    _update_weekly_baseline(
        _ALERTS_AGGREGATE,
        metadata,
        run_date,
        anchor,
        output_dir,
        date_col,
        csv_path,
        metadata_path,
        consented_physicians,
        dry_run,
    )


def _create_risk_baseline(
    deployment_start_date,
    last_week_ending,
    anchor,
    output_dir,
    date_col,
    csv_path,
    metadata_path,
    consented_physicians,
    dry_run,
):
    _create_weekly_baseline(
        _RISK_SCORES_AGGREGATE,
        deployment_start_date,
        last_week_ending,
        anchor,
        output_dir,
        date_col,
        csv_path,
        metadata_path,
        consented_physicians,
        dry_run,
    )


def _update_risk_baseline(
    metadata,
    run_date,
    anchor,
    output_dir,
    date_col,
    csv_path,
    metadata_path,
    consented_physicians,
    dry_run,
):
    _update_weekly_baseline(
        _RISK_SCORES_AGGREGATE,
        metadata,
        run_date,
        anchor,
        output_dir,
        date_col,
        csv_path,
        metadata_path,
        consented_physicians,
        dry_run,
    )


def _create_missingness_baseline(
    deployment_start_date,
    last_week_ending,
    anchor,
    output_dir,
    date_col,
    csv_path,
    metadata_path,
    consented_physicians,
    dry_run,
):
    _create_weekly_baseline(
        _MISSINGNESS_AGGREGATE,
        deployment_start_date,
        last_week_ending,
        anchor,
        output_dir,
        date_col,
        csv_path,
        metadata_path,
        consented_physicians,
        dry_run,
    )


def _update_missingness_baseline(
    metadata,
    run_date,
    anchor,
    output_dir,
    date_col,
    csv_path,
    metadata_path,
    consented_physicians,
    dry_run,
):
    _update_weekly_baseline(
        _MISSINGNESS_AGGREGATE,
        metadata,
        run_date,
        anchor,
        output_dir,
        date_col,
        csv_path,
        metadata_path,
        consented_physicians,
        dry_run,
    )


def _run_alerts_aggregate(
    deployment_start_date,
    run_date,
    anchor,
    output_dir,
    date_col,
    consented_physicians,
    dry_run,
):
    _run_weekly_aggregate(
        _ALERTS_AGGREGATE,
        deployment_start_date,
        run_date,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )


def _run_risk_scores_aggregate(
    deployment_start_date,
    run_date,
    anchor,
    output_dir,
    date_col,
    consented_physicians,
    dry_run,
):
    _run_weekly_aggregate(
        _RISK_SCORES_AGGREGATE,
        deployment_start_date,
        run_date,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )


def _run_missingness_aggregate(
    deployment_start_date,
    run_date,
    anchor,
    output_dir,
    date_col,
    consented_physicians,
    dry_run,
):
    _run_weekly_aggregate(
        _MISSINGNESS_AGGREGATE,
        deployment_start_date,
        run_date,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )


def _run_once(deployment_start_date, run_date, anchor, output_dir, consented_physician_file, dry_run):
    if anchor not in ("clinic", "treatment"):
        raise MonitoringPipelineError(f"Unsupported anchor: {anchor!r}")

    consented_physicians = None if dry_run else load_consented_physicians(consented_physician_file)
    date_col = {"clinic": "clinic_date", "treatment": "treatment_date"}[anchor]
    last_week_ending = last_eligible_friday(run_date)
    first_week_ending = first_friday_on_or_after(deployment_start_date)
    if first_week_ending > last_week_ending:
        raise MonitoringPipelineError(
            f"No complete Friday-ending week between deployment start "
            f"{deployment_start_date.isoformat()} and run date "
            f"{run_date.isoformat()}"
        )

    _, _, silent_csv_path, silent_metadata_path = _risk_paths(output_dir, anchor)
    _ensure_silent_snapshot(
        output_dir,
        anchor,
        silent_csv_path,
        silent_metadata_path,
        dry_run,
    )

    _, _, silent_missingness_csv_path, silent_missingness_metadata_path = _missingness_paths(output_dir, anchor)
    _ensure_silent_missingness_baseline(
        output_dir,
        anchor,
        silent_missingness_csv_path,
        silent_missingness_metadata_path,
        dry_run,
    )

    _run_alerts_aggregate(
        deployment_start_date,
        run_date,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )
    _run_unique_patients_aggregate(
        deployment_start_date,
        run_date,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )
    _run_risk_scores_aggregate(
        deployment_start_date,
        run_date,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )
    _run_missingness_aggregate(
        deployment_start_date,
        run_date,
        anchor,
        output_dir,
        date_col,
        consented_physicians,
        dry_run,
    )
    weekly_psi_monitoring.run_aggregate(
        deployment_start_date,
        run_date,
        anchor,
        output_dir,
        consented_physicians,
        dry_run,
    )


# Pipeline entry point


def run_pipeline(
    deployment_start_date,
    run_date,
    anchor,
    output_dir,
    consented_physician_file,
    dry_run=False,
):
    deployment_start_date = to_date(deployment_start_date)
    run_date = to_date(run_date)
    _, _, lock_path = _paths(output_dir, anchor)
    run_serialized(
        _run_once,
        lock_path=lock_path,
        dry_run=dry_run,
        deployment_start_date=deployment_start_date,
        run_date=run_date,
        anchor=anchor,
        output_dir=output_dir,
        consented_physician_file=consented_physician_file,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Maintain weekly live-deployment monitoring aggregates.")
    parser.add_argument(
        "--deployment-start-date",
        required=True,
        type=argparse_date,
        help="Live deployment monitoring start date (YYYYMMDD).",
    )
    parser.add_argument(
        "--run-date",
        type=argparse_date,
        default=None,
        help="As-of date used to aggregate weekly statistics (YYYYMMDD). Defaults to today.",
    )
    parser.add_argument(
        "--model-anchor",
        choices=["clinic", "treatment"],
        default="clinic",
    )
    parser.add_argument("--output-dir", default="./Outputs")
    parser.add_argument(
        "--consented-physician-file",
        required=True,
        help="CSV containing consented EpicProvider values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovery and planned work without writing files.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_date = args.run_date or date.today()
    try:
        run_pipeline(
            deployment_start_date=args.deployment_start_date,
            run_date=run_date,
            anchor=args.model_anchor,
            output_dir=args.output_dir,
            consented_physician_file=args.consented_physician_file,
            dry_run=args.dry_run,
        )
    except MonitoringPipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
