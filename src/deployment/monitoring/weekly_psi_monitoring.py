#!/usr/bin/env python3
"""Maintain weekly population stability index (PSI) monitoring."""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from deployer.monitoring.monitoring_common import (
    WEEKLY_INTERIM_DIR,
    MonitoringPipelineError,
    atomic_write_csv,
    atomic_write_text,
    consented_provider_mask,
    daily_prediction_path,
    discover_daily_input_files,
    friday_on_or_after,
    last_eligible_friday,
    metadata_text,
    print_files,
    read_metadata,
    to_date,
    utc_now,
    week_window_start,
)
from pandas.api.types import is_bool_dtype, is_numeric_dtype

_friday_on_or_after = friday_on_or_after
_last_eligible_friday = last_eligible_friday
_metadata_text = metadata_text

PSI_REFERENCE_SCHEMA_VERSION = 1
PSI_WEEKLY_SCHEMA_VERSION = 1
PSI_BINNING_VERSION = 1
PSI_N_QUANTILE_BINS = 10
PSI_EPSILON = 1e-6
PSI_INTERIM_DIR = WEEKLY_INTERIM_DIR
PSI_CATEGORICAL_GROUPS = {
    "cancer_site": "cancer_site_",
    "regimen": "regimen_",
    "intent": "intent_",
}
PSI_FEMALE_COLUMN = "female"
PSI_PROVIDER_COLUMN = "VISIT_PROVIDER_NAME"
PSI_SEX_FEATURE = "sex"
PSI_MISSING_LABEL = "__missing__"
PSI_OTHER_LABEL = "__other__"
PSI_NONE_LABEL = "__none__"
PSI_MULTI_LABEL = "__multi__"
PSI_SPECIAL_CATEGORICAL_LABELS = (
    PSI_NONE_LABEL,
    PSI_MULTI_LABEL,
    PSI_MISSING_LABEL,
    PSI_OTHER_LABEL,
)
PSI_COUNT_COLUMNS = (
    "week_ending",
    "reference_version",
    "feature",
    "feature_type",
    "bin_id",
    "bin_label",
    "count",
)
PSI_REFERENCE_BIN_COLUMNS = (
    "reference_version",
    "binning_version",
    "source_mtime_ns",
    "source_size",
    "activated_at",
    "first_week_ending",
    "feature",
    "feature_type",
    "bin_id",
    "bin_label",
    "lower_bound",
    "upper_bound",
    "reference_count",
    "reference_pct",
)


@dataclass(frozen=True)
class PsiFeatureSpec:
    name: str
    feature_type: str
    source_columns: tuple[str, ...]
    prefix: str | None = None
    true_label: str | None = None
    false_label: str | None = None


@dataclass(frozen=True)
class PsiReferenceVersion:
    reference_version: str
    binning_version: int
    source_mtime_ns: int
    source_size: int
    activated_at: str
    first_week_ending: date
    row_count: int
    feature_count: int
    bin_count: int
    feature_sources: dict[str, tuple[str, ...]]

    def to_dict(self):
        return {
            "reference_version": self.reference_version,
            "binning_version": self.binning_version,
            "source_mtime_ns": self.source_mtime_ns,
            "source_size": self.source_size,
            "activated_at": self.activated_at,
            "first_week_ending": self.first_week_ending.isoformat(),
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "bin_count": self.bin_count,
            "feature_sources": {feature: list(columns) for feature, columns in self.feature_sources.items()},
        }

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                reference_version=str(data["reference_version"]),
                binning_version=int(data["binning_version"]),
                source_mtime_ns=int(data["source_mtime_ns"]),
                source_size=int(data["source_size"]),
                activated_at=str(data["activated_at"]),
                first_week_ending=to_date(data["first_week_ending"]),
                row_count=int(data["row_count"]),
                feature_count=int(data["feature_count"]),
                bin_count=int(data["bin_count"]),
                feature_sources={
                    str(feature): tuple(str(col) for col in columns)
                    for feature, columns in data.get("feature_sources", {}).items()
                },
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid PSI reference version: {exc}") from exc


@dataclass(frozen=True)
class PsiReferenceMetadata:
    schema_version: int
    anchor: str
    active_reference_version: str
    source_file: str
    source_mtime_ns: int
    source_size: int
    generated_at: str
    binning_version: int
    feature_columns: tuple[str, ...]
    feature_sources: dict[str, tuple[str, ...]]
    categorical_groups: dict[str, str]
    n_quantile_bins: int
    epsilon: float
    versions: tuple[PsiReferenceVersion, ...]

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "anchor": self.anchor,
            "active_reference_version": self.active_reference_version,
            "source_file": self.source_file,
            "source_mtime_ns": self.source_mtime_ns,
            "source_size": self.source_size,
            "generated_at": self.generated_at,
            "binning_version": self.binning_version,
            "feature_columns": list(self.feature_columns),
            "feature_sources": {feature: list(columns) for feature, columns in self.feature_sources.items()},
            "categorical_groups": dict(self.categorical_groups),
            "n_quantile_bins": self.n_quantile_bins,
            "epsilon": self.epsilon,
            "versions": [version.to_dict() for version in self.versions],
        }

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                schema_version=int(data["schema_version"]),
                anchor=str(data["anchor"]),
                active_reference_version=str(data["active_reference_version"]),
                source_file=str(data["source_file"]),
                source_mtime_ns=int(data["source_mtime_ns"]),
                source_size=int(data["source_size"]),
                generated_at=str(data["generated_at"]),
                binning_version=int(data["binning_version"]),
                feature_columns=tuple(str(col) for col in data["feature_columns"]),
                feature_sources={
                    str(feature): tuple(str(col) for col in columns)
                    for feature, columns in data.get("feature_sources", {}).items()
                },
                categorical_groups={
                    str(name): str(prefix) for name, prefix in data.get("categorical_groups", {}).items()
                },
                n_quantile_bins=int(data["n_quantile_bins"]),
                epsilon=float(data["epsilon"]),
                versions=tuple(PsiReferenceVersion.from_dict(version) for version in data.get("versions", [])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid PSI reference metadata: {exc}") from exc


@dataclass(frozen=True)
class PsiWeeklyMetadata:
    schema_version: int
    anchor: str
    deployment_start_date: date
    last_week_ending: date
    reference_version: str
    row_count: int
    week_count: int
    feature_count: int
    bin_count: int
    updated_at: str

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "anchor": self.anchor,
            "deployment_start_date": self.deployment_start_date.isoformat(),
            "last_week_ending": self.last_week_ending.isoformat(),
            "reference_version": self.reference_version,
            "row_count": self.row_count,
            "week_count": self.week_count,
            "feature_count": self.feature_count,
            "bin_count": self.bin_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                schema_version=int(data["schema_version"]),
                anchor=str(data["anchor"]),
                deployment_start_date=to_date(data["deployment_start_date"]),
                last_week_ending=to_date(data["last_week_ending"]),
                reference_version=str(data["reference_version"]),
                row_count=int(data["row_count"]),
                week_count=int(data["week_count"]),
                feature_count=int(data["feature_count"]),
                bin_count=int(data["bin_count"]),
                updated_at=str(data["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid PSI weekly metadata: {exc}") from exc


def _psi_paths(output_dir, anchor):
    interim_dir = Path(output_dir) / PSI_INTERIM_DIR
    return (
        interim_dir / f"psi_reference_bins_{anchor}.csv",
        interim_dir / f"psi_reference_metadata_{anchor}.json",
        interim_dir / f"weekly_psi_counts_{anchor}.csv",
        interim_dir / f"weekly_psi_metadata_{anchor}.json",
        interim_dir / f"weekly_psi_scores_{anchor}.csv",
    )


def _reference_source_path(output_dir, anchor):
    return Path(output_dir) / f"silent_deployment_input_{anchor}.parquet"


def _reference_version_id(source_mtime_ns, source_size):
    return f"ref_{source_mtime_ns}_{source_size}_v{PSI_BINNING_VERSION}"


def _format_bound(value):
    return f"{float(value):g}"


def _feature_specs_from_reference(df):
    used_columns = set()
    specs = []

    for feature, prefix in PSI_CATEGORICAL_GROUPS.items():
        columns = tuple(str(column) for column in df.columns if str(column).startswith(prefix))
        if not columns:
            continue
        specs.append(
            PsiFeatureSpec(
                name=feature,
                feature_type="categorical",
                source_columns=columns,
                prefix=prefix,
            )
        )
        used_columns.update(columns)

    if PSI_FEMALE_COLUMN in df.columns:
        specs.append(
            PsiFeatureSpec(
                name=PSI_SEX_FEATURE,
                feature_type="binary",
                source_columns=(PSI_FEMALE_COLUMN,),
                true_label="female",
                false_label="male",
            )
        )
        used_columns.add(PSI_FEMALE_COLUMN)

    for column in df.columns:
        if column in used_columns:
            continue
        if str(column).endswith("_is_missing") or is_bool_dtype(df[column].dtype):
            specs.append(
                PsiFeatureSpec(
                    name=str(column),
                    feature_type="binary",
                    source_columns=(str(column),),
                    true_label="1",
                    false_label="0",
                )
            )
        elif is_numeric_dtype(df[column].dtype):
            specs.append(
                PsiFeatureSpec(
                    name=str(column),
                    feature_type="continuous",
                    source_columns=(str(column),),
                )
            )
        else:
            raise MonitoringPipelineError(f"Unsupported PSI reference feature {column!r} with dtype {df[column].dtype}")

    if not specs:
        raise MonitoringPipelineError("PSI reference has no feature columns")
    return specs


def _specs_from_bins(bins, metadata=None):
    specs = []
    feature_sources = metadata.feature_sources if metadata is not None else {}
    for feature in bins["feature"].drop_duplicates().tolist():
        feature = str(feature)
        rows = bins.loc[bins["feature"] == feature]
        feature_type = str(rows["feature_type"].iloc[0])
        if feature in PSI_CATEGORICAL_GROUPS:
            specs.append(
                PsiFeatureSpec(
                    name=feature,
                    feature_type=feature_type,
                    source_columns=tuple(feature_sources.get(feature, ())),
                    prefix=PSI_CATEGORICAL_GROUPS[feature],
                )
            )
        elif feature == PSI_SEX_FEATURE:
            specs.append(
                PsiFeatureSpec(
                    name=feature,
                    feature_type=feature_type,
                    source_columns=(PSI_FEMALE_COLUMN,),
                    true_label="female",
                    false_label="male",
                )
            )
        elif feature_type == "binary":
            specs.append(
                PsiFeatureSpec(
                    name=feature,
                    feature_type=feature_type,
                    source_columns=(feature,),
                    true_label="1",
                    false_label="0",
                )
            )
        else:
            specs.append(
                PsiFeatureSpec(
                    name=feature,
                    feature_type=feature_type,
                    source_columns=(feature,),
                )
            )
    return specs


def _categorical_group_labels(df, prefix, source_columns=()):
    columns = [str(column) for column in df.columns if str(column).startswith(prefix)]
    if not columns:
        raise MonitoringPipelineError(f"No live columns found for categorical PSI group {prefix!r}")

    missing = [column for column in source_columns if column not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"Missing categorical PSI source column(s) {missing}")

    values = df[columns].apply(pd.to_numeric, errors="coerce")
    all_missing = values.isna().all(axis=1)
    active = values.fillna(0) > 0
    active_count = active.sum(axis=1)

    labels = pd.Series(
        PSI_MISSING_LABEL,
        index=df.index,
        dtype="object",
    )
    labels.loc[~all_missing & (active_count == 0)] = PSI_NONE_LABEL

    one_active = ~all_missing & (active_count == 1)
    if one_active.any():
        chosen = active.loc[one_active].idxmax(axis=1)
        labels.loc[one_active] = chosen.map(lambda column: str(column)[len(prefix) :])

    labels.loc[~all_missing & (active_count > 1)] = PSI_MULTI_LABEL
    return labels


def _binary_labels(series, true_label, false_label):
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "1.0": 1.0,
            "0": 0.0,
            "0.0": 0.0,
        }
    )
    invalid = mapped.isna() & series.notna()
    if invalid.any():
        raise MonitoringPipelineError(f"Invalid binary PSI values in column {series.name!r}")

    labels = mapped.map({1.0: true_label, 0.0: false_label})
    labels = labels.astype("object")
    labels.loc[series.isna()] = PSI_MISSING_LABEL
    return labels


def _fit_continuous_feature(series):
    numeric = pd.to_numeric(series, errors="raise")
    non_missing = numeric.dropna()
    missing_count = int(numeric.isna().sum())
    total = len(numeric)
    bins = []

    if non_missing.empty or non_missing.nunique() <= 1:
        bins.append(
            {
                "bin_id": 0,
                "bin_label": "all",
                "lower_bound": np.nan,
                "upper_bound": np.nan,
                "reference_count": int(len(non_missing)),
            }
        )
    else:
        _, edges = pd.qcut(
            non_missing,
            q=PSI_N_QUANTILE_BINS,
            retbins=True,
            duplicates="drop",
        )
        edges = np.unique(np.asarray(edges, dtype=float))
        inner_edges = edges[1:-1]
        cut_edges = [-np.inf, *inner_edges.tolist(), np.inf]
        n_bins = len(cut_edges) - 1
        if n_bins == 1:
            bins.append(
                {
                    "bin_id": 0,
                    "bin_label": "all",
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                    "reference_count": int(len(non_missing)),
                }
            )
        else:
            codes = pd.cut(
                numeric,
                bins=cut_edges,
                labels=False,
                include_lowest=True,
            )
            counts = codes.value_counts().reindex(range(n_bins), fill_value=0).sort_index()

            for bin_id in range(n_bins):
                lower = None if bin_id == 0 else cut_edges[bin_id]
                upper = None if bin_id == n_bins - 1 else cut_edges[bin_id + 1]
                if bin_id == 0:
                    label = f"(-inf, {_format_bound(upper)}]"
                elif bin_id == n_bins - 1:
                    label = f"({_format_bound(lower)}, inf)"
                else:
                    label = f"({_format_bound(lower)}, {_format_bound(upper)}]"
                bins.append(
                    {
                        "bin_id": bin_id,
                        "bin_label": label,
                        "lower_bound": np.nan if lower is None else lower,
                        "upper_bound": np.nan if upper is None else upper,
                        "reference_count": int(counts.loc[bin_id]),
                    }
                )

    bins.append(
        {
            "bin_id": len(bins),
            "bin_label": PSI_MISSING_LABEL,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "reference_count": missing_count,
        }
    )

    rows = []
    for item in bins:
        rows.append(
            {
                **item,
                "reference_pct": (item["reference_count"] / total if total else 0.0),
            }
        )
    return rows


def _fit_categorical_feature(labels, total):
    observed = sorted(str(value) for value in labels.dropna().unique())
    categories = observed + [label for label in PSI_SPECIAL_CATEGORICAL_LABELS if label not in observed]
    counts = labels.value_counts().reindex(categories, fill_value=0)
    return [
        {
            "bin_id": bin_id,
            "bin_label": category,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "reference_count": int(counts.loc[category]),
            "reference_pct": (int(counts.loc[category]) / total if total else 0.0),
        }
        for bin_id, category in enumerate(categories)
    ]


def _fit_binary_feature(labels, total, true_label, false_label):
    categories = [false_label, true_label, PSI_MISSING_LABEL]
    counts = labels.value_counts().reindex(categories, fill_value=0)
    return [
        {
            "bin_id": bin_id,
            "bin_label": category,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "reference_count": int(counts.loc[category]),
            "reference_pct": (int(counts.loc[category]) / total if total else 0.0),
        }
        for bin_id, category in enumerate(categories)
    ]


def _build_reference_bins(df, specs, version_record):
    rows = []
    total = len(df)

    for spec in specs:
        if spec.feature_type == "continuous":
            print(spec)
            feature_bins = _fit_continuous_feature(df[spec.source_columns[0]])
        elif spec.feature_type == "categorical":
            labels = _categorical_group_labels(
                df,
                spec.prefix,
                spec.source_columns,
            )
            feature_bins = _fit_categorical_feature(labels, total)
        elif spec.feature_type == "binary":
            labels = _binary_labels(
                df[spec.source_columns[0]],
                spec.true_label,
                spec.false_label,
            )
            feature_bins = _fit_binary_feature(
                labels,
                total,
                spec.true_label,
                spec.false_label,
            )
        else:
            raise MonitoringPipelineError(f"Unsupported PSI feature type: {spec.feature_type!r}")

        for item in feature_bins:
            rows.append(
                {
                    "reference_version": version_record.reference_version,
                    "binning_version": version_record.binning_version,
                    "source_mtime_ns": version_record.source_mtime_ns,
                    "source_size": version_record.source_size,
                    "activated_at": version_record.activated_at,
                    "first_week_ending": pd.Timestamp(version_record.first_week_ending),
                    "feature": spec.name,
                    "feature_type": spec.feature_type,
                    **item,
                }
            )

    return pd.DataFrame(rows, columns=list(PSI_REFERENCE_BIN_COLUMNS))


def _read_reference_bins(path):
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise MonitoringPipelineError(f"Unable to read PSI reference bins {path}: {exc}") from exc

    missing = [column for column in PSI_REFERENCE_BIN_COLUMNS if column not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"PSI reference bins file is missing column(s) {missing}: {path}")

    try:
        df["first_week_ending"] = pd.to_datetime(
            df["first_week_ending"],
            errors="raise",
        )
        for column in (
            "binning_version",
            "source_mtime_ns",
            "source_size",
            "bin_id",
            "reference_count",
        ):
            df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
        for column in ("lower_bound", "upper_bound", "reference_pct"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
    except (TypeError, ValueError) as exc:
        raise MonitoringPipelineError(f"Invalid PSI reference bins values in {path}: {exc}") from exc

    if df["reference_pct"].isna().any():
        raise MonitoringPipelineError(f"PSI reference bins file contains missing proportions: {path}")

    keys = ["reference_version", "feature", "bin_id"]
    if df.duplicated(subset=keys).any():
        raise MonitoringPipelineError(f"PSI reference bins file contains duplicate bin keys: {path}")

    return df.sort_values(
        ["first_week_ending", "feature", "bin_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _read_reference_metadata(path):
    return read_metadata(
        path,
        PsiReferenceMetadata,
        PSI_REFERENCE_SCHEMA_VERSION,
        label="PSI reference metadata",
    )


def _read_weekly_metadata(path):
    return read_metadata(
        path,
        PsiWeeklyMetadata,
        PSI_WEEKLY_SCHEMA_VERSION,
        label="PSI weekly metadata",
    )


def _versions_from_bins(bins):
    versions = []
    for version, rows in bins.groupby("reference_version", sort=False):
        first = rows.iloc[0]
        versions.append(
            PsiReferenceVersion(
                reference_version=str(version),
                binning_version=int(first["binning_version"]),
                source_mtime_ns=int(first["source_mtime_ns"]),
                source_size=int(first["source_size"]),
                activated_at=str(first["activated_at"]),
                first_week_ending=to_date(first["first_week_ending"]),
                row_count=int(rows[["feature", "bin_id"]].drop_duplicates().shape[0]),
                feature_count=int(rows["feature"].nunique()),
                bin_count=int(len(rows)),
                feature_sources={},
            )
        )
    versions.sort(key=lambda item: item.first_week_ending)
    return tuple(versions)


def _metadata_from_bins(
    bins,
    anchor,
    active_version,
    source_path,
):
    active_rows = bins.loc[bins["reference_version"] == active_version]
    if active_rows.empty:
        raise MonitoringPipelineError(f"Active PSI reference version not found in bins: {active_version}")

    first = active_rows.iloc[0]
    specs = _specs_from_bins(active_rows)
    feature_sources = {spec.name: spec.source_columns for spec in specs if spec.source_columns}
    versions = list(_versions_from_bins(bins))
    versions = [version for version in versions if version.reference_version != active_version]
    versions.append(
        PsiReferenceVersion(
            reference_version=active_version,
            binning_version=int(first["binning_version"]),
            source_mtime_ns=int(first["source_mtime_ns"]),
            source_size=int(first["source_size"]),
            activated_at=str(first["activated_at"]),
            first_week_ending=to_date(first["first_week_ending"]),
            row_count=int(active_rows[["feature", "bin_id"]].drop_duplicates().shape[0]),
            feature_count=int(active_rows["feature"].nunique()),
            bin_count=int(len(active_rows)),
            feature_sources=feature_sources,
        )
    )
    versions.sort(key=lambda item: item.first_week_ending)

    return PsiReferenceMetadata(
        schema_version=PSI_REFERENCE_SCHEMA_VERSION,
        anchor=anchor,
        active_reference_version=active_version,
        source_file=str(source_path),
        source_mtime_ns=int(first["source_mtime_ns"]),
        source_size=int(first["source_size"]),
        generated_at=utc_now(),
        binning_version=int(first["binning_version"]),
        feature_columns=tuple(active_rows["feature"].drop_duplicates().tolist()),
        feature_sources=feature_sources,
        categorical_groups=dict(PSI_CATEGORICAL_GROUPS),
        n_quantile_bins=PSI_N_QUANTILE_BINS,
        epsilon=PSI_EPSILON,
        versions=tuple(versions),
    )


def _load_reference_source(path):
    try:
        df = pd.read_parquet(path)
    except (FileNotFoundError, ValueError) as exc:
        raise MonitoringPipelineError(f"Unable to load PSI reference source {path}: {exc}") from exc

    if df.empty:
        raise MonitoringPipelineError(f"PSI reference source has no rows: {path}")

    drop_columns = [
        column
        for column in (
            "mrn",
            "clinic_date",
            "treatment_date",
            "idx",
            "first_trt",
            "Unnamed: 0",
        )
        if column in df.columns
    ]
    if drop_columns:
        df = df.drop(columns=drop_columns)
    if df.empty or len(df.columns) == 0:
        raise MonitoringPipelineError(f"PSI reference source has no feature columns: {path}")
    return df


def _ensure_reference(output_dir, run_date, anchor, dry_run):
    bins_path, metadata_path, _, _, _ = _psi_paths(output_dir, anchor)
    source_path = _reference_source_path(output_dir, anchor)

    bins = None
    if bins_path.exists():
        try:
            bins = _read_reference_bins(bins_path)
        except MonitoringPipelineError:
            bins = None

    metadata = None
    if metadata_path.exists():
        try:
            metadata = _read_reference_metadata(metadata_path)
        except MonitoringPipelineError:
            metadata = None

    source_exists = source_path.exists()
    source_stat = source_path.stat() if source_exists else None

    if not source_exists:
        if bins is not None and metadata is not None:
            print(f"[WARN] PSI reference source file not found: {source_path}; keeping existing reference")
            return metadata, bins
        print(f"[WARN] PSI reference source file not found: {source_path}; skipping PSI aggregation")
        return None, None

    active_version = None
    if (
        metadata is not None
        and bins is not None
        and metadata.anchor == anchor
        and metadata.binning_version == PSI_BINNING_VERSION
        and metadata.source_mtime_ns == source_stat.st_mtime_ns
        and metadata.source_size == source_stat.st_size
        and metadata.active_reference_version in set(bins["reference_version"])
    ):
        active_version = metadata.active_reference_version

    if active_version is not None:
        return metadata, bins

    if bins is not None:
        matching = bins.loc[
            (bins["source_mtime_ns"] == source_stat.st_mtime_ns)
            & (bins["source_size"] == source_stat.st_size)
            & (bins["binning_version"] == PSI_BINNING_VERSION)
        ]
        if not matching.empty:
            active_version = str(matching["reference_version"].iloc[0])
            metadata = _metadata_from_bins(
                bins,
                anchor,
                active_version,
                source_path,
            )
            if not dry_run:
                atomic_write_text(_metadata_text(metadata), metadata_path)
            return metadata, bins

    if dry_run:
        print(f"[INFO] Would rebuild PSI reference bins from {source_path}")
        return None, bins

    reference_df = _load_reference_source(source_path)
    specs = _feature_specs_from_reference(reference_df)
    feature_sources = {spec.name: spec.source_columns for spec in specs}

    activated_at = utc_now()
    activation_date = to_date(run_date)
    first_week_ending = _friday_on_or_after(activation_date)
    reference_version = _reference_version_id(
        source_stat.st_mtime_ns,
        source_stat.st_size,
    )
    version_record = PsiReferenceVersion(
        reference_version=reference_version,
        binning_version=PSI_BINNING_VERSION,
        source_mtime_ns=source_stat.st_mtime_ns,
        source_size=source_stat.st_size,
        activated_at=activated_at,
        first_week_ending=first_week_ending,
        row_count=len(reference_df),
        feature_count=len(specs),
        bin_count=0,
        feature_sources=feature_sources,
    )
    new_bins = _build_reference_bins(reference_df, specs, version_record)
    version_record = PsiReferenceVersion(
        **{
            **version_record.__dict__,
            "bin_count": len(new_bins),
        }
    )

    if bins is not None:
        combined_bins = pd.concat([bins, new_bins], ignore_index=True)
    else:
        combined_bins = new_bins
    combined_bins = combined_bins.drop_duplicates(
        subset=["reference_version", "feature", "bin_id"],
        keep="last",
    ).reset_index(drop=True)

    if metadata is not None:
        versions = list(metadata.versions)
    elif bins is not None:
        versions = list(_versions_from_bins(bins))
    else:
        versions = []
    versions = [version for version in versions if version.reference_version != reference_version]
    versions.append(version_record)
    versions.sort(key=lambda item: item.first_week_ending)

    updated_metadata = PsiReferenceMetadata(
        schema_version=PSI_REFERENCE_SCHEMA_VERSION,
        anchor=anchor,
        active_reference_version=reference_version,
        source_file=str(source_path),
        source_mtime_ns=source_stat.st_mtime_ns,
        source_size=source_stat.st_size,
        generated_at=utc_now(),
        binning_version=PSI_BINNING_VERSION,
        feature_columns=tuple(spec.name for spec in specs),
        feature_sources=feature_sources,
        categorical_groups=dict(PSI_CATEGORICAL_GROUPS),
        n_quantile_bins=PSI_N_QUANTILE_BINS,
        epsilon=PSI_EPSILON,
        versions=tuple(versions),
    )

    atomic_write_csv(combined_bins, bins_path)
    atomic_write_text(_metadata_text(updated_metadata), metadata_path)
    print(f"[INFO] PSI reference bins written to {bins_path} (version={reference_version})")
    return updated_metadata, combined_bins


def _version_for_week(versions, week_ending):
    week_ending = to_date(week_ending)
    eligible = [version for version in versions if version.first_week_ending <= week_ending]
    if eligible:
        return max(eligible, key=lambda item: item.first_week_ending)
    return min(versions, key=lambda item: item.first_week_ending)


def _validate_live_columns(df, specs, warned_extra):
    required = []
    for spec in specs:
        required.extend(spec.source_columns)

    missing = sorted({column for column in required if column not in df.columns})
    if missing:
        raise MonitoringPipelineError(f"Live input is missing PSI feature column(s): {missing}")

    used = set(required)
    for spec in specs:
        if spec.feature_type == "categorical":
            used.update(str(column) for column in df.columns if str(column).startswith(spec.prefix))

    extra = sorted(str(column) for column in df.columns if str(column) not in used and str(column) != "idx")
    if extra and not warned_extra:
        print(f"[WARN] Ignoring live input column(s) without a PSI reference: {extra}")
        warned_extra = True
    return warned_extra


def _count_continuous_feature(series, bins_feature):
    numeric = pd.to_numeric(series, errors="raise")
    missing_rows = bins_feature.loc[bins_feature["bin_label"] == PSI_MISSING_LABEL]
    if missing_rows.empty:
        raise MonitoringPipelineError("Continuous PSI reference bins are missing __missing__ bin")
    missing_id = int(missing_rows["bin_id"].iloc[0])

    value_bins = bins_feature.loc[bins_feature["bin_label"] != PSI_MISSING_LABEL].sort_values("bin_id")
    if value_bins.empty:
        raise MonitoringPipelineError("Continuous PSI reference bins have no value bins")

    if len(value_bins) == 1 and value_bins["lower_bound"].isna().all() and value_bins["upper_bound"].isna().all():
        codes = pd.Series(0, index=numeric.index, dtype="float")
    else:
        upper_bounds = value_bins["upper_bound"].iloc[:-1].tolist()
        cut_edges = [-np.inf, *upper_bounds, np.inf]
        codes = pd.cut(
            numeric,
            bins=cut_edges,
            labels=False,
            include_lowest=True,
        ).astype("float")

    codes.loc[numeric.isna()] = missing_id
    counts = codes.value_counts().reindex(
        bins_feature["bin_id"],
        fill_value=0,
    )
    return counts.astype(int)


def _bin_counts_by_id(labels, bins_feature):
    ordered = bins_feature.sort_values("bin_id")
    categories = ordered["bin_label"].tolist()
    bin_ids = ordered["bin_id"].astype(int).tolist()
    counts = labels.value_counts().reindex(categories, fill_value=0)
    return pd.Series(counts.to_numpy(), index=bin_ids, dtype=int)


def _count_categorical_feature(labels, bins_feature):
    categories = bins_feature.sort_values("bin_id")["bin_label"].tolist()
    if PSI_OTHER_LABEL not in categories:
        raise MonitoringPipelineError("Categorical PSI reference bins are missing __other__ bin")
    labels = labels.astype("object")
    labels.loc[~labels.isin(categories)] = PSI_OTHER_LABEL
    return _bin_counts_by_id(labels, bins_feature)


def _count_binary_feature(labels, bins_feature):
    return _bin_counts_by_id(labels, bins_feature)


def _count_feature(df, spec, bins_feature):
    if spec.feature_type == "continuous":
        return _count_continuous_feature(
            df[spec.source_columns[0]],
            bins_feature,
        )
    if spec.feature_type == "categorical":
        labels = _categorical_group_labels(
            df,
            spec.prefix,
            spec.source_columns,
        )
        return _count_categorical_feature(labels, bins_feature)
    if spec.feature_type == "binary":
        labels = _binary_labels(
            df[spec.source_columns[0]],
            spec.true_label,
            spec.false_label,
        )
        return _count_binary_feature(labels, bins_feature)
    raise MonitoringPipelineError(f"Unsupported PSI feature type: {spec.feature_type!r}")


def _read_daily_pair(input_path, output_path, consented_physicians):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not output_path.exists():
        raise MonitoringPipelineError(f"Missing daily output file for PSI aggregation: {output_path}")

    try:
        input_df = pd.read_csv(input_path)
        output_df = pd.read_csv(output_path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise MonitoringPipelineError(
            f"Unable to read daily PSI input/output pair {input_path}, {output_path}: {exc}"
        ) from exc

    if "idx" in input_df.columns:
        input_df = input_df.drop(columns=["idx"])
    required_output_columns = ["first_trt", PSI_PROVIDER_COLUMN]
    missing_output_columns = [column for column in required_output_columns if column not in output_df.columns]
    if missing_output_columns:
        raise MonitoringPipelineError(
            f"Daily output file is missing required column(s) {missing_output_columns}: {output_path}"
        )
    if len(input_df) != len(output_df):
        raise MonitoringPipelineError(
            "Daily input/output row count mismatch for PSI aggregation: "
            f"{input_path} ({len(input_df)}) vs "
            f"{output_path} ({len(output_df)})"
        )

    first_trt = pd.to_numeric(output_df["first_trt"], errors="coerce").fillna(0).eq(1).to_numpy()
    consented = consented_provider_mask(output_df, consented_physicians, column=PSI_PROVIDER_COLUMN).to_numpy()
    return input_df.loc[first_trt & consented].reset_index(drop=True)


def _build_counts_for_weeks(
    start_week_ending,
    last_week_ending,
    deployment_start_date,
    anchor,
    output_dir,
    reference_metadata,
    reference_bins,
    consented_physicians,
    dry_run,
):
    window_start = week_window_start(
        start_week_ending,
        deployment_start_date,
        lookback_days=4,
    )
    files = discover_daily_input_files(
        output_dir,
        anchor,
        window_start,
        last_week_ending,
    )
    print(
        "[INFO] Aggregating weekly PSI: "
        f"weeks={to_date(start_week_ending).isoformat()}.."
        f"{to_date(last_week_ending).isoformat()} "
        f"daily_files={len(files)}"
    )
    print_files([path for _, path in files])

    if not files:
        print(
            "[WARN] No daily input files found for weekly PSI window "
            f"{window_start.isoformat()} to "
            f"{to_date(last_week_ending).isoformat()}; skipping weekly PSI"
        )
        return None
    if dry_run:
        return None

    versions = list(reference_metadata.versions)
    specs_by_version = {}
    bins_by_version = {}
    for version in versions:
        version_bins = reference_bins.loc[reference_bins["reference_version"] == version.reference_version].reset_index(
            drop=True
        )
        bins_by_version[version.reference_version] = version_bins
        specs_by_version[version.reference_version] = _specs_from_bins(
            version_bins,
            reference_metadata,
        )

    count_rows = []
    warned_extra = False

    for file_date, input_path in files:
        week_ending = _friday_on_or_after(file_date)
        if week_ending < to_date(start_week_ending):
            continue
        if week_ending > to_date(last_week_ending):
            continue

        version = _version_for_week(versions, week_ending)
        version_bins = bins_by_version[version.reference_version]
        specs = specs_by_version[version.reference_version]

        output_path = daily_prediction_path(output_dir, file_date, anchor)
        filtered = _read_daily_pair(input_path, output_path, consented_physicians)
        warned_extra = _validate_live_columns(
            filtered,
            specs,
            warned_extra,
        )

        for spec in specs:
            feature_bins = version_bins.loc[version_bins["feature"] == spec.name].sort_values("bin_id")
            if feature_bins.empty:
                raise MonitoringPipelineError(
                    f"PSI reference bins are missing feature {spec.name!r} for version {version.reference_version}"
                )
            counts = _count_feature(filtered, spec, feature_bins)
            for bin_id, count in counts.items():
                bin_rows = feature_bins.loc[feature_bins["bin_id"] == int(bin_id)]
                if len(bin_rows) != 1:
                    raise MonitoringPipelineError(
                        f"Unexpected PSI reference bin definition for feature {spec.name!r}, bin {bin_id}"
                    )
                count_rows.append(
                    {
                        "week_ending": week_ending,
                        "reference_version": version.reference_version,
                        "feature": spec.name,
                        "feature_type": spec.feature_type,
                        "bin_id": int(bin_id),
                        "bin_label": str(bin_rows["bin_label"].iloc[0]),
                        "count": int(count),
                    }
                )

    actual = pd.DataFrame(count_rows, columns=list(PSI_COUNT_COLUMNS))
    if actual.empty:
        print("[WARN] No daily input rows mapped to weekly PSI window; skipping weekly PSI")
        return None
    actual = actual.groupby(list(PSI_COUNT_COLUMNS[:-1]), sort=True, as_index=False)["count"].sum()

    grid_rows = []
    week = to_date(start_week_ending)
    last_week = to_date(last_week_ending)
    while week <= last_week:
        version = _version_for_week(versions, week)
        version_bins = bins_by_version[version.reference_version]
        for row in version_bins.itertuples(index=False):
            grid_rows.append(
                {
                    "week_ending": week,
                    "reference_version": version.reference_version,
                    "feature": row.feature,
                    "feature_type": row.feature_type,
                    "bin_id": int(row.bin_id),
                    "bin_label": row.bin_label,
                    "count": 0,
                }
            )
        week += timedelta(days=7)

    grid = pd.DataFrame(grid_rows, columns=list(PSI_COUNT_COLUMNS))
    keys = list(PSI_COUNT_COLUMNS[:-1])
    combined = grid.merge(
        actual.rename(columns={"count": "actual_count"}),
        on=keys,
        how="left",
    )
    combined["count"] = combined["actual_count"].fillna(combined["count"]).astype(int)
    combined = combined.drop(columns=["actual_count"])
    return combined.sort_values(
        ["week_ending", "feature", "bin_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _read_weekly_counts_csv(path):
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise MonitoringPipelineError(f"Unable to read weekly PSI counts {path}: {exc}") from exc

    missing = [column for column in PSI_COUNT_COLUMNS if column not in df.columns]
    if missing:
        raise MonitoringPipelineError(f"Weekly PSI counts file is missing column(s) {missing}: {path}")

    try:
        df["week_ending"] = pd.to_datetime(
            df["week_ending"],
            errors="raise",
        )
        df["bin_id"] = pd.to_numeric(
            df["bin_id"],
            errors="raise",
        ).astype(int)
        df["count"] = pd.to_numeric(
            df["count"],
            errors="raise",
        ).astype(int)
    except (TypeError, ValueError) as exc:
        raise MonitoringPipelineError(f"Invalid weekly PSI count values in {path}: {exc}") from exc

    if (df["count"] < 0).any():
        raise MonitoringPipelineError(f"Weekly PSI counts contain negative values: {path}")
    keys = ["week_ending", "reference_version", "feature", "bin_id"]
    if df.duplicated(subset=keys).any():
        raise MonitoringPipelineError(f"Weekly PSI counts file contains duplicate rows: {path}")
    return df


def _weekly_counts_consistent(df, metadata):
    if len(df) != metadata.row_count:
        return False
    if df["week_ending"].nunique() != metadata.week_count:
        return False
    if df["feature"].nunique() != metadata.feature_count:
        return False
    if not df.empty and to_date(df["week_ending"].max()) != (metadata.last_week_ending):
        return False
    return True


def _write_weekly_counts_csv(df, path):
    output = df.copy()
    output["week_ending"] = pd.to_datetime(output["week_ending"]).dt.strftime("%Y-%m-%d")
    atomic_write_csv(
        output[list(PSI_COUNT_COLUMNS)],
        path,
    )


def _psi_value(current_counts, reference_pct):
    current_total = float(np.sum(current_counts))
    if current_total <= 0:
        return np.nan

    current_pct = np.asarray(current_counts, dtype=float) / current_total
    reference_pct = np.asarray(reference_pct, dtype=float)
    current_pct = np.maximum(current_pct, PSI_EPSILON)
    reference_pct = np.maximum(reference_pct, PSI_EPSILON)
    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def _compute_psi_scores(counts, reference_bins):
    if counts.empty:
        return pd.DataFrame(
            columns=[
                "week_ending",
                "reference_version",
                "feature",
                "row_count",
                "psi_week",
                "psi_cumulative",
                "cumulative_row_count",
            ]
        )

    reference = reference_bins[["reference_version", "feature", "bin_id", "reference_pct"]]
    merged = counts.merge(
        reference,
        on=["reference_version", "feature", "bin_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["reference_pct"].isna().any():
        raise MonitoringPipelineError("Weekly PSI counts reference bin definitions that are missing")

    score_rows = []
    for (version, feature), rows in merged.groupby(
        ["reference_version", "feature"],
        sort=False,
    ):
        reference_rows = reference_bins.loc[
            (reference_bins["reference_version"] == version) & (reference_bins["feature"] == feature)
        ].sort_values("bin_id")
        ref_pct = reference_rows["reference_pct"].to_numpy(dtype=float)
        bin_ids = reference_rows["bin_id"].tolist()

        weekly = rows.pivot_table(
            index="week_ending",
            columns="bin_id",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        weekly = weekly.reindex(columns=bin_ids, fill_value=0).sort_index()
        cumulative = weekly.cumsum()

        for week_ending in weekly.index:
            week_counts = weekly.loc[week_ending].to_numpy(dtype=float)
            cumulative_counts = cumulative.loc[week_ending].to_numpy(dtype=float)
            score_rows.append(
                {
                    "week_ending": week_ending,
                    "reference_version": version,
                    "feature": feature,
                    "row_count": int(week_counts.sum()),
                    "psi_week": _psi_value(week_counts, ref_pct),
                    "psi_cumulative": _psi_value(
                        cumulative_counts,
                        ref_pct,
                    ),
                    "cumulative_row_count": int(cumulative_counts.sum()),
                }
            )

    scores = pd.DataFrame(score_rows)
    return scores.sort_values(
        ["week_ending", "reference_version", "feature"],
        kind="mergesort",
    ).reset_index(drop=True)


def _write_scores_csv(scores, path):
    output = scores.copy()
    if "week_ending" in output.columns:
        output["week_ending"] = pd.to_datetime(output["week_ending"]).dt.strftime("%Y-%m-%d")
    atomic_write_csv(output, path)


def _weekly_metadata_for(
    anchor,
    deployment_start_date,
    last_week_ending,
    counts,
    reference_metadata,
):
    versions = list(reference_metadata.versions)
    version = _version_for_week(versions, last_week_ending)
    return PsiWeeklyMetadata(
        schema_version=PSI_WEEKLY_SCHEMA_VERSION,
        anchor=anchor,
        deployment_start_date=to_date(deployment_start_date),
        last_week_ending=to_date(last_week_ending),
        reference_version=version.reference_version,
        row_count=len(counts),
        week_count=int(counts["week_ending"].nunique()),
        feature_count=int(counts["feature"].nunique()),
        bin_count=int(counts[["reference_version", "feature", "bin_id"]].drop_duplicates().shape[0]),
        updated_at=utc_now(),
    )


def _write_scores_from_counts(counts, reference_bins, scores_path):
    scores = _compute_psi_scores(counts, reference_bins)
    _write_scores_csv(scores, scores_path)


def run_aggregate(
    deployment_start_date,
    run_date,
    anchor,
    output_dir,
    consented_physicians,
    dry_run=False,
):
    deployment_start_date = to_date(deployment_start_date)
    run_date = to_date(run_date)
    first_week_ending = _friday_on_or_after(deployment_start_date)
    last_week_ending = _last_eligible_friday(run_date)
    if first_week_ending > last_week_ending:
        return

    (
        bins_path,
        _,
        counts_path,
        weekly_metadata_path,
        scores_path,
    ) = _psi_paths(output_dir, anchor)

    reference_metadata, reference_bins = _ensure_reference(
        output_dir,
        run_date,
        anchor,
        dry_run,
    )
    if reference_metadata is None or reference_bins is None:
        print("[INFO] Weekly PSI skipped because no PSI reference is available")
        return

    counts_exists = counts_path.exists()
    weekly_metadata_exists = weekly_metadata_path.exists()

    if counts_exists and weekly_metadata_exists:
        try:
            counts = _read_weekly_counts_csv(counts_path)
            weekly_metadata = _read_weekly_metadata(weekly_metadata_path)
        except MonitoringPipelineError:
            counts = None
            weekly_metadata = None

        if counts is not None and weekly_metadata is not None and _weekly_counts_consistent(counts, weekly_metadata):
            first_new_week = weekly_metadata.last_week_ending + timedelta(days=7)
            if first_new_week > last_week_ending:
                print(f"[INFO] No new complete PSI week after {weekly_metadata.last_week_ending.isoformat()}")
                if not dry_run:
                    _write_scores_from_counts(
                        counts,
                        reference_bins,
                        scores_path,
                    )
                return

            print(f"[INFO] Updating weekly PSI: weeks={first_new_week.isoformat()}..{last_week_ending.isoformat()}")
            new_counts = _build_counts_for_weeks(
                first_new_week,
                last_week_ending,
                deployment_start_date,
                anchor,
                output_dir,
                reference_metadata,
                reference_bins,
                consented_physicians,
                dry_run,
            )
            if new_counts is None:
                return

            combined = pd.concat([counts, new_counts], ignore_index=True)
            combined["week_ending"] = pd.to_datetime(combined["week_ending"])
            combined = combined.drop_duplicates(
                subset=[
                    "week_ending",
                    "reference_version",
                    "feature",
                    "bin_id",
                ],
                keep="last",
            )
            combined = combined.sort_values(
                ["week_ending", "feature", "bin_id"],
                kind="mergesort",
            ).reset_index(drop=True)

            updated_metadata = _weekly_metadata_for(
                anchor,
                deployment_start_date,
                last_week_ending,
                combined,
                reference_metadata,
            )
            _write_weekly_counts_csv(combined, counts_path)
            atomic_write_text(
                _metadata_text(updated_metadata),
                weekly_metadata_path,
            )
            _write_scores_from_counts(
                combined,
                reference_bins,
                scores_path,
            )
            print(f"[INFO] Weekly PSI updated through {last_week_ending.isoformat()}; rows={len(combined)}")
            return

        print(
            "[WARN] Weekly PSI counts and metadata are inconsistent; "
            "rebuilding from daily inputs with versioned reference bins"
        )
    elif counts_exists != weekly_metadata_exists:
        print(
            "[WARN] Weekly PSI counts and metadata are inconsistent; "
            "rebuilding from daily inputs with versioned reference bins"
        )

    print(f"[INFO] Creating weekly PSI baseline: weeks={first_week_ending.isoformat()}..{last_week_ending.isoformat()}")
    counts = _build_counts_for_weeks(
        first_week_ending,
        last_week_ending,
        deployment_start_date,
        anchor,
        output_dir,
        reference_metadata,
        reference_bins,
        consented_physicians,
        dry_run,
    )
    if counts is None:
        return

    metadata = _weekly_metadata_for(
        anchor,
        deployment_start_date,
        last_week_ending,
        counts,
        reference_metadata,
    )
    _write_weekly_counts_csv(counts, counts_path)
    atomic_write_text(_metadata_text(metadata), weekly_metadata_path)
    _write_scores_from_counts(counts, reference_bins, scores_path)
    print(f"[INFO] Weekly PSI baseline written to {counts_path}; rows={len(counts)}")
