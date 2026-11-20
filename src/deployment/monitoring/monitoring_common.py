"""Shared helpers for deployment monitoring pipelines."""

import argparse
import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

REQUIRED_PREDICTION_COLUMNS = ("mrn", "regimen", "ed_pred_prob")
CONSENTED_PHYSICIAN_PROVIDER_COLUMN = "EpicProvider"
CONSENTED_PHYSICIAN_ENABLED_COLUMN = "Enabled"
WEEK_ENDING_WEEKDAY = 4  # Friday, with Monday == 0
WEEKLY_INTERIM_DIR = "interim_weekly_monitoring"
MONTHLY_INTERIM_DIR = "interim_monthly_monitoring"


class MonitoringPipelineError(RuntimeError):
    pass


def to_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()

    text = str(value)
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise MonitoringPipelineError(f"Unsupported date value: {value!r}")


def argparse_date(value):
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYYMMDD."
        ) from exc


def friday_on_or_after(value):
    day = to_date(value)
    return day + timedelta(days=(WEEK_ENDING_WEEKDAY - day.weekday()) % 7)


def last_eligible_friday(run_date):
    day = to_date(run_date) - timedelta(days=1)
    return day - timedelta(days=(day.weekday() - WEEK_ENDING_WEEKDAY) % 7)


def week_ending_range(first_week_ending, last_week_ending):
    return pd.date_range(
        first_week_ending,
        last_week_ending,
        freq="7D",
        name="week_ending",
    )


def week_window_start(first_week_ending, deployment_start_date, lookback_days=4):
    return max(
        to_date(deployment_start_date),
        to_date(first_week_ending) - timedelta(days=lookback_days),
    )


def utc_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def metadata_text(metadata):
    return json.dumps(metadata.to_dict(), indent=2) + "\n"


def read_metadata(path, cls, schema_version, *, label):
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise MonitoringPipelineError(f"Unable to read {label} {path}: {exc}") from exc

    metadata = cls.from_dict(data)
    if metadata.schema_version != schema_version:
        raise MonitoringPipelineError(
            f"Unsupported {label} schema version: {metadata.schema_version}"
        )
    return metadata


def validate_metadata_identity(metadata, anchor, deployment_start_date):
    if metadata.anchor != anchor:
        raise MonitoringPipelineError(
            f"Metadata anchor {metadata.anchor!r} does not match {anchor!r}"
        )
    if metadata.deployment_start_date != deployment_start_date:
        raise MonitoringPipelineError(
            "Metadata deployment start date "
            f"{metadata.deployment_start_date.isoformat()} does not match "
            f"{deployment_start_date.isoformat()}"
        )


def atomic_write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as temp_file:
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


# This lock only prevents two monitoring pipeline runs from updating the same
# files at the same time. Without it, overlapping runs could both read the old
# cumulative file and overwrite each other's updates.
@contextmanager
def pipeline_lock(lock_path):
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MonitoringPipelineError(
                f"Another monitoring pipeline run holds {lock_path}"
            ) from exc

        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def run_serialized(run_once, *, lock_path, dry_run, **kwargs):
    if dry_run:
        run_once(dry_run=True, **kwargs)
        return

    with pipeline_lock(lock_path):
        run_once(dry_run=False, **kwargs)


def prediction_file_pattern(anchor):
    return re.compile(
        rf"^output_(\d{{8}})_(\d{{8}})_{re.escape(anchor)}\.csv$"
    )


def daily_input_file_pattern(anchor):
    return re.compile(rf"^input_(\d{{8}})_{re.escape(anchor)}\.csv$")


def daily_prediction_path(output_dir, day, anchor):
    day = to_date(day)
    return (
        Path(output_dir)
        / f"output_{day:%Y%m%d}_{day:%Y%m%d}_{anchor}.csv"
    )


def discover_daily_input_files(output_dir, anchor, start_date, end_date):
    input_path = Path(output_dir) / "daily_inputs"
    if not input_path.is_dir():
        return []

    start_date = to_date(start_date)
    end_date = to_date(end_date)
    pattern = daily_input_file_pattern(anchor)
    found = []

    for path in input_path.glob(f"input_*_{anchor}.csv"):
        match = pattern.match(path.name)
        if not match:
            continue

        file_date = to_date(match.group(1))
        if start_date <= file_date <= end_date:
            found.append((file_date, path))

    found.sort(key=lambda item: item[0])
    return found


def discover_prediction_files(output_dir, anchor, start_date, end_date):
    output_path = Path(output_dir)
    if not output_path.is_dir():
        raise MonitoringPipelineError(f"Output directory not found: {output_path}")

    start_date = to_date(start_date)
    end_date = to_date(end_date)
    pattern = prediction_file_pattern(anchor)
    found = []

    for path in output_path.glob(f"output_*_*_{anchor}.csv"):
        match = pattern.match(path.name)
        if not match:
            continue

        file_start = to_date(match.group(1))
        file_end = to_date(match.group(2))
        if file_start != file_end:
            continue
        if start_date <= file_start <= end_date:
            found.append((file_start, path))

    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def load_prediction_files(paths, date_col):
    if not paths:
        return pd.DataFrame()

    from deployer.evaluation import load_prediction_csv

    frames = []
    for path in paths:
        try:
            frames.append(load_prediction_csv(path, date_col))
        except (FileNotFoundError, ValueError) as exc:
            raise MonitoringPipelineError(
                f"Unable to load prediction file {path}: {exc}"
            ) from exc
    return pd.concat(frames, ignore_index=True)


def filter_prediction_frame(df, date_col, start_date, end_date):
    if df.empty:
        return df.copy()
    if date_col not in df.columns:
        raise MonitoringPipelineError(
            f"Missing required prediction column: {date_col!r}"
        )

    filtered = df.copy()
    filtered[date_col] = pd.to_datetime(filtered[date_col])
    return filtered[
        filtered[date_col].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()


def validate_prediction_columns(df, date_col):
    required = (*REQUIRED_PREDICTION_COLUMNS, date_col)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise MonitoringPipelineError(
            f"Missing required prediction column(s): {missing}"
        )


def load_consented_physicians(path):
    frame = pd.read_csv(path)
    required = [CONSENTED_PHYSICIAN_PROVIDER_COLUMN, CONSENTED_PHYSICIAN_ENABLED_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MonitoringPipelineError(f"Consented physician file is missing required column(s): {missing}")

    provider = frame[CONSENTED_PHYSICIAN_PROVIDER_COLUMN].astype("string").str.strip()
    enabled = (
        frame[CONSENTED_PHYSICIAN_ENABLED_COLUMN].astype("string").str.strip().str.upper().eq("TRUE").fillna(False)
    )
    valid_provider = provider.notna() & provider.ne("").fillna(False)
    consented = set(provider.loc[enabled & valid_provider].tolist())
    if not consented:
        raise MonitoringPipelineError(f"No consented physicians found in {path}.")
    return consented


def consented_provider_mask(df, consented_physicians, column="VISIT_PROVIDER_NAME"):
    if column not in df.columns:
        raise MonitoringPipelineError(f"Missing required consent column: {column!r}")
    provider = df[column].astype("string").str.strip()
    valid_provider = provider.notna() & provider.ne("").fillna(False)
    return valid_provider & provider.isin(consented_physicians)


def load_prediction_window(paths, date_col, start_date, end_date):
    df = load_prediction_files(paths, date_col)
    validate_prediction_columns(df, date_col)
    return filter_prediction_frame(df, date_col, start_date, end_date)


def print_files(files):
    for path in files:
        print(f"[INFO]   {Path(path).name}")
