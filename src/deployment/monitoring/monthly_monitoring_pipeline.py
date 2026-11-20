#!/usr/bin/env python3
"""Maintain cumulative live-deployment predictions with ED labels."""

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from deployer.data_prep.constants import MONTHLY_POSTFIX_MAP, PROJ_NAME
from deployer.monitoring.monitoring_common import (
    MONTHLY_INTERIM_DIR,
    MonitoringPipelineError,
    argparse_date,
    atomic_write_csv,
    atomic_write_text,
    discover_prediction_files,
    load_prediction_window,
    metadata_text,
    print_files,
    read_metadata,
    run_serialized,
    to_date,
    utc_now,
    validate_metadata_identity,
)

METADATA_SCHEMA_VERSION = 1
DEFAULT_LOOKAHEAD_DAYS = 30


@dataclass(frozen=True)
class MonthlyPull:
    pull_date: date
    chemo_file: Path
    ed_file: Path


@dataclass(frozen=True)
class MonitoringMetadata:
    schema_version: int
    anchor: str
    deployment_start_date: date
    end_date: date
    monthly_pull_date: date
    updated_at: str

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "anchor": self.anchor,
            "deployment_start_date": self.deployment_start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "monthly_pull_date": self.monthly_pull_date.isoformat(),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                schema_version=int(data["schema_version"]),
                anchor=str(data["anchor"]),
                deployment_start_date=to_date(data["deployment_start_date"]),
                end_date=to_date(data["end_date"]),
                monthly_pull_date=to_date(data["monthly_pull_date"]),
                updated_at=str(data["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringPipelineError(f"Invalid metadata: {exc}") from exc


def derive_end_date(monthly_pull_date, lookahead_days=DEFAULT_LOOKAHEAD_DAYS):
    """Return the latest month-end whose full lookahead window is before the pull."""
    pull_date = to_date(monthly_pull_date)
    last_eligible_day = pull_date - timedelta(days=lookahead_days)
    return last_eligible_day.replace(day=1) - timedelta(days=1)


def _monthly_file_pattern(anchor, stem):
    postfix = MONTHLY_POSTFIX_MAP[anchor]
    return re.compile(
        rf"^{re.escape(PROJ_NAME)}_{re.escape(stem)}_{re.escape(postfix)}(\d{{8}})\.csv$"
    )


def _scan_monthly_files(data_dir, anchor, stem):
    pattern = _monthly_file_pattern(anchor, stem)
    files = {}
    for path in Path(data_dir).glob("*.csv"):
        match = pattern.match(path.name)
        if match:
            files[to_date(match.group(1))] = path
    return files


def discover_monthly_pull(data_dir, anchor, run_date):
    if anchor not in MONTHLY_POSTFIX_MAP:
        raise MonitoringPipelineError(f"Unsupported anchor: {anchor!r}")

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise MonitoringPipelineError(f"Data directory not found: {data_path}")

    run_date = to_date(run_date)
    chemo_files = _scan_monthly_files(data_path, anchor, "chemo")
    ed_files = _scan_monthly_files(data_path, anchor, "ED_visits")
    common_dates = sorted(set(chemo_files) & set(ed_files))
    eligible_dates = [pull_date for pull_date in common_dates if pull_date <= run_date]

    if not eligible_dates:
        raise MonitoringPipelineError(
            f"No monthly chemo + ED pull found on or before {run_date.isoformat()} "
            f"in {data_path}"
        )

    pull_date = eligible_dates[-1]
    return MonthlyPull(
        pull_date=pull_date,
        chemo_file=chemo_files[pull_date],
        ed_file=ed_files[pull_date],
    )


def validate_existing_labels(df, date_col):
    required = ("mrn", date_col, "actual_trt_date", "target_ED_30d")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise MonitoringPipelineError(
            f"Existing labels file is missing required column(s): {missing}"
        )


def _build_labels(df, start_date, end_date, monthly_pull_date, anchor, data_dir, info_dir):
    from deployer.evaluation import build_eval_frame
    from deployer.loader import Config

    config = Config(info_dir=info_dir)
    return build_eval_frame(
        df,
        start_date,
        end_date,
        monthly_pull_date,
        anchor,
        config,
        data_dir,
    )


def _combine_labels(existing, new, date_col):
    from deployer.evaluation import combine_eval_frames

    return combine_eval_frames([existing, new], date_col)


def _paths(output_dir, anchor):
    interim_dir = Path(output_dir) / MONTHLY_INTERIM_DIR
    return (
        interim_dir / f"monthly_monitoring_{anchor}_pred_w_ED_labels.csv",
        interim_dir / f"monthly_monitoring_{anchor}_pred_w_ED_labels_metadata.json",
        interim_dir / f"monthly_monitoring_{anchor}.lock",
    )


def _metadata_text(metadata):
    return metadata_text(metadata)


def _read_metadata(path):
    return read_metadata(
        path,
        MonitoringMetadata,
        METADATA_SCHEMA_VERSION,
        label="metadata",
    )


def _validate_metadata(metadata, anchor, deployment_start_date):
    validate_metadata_identity(metadata, anchor, deployment_start_date)


def _create_baseline(
    pull,
    deployment_start_date,
    anchor,
    data_dir,
    output_dir,
    info_dir,
    labels_path,
    metadata_path,
    date_col,
    dry_run,
):
    end_date = derive_end_date(pull.pull_date)
    if end_date < deployment_start_date:
        raise MonitoringPipelineError(
            f"Derived end date {end_date.isoformat()} is before deployment start "
            f"{deployment_start_date.isoformat()}. Most recent eligible monthly pull: "
            f"{pull.pull_date.isoformat()} ({DEFAULT_LOOKAHEAD_DAYS}-day lookahead). "
            "Use a later run date or an earlier deployment start date."
        )

    files = discover_prediction_files(
        output_dir, anchor, deployment_start_date, end_date
    )
    print(
        f"[INFO] Creating baseline: pull={pull.pull_date.isoformat()} "
        f"end={end_date.isoformat()} daily_files={len(files)}"
    )
    print_files(files)

    if not files:
        raise MonitoringPipelineError(
            "No daily prediction files found for baseline interval "
            f"{deployment_start_date.isoformat()} to {end_date.isoformat()}"
        )
    if dry_run:
        return

    df = load_prediction_window(
        files, date_col, deployment_start_date, end_date
    )
    if df.empty:
        raise MonitoringPipelineError(
            "No prediction rows found for baseline interval "
            f"{deployment_start_date.isoformat()} to {end_date.isoformat()}"
        )

    labels = _build_labels(
        df,
        deployment_start_date,
        end_date,
        pull.pull_date,
        anchor,
        data_dir,
        info_dir,
    )
    atomic_write_csv(labels, labels_path)

    metadata = MonitoringMetadata(
        schema_version=METADATA_SCHEMA_VERSION,
        anchor=anchor,
        deployment_start_date=deployment_start_date,
        end_date=end_date,
        monthly_pull_date=pull.pull_date,
        updated_at=utc_now(),
    )
    atomic_write_text(_metadata_text(metadata), metadata_path)
    print(f"[INFO] Baseline labels written to {labels_path}")


def _update_baseline(
    pull,
    metadata,
    anchor,
    data_dir,
    output_dir,
    info_dir,
    labels_path,
    metadata_path,
    date_col,
    dry_run,
):
    if pull.pull_date <= metadata.monthly_pull_date:
        print(
            f"[INFO] No newer monthly pull than "
            f"{metadata.monthly_pull_date.isoformat()}"
        )
        return

    new_end_date = derive_end_date(pull.pull_date)
    if new_end_date <= metadata.end_date:
        print(
            f"[INFO] New pull {pull.pull_date.isoformat()} does not extend the "
            f"labelable end date beyond {metadata.end_date.isoformat()}"
        )
        return

    new_start_date = metadata.end_date + timedelta(days=1)
    files = discover_prediction_files(
        output_dir, anchor, new_start_date, new_end_date
    )
    print(
        f"[INFO] Updating labels: pull={pull.pull_date.isoformat()} "
        f"window={new_start_date.isoformat()}..{new_end_date.isoformat()} "
        f"daily_files={len(files)}"
    )
    print_files(files)

    if not files:
        print(
            "[INFO] No daily prediction files in update interval; "
            "leaving metadata unchanged"
        )
        return
    if dry_run:
        return

    new_df = load_prediction_window(files, date_col, new_start_date, new_end_date)
    if new_df.empty:
        print(
            "[INFO] No prediction rows in update interval; "
            "leaving metadata unchanged"
        )
        return

    new_labels = _build_labels(
        new_df,
        new_start_date,
        new_end_date,
        pull.pull_date,
        anchor,
        data_dir,
        info_dir,
    )
    existing = pd.read_csv(labels_path)
    validate_existing_labels(existing, date_col)
    combined = _combine_labels(existing, new_labels, date_col)
    atomic_write_csv(combined, labels_path)

    updated_metadata = MonitoringMetadata(
        schema_version=METADATA_SCHEMA_VERSION,
        anchor=metadata.anchor,
        deployment_start_date=metadata.deployment_start_date,
        end_date=new_end_date,
        monthly_pull_date=pull.pull_date,
        updated_at=utc_now(),
    )
    atomic_write_text(_metadata_text(updated_metadata), metadata_path)
    print(
        f"[INFO] Labels updated through {new_end_date.isoformat()}; "
        f"rows={len(combined)}"
    )


def _run_once(
    deployment_start_date,
    run_date,
    anchor,
    data_dir,
    output_dir,
    info_dir,
    dry_run,
):
    labels_path, metadata_path, _ = _paths(output_dir, anchor)
    date_col = {"clinic": "clinic_date", "treatment": "treatment_date"}[anchor]
    pull = discover_monthly_pull(data_dir, anchor, run_date)

    labels_exist = labels_path.exists()
    metadata_exist = metadata_path.exists()

    if not labels_exist or not metadata_exist:
        if labels_exist != metadata_exist:
            print(
                "[WARN] Baseline labels and metadata are inconsistent; "
                "rebuilding the baseline"
            )
        _create_baseline(
            pull=pull,
            deployment_start_date=deployment_start_date,
            anchor=anchor,
            data_dir=data_dir,
            output_dir=output_dir,
            info_dir=info_dir,
            labels_path=labels_path,
            metadata_path=metadata_path,
            date_col=date_col,
            dry_run=dry_run,
        )
        return

    metadata = _read_metadata(metadata_path)
    _validate_metadata(metadata, anchor, deployment_start_date)
    _update_baseline(
        pull=pull,
        metadata=metadata,
        anchor=anchor,
        data_dir=data_dir,
        output_dir=output_dir,
        info_dir=info_dir,
        labels_path=labels_path,
        metadata_path=metadata_path,
        date_col=date_col,
        dry_run=dry_run,
    )


def run_pipeline(
    deployment_start_date,
    run_date,
    anchor,
    data_dir,
    output_dir,
    info_dir,
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
        data_dir=data_dir,
        output_dir=output_dir,
        info_dir=info_dir,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Maintain cumulative live-deployment predictions with ED labels."
    )
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
        help="As-of date used to discover monthly pulls (YYYYMMDD). Defaults to today.",
    )
    parser.add_argument(
        "--model-anchor",
        choices=["clinic", "treatment"],
        default="clinic",
    )
    parser.add_argument("--data-dir", default="./Data")
    parser.add_argument("--output-dir", default="./Outputs")
    parser.add_argument("--info-dir", default="./Infos")
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
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            info_dir=args.info_dir,
            dry_run=args.dry_run,
        )
    except MonitoringPipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
