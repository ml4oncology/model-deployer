#!/usr/bin/env python3
"""Run the weekly, monthly, and dashboard monitoring stages in sequence."""

import argparse
import sys
from datetime import date
from pathlib import Path

from deployer.monitoring import monthly_monitoring_pipeline as monthly
from deployer.monitoring import weekly_monitoring_pipeline as weekly
from deployer.monitoring.monitoring_common import (
    MONTHLY_INTERIM_DIR,
    WEEKLY_INTERIM_DIR,
    MonitoringPipelineError,
    argparse_date,
    to_date,
)


def _dashboard_module():
    from deployer.monitoring import generate_monitoring_dashboard_pipeline as dashboard

    return dashboard


def run_pipeline(
    deployment_start_date,
    run_date,
    anchor,
    data_dir,
    output_dir,
    info_dir,
    dashboard_output_dir,
    max_alerts,
    dry_run=False,
):
    run_date = to_date(run_date)

    print("[INFO] Running weekly monitoring pipeline...")
    weekly.run_pipeline(
        deployment_start_date=deployment_start_date,
        run_date=run_date,
        anchor=anchor,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    print("[INFO] Running monthly monitoring pipeline...")
    monthly.run_pipeline(
        deployment_start_date=deployment_start_date,
        run_date=run_date,
        anchor=anchor,
        data_dir=data_dir,
        output_dir=output_dir,
        info_dir=info_dir,
        dry_run=dry_run,
    )

    weekly_dir = Path(output_dir) / WEEKLY_INTERIM_DIR
    monthly_dir = Path(output_dir) / MONTHLY_INTERIM_DIR

    if dry_run:
        print(
            f"[INFO] Dry run: would generate dashboard in {dashboard_output_dir} from {weekly_dir} and {monthly_dir}."
        )
        return

    print("[INFO] Generating monitoring dashboard...")
    dashboard = _dashboard_module()
    output_paths = dashboard.run_pipeline(
        max_alerts=max_alerts,
        run_date=run_date,
        weekly_dir=weekly_dir,
        monthly_dir=monthly_dir,
        anchor=anchor,
        output_dir=dashboard_output_dir,
    )
    if not output_paths:
        raise MonitoringPipelineError("Dashboard metadata gates blocked generation; no images produced.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run all live monitoring stages in sequence.")
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
        help="As-of date used by all monitoring stages (YYYYMMDD). Defaults to today.",
    )
    parser.add_argument("--model-anchor", choices=["clinic", "treatment"], default="clinic")
    parser.add_argument("--data-dir", default="./Data")
    parser.add_argument("--output-dir", default="./Outputs")
    parser.add_argument("--info-dir", default="./Infos")
    parser.add_argument("--dashboard-output-dir", default="./Outputs")
    parser.add_argument(
        "--max-alerts",
        required=True,
        type=int,
        help="Cumulative alert target for the weekly dashboard.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned weekly/monthly work without writing files; skip dashboard generation.",
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
            dashboard_output_dir=args.dashboard_output_dir,
            max_alerts=args.max_alerts,
            dry_run=args.dry_run,
        )
    except (MonitoringPipelineError, FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
