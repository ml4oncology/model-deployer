from pathlib import Path

import pandas as pd

from deployer.data_prep.constants import DAILY_POSTFIX_MAP, PROJ_NAME


def get_dashboard_keep_mask(
    df_output: pd.DataFrame,
    data_dir: str | Path,
    data_pull_date: str,
    anchor: str,
) -> pd.Series:
    postfix = DAILY_POSTFIX_MAP[anchor]
    chemo_file = Path(data_dir) / f"{PROJ_NAME}_chemo_{postfix}{data_pull_date}.csv"
    df_chemo = pd.read_csv(chemo_file)
    df_chemo.columns = df_chemo.columns.str.lower()
    df_chemo["first_trt_date_utc"] = pd.to_datetime(df_chemo["first_trt_date_utc"], errors="coerce")

    latest_first_treatment = df_chemo.groupby("research_id")["first_trt_date_utc"].max()
    data_pull_ts = pd.to_datetime(data_pull_date)
    upper_bound = data_pull_ts + pd.Timedelta(days=5)

    latest_dates = df_output["mrn"].map(latest_first_treatment)
    keep_mask = latest_dates.isna() | latest_dates.between(data_pull_ts, upper_bound, inclusive="both")
    return keep_mask.astype(int)
