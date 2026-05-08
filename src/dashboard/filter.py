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
    df_chemo["tx_sched_date"] = pd.to_datetime(df_chemo["tx_sched_date"], errors="coerce")
    df_chemo["first_trt_date_utc"] = pd.to_datetime(df_chemo["first_trt_date_utc"], errors="coerce")

    data_pull_ts = pd.to_datetime(data_pull_date)
    upper_bound = data_pull_ts + pd.Timedelta(days=5)

    def should_keep(group: pd.DataFrame) -> bool:
        if group["tx_sched_date"].isna().all():
            return True

        eligible_rows = group[group["tx_sched_date"] >= data_pull_ts]
        latest_first_treatment = eligible_rows["first_trt_date_utc"].max()

        if pd.isna(latest_first_treatment):
            return True

        return data_pull_ts <= latest_first_treatment <= upper_bound

    keep_by_mrn = df_chemo.groupby("research_id").apply(should_keep, include_groups=False)
    keep_mask = df_output["mrn"].map(keep_by_mrn).fillna(True)
    return keep_mask.astype(int)
