from pathlib import Path

import pandas as pd

from deployer.data_prep.constants import DAILY_POSTFIX_MAP, PROJ_NAME


def get_dashboard_keep_mask(
    df_output: pd.DataFrame | None,
    data_dir: str | Path,
    data_pull_date: str,
    anchor: str,
) -> pd.Series | pd.DataFrame:
    postfix = DAILY_POSTFIX_MAP[anchor]
    chemo_file = Path(data_dir) / f"{PROJ_NAME}_chemo_{postfix}{data_pull_date}.csv"
    df_chemo = pd.read_csv(chemo_file)
    df_chemo.columns = df_chemo.columns.str.lower()
    df_chemo["tx_sched_date"] = pd.to_datetime(df_chemo["tx_sched_date"], errors="coerce")
    df_chemo["first_trt_date_utc"] = pd.to_datetime(df_chemo["first_trt_date_utc"], errors="coerce")

    data_pull_ts = pd.to_datetime(data_pull_date, format="%Y%m%d")
    upper_bound = data_pull_ts + pd.Timedelta(days=5)

    def should_keep(group: pd.DataFrame) -> bool:
        eligible_rows = group[
            group["tx_sched_date"].between(data_pull_ts, upper_bound, inclusive="both")
        ]
        if eligible_rows.empty:
            return False

        return eligible_rows["first_trt_date_utc"].isna().any()

    keep_by_mrn = df_chemo.groupby("research_id").apply(should_keep, include_groups=False)
    if df_output is None:
        keep_df = keep_by_mrn[keep_by_mrn].reset_index()[["research_id"]].rename(
            columns={"research_id": "mrn"}
        )
        keep_df["clinic_date"] = data_pull_ts
        return keep_df

    keep_mask = df_output["mrn"].map(keep_by_mrn).fillna(False)
    return keep_mask.astype(int)
