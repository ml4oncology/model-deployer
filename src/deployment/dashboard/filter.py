from pathlib import Path

import pandas as pd

from deployer.data_prep.constants import DAILY_POSTFIX_MAP, PROJ_NAME


def get_dashboard_keep_mask(
    df_output: pd.DataFrame | None,
    data_dir: str | Path,
    data_pull_date: str,
    anchor: str,
) -> pd.Series | pd.DataFrame:
    """
    Identify patients (MRNs) who are candidates for "first treatment" dashboard
    inclusion based on scheduled chemo and whether a first treatment has been
    recorded yet.

    The rule is evaluated per MRN (research_id) using the chemo snapshot for
    this anchor/data pull:
    - Load chemo CSV for (anchor, data_pull_date): uses DAILY_POSTFIX_MAP
      (treatment -> "", clinic -> "weekly_").
    - Define a time window relative to data_pull_date: [data_pull_ts,
      data_pull_ts + 5 days] inclusive (upper_bound = pull + 5d).
    - For each MRN, find rows with tx_sched_date inside that window
      (eligible_rows). If none, MRN is not kept.
    - Keep MRN if any eligible row has first_trt_date_utc NaN (no first
      treatment date recorded yet). That means there is a scheduled treatment
      within the next 5 days of the pull but the first treatment time has not
      been populated in chemo data.

    Returns:
    - If df_output is None: DataFrame with columns ["mrn", "clinic_date"]
      where mrn = research_id of kept patients, clinic_date = data_pull_ts
      (used by main.py to write interim/first_trt_{data_pull_date}.csv).
    - If df_output is provided: int Series (0/1) aligned to df_output["mrn"]
      (True -> 1) indicating which model-output rows should be kept for the
      dashboard (NaNs mapped to False).
    """
    postfix = DAILY_POSTFIX_MAP[anchor]
    chemo_file = Path(data_dir) / f"{PROJ_NAME}_chemo_{postfix}{data_pull_date}.csv"
    df_chemo = pd.read_csv(chemo_file)
    df_chemo.columns = df_chemo.columns.str.lower()
    df_chemo["tx_sched_date"] = pd.to_datetime(df_chemo["tx_sched_date"], errors="coerce")
    df_chemo["first_trt_date_utc"] = pd.to_datetime(df_chemo["first_trt_date_utc"], errors="coerce")

    data_pull_ts = pd.to_datetime(data_pull_date, format="%Y%m%d")
    upper_bound = data_pull_ts + pd.Timedelta(days=5)

    def should_keep(group: pd.DataFrame) -> bool:
        # Rows with scheduled treatment date in [pull, pull+5 days] inclusive
        eligible_rows = group[
            group["tx_sched_date"].between(data_pull_ts, upper_bound, inclusive="both")
        ]
        if eligible_rows.empty:
            return False
        # Keep if any eligible scheduled row has no first treatment recorded yet
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
