import numpy as np
import pandas as pd
import plotly.figure_factory as ff

def risk_dist_plot(mrn: int, df_output: pd.DataFrame, df_meta: pd.DataFrame):
    """
    Compute risk score distribution plot and percentile ranks for a given patient.

    Args:
        mrn:        Target patient MRN.
        df_output:  DataFrame with columns: mrn, next_sched_trt_date, clinic_date, ed_pred_prob.
        df_meta:    DataFrame with columns: mrn, clinic_date, cancer (and others).

    Returns:
        percentile_all:  Patient's risk percentile vs. all patients in the cohort.
        percentile_same: Patient's risk percentile vs. same-cancer patients.
        fig:             Plotly figure object of the risk distribution.
    """
    pred_col = "ed_pred_prob"

    # Determine the time window from the target patient's next_sched_trt_date
    next_trt_date = pd.to_datetime(df_output.loc[df_output["mrn"] == mrn, "next_sched_trt_date"].iloc[-1])
    low, high = next_trt_date - pd.Timedelta(days=30), next_trt_date

    # Filter df_output to ±30-day window and deduplicate to one row per MRN (most recent clinic visit)
    window_mask = df_output["next_sched_trt_date"].between(low, high)
    risk = df_output[window_mask].copy()
    risk = risk.groupby("mrn").nth(-1).reset_index()

    # Attach cancer type from df_meta (most recent clinic_date per MRN)
    meta = df_meta.sort_values("clinic_date").groupby("mrn").nth(-1).reset_index()[["mrn", "cancer"]]
    risk = pd.merge(risk, meta, how="left", on="mrn")

    # Split into target patient vs. all others
    mask = risk["mrn"] == mrn
    main, other = risk[mask], risk[~mask]
    main_cancer = main["cancer"].iloc[0]
    same_cancer = other["cancer"] == main_cancer

    MIN_SAME_CANCER = 3  # minimum samples needed to plot same-cancer KDE

    same_cancer_data = list(other.loc[same_cancer, pred_col])
    if len(same_cancer_data) >= MIN_SAME_CANCER:
        hist_data = [list(other[pred_col]), same_cancer_data]
        group_labels = [
            f"All patients<br>N={len(other)}",
            f"{main_cancer} cancer patients<br>N={sum(same_cancer)}",
        ]
        colors = ["#374151", "#2dd4bf"]
    else:
        hist_data = [list(other[pred_col])]
        group_labels = [f"All patients<br>N={len(other)}"]
        colors = ["#374151"]

    fig = ff.create_distplot(hist_data, group_labels, show_rug=False, show_hist=False, colors=colors)
    fig.update_traces(fill="tozeroy")

    # Mark target patient's risk score
    x = main[pred_col].iloc[-1]
    kwargs = dict(type="line", line=dict(color="gray", width=2, dash="dot"))
    fig.add_shape(
        x0=x, x1=x,
        yref="paper", y0=0, y1=1,
        label=dict(text=f"Patient risk score = {x:.3f}", font=dict(size=12, color="gray")),
        **kwargs,
    )

    title = dict(
        text="Risk Score Comparison",
        font=dict(size=20, color="#222", family="Helvetica Neue"),
        x=0.01,
        xanchor="left",
        subtitle=dict(text="Risk percentile rank among the other patients assessed in the past month"),
    )
    fig.update_layout(
        title=title,
        legend=dict(xanchor="right"),
        xaxis_title="Probability of Patients Ending up in ED",
        yaxis_title="Density of Patients",
        template="plotly_white",
        hovermode=False,
    )

    # Percentile: same-cancer percentile falls back to all-patients when insufficient data
    percentile_all = int(round(np.mean(np.array(hist_data[0]) < x) * 100))
    percentile_same = (
        int(round(np.mean(np.array(same_cancer_data) < x) * 100))
        if len(same_cancer_data) >= MIN_SAME_CANCER
        else np.nan  # fallback: use all-patients percentile
    )

    return percentile_all, percentile_same, fig