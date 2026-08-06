import numpy as np
import pandas as pd
import plotly.figure_factory as ff

PLOT_FONT_FAMILY = "Helvetica Neue, sans-serif"
PLOT_TITLE_COLOR = "#222"
PLOT_BODY_COLOR = "#333"
PLOT_MUTED_COLOR = "#666"


def _same_cancer_group_label(cancer: str) -> str:
    if cancer.strip().lower() == "other":
        return "Patients with cancer type 'Other' starting new treatment"
    return f"Patients with {cancer} cancer starting new treatment"


def risk_dist_plot(
    mrn: int,
    df_output: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_baseline: pd.DataFrame,
    font_scale: float = 1.0,
):
    """
    Compute risk score distribution plot and percentile ranks for a given patient.

    Args:
        mrn:        Target patient MRN.
        df_output:  DataFrame with columns: mrn, next_sched_trt_date, clinic_date, ed_pred_prob.
        df_meta:    DataFrame with columns: mrn, clinic_date, cancer (and others).
        df_baseline: DataFrame with columns: mrn, ed_pred_prob, cancer.

    Returns:
        percentile_all:  Patient's risk percentile vs. all patients in the cohort.
        percentile_same: Patient's risk percentile vs. same-cancer patients.
        fig:             Plotly figure object of the risk distribution.
    """
    pred_col = "ed_pred_prob"

    risk = df_baseline[["mrn", pred_col, "cancer"]].copy()

    # Attach cancer type by exact mrn + clinic_date match, then keep the latest row for this MRN.
    main = pd.merge(
        df_output.loc[df_output["mrn"] == mrn, ["mrn", "clinic_date", pred_col]],
        df_meta[["mrn", "clinic_date", "cancer"]],
        how="left",
        on=["mrn", "clinic_date"],
    )
    main = main.sort_values("clinic_date").tail(1)

    # Use the fixed silent-deployment population as the reference cohort.
    main_cancer = main["cancer"].iloc[0]
    same_cancer = risk["cancer"] == main_cancer

    MIN_SAME_CANCER = 3  # minimum samples needed to plot same-cancer KDE

    same_cancer_data = list(risk.loc[same_cancer, pred_col])
    if len(same_cancer_data) >= MIN_SAME_CANCER:
        same_cancer_label = _same_cancer_group_label(main_cancer)
        hist_data = [list(risk[pred_col]), same_cancer_data]
        group_labels = [
            f"All patients starting new treatment<br>N={len(risk)}",
            f"{same_cancer_label}<br>N={sum(same_cancer)}",
        ]
        colors = ["#374151", "#2dd4bf"]
    else:
        hist_data = [list(risk[pred_col])]
        group_labels = [f"All patients starting new treatment<br>N={len(risk)}"]
        colors = ["#374151"]

    fig = ff.create_distplot(hist_data, group_labels, show_rug=False, show_hist=False, colors=colors)
    fig.update_traces(fill="tozeroy")

    # Mark target patient's risk score
    x = main[pred_col].iloc[-1]
    kwargs = dict(type="line", line=dict(color="black", width=2, dash="dot"))
    fig.add_shape(
        x0=x, x1=x,
        yref="paper", y0=0, y1=1,
        label=dict(
            text=f"Patient risk score = {x:.3f}",
            font=dict(size=14 * font_scale, color=PLOT_BODY_COLOR, family=PLOT_FONT_FAMILY),
        ),
        **kwargs,
    )

    title = dict(
        text="<b>Risk Score Comparison</b>",
        font=dict(size=18 * font_scale, color=PLOT_TITLE_COLOR, family=PLOT_FONT_FAMILY),
        x=0.01,
        xanchor="left",
        subtitle=dict(
            text="Risk percentile rank among other patients in the silent deployment baseline",
            font=dict(size=14 * font_scale, color=PLOT_MUTED_COLOR, family=PLOT_FONT_FAMILY),
        ),
    )
    fig.update_layout(
        title=title,
        legend=dict(
            xanchor="right",
            font=dict(size=14 * font_scale, color=PLOT_BODY_COLOR, family=PLOT_FONT_FAMILY),
        ),
        xaxis_title="Probability of Patients Ending up in ED",
        yaxis_title="Density of Patients",
        template="plotly_white",
        hovermode=False,
        font=dict(size=14 * font_scale, color=PLOT_BODY_COLOR, family=PLOT_FONT_FAMILY),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=40, t=100, b=60),
    )
    fig.update_xaxes(
        title_font=dict(size=16 * font_scale, color=PLOT_BODY_COLOR, family=PLOT_FONT_FAMILY),
        tickfont=dict(size=14 * font_scale, color=PLOT_MUTED_COLOR, family=PLOT_FONT_FAMILY),
    )
    fig.update_yaxes(
        title_font=dict(size=16 * font_scale, color=PLOT_BODY_COLOR, family=PLOT_FONT_FAMILY),
        tickfont=dict(size=14 * font_scale, color=PLOT_MUTED_COLOR, family=PLOT_FONT_FAMILY),
    )

    # Percentile: same-cancer percentile falls back to all-patients when insufficient data
    percentile_all = int(round(np.mean(np.array(hist_data[0]) < x) * 100))
    percentile_same = (
        int(round(np.mean(np.array(same_cancer_data) < x) * 100))
        if len(same_cancer_data) >= MIN_SAME_CANCER
        else np.nan  # fallback: use all-patients percentile
    )

    return percentile_all, percentile_same, fig
