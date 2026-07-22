import argparse
import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats.contingency import odds_ratio
from sklearn.calibration import calibration_curve
from deployer.data_prep.constants import MONTHLY_POSTFIX_MAP, PROJ_NAME
from deployer.data_prep.preprocess.chemo import get_treatment_data
from deployer.data_prep.preprocess.emergency import get_emergency_room_data
from deployer.loader import Config
from make_clinical_dataset.epr.label import get_ED_labels
from ml_common.eval import get_model_performance
from seismometer.data.performance import calculate_bin_stats, calculate_eval_ci
from seismometer.plot.mpl.binary_classifier import evaluation

warnings.filterwarnings("ignore")

DATE_COL_MAP = {"treatment": "treatment_date", "clinic": "clinic_date"}


def filter_intent_to_treat(df, chemo_file, config, anchor, date_col):
    """Retain only rows where patients had intent-to-treat matched treatment.

    Filters:
    - Patients who did not receive any treatment
    - Patients who received treatment more than 5 days after the visit
    - Patients whose treatment regimen did not match the predicted regimen
    - Patients whose ED visit occurred on or before treatment date
    - Patients whose visit date equals treatment date (treatment day, not pre-treatment)
    """
    data_pull_day = None
    chemo_data = get_treatment_data(chemo_file, config, data_pull_day, anchor, mode="evaluation")
    chemo_data["actual_trt_date"] = pd.to_datetime(pd.to_datetime(chemo_data["treatment_date"]).dt.date)
    chemo_data["actual_regimen"] = chemo_data["regimen"]

    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    # make sure index of df is matched with fwd_merge
    df = df.reset_index(drop=True) 

    lookup = chemo_data[["mrn", "actual_trt_date", "actual_regimen"]].drop_duplicates()
    lookup.sort_values("actual_trt_date", inplace=True)

    fwd_merge = pd.merge_asof(
        df, lookup,
        left_on=date_col, right_on="actual_trt_date",
        direction="forward", by="mrn", allow_exact_matches=True,
    )

    good = (
        # Did not receive any treatment
        fwd_merge["actual_trt_date"].notna()
        # Received treatment more than 5 days after the visit
        & ((fwd_merge["actual_trt_date"] - fwd_merge[date_col]).dt.days <= 5)
        # Treatment regimen did not match the predicted regimen
        & (fwd_merge["actual_regimen"] == fwd_merge["regimen"])
        # ED visit occurred on or before treatment date (keep if no ED date)
        & (fwd_merge["target_ED_date"].isna() | (fwd_merge["target_ED_date"] > fwd_merge["actual_trt_date"]))
        # # Visit date equals treatment date (treatment day, not pre-treatment)
        # & (fwd_merge[date_col] != fwd_merge["actual_trt_date"])
    )

    return df.loc[good].copy()


def quartile_odds_ratios(df, prob_col="ed_pred_prob", outcome_col="target_ED_31d"):
    df = df.copy()
    df["quartile"] = pd.qcut(df[prob_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    counts = df.groupby("quartile")[outcome_col].agg(events="sum", n="count")
    counts["non_events"] = counts["n"] - counts["events"]

    ref_events = counts.loc["Q1", "events"]
    ref_non_events = counts.loc["Q1", "non_events"]

    results = []
    for q in ["Q2", "Q3", "Q4"]:
        exposed_cases = counts.loc[q, "events"]
        exposed_noncases = counts.loc[q, "non_events"]
        unexposed_cases = ref_events
        unexposed_noncases = ref_non_events

        table = np.array([
            [exposed_cases, unexposed_cases],
            [exposed_noncases, unexposed_noncases]
        ])

        res = odds_ratio(table, kind="conditional")
        ci = res.confidence_interval(confidence_level=0.95)

        results.append({
            "quartile": q,
            "odds_ratio": res.statistic,
            "ci_low": ci.low,
            "ci_high": ci.high,
            "n": counts.loc[q, "n"],
            "events": exposed_cases,
        })

    results.insert(0, {
        "quartile": "Q1 (ref)", "odds_ratio": 1.0,
        "ci_low": np.nan, "ci_high": np.nan,
        "n": counts.loc["Q1", "n"], "events": ref_events,
    })

    return pd.DataFrame(results), counts


def plot_odds_ratios(results_df, title="Odds Ratio of ED Visit by Predicted Risk Quartile"):
    fig, ax = plt.subplots(figsize=(7, 5))

    quartiles = results_df["quartile"]
    ors = results_df["odds_ratio"]

    lower_err = (ors - results_df["ci_low"]).fillna(0)
    upper_err = (results_df["ci_high"] - ors).fillna(0)

    bars = ax.bar(quartiles, ors, color="#4C72B0", edgecolor="black")
    ax.errorbar(quartiles, ors, yerr=[lower_err, upper_err],
                fmt="none", ecolor="black", capsize=5, linewidth=1.2)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="OR = 1 (reference)")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    ax.set_ylabel("Odds Ratio (95% CI, log scale)")
    ax.set_xlabel("Predicted Risk Quartile")
    ax.set_title(title)

    for bar, or_val in zip(bars, ors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                f"{or_val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.legend()
    plt.tight_layout()
    return fig


def plot_calibration(df, prob_col="ed_pred_prob", outcome_col="target_ED_31d",
                     n_bins=10, strategy="quantile"):
    y_true = df[outcome_col].values
    y_prob = df[prob_col].values

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax1.plot(prob_pred, prob_true, marker="o", color="#4C72B0", label="Model")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title("Calibration Plot (Reliability Diagram)")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.hist(y_prob, bins=30, color="#4C72B0", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    return fig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="20240904")
    parser.add_argument("--end-date", type=str, default="20241130")
    parser.add_argument("--monthly-pull-date", type=str, default="20250103")
    parser.add_argument("--model-anchor", type=str, choices=["clinic", "treatment"], default="clinic")
    parser.add_argument("--prediction-file-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./Outputs")
    parser.add_argument("--data-dir", type=str, default="./Data")
    parser.add_argument("--info-dir", type=str, default="./Infos")
    parser.add_argument("--model-dir", type=str, default="./Models")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    monthly_pull_date = args.monthly_pull_date
    anchor = args.model_anchor

    output_dir = args.output_dir
    data_dir = args.data_dir
    info_dir = args.info_dir
    model_dir = args.model_dir

    prediction_file_path = args.prediction_file_path
    pred_file_ED = f"{anchor}_pred_w_ED_labels.csv"
    perf_file = f"{anchor}_model_perf.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    postfix = MONTHLY_POSTFIX_MAP[anchor]
    date_col = DATE_COL_MAP[anchor]
    chemo_file = f"{data_dir}/{PROJ_NAME}_chemo_{postfix}{monthly_pull_date}.csv"
    ED_visits_file = f"{data_dir}/{PROJ_NAME}_ED_visits_{postfix}{monthly_pull_date}.csv"

    config = Config(info_dir=info_dir)

    ############################ Analyze data #################################

    # Model Prediction file
    df = pd.read_csv(f"{prediction_file_path}", parse_dates=[date_col])
    df = df[df[date_col].between(start_date, end_date)]
    df["assessment_date"] = df[date_col]
    df["last_seen_date"] = pd.Timestamp.max

    # Merge ED visit dates and true labels to Model prediction file
    ed_visit = get_emergency_room_data(ED_visits_file)
    df = get_ED_labels(df, ed_visit, lookahead_window=31)

    # filter out cases where ED visit occurred on the same day
    df = df[(df["target_ED_date"] - df["assessment_date"]).dt.days != 0]

    # Retain only intent-to-treat matched patients
    df = filter_intent_to_treat(df, chemo_file, config, anchor, date_col)

    print(f"Evaluation sample size: {len(df)}")

    df["target_ED_31d"] = df["target_ED_31d"].astype(int)
    pd.DataFrame(df).to_csv(f"{output_dir}/{pred_file_ED}", index=False)

    ######################  Check model Performance ###########################

    print(f"=========== Anchored on {date_col} ====================")

    event_col = "target_ED_date"
    label_col = "target_ED_31d"
    pred_col = "ed_pred_prob"

    # Get pre-defined prediction thresholds
    thresholds = config.thresholds
    thresholds = thresholds.query(f'model_anchor == "{anchor.title()}-anchored"')
    thresholds.columns = thresholds.columns.str.lower()

    model_results = []
    for _, row in thresholds.iterrows():
        assert row["labels"] == "ED_visit"
        performance_metrics = get_model_performance(
            df,
            label_col,
            pred_col,
            pred_thresh=row["prediction_threshold"],
            main_date_col=date_col,
            event_date_col=event_col,
        )
        performance_metrics["Anchor"] = anchor
        performance_metrics["Alarm rate"] = row["alarm_rate"]
        model_results.append(performance_metrics)

    ######################  Model Performance using seismometer ###########################

    y_true = df[label_col].to_numpy()
    y_pred = df[pred_col].to_numpy()

    stats = calculate_bin_stats(y_true, y_pred)
    ci_data = calculate_eval_ci(stats, y_true, y_pred)
    fig = evaluation(stats=stats, ci_data=ci_data, truth=y_true, output=y_pred)

    ######################  AUROC plot with CI ###########################

    fpr_vals = ci_data["roc"]["FPR"]
    tpr_vals = ci_data["roc"]["TPR"]
    auroc_interval = ci_data["roc"]["interval"]
    auroc_val = auroc_interval.value
    auroc_lower = auroc_interval.lower
    auroc_upper = auroc_interval.upper
    conf_level = int(ci_data["conf"]["roc"]["level"] * 100)

    region = ci_data["roc"]["region"]
    upper_xy = np.column_stack([region.upper_fpr, region.upper_tpr])
    lower_xy = np.column_stack([region.lower_fpr[::-1], region.lower_tpr[::-1]])
    ci_polygon_xy = np.vstack([upper_xy, lower_xy])

    NAVY = "#1565C0"
    SKY = "#90CAF9"

    auroc_fig, ax = plt.subplots(figsize=(7, 6))
    auroc_fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F7F9FC")

    ci_patch = mpatches.Polygon(
        ci_polygon_xy, closed=True, facecolor=SKY, edgecolor="none", alpha=0.35,
        label=f"{conf_level}% confidence region",
    )
    ax.add_patch(ci_patch)
    ax.plot([0, 1], [0, 1], "--", color="#9E9E9E", linewidth=1.5) #, label="No-skill (AUC = 0.50)"
    ax.plot(
        fpr_vals, tpr_vals, color=NAVY, linewidth=2.5,
        label=f"AUC = {auroc_val:.3f}  (95% CI: {auroc_lower:.3f}–{auroc_upper:.3f})",
    )

    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=13)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    ax.set_title(
        f"ROC Curve — {anchor.title()}-Anchored Model\n{start_date} to {end_date}",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95, edgecolor="#E0E0E0")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#B0BEC5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    auroc_plot_file = f"{anchor}_auroc_ci.png"
    auroc_fig.savefig(f"{output_dir}/{auroc_plot_file}", bbox_inches="tight", dpi=150)
    plt.close(auroc_fig)
    print(f"AUROC plot saved to {auroc_plot_file}.")

    ######################  Calibration & Odds Ratio Plots ###########################

    cal_fig = plot_calibration(df, prob_col=pred_col, outcome_col=label_col)
    cal_plot_file = f"{anchor}_calibration.png"
    cal_fig.savefig(f"{output_dir}/{cal_plot_file}", bbox_inches="tight", dpi=150)
    plt.close(cal_fig)
    print(f"Calibration plot saved to {cal_plot_file}.")

    or_df, quartile_counts = quartile_odds_ratios(df, prob_col=pred_col, outcome_col=label_col)
    or_fig = plot_odds_ratios(or_df)
    or_plot_file = f"{anchor}_odds_ratios.png"
    or_fig.savefig(f"{output_dir}/{or_plot_file}", bbox_inches="tight", dpi=150)
    plt.close(or_fig)
    print(f"Odds ratio plot saved to {or_plot_file}.")

    ######################  Save Output ###########################
    pd.DataFrame(model_results).to_csv(f"{output_dir}/{perf_file}", index=False)
    print(f"Performance metrics saved to {perf_file}.")
