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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from deployer.data_prep.constants import MONTHLY_POSTFIX_MAP, PROJ_NAME
from deployer.data_prep.preprocess.chemo import get_treatment_data
from deployer.data_prep.preprocess.emergency import get_emergency_room_data
from deployer.loader import Config
from make_clinical_dataset.epr.label import get_ED_labels
from ml_common.eval import get_model_performance
from seismometer.data.performance import calculate_bin_stats, calculate_eval_ci
from seismometer.plot.mpl.binary_classifier import evaluation
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DATE_COL_MAP = {"treatment": "treatment_date", "clinic": "clinic_date"}

# ---------------------------------------------------------------------------
# Shared plot style settings (kept consistent across all figures so they can
# be dropped into a manuscript together)
# ---------------------------------------------------------------------------
FONT_SANS = ["Arial", "DejaVu Sans"]
FONT_SIZE_LABEL = 13
FONT_SIZE_TICK = 11
FONT_SIZE_LEGEND = 11
FONT_SIZE_ANNOTATION = 10

COLOR_PRIMARY = "#1565C0"    # navy - main line/bar color
COLOR_SECONDARY = "#90CAF9"  # sky - shaded regions / secondary bars
COLOR_REFERENCE = "#C62828"  # red - reference lines / highlighted estimates

SAVEFIG_DPI = 300

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_SANS,
    "font.size": FONT_SIZE_TICK,
    "axes.labelsize": FONT_SIZE_LABEL,
    "axes.titlesize": FONT_SIZE_LABEL,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "axes.grid": False,
    "savefig.dpi": SAVEFIG_DPI,
})


def style_axis(ax):
    """Apply shared manuscript-style axis formatting."""
    ax.grid(False)

    # Remove the default box
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Use bottom and left spines as the x- and y-axes
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    ax.spines["bottom"].set_position(("outward", 0))
    ax.spines["left"].set_position(("outward", 0))

    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")

    # Make sure ticks and tick labels are visible
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
        direction="out",
        length=4,
        width=1,
    )

    ax.tick_params(axis="x", top=False, labeltop=False)
    ax.tick_params(axis="y", right=False, labelright=False)

    return ax


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


def quartile_odds_ratios(df, prob_col="ed_pred_prob", outcome_col="target_ED_30d"):
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


def plot_odds_ratios(results_df):
    fig, ax = plt.subplots(figsize=(7, 5))

    quartiles = results_df["quartile"]
    ors = results_df["odds_ratio"]

    lower_err = (ors - results_df["ci_low"]).fillna(0)
    upper_err = (results_df["ci_high"] - ors).fillna(0)

    bars = ax.bar(quartiles, ors, color=COLOR_PRIMARY, edgecolor="black")
    ax.errorbar(quartiles, ors, yerr=[lower_err, upper_err],
                fmt="none", ecolor="black", capsize=5, linewidth=1.2)

    ax.axhline(1.0, color=COLOR_REFERENCE, linestyle="--", linewidth=1, label="OR = 1 (reference)")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    ax.set_ylabel("odds ratio (95% CI, log scale)")
    ax.set_xlabel("predicted ED risk quartile")

    # Annotate each bar with its OR and 95% CI, placed above the CI whisker
    # (instead of above the bar, where it used to overlap the error bar).
    for bar, or_val, ci_low, ci_high in zip(bars, ors, results_df["ci_low"], results_df["ci_high"]):
        if np.isnan(ci_high):
            label = f"{or_val:.2f} (ref)"
            y_pos = bar.get_height()
        else:
            label = f"{or_val:.2f} ({ci_low:.2f}\u2013{ci_high:.2f})"
            y_pos = ci_high
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos * 1.05,
                label, ha="center", va="bottom", fontsize=FONT_SIZE_ANNOTATION)

    # Leave extra headroom on the log-scaled y-axis so the top annotation isn't clipped
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    ax.legend()
    style_axis(ax)
    plt.tight_layout()
    return fig


def plot_calibration(df, prob_col="ed_pred_prob", outcome_col="target_ED_30d",
                     n_bins=10, strategy="quantile"):
    y_true = df[outcome_col].values
    y_prob = df[prob_col].values

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", color=COLOR_PRIMARY, label="model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    style_axis(ax)
    plt.tight_layout()
    return fig


def plot_prediction_histogram(df, prob_col="ed_pred_prob", n_bins=30):
    """Standalone histogram of predicted probabilities (previously the bottom
    panel of the calibration plot; split out into its own figure)."""
    y_prob = df[prob_col].values

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(y_prob, bins=n_bins, color=COLOR_PRIMARY, edgecolor="black", alpha=0.7)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("count")
    ax.set_xlim(0, 1)

    style_axis(ax)
    plt.tight_layout()
    return fig


def calibration_intercept_slope(y_true, y_prob):
    """Calibration intercept & slope from a logistic regression of the outcome
    on the logit of the predicted probability (Cox calibration regression).
    Slope = 1 and intercept = 0 indicate perfect calibration.
    """
    def logit(p, eps=1e-15):
        """
        Numerically stable logit transformation.

        Probabilities are clipped away from exactly 0 and 1
        because logit(0) and logit(1) are undefined.
        """
        p = np.asarray(p, dtype=float)
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    # Transform predicted probabilities to log-odds.
    logit_p = logit(y_prob)

    lr = LogisticRegression(solver="lbfgs", penalty=None)
    lr.fit(logit_p.reshape(-1, 1), y_true)

    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    return intercept, slope


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Standard equal-width-bin Expected Calibration Error (ECE)."""
    # note: this is dependent on the number of bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)

    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }


def bootstrap_calibration_metrics_ci(y_true, y_prob, n_boot=1000, random_state=42, n_bins=10, ci=95):
    """Percentile bootstrap 95% CIs for calibration intercept, slope, Brier
    score, and ECE."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct

    boot_vals = {"calibration_intercept": [], "calibration_slope": [], "brier_score": [], "ece": []}

    i = 0
    while i < n_boot:
        idx = rng.integers(0, n, n)
        y_true_bs = y_true[idx]
        y_prob_bs = y_prob[idx]
        # a resample with only one class can't produce a calibration curve; redraw
        if len(np.unique(y_true_bs)) < 2:
            continue
        m = compute_calibration_metrics(y_true_bs, y_prob_bs, n_bins=n_bins)
        for k, v in m.items():
            boot_vals[k].append(v)
        i += 1

    return {
        k: (np.percentile(v, lower_pct), np.percentile(v, upper_pct))
        for k, v in boot_vals.items()
    }


def bootstrap_auc_distribution(y_true, y_pred, n_boot=1000, random_state=42):
    """Generate a bootstrap distribution of AUC estimates via resampling with replacement."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_aucs = np.empty(n_boot)

    i = 0
    while i < n_boot:
        idx = rng.integers(0, n, n)
        y_true_bs = y_true[idx]
        # a resample with only one class can't produce an AUC; redraw
        if len(np.unique(y_true_bs)) < 2:
            continue
        boot_aucs[i] = roc_auc_score(y_true_bs, y_pred[idx])
        i += 1

    return boot_aucs


def plot_auc_bootstrap_distribution(boot_aucs, auc_estimate, title="Bootstrap Distribution of AUC"):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(boot_aucs, bins=30, color=COLOR_PRIMARY, edgecolor="black", alpha=0.8)
    ax.axvline(auc_estimate, color=COLOR_REFERENCE, linestyle="--", linewidth=1.5,
               label=f"AUC estimate = {auc_estimate:.3f}")

    ax.set_xlabel("bootstrapped AUC")
    ax.set_ylabel("frequency")
    ax.set_title(title)
    ax.legend()

    style_axis(ax)
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
    df = get_ED_labels(df, ed_visit, lookahead_window=30)

    # filter out cases where ED visit occurred on the same day
    df = df[(df["target_ED_date"] - df["assessment_date"]).dt.days != 0]

    # Retain only intent-to-treat matched patients
    df = filter_intent_to_treat(df, chemo_file, config, anchor, date_col)

    print(f"Evaluation sample size: {len(df)}")

    df["target_ED_30d"] = df["target_ED_30d"].astype(int)
    pd.DataFrame(df).to_csv(f"{output_dir}/{pred_file_ED}", index=False)

    ######################  Check model Performance ###########################

    print(f"=========== Anchored on {date_col} ====================")

    event_col = "target_ED_date"
    label_col = "target_ED_30d"
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

    print(f"\nAUC = {auroc_val:.4f} (95% CI: {auroc_lower:.4f}-{auroc_upper:.4f})")

    region = ci_data["roc"]["region"]
    upper_xy = np.column_stack([region.upper_fpr, region.upper_tpr])
    lower_xy = np.column_stack([region.lower_fpr[::-1], region.lower_tpr[::-1]])
    ci_polygon_xy = np.vstack([upper_xy, lower_xy])

    auroc_fig, ax = plt.subplots(figsize=(7, 6))
    auroc_fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F7F9FC")

    ci_patch = mpatches.Polygon(
        ci_polygon_xy, closed=True, facecolor=COLOR_SECONDARY, edgecolor="none", alpha=0.35,
        label=f"{conf_level}% confidence region",
    )
    ax.add_patch(ci_patch)
    ax.plot([0, 1], [0, 1], "--", color="#9E9E9E", linewidth=1.5) #, label="No-skill (AUC = 0.50)"
    ax.plot(
        fpr_vals, tpr_vals, color=COLOR_PRIMARY, linewidth=2.5,
        label=f"AUC = {auroc_val:.3f}  (95% CI: {auroc_lower:.3f}–{auroc_upper:.3f})",
    )

    ax.set_xlabel("false positive rate (1 − specificity)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("true positive rate (sensitivity)", fontsize=FONT_SIZE_LABEL)
    # ax.set_title(
    #     f"ROC Curve — {anchor.title()}-Anchored Model\n{start_date} to {end_date}",
    #     fontsize=14, fontweight="bold", pad=12,
    # )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND, framealpha=0.95, edgecolor="#E0E0E0")
    style_axis(ax)

    plt.tight_layout()
    auroc_plot_file = f"{anchor}_auroc_ci.png"
    auroc_fig.savefig(f"{output_dir}/{auroc_plot_file}", bbox_inches="tight", dpi=SAVEFIG_DPI)
    plt.close(auroc_fig)
    print(f"AUROC plot saved to {auroc_plot_file}.")

    ######################  Bootstrap AUC Distribution ###########################

    boot_aucs = bootstrap_auc_distribution(y_true, y_pred, n_boot=1000, random_state=42)
    boot_fig = plot_auc_bootstrap_distribution(
        boot_aucs, auroc_val,
        title=f"Bootstrap Distribution of AUC — {anchor.title()}-Anchored Model",
    )
    boot_plot_file = f"{anchor}_auc_bootstrap.png"
    boot_fig.savefig(f"{output_dir}/{boot_plot_file}", bbox_inches="tight", dpi=SAVEFIG_DPI)
    plt.close(boot_fig)
    print(f"AUC bootstrap distribution plot saved to {boot_plot_file}.")

    ######################  Calibration & Odds Ratio Plots ###########################

    cal_fig = plot_calibration(df, prob_col=pred_col, outcome_col=label_col)
    cal_plot_file = f"{anchor}_calibration.png"
    cal_fig.savefig(f"{output_dir}/{cal_plot_file}", bbox_inches="tight", dpi=SAVEFIG_DPI)
    plt.close(cal_fig)
    print(f"Calibration plot saved to {cal_plot_file}.")

    hist_fig = plot_prediction_histogram(df, prob_col=pred_col)
    hist_plot_file = f"{anchor}_prediction_histogram.png"
    hist_fig.savefig(f"{output_dir}/{hist_plot_file}", bbox_inches="tight", dpi=SAVEFIG_DPI)
    plt.close(hist_fig)
    print(f"Prediction histogram saved to {hist_plot_file}.")

    or_df, quartile_counts = quartile_odds_ratios(df, prob_col=pred_col, outcome_col=label_col)

    print("\nQuartile observed frequencies:")
    print(f"  {'Quartile':<12} {'Observed Freq':<16} {'Sample Size':<12}")
    print(f"  {'-'*12} {'-'*16} {'-'*12}")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        row = quartile_counts.loc[q]
        freq = row["events"] / row["n"]
        print(f"  {q:<12} {freq:<12.4f}     {row['n']:<12}")
    print(f"  {'Total':<12} {'':<16} {quartile_counts['n'].sum():<12}\n")

    or_fig = plot_odds_ratios(or_df)
    or_plot_file = f"{anchor}_odds_ratios.png"
    or_fig.savefig(f"{output_dir}/{or_plot_file}", bbox_inches="tight", dpi=SAVEFIG_DPI)
    plt.close(or_fig)
    print(f"Odds ratio plot saved to {or_plot_file}.")

    ######################  Calibration Metrics ###########################

    cal_metrics = compute_calibration_metrics(y_true, y_pred, n_bins=10)
    print("\nCalibration metrics:")
    print(f"  Calibration intercept: {cal_metrics['calibration_intercept']:.4f}")
    print(f"  Calibration slope:     {cal_metrics['calibration_slope']:.4f}")
    print(f"  Brier score:           {cal_metrics['brier_score']:.4f}")
    print(f"  ECE:                   {cal_metrics['ece']:.4f}")

    print("\nComputing bootstrap 95% CIs for calibration metrics...")
    cal_metric_cis = bootstrap_calibration_metrics_ci(
        y_true, y_pred, n_boot=1000, random_state=42, n_bins=10,
    )
    for metric_name, (lo, hi) in cal_metric_cis.items():
        print(f"  {metric_name}: {cal_metrics[metric_name]:.4f} (95% CI: {lo:.4f}-{hi:.4f})")

    cal_metrics_df = pd.DataFrame([{
        "Anchor": anchor,
        **cal_metrics,
        **{f"{k} CI Low": ci_low for k, (ci_low, ci_high) in cal_metric_cis.items()},
        **{f"{k} CI High": ci_high for k, (ci_low, ci_high) in cal_metric_cis.items()},
    }])
    cal_metrics_file = f"{anchor}_calibration_metrics.csv"
    cal_metrics_df.to_csv(f"{output_dir}/{cal_metrics_file}", index=False)
    print(f"Calibration metrics saved to {cal_metrics_file}.")

    ######################  Save Output ###########################
    pd.DataFrame(model_results).to_csv(f"{output_dir}/{perf_file}", index=False)
    print(f"Performance metrics saved to {perf_file}.")