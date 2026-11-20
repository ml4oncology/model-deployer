#!/usr/bin/env python3
"""Generate the weekly ED-risk monitoring dashboard from interim monitoring data."""

import argparse
import json
import sys
import textwrap
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deployer.evaluation import DATE_COL_MAP
from deployer.monitoring.monitoring_common import (
    MonitoringPipelineError,
    argparse_date,
    consented_provider_mask,
    last_eligible_friday,
    load_consented_physicians,
    to_date,
)
from deployer.monitoring.subgroup_metrics import CATEGORY_REGISTRY, compute_category_metrics, compute_subgroup_metrics
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy.stats import gaussian_kde

matplotlib.use("Agg", force=True)

W = 10
DPI = 160
BG = "#F3F7FC"
NAVY = "#122341"
MID = "#24669C"
BLUE = "#287AB7"
ORANGE = "#E58A49"
SILENT = "#667789"
MUTED = "#53677C"
GRID = "#DDE6EF"
EDGE = "#D7E4F0"
WHITE = "#FFFFFF"
WEEKLY_RISK_BINS = np.linspace(0, 1, 11)
CLINICIAN_LABEL_WIDTH = 16
FEATURE_LABEL_WIDTH = 24
PSI_LABEL_WIDTH = 16
PSI_X_LIMIT = 0.30
PSI_CLAMP_MARGIN = 0.005
EXCLUDED_CONTINUOUS_PSI_FEATURES = {"visit_month_sin", "visit_month_cos"}
MONTHLY_CLINICAL_CATEGORIES = ("age_over_65", "sex", "treatment_intent")
MONTHLY_CANCER_CATEGORIES = ("cancer_type",)
MONTHLY_N_BOOT = 2000
MONTHLY_RANDOM_STATE = 42
PSI_FEATURE_LABELS = {
    "cancer_site": "Cancer group",
    "regimen": "Treatment regimen",
    "intent": "Treatment intent",
    "sex": "Sex",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.3,
        "axes.edgecolor": "#ABBCCA",
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": NAVY,
        "savefig.facecolor": BG,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Shared plotting helpers: identical geometry and styling to the prototype.
# ---------------------------------------------------------------------------


def figure(height):
    return plt.figure(figsize=(W, height), facecolor=BG)


def render_figure(fig, scale=1.0):
    buffer = BytesIO()
    fig.savefig(buffer, format="PNG", dpi=DPI * scale, facecolor=BG)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    buffer.close()
    return image


def card(fig, x, y, w, h, color=WHITE, edge=EDGE, r=0.018):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=fig.transFigure,
        boxstyle=f"round,pad=0.000,rounding_size={r}",
        fc=color,
        ec=edge,
        lw=0.85,
        zorder=0,
    )
    fig.patches.append(patch)


def txt(fig, x, y, s, size=10, weight="normal", color=NAVY, ha="left", va="center"):
    fig.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=10)


def header(fig, section, subtitle, tag=None):
    card(fig, 0.025, 0.905, 0.95, 0.075, color="#E3EEF9", edge="#E3EEF9")
    txt(fig, 0.045, 0.956, section, 15.2, "bold")
    txt(fig, 0.045, 0.925, subtitle, 8.4, color=MUTED)
    if tag:
        txt(fig, 0.949, 0.95, tag, 8.7, "bold", MID, "right")


def axis(fig, rect):
    return fig.add_axes(rect, facecolor=WHITE, zorder=3)


def clean(ax, xgrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#ACBFD0")
    ax.spines["bottom"].set_color("#ACBFD0")
    ax.tick_params(length=0, labelsize=8.5, pad=5)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def kde_curve(vals, x):
    if len(vals) < 3 or np.allclose(vals, vals[0]):
        return np.zeros_like(x)
    return gaussian_kde(vals, bw_method=0.25)(x)


def psi_plot(fig, region, vals):
    ax = axis(fig, region)
    ypos = np.arange(len(vals))[::-1]
    clamp_at = PSI_X_LIMIT - PSI_CLAMP_MARGIN
    for i, (_, weekly, cumulative) in enumerate(vals):
        yi = ypos[i]
        weekly_value = float(weekly)
        cumulative_value = float(cumulative)
        weekly_display = min(weekly_value, clamp_at)
        cumulative_display = min(cumulative_value, clamp_at)
        ax.plot([weekly_display, cumulative_display], [yi, yi], color="#D1DCE5", lw=3, zorder=1)
        ax.scatter(
            [weekly_display],
            [yi],
            color=BLUE,
            s=44,
            marker=">" if weekly_value > PSI_X_LIMIT else "o",
            zorder=4,
        )
        ax.scatter(
            [cumulative_display],
            [yi],
            color=ORANGE,
            s=44,
            marker=">" if cumulative_value > PSI_X_LIMIT else "D",
            zorder=4,
        )
        if weekly_value > PSI_X_LIMIT or cumulative_value > PSI_X_LIMIT:
            ax.annotate(
                ">0.30",
                (clamp_at, yi),
                xytext=(-3, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=6.8,
                color=MUTED,
            )
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [wrap_label(item[0], PSI_LABEL_WIDTH, inclusive=True) for item in vals],
        fontsize=8.9,
        linespacing=1.3,
    )
    ax.set_xlim(0, PSI_X_LIMIT)
    ax.set_xticks([0, 0.10, 0.20, 0.30])
    clean(ax)
    for cut in [0.1, 0.25]:
        ax.axvline(cut, color="#B0BCC7", ls=":", lw=1.1, zorder=2)
    ax.tick_params(axis="y", pad=8)
    return ax


# ---------------------------------------------------------------------------
# Inputs, paths, and metadata gate: proceed only when all weekly metadata
# files share the same update date.
# ---------------------------------------------------------------------------


def weekly_paths(weekly_dir, anchor):
    interim = Path(weekly_dir)
    return {
        "alerts": interim / f"weekly_alerts_{anchor}.csv",
        "alerts_metadata": interim / f"weekly_alerts_{anchor}_metadata.json",
        "risk_scores": interim / f"weekly_risk_scores_{anchor}.csv",
        "risk_metadata": interim / f"weekly_risk_scores_{anchor}_metadata.json",
        "missingness": interim / f"weekly_missingness_counts_{anchor}.csv",
        "missingness_metadata": interim / f"weekly_missingness_counts_{anchor}_metadata.json",
        "psi_scores": interim / f"weekly_psi_scores_{anchor}.csv",
        "psi_counts": interim / f"weekly_psi_counts_{anchor}.csv",
        "psi_metadata": interim / f"weekly_psi_metadata_{anchor}.json",
        "silent_scores": interim / f"silent_deployment_risk_scores_{anchor}.csv",
        "silent_missingness": interim / f"silent_deployment_missingness_{anchor}.csv",
    }


def read_metadata_field(path, field):
    with Path(path).open() as handle:
        return json.load(handle)[field]


def metadata_date(path):
    value = str(read_metadata_field(path, "updated_at")).replace("Z", "+00:00")
    return datetime.fromisoformat(value).date()


def metadata_gate_passes(paths):
    metadata_keys = ("alerts_metadata", "risk_metadata", "missingness_metadata", "psi_metadata")
    dates = [(key, metadata_date(paths[key])) for key in metadata_keys]
    unique_dates = {date for _, date in dates}
    if len(unique_dates) != 1:
        details = ", ".join(f"{Path(paths[key]).name}={date.isoformat()}" for key, date in dates)
        print(f"[INFO] Dashboard not generated: weekly metadata dates differ ({details})")
        return False
    return True


# ---------------------------------------------------------------------------
# Monthly paths and reporting window: the reported month is the latest
# available month that starts before the run-date month.
# ---------------------------------------------------------------------------


def monthly_paths(monthly_dir, anchor):
    interim = Path(monthly_dir)
    return {
        "monthly_labels": interim / f"monthly_monitoring_{anchor}_pred_w_ED_labels.csv",
    }


def available_month_window(df, date_col, run_date):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    run_month_start = pd.Timestamp(run_date.year, run_date.month, 1)
    eligible = dates.loc[dates < run_month_start].dropna()
    if eligible.empty:
        raise MonitoringPipelineError(
            f"No monthly dates available before {run_month_start:%b %Y} in column '{date_col}'."
        )
    month_start = eligible.max().to_period("M").to_timestamp()
    return month_start, month_start + pd.offsets.MonthBegin(1)


# ---------------------------------------------------------------------------
# Weekly data loading: the report week is the prior week ending when available,
# otherwise the latest week ending present in all weekly result files;
# cumulative values use all available weeks up to that week.
# ---------------------------------------------------------------------------

WEEKLY_REPORTING_KEYS = ("alerts", "risk_scores", "missingness", "psi_scores")


def weekly_report_week(paths, run_date):
    expected = pd.Timestamp(last_eligible_friday(run_date))
    available = None
    for key in WEEKLY_REPORTING_KEYS:
        frame = pd.read_csv(paths[key], usecols=["week_ending"])
        weeks = set(pd.to_datetime(frame["week_ending"], errors="coerce").dropna().dt.normalize())
        eligible = {week for week in weeks if week <= expected}
        if not eligible:
            raise MonitoringPipelineError(
                f"No weekly results available on or before {expected:%b %d, %Y} in {Path(paths[key]).name}."
            )
        available = eligible if available is None else available & eligible
        if not available:
            raise MonitoringPipelineError(
                f"No weekly results are available in all weekly files on or before {expected:%b %d, %Y}."
            )
    return expected if expected in available else max(available)


def display_feature(name):
    name = str(name)
    return PSI_FEATURE_LABELS.get(name, name.replace("_", " ").title())


def upper_limit(values, headroom=1.17):
    return max(float(np.max(values, initial=0)) * headroom, 1.0)


def wrap_label(label, width, inclusive=False):
    value = str(label)
    if len(value) < width or (len(value) == width and not inclusive):
        return value
    wrap_width = width - 1 if inclusive and len(value) == width else width
    return "\n".join(textwrap.wrap(value, width=wrap_width, break_long_words=False, break_on_hyphens=False)) or value


def format_clinician_label(name):
    value = str(name)
    if "," in value:
        last, first = value.split(",", 1)
        value = f"{last.strip()}, {first.strip().split(' ', 1)[0]}".rstrip(", ")
    if len(value) <= CLINICIAN_LABEL_WIDTH:
        return value
    if "," in value:
        last, first = value.split(",", 1)
        return f"{last.strip()},\n{first.strip()}"
    return wrap_label(value, CLINICIAN_LABEL_WIDTH)


def with_datetime_week(df):
    table = df.copy()
    table["week_ending"] = pd.to_datetime(table["week_ending"])
    return table


def load_weekly_data(paths, report_week):
    report_timestamp = pd.Timestamp(report_week)

    alerts = with_datetime_week(pd.read_csv(paths["alerts"]))
    alerts = alerts.loc[alerts["week_ending"] <= report_timestamp].sort_values("week_ending")
    clinician_columns = [column for column in alerts.columns if column != "week_ending"]
    alert_totals = alerts[clinician_columns].sum(axis=1).to_numpy(dtype=float)
    report_alert_row = alerts.loc[alerts["week_ending"] == report_timestamp].iloc[0]
    clinician_cumulative = alerts[clinician_columns].sum().sort_values(ascending=False)
    top_clinicians = clinician_cumulative.head(15).index.tolist()

    risk_scores = with_datetime_week(pd.read_csv(paths["risk_scores"]))
    risk_scores = risk_scores.loc[risk_scores["week_ending"] <= report_timestamp]
    weekly_risk_rows = risk_scores.loc[
        risk_scores["week_ending"] == report_timestamp,
        "ed_pred_prob",
    ]
    weekly_scores = pd.to_numeric(weekly_risk_rows, errors="coerce").dropna().to_numpy(dtype=float)
    cumulative_scores = pd.to_numeric(risk_scores["ed_pred_prob"], errors="coerce").dropna().to_numpy(dtype=float)
    silent_scores = (
        pd.to_numeric(
            pd.read_csv(paths["silent_scores"])["ed_pred_prob"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    missingness = with_datetime_week(pd.read_csv(paths["missingness"]))
    missingness = missingness.loc[missingness["week_ending"] <= report_timestamp]
    feature_columns = [column for column in missingness.columns if column not in ("week_ending", "encounter_count")]
    report_missing_row = missingness.loc[missingness["week_ending"] == report_timestamp].iloc[0]
    weekly_encounters = float(report_missing_row["encounter_count"]) or 1.0
    cumulative_encounters = float(missingness["encounter_count"].sum()) or 1.0

    weekly_missing_pct = {}
    cumulative_missing_pct = {}
    for feature in feature_columns:
        weekly_missing_pct[feature] = float(report_missing_row[feature]) / weekly_encounters * 100
        cumulative_missing_pct[feature] = float(missingness[feature].sum()) / cumulative_encounters * 100

    silent_missingness = pd.read_csv(paths["silent_missingness"])
    silent_missing_pct = dict(
        zip(
            silent_missingness["feature"],
            pd.to_numeric(silent_missingness["missing_pct"], errors="coerce").fillna(0) * 100,
            strict=False,
        )
    )
    top_features = sorted(feature_columns, key=lambda feature: cumulative_missing_pct[feature], reverse=True)[:10]
    missingness_rows = [
        (
            display_feature(feature),
            weekly_missing_pct[feature],
            cumulative_missing_pct[feature],
            float(silent_missing_pct.get(feature, 0)),
        )
        for feature in top_features
    ]

    psi_scores = with_datetime_week(pd.read_csv(paths["psi_scores"]))
    psi_scores = psi_scores.loc[psi_scores["week_ending"] <= report_timestamp]
    report_psi = psi_scores.loc[psi_scores["week_ending"] == report_timestamp].copy()
    psi_counts = pd.read_csv(paths["psi_counts"])
    feature_types = psi_counts.drop_duplicates("feature").set_index("feature")["feature_type"].to_dict()
    report_psi["feature_type"] = report_psi["feature"].map(feature_types)
    report_psi = report_psi.dropna(subset=["psi_week", "psi_cumulative"])
    report_psi = report_psi.sort_values("psi_cumulative", ascending=False).drop_duplicates("feature")

    continuous = report_psi.loc[
        (report_psi["feature_type"] == "continuous") & (~report_psi["feature"].isin(EXCLUDED_CONTINUOUS_PSI_FEATURES))
    ].head(5)
    categorical = report_psi.loc[report_psi["feature_type"].isin(["categorical", "binary"])].head(5)

    def psi_rows(frame):
        return [
            (
                display_feature(row.feature),
                float(row.psi_week),
                float(row.psi_cumulative),
            )
            for row in frame.itertuples()
        ]

    return {
        "weeks": alerts["week_ending"].to_numpy(),
        "weekly_counts": alert_totals,
        "cumulative_counts": np.cumsum(alert_totals),
        "alerts_this_week": float(report_alert_row[clinician_columns].sum()),
        "cumulative_alerts": float(alerts[clinician_columns].to_numpy().sum()),
        "clinicians": top_clinicians,
        "clinician_weekly": report_alert_row[top_clinicians].to_numpy(dtype=float),
        "clinician_cumulative": clinician_cumulative.loc[top_clinicians].to_numpy(dtype=float),
        "weekly_scores": weekly_scores,
        "cumulative_scores": cumulative_scores,
        "silent_scores": silent_scores,
        "missingness": missingness_rows,
        "psi_continuous": psi_rows(continuous),
        "psi_categorical": psi_rows(categorical),
    }


# ---------------------------------------------------------------------------
# Monthly metric preparation: filter to consented first-treatment rows first,
# use the latest available pre-run-month cohort, and cap cumulative rows at
# the end of that cohort's month.
# ---------------------------------------------------------------------------


def load_monthly_data(paths, run_date, anchor, consented_physicians):
    date_col = DATE_COL_MAP[anchor]
    frame = pd.read_csv(paths["monthly_labels"], parse_dates=[date_col])
    required = [
        date_col,
        "ed_pred_prob",
        "target_ED_30d",
        "first_trt",
        "VISIT_PROVIDER_NAME",
        "cancer",
        "age",
        "gender",
        "intent",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MonitoringPipelineError(f"Monthly labels file is missing required column(s): {missing}")

    first_trt = pd.to_numeric(frame["first_trt"], errors="coerce")
    frame = frame.loc[first_trt.eq(1)].copy()
    if frame.empty:
        raise MonitoringPipelineError("Monthly labels contain no first-treatment patients (first_trt == 1).")

    frame = frame.loc[consented_provider_mask(frame, consented_physicians)].copy()
    if frame.empty:
        raise MonitoringPipelineError("Monthly labels contain no first-treatment patients with a consented physician.")

    month_start, month_end_exclusive = available_month_window(frame, date_col, run_date)
    monthly = frame.loc[(frame[date_col] >= month_start) & (frame[date_col] < month_end_exclusive)].copy()
    if monthly.empty:
        raise MonitoringPipelineError(f"No monthly rows found for {month_start:%b %Y} in column '{date_col}'.")

    return {
        "all": frame.loc[frame[date_col] < month_end_exclusive].copy(),
        "monthly": monthly,
        "month_start": month_start,
        "date_col": date_col,
        "month_label": month_start.strftime("%b %Y"),
    }


def compute_overall_metrics(frame):
    labelled = frame.assign(_overall="All")
    return compute_subgroup_metrics(
        labelled,
        label_col="target_ED_30d",
        score_col="ed_pred_prob",
        subgroup_col="_overall",
        n_boot=MONTHLY_N_BOOT,
        random_state=MONTHLY_RANDOM_STATE,
    ).iloc[0]


def selected_categories(names):
    by_name = {category.name: category for category in CATEGORY_REGISTRY}
    return [by_name[name] for name in names]


def compute_category_pairs(data, names):
    pairs = []
    for category in selected_categories(names):
        monthly = compute_category_metrics(
            data["monthly"],
            category,
            n_boot=MONTHLY_N_BOOT,
            random_state=MONTHLY_RANDOM_STATE,
        )
        cumulative = compute_category_metrics(
            data["all"],
            category,
            n_boot=MONTHLY_N_BOOT,
            random_state=MONTHLY_RANDOM_STATE,
        )
        pairs.append((category.display_name, monthly, cumulative))
    return pairs


def subgroup_union(monthly, cumulative):
    order = monthly["subgroup"].astype(str).tolist()
    for value in cumulative["subgroup"].astype(str):
        if value not in order:
            order.append(value)
    return order


def metric_row(frame, subgroup):
    matches = frame.loc[frame["subgroup"].astype(str) == str(subgroup)]
    return matches.iloc[0] if not matches.empty else None


# ---------------------------------------------------------------------------
# Section 1: prediction-space dashboard (alerts and predicted risk scores).
# ---------------------------------------------------------------------------


def draw_prediction_space(data, report_week, max_alerts, scale=1.0):
    fig = figure(13.7)
    header(
        fig,
        "ED VISIT RISK PREDICTION  ·  MONITORING",
        f"Week ending {report_week:%a %d %b %Y}  •  Weekly report",
        "PREDICTION SPACE",
    )
    card(fig, 0.025, 0.836, 0.461, 0.074)
    card(fig, 0.497, 0.836, 0.478, 0.074)
    txt(fig, 0.045, 0.892, "ALERTS THIS WEEK", 9.8, "bold", MUTED)
    txt(fig, 0.045, 0.854, f"{int(data['alerts_this_week']):,}", 27, "bold")
    txt(fig, 0.22, 0.857, "First-treatment patients emailed", 9, color=MUTED)
    txt(fig, 0.518, 0.892, "CUMULATIVE ALERTS", 9.8, "bold", MUTED)
    txt(fig, 0.518, 0.854, f"{int(data['cumulative_alerts']):,}", 27, "bold")
    txt(fig, 0.704, 0.857, "Since live deployment", 9, color=MUTED)

    card(fig, 0.025, 0.601, 0.95, 0.225)
    txt(fig, 0.045, 0.807, "Alerts sent by week", 11.5, "bold")
    txt(
        fig,
        0.045,
        0.784,
        "Weekly count (left axis)  ·  Cumulative count (right axis)",
        8.1,
        color=MUTED,
    )
    weeks = pd.to_datetime(data["weeks"])
    weekly = np.asarray(data["weekly_counts"], dtype=float)
    cumulative = np.asarray(data["cumulative_counts"], dtype=float)
    ax = axis(fig, [0.102, 0.653, 0.777, 0.114])
    ax.plot(weeks, weekly, "o-", color=BLUE, lw=2.1, ms=4.4, zorder=4)
    ax.set_ylim(0, upper_limit(weekly))
    ax.locator_params(axis="y", nbins=4)
    ax.set_ylabel("Weekly alerts", fontsize=8.1, color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.grid(axis="y", alpha=0.55, color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8)

    tick_idx = np.unique(np.linspace(0, len(weeks) - 1, min(8, len(weeks)), dtype=int))
    tick_values = weeks[tick_idx]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(
        [value.strftime("%b %d, %Y") for value in tick_values],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=7.6,
    )
    ax.tick_params(axis="x", pad=5)
    ax2 = ax.twinx()
    ax2.set_zorder(4)
    ax2.patch.set_visible(False)
    ax2.plot(weeks, cumulative, "s-", color=ORANGE, lw=2, ms=3.7, zorder=4)
    right_upper = float(cumulative.max()) * 1.07 or 1.0
    ax2.set_ylim(max(0, float(cumulative.min()) * 0.94), right_upper)
    ax2.locator_params(axis="y", nbins=4)
    ax2.tick_params(axis="y", labelsize=8, labelcolor=ORANGE, length=0)
    ax2.set_ylabel(
        f"Cumulative alerts\nTarget: {max_alerts:,}",
        fontsize=8.1,
        color=ORANGE,
        labelpad=10,
    )
    ax2.spines["top"].set_visible(False)
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=2, marker="o", ms=4, label="Weekly"),
            Line2D([0], [0], color=ORANGE, lw=2, marker="s", ms=3.5, label="Cumulative"),
        ],
        loc="center right",
        bbox_to_anchor=(0.961, 0.805),
        ncol=2,
        frameon=False,
        fontsize=8.0,
    )

    card(fig, 0.025, 0.336, 0.95, 0.254)
    txt(fig, 0.045, 0.569, "Alerts by clinician", 11.5, "bold")
    txt(
        fig,
        0.045,
        0.548,
        "Sorted by cumulative count  ·  Weekly (left axis) and cumulative (right axis)",
        8.1,
        color=MUTED,
    )
    clinicians = data["clinicians"]
    clinician_weekly = np.asarray(data["clinician_weekly"], dtype=float)
    clinician_cumulative = np.asarray(data["clinician_cumulative"], dtype=float)
    x = np.arange(len(clinicians))
    width = 0.38
    ax = axis(fig, [0.1, 0.403, 0.79, 0.128])
    ax.bar(x - width / 2, clinician_weekly, width=width, color=BLUE, zorder=3)
    ax.set_ylim(0, upper_limit(clinician_weekly))
    ax.set_ylabel("Weekly", fontsize=8.4, color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_clinician_label(name) for name in clinicians],
        rotation=45,
        ha="right",
        fontsize=7.2,
        linespacing=1.35,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(length=0, pad=2)
    ax2 = ax.twinx()
    ax2.patch.set_visible(False)
    ax2.bar(x + width / 2, clinician_cumulative, width=width, color=ORANGE, zorder=3)
    ax2.set_ylim(0, upper_limit(clinician_cumulative))
    ax2.set_ylabel("Cumulative", fontsize=8.4, color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE, length=0, labelsize=8)
    ax2.spines["top"].set_visible(False)
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=5, label="Weekly"),
            Line2D([0], [0], color=ORANGE, lw=5, label="Cumulative"),
        ],
        loc="center right",
        bbox_to_anchor=(0.96, 0.568),
        ncol=2,
        frameon=False,
        fontsize=8.3,
    )

    card(fig, 0.025, 0.035, 0.95, 0.29)
    txt(fig, 0.046, 0.304, "Predicted ED visit risk distribution", 12, "bold")
    txt(
        fig,
        0.046,
        0.28,
        "First-treatment patients only  ·  Area-normalized densities (not raw counts)",
        8.3,
        color=MUTED,
    )
    ax = axis(fig, [0.085, 0.077, 0.82, 0.177])
    xx = np.linspace(0, 1, 250)
    weekly_scores = data["weekly_scores"]
    ax.hist(
        weekly_scores,
        bins=WEEKLY_RISK_BINS,
        density=True,
        color=BLUE,
        edgecolor=BLUE,
        linewidth=1.05,
        alpha=0.30,
        zorder=2,
        label=f"This week (n={len(weekly_scores):,}; binned)",
    )
    for vals, color, name, linestyle in [
        (data["silent_scores"], SILENT, "Silent deployment", "--"),
        (data["cumulative_scores"], ORANGE, "Cumulative live", "-"),
    ]:
        ax.plot(
            xx,
            kde_curve(vals, xx),
            c=color,
            lw=2.2,
            ls=linestyle,
            zorder=4,
            label=f"{name} (n={len(vals):,})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Predicted risk score", fontsize=8.8)
    ax.set_ylabel("Density", fontsize=8.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, alpha=0.7)
    ax.tick_params(length=0, labelsize=8.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [
        next(index for index, label in enumerate(labels) if label.startswith(prefix))
        for prefix in ["Silent deployment", "This week", "Cumulative live"]
    ]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        ncol=3,
        loc="upper right",
        fontsize=7.9,
    )

    return render_figure(fig, scale)


# ---------------------------------------------------------------------------
# Section 2: input-space dashboard (missingness and PSI).
# ---------------------------------------------------------------------------


def draw_input_space(data, scale=1.0):
    fig = figure(9.6)
    header(
        fig,
        "INPUT QUALITY & DISTRIBUTION SHIFT",
        "Weekly and cumulative  •  First-treatment patients only",
        "INPUT SPACE",
    )
    card(fig, 0.025, 0.506, 0.95, 0.383)
    txt(fig, 0.046, 0.859, "Feature-level missingness", 12.1, "bold")
    txt(fig, 0.046, 0.836, "Top 10 of ~50 input features by cumulative missingness", 8.5, color=MUTED)

    missingness = data["missingness"]
    y = np.arange(len(missingness))[::-1]
    xmax = max(64.0, max((max(row[1:]) for row in missingness), default=0) * 1.08)
    ax = axis(fig, [0.3, 0.548, 0.51, 0.245])
    for i, (_, weekly, cumulative, silent) in enumerate(missingness):
        yi = y[i]
        ax.plot(
            [min(weekly, cumulative, silent), max(weekly, cumulative, silent)],
            [yi, yi],
            color="#CCD7E1",
            lw=3,
            zorder=1,
        )
        ax.scatter(silent, yi, s=50, color=SILENT, marker="^", zorder=3)
        ax.scatter(cumulative, yi, s=42, color=ORANGE, marker="D", zorder=4)
        ax.scatter(weekly, yi, s=49, color=BLUE, marker="o", zorder=5)
        ax.text(
            1.02,
            yi,
            f"{weekly:.0f}% / {cumulative:.0f}% / {silent:.0f}%",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.0,
            color=NAVY,
            clip_on=False,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [wrap_label(row[0], FEATURE_LABEL_WIDTH) for row in missingness],
        fontsize=9,
        linespacing=1.25,
    )
    xticks = np.arange(0, xmax + 1, 10)
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(value)}%" for value in xticks])
    clean(ax)
    fig.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, ms=6, label="This week"
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=ORANGE,
                markeredgecolor=ORANGE,
                ms=6,
                label="Cumulative",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor=SILENT,
                markeredgecolor=SILENT,
                ms=6,
                label="Silent deployment",
            ),
        ],
        bbox_to_anchor=(0.67, 0.812),
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=8.1,
    )
    txt(
        fig,
        0.046,
        0.525,
        "Right labels = weekly / cumulative / silent  •  Fraction missing for each feature",
        8,
        color=MUTED,
    )

    card(fig, 0.025, 0.035, 0.95, 0.451)
    txt(fig, 0.046, 0.456, "Population Stability Index (PSI)", 12.1, "bold")
    txt(
        fig,
        0.046,
        0.432,
        "Primary reference: silent deployment — first-treatment subset, fixed bins/categories",
        8.1,
        color=MUTED,
    )
    txt(fig, 0.052, 0.401, "Top 5 continuous by cumulative PSI", 9.4, "bold")
    txt(fig, 0.544, 0.401, "Top 5 categorical/binary by cumulative PSI", 9.4, "bold")
    psi_plot(fig, [0.17, 0.123, 0.262, 0.245], data["psi_continuous"])
    psi_plot(fig, [0.668, 0.123, 0.258, 0.245], data["psi_categorical"])
    fig.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, ms=6, label="Weekly PSI"
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=ORANGE,
                markeredgecolor=ORANGE,
                ms=6,
                label="Cumulative PSI",
            ),
        ],
        bbox_to_anchor=(0.5, 0.377),
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    txt(
        fig,
        0.048,
        0.076,
        "PSI reference is fixed. Dotted 0.10 / 0.25 lines are rough heuristics, not validated alarms.",
        7.9,
        color=MUTED,
    )

    return render_figure(fig, scale)


# ---------------------------------------------------------------------------
# Monthly Section 3 and Section 4: clinical and cancer subgroup forests with
# prototype geometry, monthly/cumulative estimates, and Wilson/bootstrap CIs.
# ---------------------------------------------------------------------------


def safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def metric_values(row, kind):
    columns = {
        "event": ("event_rate", "event_rate_ci_low", "event_rate_ci_high"),
        "auc": ("auc", "auc_ci_low", "auc_ci_high"),
        "brier": ("brier", "brier_ci_low", "brier_ci_high"),
    }
    if row is None:
        return {"n": 0, "n_pos": 0, "value": np.nan, "low": np.nan, "high": np.nan}
    value_col, low_col, high_col = columns[kind]
    return {
        "n": int(row["n"]),
        "n_pos": int(row["n_pos"]),
        "value": safe_float(row[value_col]),
        "low": safe_float(row[low_col]),
        "high": safe_float(row[high_col]),
    }


def format_metric(value, kind):
    if not np.isfinite(value):
        return "n/a"
    if kind == "event":
        return f"{value:.1%}"
    if kind == "brier":
        return f"{value:.3f}"
    return f"{value:.2f}"


def metric_summary(fig, region, label, kind, monthly, cumulative):
    a, b, w, h = region
    monthly_values = metric_values(monthly, kind)
    cumulative_values = metric_values(cumulative, kind)
    card(fig, a, b, w, h)
    txt(fig, a + 0.018, b + h - 0.030, label, 9.9, "bold", MUTED)
    txt(fig, a + 0.018, b + h - 0.078, format_metric(monthly_values["value"], kind), 21, "bold", BLUE)
    txt(
        fig,
        a + 0.018,
        b + 0.037,
        f"Cumulative: {format_metric(cumulative_values['value'], kind)}",
        9.4,
        color=ORANGE,
    )
    if np.isfinite(monthly_values["low"]) and np.isfinite(monthly_values["high"]):
        ci_text = f"95% CI {format_metric(monthly_values['low'], kind)}–{format_metric(monthly_values['high'], kind)}"
        txt(fig, a + w - 0.018, b + 0.065, ci_text, 7.6, color=MUTED, ha="right")


def category_separators(category_pairs):
    separators = []
    total = 0
    for _, monthly, cumulative in category_pairs[:-1]:
        total += len(subgroup_union(monthly, cumulative))
        separators.append(total)
    return separators


def draw_forest(fig, region, category_pairs, kind, show_left, separators=None, wrap_labels=False):
    groups = []
    for _, monthly, cumulative in category_pairs:
        for subgroup in subgroup_union(monthly, cumulative):
            groups.append((subgroup, metric_row(monthly, subgroup), metric_row(cumulative, subgroup)))

    ax = axis(fig, region)
    ypos = np.arange(len(groups))[::-1]
    limits = {"event": (0, 0.50), "auc": (0, 1.0), "brier": (0, 0.40)}
    lower, upper = limits[kind]
    ax.set_xlim(lower, upper)
    ax.set_ylim(-0.58, len(groups) - 0.42)
    ax.grid(axis="x", color=GRID, lw=0.7)
    if kind == "auc":
        ax.axvline(0.5, ls=":", lw=1.2, color="#ADB7C3", zorder=1)

    def clip(value):
        return min(max(float(value), lower), upper)

    for i, (_, monthly, cumulative) in enumerate(groups):
        y = ypos[i]
        for values, offset, color, marker in (
            (metric_values(monthly, kind), 0.13, BLUE, "o"),
            (metric_values(cumulative, kind), -0.13, ORANGE, "D"),
        ):
            if not np.isfinite(values["value"]):
                continue
            low = values["low"]
            high = values["high"]
            value = values["value"]
            if np.isfinite(low) and np.isfinite(high):
                if low != high:
                    ax.plot(
                        [clip(low), clip(high)],
                        [y + offset, y + offset],
                        color=color,
                        lw=1.45,
                        zorder=3,
                        solid_capstyle="round",
                    )
                    ax.plot([clip(low)] * 2, [y + offset - 0.075, y + offset + 0.075], color=color, lw=1.1, zorder=3)
                    ax.plot(
                        [clip(high)] * 2,
                        [y + offset - 0.075, y + offset + 0.075],
                        color=color,
                        lw=1.1,
                        zorder=3,
                    )
                if low < lower:
                    ax.annotate(
                        f"<{lower:.2f}",
                        (clip(lower), y + offset),
                        xytext=(3, 6 if offset > 0 else -8),
                        textcoords="offset points",
                        ha="left",
                        va="bottom" if offset > 0 else "top",
                        fontsize=6.5,
                        color=MUTED,
                        zorder=6,
                        annotation_clip=False,
                    )
                if high > upper:
                    ax.annotate(
                        f">{upper:.2f}",
                        (clip(upper), y + offset),
                        xytext=(-3, 6 if offset > 0 else -8),
                        textcoords="offset points",
                        ha="right",
                        va="bottom" if offset > 0 else "top",
                        fontsize=6.5,
                        color=MUTED,
                        zorder=6,
                        annotation_clip=False,
                    )
            at_lower = value < lower
            at_upper = value > upper
            point_marker = "<" if at_lower else ">" if at_upper else marker
            ax.scatter(
                [clip(value)],
                [y + offset],
                color=color,
                edgecolors=WHITE,
                linewidths=0.8,
                s=30,
                marker=point_marker,
                zorder=5,
                clip_on=False,
            )
            numeric_xytext = (5, -12 if offset > 0 else 12) if at_lower or at_upper else (5, 6 if offset > 0 else -8)
            ax.annotate(
                format_metric(value, kind),
                (clip(value), y + offset),
                xytext=numeric_xytext,
                textcoords="offset points",
                ha="left",
                va="bottom" if offset > 0 else "top",
                fontsize=6.3,
                color=color,
                zorder=7,
                annotation_clip=False,
            )

    if separators:
        for separator in separators:
            ax.axhline(len(groups) - separator - 0.5, color="#BCCEDA", lw=0.85, zorder=0)

    ax.set_yticks(ypos)
    if show_left:
        labels = []
        for subgroup, monthly, cumulative in groups:
            monthly_values = metric_values(monthly, "event")
            cumulative_values = metric_values(cumulative, "event")
            label_name = wrap_label(subgroup, 18, inclusive=True) if wrap_labels else subgroup
            labels.append(
                f"{label_name}\nn={monthly_values['n']:,}/{cumulative_values['n']:,} "
                f"· ED={monthly_values['n_pos']:,}/{cumulative_values['n_pos']:,}"
            )
        ax.set_yticklabels(labels, fontsize=8.2, linespacing=1.5)
    else:
        ax.set_yticklabels([])
    clean(ax)
    ax.tick_params(axis="y", pad=8)

    if kind == "event":
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    elif kind == "auc":
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"])
    else:
        ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
        ax.set_xticklabels(["0", ".05", ".10", ".15", ".20", ".25", ".30", ".35", ".40"])
    return ax


def monthly_header(fig, run_date, month_label, cumulative_label):
    header(
        fig,
        "OUTCOME PERFORMANCE  ·  CLINICAL SUBGROUPS",
        f"Most recent available month before {run_date:%b %Y}: {month_label}  •  {cumulative_label}",
        "MONTHLY ONLY",
    )


def draw_monthly_clinical_subgroups(data, run_date, scale=1.0):
    fig = figure(9.3)
    monthly_header(fig, run_date, data["month_label"], "Cumulative: all first-treatment patients")
    monthly_overall = compute_overall_metrics(data["monthly"])
    cumulative_overall = compute_overall_metrics(data["all"])
    monthly_event = metric_values(monthly_overall, "event")
    cumulative_event = metric_values(cumulative_overall, "event")
    metric_summary(fig, (0.025, 0.744, 0.300, 0.139), "Event rate", "event", monthly_overall, cumulative_overall)
    metric_summary(fig, (0.337, 0.744, 0.300, 0.139), "AUROC", "auc", monthly_overall, cumulative_overall)
    metric_summary(
        fig,
        (0.649, 0.744, 0.326, 0.139),
        "Brier score  ↓ better",
        "brier",
        monthly_overall,
        cumulative_overall,
    )
    txt(
        fig,
        0.037,
        0.719,
        f"Monthly: n={monthly_event['n']:,} patients, {monthly_event['n_pos']:,} ED events"
        f"    •    Cumulative: n={cumulative_event['n']:,}, {cumulative_event['n_pos']:,} ED events",
        8.8,
        color=MUTED,
    )
    card(fig, 0.025, 0.067, 0.950, 0.618)
    txt(fig, 0.045, 0.655, "Monthly vs cumulative estimates by subgroup", 12, "bold")
    txt(
        fig,
        0.045,
        0.625,
        "Dots = point estimates; horizontal whiskers = 95% CI. Numbers in labels: monthly/cumulative.",
        8.5,
        color=MUTED,
    )
    txt(fig, 0.045, 0.586, "Subgroup / n (sample size)  •  ED events", 9, "bold")
    txt(fig, 0.28, 0.565, "EVENT RATE", 9.9, "bold", ha="center")
    txt(fig, 0.545, 0.565, "AUROC", 9.9, "bold", ha="center")
    txt(fig, 0.83, 0.565, "BRIER", 9.9, "bold", ha="center")
    category_pairs = compute_category_pairs(data, MONTHLY_CLINICAL_CATEGORIES)
    forests = [(0.18, 0.145, 0.20, 0.387), (0.41, 0.145, 0.27, 0.387), (0.71, 0.145, 0.24, 0.387)]
    for region, kind in zip(forests, ("event", "auc", "brier"), strict=True):
        draw_forest(
            fig,
            region,
            category_pairs,
            kind,
            show_left=(kind == "event"),
            separators=category_separators(category_pairs),
        )
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE, ms=6, lw=1.6, label=f"{data['month_label']} (monthly)"),
            Line2D([0], [0], marker="D", color=ORANGE, ms=6, lw=1.6, label="Cumulative"),
        ],
        bbox_to_anchor=(0.54, 0.102),
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=8.6,
    )
    return render_figure(fig, scale)


def draw_monthly_cancer_groups(data, run_date, scale=1.0):
    fig = figure(7.45)
    header(
        fig,
        "OUTCOME PERFORMANCE  ·  CANCER GROUPS",
        f"Most recent available month before {run_date:%b %Y}: {data['month_label']}  •  Same cohort and CI convention",
        "MONTHLY ONLY",
    )
    card(fig, 0.025, 0.082, 0.950, 0.800)
    txt(fig, 0.045, 0.846, "Cancer-group outcome metrics", 12.5, "bold")
    txt(fig, 0.045, 0.818, "Monthly and cumulative; subgroup labels show n and ED-event counts.", 8.3, color=MUTED)
    txt(fig, 0.045, 0.771, "Cancer type / n (sample size)  •  ED events", 9, "bold")
    txt(fig, 0.28, 0.741, "EVENT RATE", 9.9, "bold", ha="center")
    txt(fig, 0.545, 0.741, "AUROC", 9.9, "bold", ha="center")
    txt(fig, 0.83, 0.741, "BRIER", 9.9, "bold", ha="center")
    category_pairs = compute_category_pairs(data, MONTHLY_CANCER_CATEGORIES)
    forests = [(0.18, 0.218, 0.20, 0.475), (0.41, 0.218, 0.27, 0.475), (0.71, 0.218, 0.24, 0.475)]
    for region, kind in zip(forests, ("event", "auc", "brier"), strict=True):
        draw_forest(fig, region, category_pairs, kind, show_left=(kind == "event"), wrap_labels=True)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE, ms=6, lw=1.6, label=f"{data['month_label']} (monthly)"),
            Line2D([0], [0], marker="D", color=ORANGE, ms=6, lw=1.6, label="Cumulative"),
        ],
        bbox_to_anchor=(0.54, 0.158),
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    return render_figure(fig, scale)


# ---------------------------------------------------------------------------
# Output combination: stitch the four rendered sections into the single full
# dashboard image.
# ---------------------------------------------------------------------------


def combine_images(images, output_dir, filename, scale=1.0):
    separator = max(1, round(8 * scale))
    combined = Image.new(
        "RGB",
        (images[0].width, sum(image.height for image in images) + separator * (len(images) - 1)),
        BG,
    )
    position = 0
    for image in images:
        combined.paste(image, (0, position))
        position += image.height + separator
    path = Path(output_dir) / filename
    combined.save(path, optimize=True)
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point.
# ---------------------------------------------------------------------------


def run_pipeline(
    max_alerts,
    run_date,
    weekly_dir,
    monthly_dir,
    consented_physician_file,
    anchor,
    output_dir,
    image_scale=1.0,
):
    run_date = to_date(run_date)
    weekly_input_paths = weekly_paths(weekly_dir, anchor)
    monthly_input_paths = monthly_paths(monthly_dir, anchor)
    if not metadata_gate_passes(weekly_input_paths):
        return []

    report_week = weekly_report_week(weekly_input_paths, run_date)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    consented_physicians = load_consented_physicians(consented_physician_file)
    weekly_data = load_weekly_data(weekly_input_paths, report_week)
    monthly_data = load_monthly_data(monthly_input_paths, run_date, anchor, consented_physicians)
    sections = [
        draw_prediction_space(weekly_data, report_week, max_alerts, scale=image_scale),
        draw_input_space(weekly_data, scale=image_scale),
        draw_monthly_clinical_subgroups(monthly_data, run_date, scale=image_scale),
        draw_monthly_cancer_groups(monthly_data, run_date, scale=image_scale),
    ]
    full_combined_path = combine_images(
        sections,
        output_dir,
        f"ED_risk_monitoring_full_{run_date:%Y%m%d}.png",
        scale=image_scale,
    )
    print(f"[INFO] Saved {full_combined_path}")
    return [full_combined_path]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate the weekly and monthly ED-risk monitoring dashboard.")
    parser.add_argument(
        "--max-alerts", required=True, type=int, help="Cumulative alert target for the weekly dashboard."
    )
    parser.add_argument("--run-date", required=True, type=argparse_date, help="Metadata run date (YYYYMMDD).")
    parser.add_argument("--interim-weekly-dir", required=True, type=Path)
    parser.add_argument("--interim-monthly-dir", required=True, type=Path)
    parser.add_argument(
        "--consented-physician-file",
        required=True,
        type=Path,
        help="CSV with EpicProvider and Enabled columns defining consented physicians.",
    )
    parser.add_argument("--model-anchor", choices=("clinic", "treatment"), default="clinic")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="Multiplier for rendered image dimensions (e.g. 1.5 or 2).",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.image_scale) or args.image_scale <= 0:
        parser.error("--image-scale must be a positive finite number")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        run_pipeline(
            max_alerts=args.max_alerts,
            run_date=args.run_date,
            weekly_dir=args.interim_weekly_dir,
            monthly_dir=args.interim_monthly_dir,
            consented_physician_file=args.consented_physician_file,
            anchor=args.model_anchor,
            output_dir=args.output_dir,
            image_scale=args.image_scale,
        )
    except (MonitoringPipelineError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
