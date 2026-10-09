#!/usr/bin/env python3
"""
Subgroup metrics analysis tool.

For each subgroup category, computes event rate, AUC, and Brier score
(each with bootstrap 95% CIs), prints a results table, saves it to CSV,
and produces a 3-panel bar chart (event rate | AUC | Brier score) per
category.

Designed to be extended: to add a new subgroup category in the future,
add a CategoryConfig to CATEGORY_REGISTRY below. No other code needs
to change.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

CANCER_MAPPING = {
    'hepatocellular': 'hepatobiliary',
    'biliary': 'hepatobiliary',
}

INTENT_MAPPING = {
    'CURATIVE': 'NON-PALLIATIVE',
    'NEOADJUVANT': 'NON-PALLIATIVE',
    'ADJUVANT': 'NON-PALLIATIVE',
    'PALLIATIVE': 'PALLIATIVE',
}

FIRST_TRT_MAPPING = {
    1: "New treatment",
    0: "Continuing treatment",
}

@dataclass
class CategoryConfig:
    """Defines one subgroup category to analyze.

    name:         internal key, used in --categories and output filenames
    column:       raw column name expected in the input CSV
    display_name: human-readable label used in plot titles
    mapping:      optional dict mapping raw values -> subgroup labels
                  (raises if a raw value isn't covered)
    transform:    optional custom function(series) -> series; if given,
                  takes precedence over `mapping` (use this for things
                  like binning a continuous column, e.g. age > 65)
    """
    name: str
    column: str
    display_name: str
    mapping: Optional[dict] = None
    transform: Optional[Callable[[pd.Series], pd.Series]] = None

    def get_subgroup_labels(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            raise KeyError(
                f"Column '{self.column}' not found in input data "
                f"(required for category '{self.name}')."
            )
        series = df[self.column]
        if self.transform is not None:
            return self.transform(series)
        if self.mapping is not None:
            # Partial mapping: values not in the mapping are kept as is.
            passed_through = sorted(set(series.dropna().unique()) - set(self.mapping.keys()), key=str)
            if passed_through:
                print(f"[INFO] Category '{self.name}': values not in mapping kept as is: {passed_through}")
            return series.map(lambda v: self.mapping.get(v, v))
        return series


# Registered categories. Add new ones here as new columns become available.
CATEGORY_REGISTRY: List[CategoryConfig] = [
    CategoryConfig(
        name="cancer_type",
        column="cancer",
        display_name="Cancer Type",
        mapping=CANCER_MAPPING
    ),
    CategoryConfig(
        name="age_over_65",
        column="age",
        display_name="Age > 65",
        transform=lambda s: np.where(s > 65, ">65", "<=65"),
    ),
    CategoryConfig(
        name="sex",
        column="gender",
        display_name="Sex at Birth",
    ),
    CategoryConfig(
        name="treatment_intent",
        column="intent",
        display_name="Treatment Intent",
        mapping=INTENT_MAPPING,
    ),
    CategoryConfig(
        name="first_trt",
        column="first_trt",
        display_name="Treatment Initiation",
        mapping=FIRST_TRT_MAPPING,
    ),
]


# ---------------------------------------------------------------------------
# Bootstrap: event rate, AUC, Brier score (single resampling pass)
# ---------------------------------------------------------------------------

def bootstrap_metrics_ci(y_true, y_score, n_boot=2000, ci=0.95, random_state=None, compute_auc=True):
    """Return point estimates + percentile bootstrap CIs for AUC and
    Brier score (from the *same* bootstrap resamples), plus the event
    rate as a plain point estimate (no CI — it's not bootstrapped).

    compute_auc=False skips AUC (used when a subgroup is single-class,
    where AUC is undefined) but Brier score is still computed.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    point_event_rate = y_true.mean()
    point_brier = brier_score_loss(y_true, y_score)
    point_auc = roc_auc_score(y_true, y_score) if compute_auc else np.nan

    boot_briers = []
    boot_aucs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        boot_briers.append(brier_score_loss(yt, ys))
        if compute_auc and len(np.unique(yt)) >= 2:
            boot_aucs.append(roc_auc_score(yt, ys))

    alpha = (1 - ci) / 2

    def pct_ci(vals):
        if len(vals) == 0:
            return np.nan, np.nan
        return np.percentile(vals, 100 * alpha), np.percentile(vals, 100 * (1 - alpha))

    brier_lo, brier_hi = pct_ci(boot_briers)
    auc_lo, auc_hi = pct_ci(boot_aucs) if compute_auc else (np.nan, np.nan)

    return {
        "event_rate": point_event_rate,
        "auc": point_auc, "auc_ci_low": auc_lo, "auc_ci_high": auc_hi,
        "brier": point_brier, "brier_ci_low": brier_lo, "brier_ci_high": brier_hi,
    }


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------

def compute_subgroup_metrics(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    subgroup_col: str,
    n_boot: int = 2000,
    ci: float = 0.95,
    min_n: int = 10,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Compute event rate, AUC, and Brier score (+ bootstrap CIs) per
    unique value of subgroup_col."""
    results = []
    for subgroup_value, sub_df in df.groupby(subgroup_col, dropna=False):
        sub_df = sub_df.dropna(subset=[label_col, score_col])
        n = len(sub_df)
        n_pos = int(sub_df[label_col].sum()) if n else 0
        n_neg = n - n_pos

        if n < min_n:
            results.append({
                "subgroup": subgroup_value, "n": n, "n_pos": n_pos,
                "event_rate": np.nan,
                "auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan,
                "brier": np.nan, "brier_ci_low": np.nan, "brier_ci_high": np.nan,
                "skipped_reason": "insufficient data",
            })
            continue

        compute_auc = n_pos > 0 and n_neg > 0
        metrics = bootstrap_metrics_ci(
            sub_df[label_col], sub_df[score_col],
            n_boot=n_boot, ci=ci, random_state=random_state, compute_auc=compute_auc,
        )
        row = {"subgroup": subgroup_value, "n": n, "n_pos": n_pos, **metrics}
        row["skipped_reason"] = None if compute_auc else "single class only (AUC undefined)"
        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_subgroup_metrics(results: pd.DataFrame, category: CategoryConfig, output_dir: str, ci_pct: int, title_note=""):
    plot_df = results[results["skipped_reason"] != "insufficient data"].copy()
    if plot_df.empty:
        print(f"[WARN] No valid subgroups to plot for category '{category.name}'. Skipping plot.")
        return None

    # Order subgroups consistently across all three panels, AUC descending
    # (subgroups with undefined AUC sort to the end).
    plot_df = plot_df.sort_values("auc", ascending=False, na_position="last").reset_index(drop=True)
    x = np.arange(len(plot_df))
    labels = plot_df["subgroup"].astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(max(14, 3.2 * len(plot_df)), 6))

    # (value_col, ci_low_col, ci_high_col, title, ylim) — ci cols are None
    # for panels with no CI (just event rate, for now).
    panel_specs = [
        ("event_rate", None, None, "Event Rate", (0, 1.05)),
        ("auc", "auc_ci_low", "auc_ci_high", "AUC", (0, 1.05)),
        ("brier", "brier_ci_low", "brier_ci_high", "Brier Score", None),
    ]

    for panel_idx, (ax, (val_col, lo_col, hi_col, title, ylim)) in enumerate(zip(axes, panel_specs)):
        values = plot_df[val_col].fillna(0)
        has_ci = lo_col is not None and hi_col is not None
        if has_ci:
            yerr_low = (plot_df[val_col] - plot_df[lo_col]).clip(lower=0).fillna(0)
            yerr_high = (plot_df[hi_col] - plot_df[val_col]).clip(lower=0).fillna(0)
            yerr = [yerr_low, yerr_high]
        else:
            yerr = None

        bars = ax.bar(
            x, values, yerr=yerr,
            capsize=4, color="#4C72B0", edgecolor="black",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        title_suffix = f"\n({ci_pct}% bootstrap CI)" if has_ci else ""
        ax.set_title(f"{title}{title_suffix}")
        ax.set_ylim(bottom=0)
        if ylim:
            ax.set_ylim(*ylim)
        if title == "AUC":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        top = max(values.max(), 1e-6)
        for bar, (_, row) in zip(bars, plot_df.iterrows()):
            height = bar.get_height()
            if pd.isna(row[val_col]):
                ax.annotate("n/a", xy=(bar.get_x() + bar.get_width() / 2, 0.02 * top),
                            ha="center", va="bottom", fontsize=8, color="black")
            elif panel_idx == 0:
                # Sample size only needs to be shown once; put it on the first panel.
                ax.annotate(f"n={int(row['n'])}", xy=(bar.get_x() + bar.get_width() / 2, 0.02 * top),
                            ha="center", va="bottom", fontsize=8,
                            color="white" if height > 0.08 * top else "black")

    fig.suptitle(f"Subgroup metrics by {category.display_name}{title_note}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"metrics_by_{category.name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved plot for '{category.name}' to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Subgroup event rate / AUC / Brier score analysis with bootstrap CIs.")
    parser.add_argument("csv_path", help="Path to input CSV file.")
    parser.add_argument("--output-dir", default="./subgroup_auc_output",
                         help="Directory to save figures and result tables (created if missing).")
    parser.add_argument("--score-col", default="ed_pred_prob",
                         help="Column with predicted probabilities.")
    parser.add_argument("--label-col", default="target_ED_30d",
                         help="Column with binary ground-truth label.")
    parser.add_argument("--categories", nargs="+", default=None,
                         help="Subset of registered category names to run (default: all).")
    parser.add_argument("--n-boot", type=int, default=2000,
                         help="Number of bootstrap resamples per subgroup.")
    parser.add_argument("--ci", type=float, default=0.95,
                         help="Confidence interval width, e.g. 0.95 for 95%%.")
    parser.add_argument("--min-n", type=int, default=10,
                         help="Minimum subgroup size required to compute metrics.")
    parser.add_argument("--random-state", type=int, default=42,
                         help="Random seed for reproducible bootstrapping.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    for col in (args.score_col, args.label_col):
        if col not in df.columns:
            sys.exit(f"[ERROR] Required column '{col}' not found in {args.csv_path}.")

    has_first_trt = "first_trt" in df.columns

    categories_to_run = CATEGORY_REGISTRY
    if not has_first_trt:
        categories_to_run = [c for c in categories_to_run if c.name != "first_trt"]
    if args.categories:
        available = [c.name for c in categories_to_run]
        wanted = set(args.categories)
        categories_to_run = [c for c in categories_to_run if c.name in wanted]
        missing = wanted - {c.name for c in categories_to_run}
        if missing:
            sys.exit(f"[ERROR] Unknown categories: {sorted(missing)}. Available: {available}")

    os.makedirs(args.output_dir, exist_ok=True)

    for category in categories_to_run:
        is_filtered = has_first_trt and category.name != "first_trt"
        analysis_df = df[df["first_trt"] == 1].copy() if is_filtered else df
        if is_filtered:
            print(f"[INFO] Category '{category.name}' restricted to first_trt == 1 ({len(analysis_df)} rows).")

        print(f"\n[INFO] Running subgroup metrics analysis for category: {category.name}")
        subgroup_col = f"_subgroup_{category.name}"
        try:
            analysis_df[subgroup_col] = category.get_subgroup_labels(analysis_df)
        except (KeyError, ValueError) as e:
            print(f"[WARN] Skipping category '{category.name}': {e}")
            continue

        results = compute_subgroup_metrics(
            analysis_df, label_col=args.label_col, score_col=args.score_col, subgroup_col=subgroup_col,
            n_boot=args.n_boot, ci=args.ci, min_n=args.min_n, random_state=args.random_state,
        )

        print(results.to_string(index=False))

        csv_out = os.path.join(args.output_dir, f"metrics_by_{category.name}.csv")
        results.to_csv(csv_out, index=False)
        print(f"[INFO] Saved results table to {csv_out}")

        plot_subgroup_metrics(
            results, category, args.output_dir, ci_pct=int(round(args.ci * 100)),
            title_note="",
        )


if __name__ == "__main__":
    main()