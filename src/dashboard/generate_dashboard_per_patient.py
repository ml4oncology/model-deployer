"""
Module to assemble and save patient dashboards as PNGs using Playwright.
"""

import base64
import os
from pathlib import Path

import pandas as pd
import plotly.io as pio
from playwright.sync_api import sync_playwright

from deployer.dashboard.component import create_patient_overview, create_model_overview, create_percentile_overview
from deployer.dashboard.risk_dist_plot import risk_dist_plot

DASHBOARD_SUBDIR = "dashboards"
SHAP_SUBDIR = "shap_waterfall"
SUPPORTED_LAYOUTS = {"portrait", "landscape"}
DEFAULT_FONT_SCALE = 1.0

CSS_PATH = Path(__file__).parent / "style.css"


def _build_dashboard_html(
    mrn: int,
    clinic_date: pd.Timestamp,
    row: pd.Series,
    meta_row: pd.Series,
    df_output: pd.DataFrame,
    df_meta: pd.DataFrame,
    shap_png_path: Path,
    layout: str,
    font_scale: float,
) -> str:
    """Assemble the full dashboard HTML for one patient/clinic_date row."""

    # --- Patient card ---
    risk_level = "N/A" #"High Risk" if row["ed_pred_alarm_0.1"] == 1 else "Low Risk"
    patient_html = create_patient_overview(
        mrn=mrn,
        next_sched_trt=str(row["next_sched_trt_date"])[:10],
        cancer=meta_row["cancer"],
        age=meta_row["age"],
        gender=meta_row["gender"],
        risk_score=row["ed_pred_prob"],
        risk_level=risk_level,
    )

    # --- Risk distribution plot ---
    percentile_all, percentile_same, fig = risk_dist_plot(mrn, df_output, df_meta, font_scale=font_scale)
    risk_dist_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    percentile_html = create_percentile_overview(percentile_all, percentile_same, meta_row["cancer"])

    # --- Model overview ---
    model_html = create_model_overview()

    # --- SHAP waterfall: embed as base64 so HTML is fully self-contained ---
    with open(shap_png_path, "rb") as f:
        shap_b64 = base64.b64encode(f.read()).decode("utf-8")
    shap_img_tag = (
        f'<img src="data:image/png;base64,{shap_b64}" '
        f'alt="SHAP waterfall MRN {mrn}" style="width:100%; height:auto;" />'
    )

    # --- Read CSS inline so the HTML is self-contained ---
    css = CSS_PATH.read_text()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Dashboard MRN {mrn} — {str(clinic_date)[:10]}</title>
  <style>:root {{ --dashboard-font-scale: {font_scale}; }}</style>
  <style>{css}</style>
</head>
<body class="dashboard-page dashboard-layout-{layout}">

  <!-- Patient card -->
  <div class="dashboard-content">
    {patient_html}

    <!-- Risk distribution row: plot (4 cols) + percentile panel (1 col) -->
    <div class="dashboard-section dashboard-section--risk">
      <div class="dashboard-risk-plot">{risk_dist_html}</div>
      <div class="dashboard-risk-percentiles">{percentile_html}</div>
    </div>

    <!-- SHAP + model card row -->
    <div class="dashboard-section dashboard-section--bottom">
      <div class="dashboard-shap-section">
        <div class="section-title">Feature Contribution</div>
        <div class="dashboard-shap-panel">
          <div class="dashboard-shap-image">{shap_img_tag}</div>
        </div>
      </div>
      <div>{model_html}</div>
    </div>
  </div>

</body>
</html>"""
    return html


def save_dashboard_png(
    df_output: pd.DataFrame,
    df_meta: pd.DataFrame,
    output_dir: str | Path,
    layout: str = "portrait",
    font_scale: float = DEFAULT_FONT_SCALE,
) -> None:
    """
    Loop over every row in df_output, assemble the dashboard HTML,
    render it to PNG via Playwright, and save to output_dir/dashboards/.
    The SHAP waterfall PNG is deleted after use.

    Args:
        df_output: DataFrame with columns: mrn, next_sched_trt_date, clinic_date,
                   ed_pred_prob, ed_pred_alarm_0.1
        df_meta:   DataFrame with columns: mrn, clinic_date, age, gender, cancer
        output_dir: Root output directory (same one passed to get_model_output).
        layout: Dashboard layout style. Supported values are "portrait" and "landscape".
        font_scale: Multiplier applied to clinician-facing dashboard text.
    """
    if layout not in SUPPORTED_LAYOUTS:
        supported = ", ".join(sorted(SUPPORTED_LAYOUTS))
        raise ValueError(f"Unsupported layout '{layout}'. Expected one of: {supported}")
    if font_scale <= 0:
        raise ValueError("font_scale must be greater than 0")

    output_dir = Path(output_dir)
    dashboard_dir = output_dir / DASHBOARD_SUBDIR
    shap_dir = output_dir / SHAP_SUBDIR
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    # Index df_meta by (mrn, clinic_date) for fast lookup
    meta_indexed = df_meta.set_index(["mrn", "clinic_date"])
    viewport = {"width": 1280, "height": 1600} if layout == "portrait" else {"width": 1600, "height": 900}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=viewport)

        df_meta["clinic_date"] = pd.to_datetime(df_meta["clinic_date"])

        for _, row in df_output.iterrows():
            mrn = int(row["mrn"])
            clinic_date = pd.Timestamp(row["clinic_date"])
            clinic_date_str = clinic_date.strftime("%Y%m%d")

            # Locate SHAP PNG
            shap_png_path = shap_dir / f"shap_mrn_{mrn}_clinic_{clinic_date_str}.png"
            if not shap_png_path.exists():
                print(f"[WARN] SHAP plot not found, skipping: {shap_png_path}")
                continue

            # Fetch matching meta row (most recent clinic_date <= current for this mrn)
            try:
                meta_row = meta_indexed.loc[(mrn, clinic_date)]
            except KeyError:
                # Fall back to nearest available clinic date for this mrn
                mrn_meta = df_meta[df_meta["mrn"] == mrn].copy()
                mrn_meta["date_diff"] = (mrn_meta["clinic_date"] - clinic_date).abs()
                meta_row = mrn_meta.loc[mrn_meta["date_diff"].idxmin()]

            # Build HTML and render to PNG
            html = _build_dashboard_html(
                mrn=mrn,
                clinic_date=clinic_date,
                row=row,
                meta_row=meta_row,
                df_output=df_output,
                df_meta=df_meta,
                shap_png_path=shap_png_path,
                layout=layout,
                font_scale=font_scale,
            )

            page.set_content(html, wait_until="networkidle")
            out_path = dashboard_dir / f"dashboard_mrn_{mrn}_clinic_{clinic_date_str}.png"
            page.screenshot(path=str(out_path), full_page=True)

            # Delete SHAP PNG now that it has been embedded
            shap_png_path.unlink()
            print(f"[OK] Saved dashboard: {out_path}")

        browser.close()
