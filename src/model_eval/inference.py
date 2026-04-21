"""
Module to generate predictions
"""

import matplotlib  
matplotlib.use('Agg')  

import pickle
from functools import partial 
from pathlib import Path  
from typing import Callable, TypeVar

import matplotlib.pyplot as plt  
from matplotlib.patches import FancyArrow
import numpy as np
import pandas as pd
import shap  
from deployer.loader import Model
from make_clinical_dataset.shared.constants import UNIT_MAP  
from ml_common.constants import CANCER_CODE_MAP
from sklearn.base import BaseEstimator
from dateutil.relativedelta import relativedelta
from deployer.model_eval.util import clean_feature_name  

import textwrap

def wrap_label(label, width=25):
    return "\n".join(textwrap.wrap(label, width=width))

ScikitModel = TypeVar("ScikitModel", bound=BaseEstimator)

ANCHOR_META_COLS = {"clinic": ["mrn", "next_sched_trt_date", "clinic_date"], "treatment": ["mrn", "treatment_date"]}
SHAP_SUBDIR = "shap_waterfall"  
SHAP_LABEL_FONT_SIZE = 8
SHAP_BODY_FONT_SIZE = 8
SHAP_ANNOTATION_FONT_SIZE = 8


def predict(data: pd.DataFrame, models: list[ScikitModel]):
    # average across the folds
    return np.mean([m.predict_proba(data)[:, 1] for m in models], axis=0)

def _fix_waterfall_labels(fig: plt.Figure, min_bar_width: float = 0.2) -> None:
    if not fig.axes:
        return

    ax = fig.axes[0]
    texts = [t for t in ax.texts if t.get_text().lstrip("−-+").replace(".", "").isdigit()]
    arrows = [p for p in ax.patches if isinstance(p, FancyArrow)]

    if not texts or not arrows:
        return

    x_min, x_max = ax.get_xlim()
    axis_range = x_max - x_min
    if axis_range <= 0:
        return

    text_rows = sorted(texts, key=lambda text: text.get_position()[1], reverse=True)

    arrow_rows = []
    for arrow in arrows:
        verts = arrow.get_path().vertices
        transform = arrow.get_transform()
        verts_display = transform.transform(verts)
        verts_data = ax.transData.inverted().transform(verts_display)
        width = float(np.max(verts_data[:, 0]) - np.min(verts_data[:, 0]))
        center_y = float(np.mean(verts_data[:, 1]))
        arrow_rows.append((center_y, width, verts_data, arrow))

    arrow_rows.sort(key=lambda row: row[0], reverse=True)

    for text, (_, bar_width, verts_data, arrow) in zip(text_rows, arrow_rows):
        if bar_width / axis_range >= min_bar_width:
            continue

        is_positive = arrow._dx >= 0
        bar_tip = float(np.max(verts_data[:, 0]) if is_positive else np.min(verts_data[:, 0]))
        _, tip_y = text.get_position()
        offset_points = 1 if is_positive else -1

        text.set_position((bar_tip, tip_y))
        text.set_transform(
            ax.transData + plt.matplotlib.transforms.ScaledTranslation(
                offset_points / 72.0,
                0,
                fig.dpi_scale_trans,
            )
        )
        text.set_ha("left" if is_positive else "right")
        text.set_va("center")
        text.set_clip_on(False)

        edge = arrow.get_edgecolor()
        if len(edge) >= 4 and edge[3] > 0:
            text.set_color(edge)
        else:
            text.set_color("#ff0051" if is_positive else "#008bfb")

def _add_waterfall_annotation_legend(fig: plt.Figure, font_scale: float = 1.0) -> None:
    legend_text = (
        "f(x): Patient's Predicted ED Visit Risk\n"
        "E[f(X)]: Average ED Visit Risk"
    )
    fig.text(
        0.02,
        0.985,
        legend_text,
        ha="left",
        va="top",
        fontsize=SHAP_ANNOTATION_FONT_SIZE * font_scale,
        color="#444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5", edgecolor="#d9d9d9"),
    )


def _apply_waterfall_typography(fig: plt.Figure, font_scale: float = 1.0) -> None:
    if not fig.axes:
        return

    ax = fig.axes[0]
    label_size = SHAP_LABEL_FONT_SIZE * font_scale
    body_size = SHAP_BODY_FONT_SIZE * font_scale

    for text in ax.texts:
        text.set_fontsize(body_size)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(body_size)

    ax.tick_params(axis="both", labelsize=body_size)
    ax.xaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontsize(label_size)

def _save_shap_waterfall(
    model: Model,
    model_input: pd.DataFrame,
    explainer: shap.Explainer,
    shap_dir: Path,
    row_idx: int,
    mrn: int,
    clinic_date: pd.Timestamp,
    max_display: int = 10,
    font_scale: float = 1.0,
) -> None:
    """
    Compute and save a SHAP waterfall plot for a single model input row.
    """
    # Step 1: Extract the single row and compute SHAP values
    input_row = model_input.iloc[[row_idx]].astype(float)
    shap_values = explainer(input_row)

    # Step 2: Unnormalize input features for display (from ml4u backend shap_plot)
    norm_cols = model.prep.norm_cols
    input_row[norm_cols] = model.prep.scaler.inverse_transform(input_row[norm_cols])
    input_data = input_row.iloc[0].tolist()

    # Step 3: Clean feature names + add units 
    unit_map = {feat: unit for unit, feats in UNIT_MAP.items() for feat in feats}
    feature_names = []
    for feat in shap_values.feature_names:
        res = clean_feature_name(feat)
        if feat in unit_map:
            res = f"{res} ({unit_map[feat]})"
        res = wrap_label(res, width=30)
        feature_names.append(res)

    # Step 4: Rebuild shap.Explanation with cleaned names + unnormalized data
    explanation = shap.Explanation(
        values=shap_values.values[0],
        base_values=float(shap_values.base_values[0]),
        data=np.array(input_data),
        feature_names=feature_names,
    )

    # Step 5: Plot and save (mirrors component.py shap.plots.waterfall call)
    with plt.rc_context({"font.size": SHAP_LABEL_FONT_SIZE * font_scale}):
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout(pad=0.5)

    # --- Fix clipped labels: move text outside narrow bars ---
    fig = plt.gcf()
    _apply_waterfall_typography(fig, font_scale=font_scale)
    _fix_waterfall_labels(fig)
    fig.subplots_adjust(top=0.90)
    _add_waterfall_annotation_legend(fig, font_scale=font_scale)

    clinic_date_str = pd.Timestamp(clinic_date).strftime('%Y%m%d')
    filename = shap_dir / f"shap_mrn_{mrn}_clinic_{clinic_date_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    plt.close('all')

def _compute_demographic_info(df_demographic: pd.DataFrame, 
                              df_model_output: pd.DataFrame) -> pd.DataFrame:
    
    df_model_output = df_model_output.merge(df_demographic, how="left", on=["mrn"])
    df_model_output["age"] = df_model_output.apply(
        lambda row: relativedelta(row['clinic_date'], row['date_of_birth']).years,
        axis=1
    )
    df_model_output["gender"] = df_model_output["female"].astype(int).map({1: "Female", 0: "Male"})
    df_model_output["cancer"] = df_model_output["primary_site"].map(CANCER_CODE_MAP).fillna("Other").str.split(" ").str[0]

    return df_model_output[['mrn', 'clinic_date', 'age', 'gender', 'cancer']].copy()

def get_model_output(
    model: Model,
    df: pd.DataFrame,
    demographic_info: pd.DataFrame, 
    thresholds: pd.DataFrame,
    pred_fn: Callable | None = None,
    output_dir: str | Path | None = None,  
    dashboard_font_scale: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """
    TODO: set data_pull_day as the assessment date for treatment date anchor
    """
    if pred_fn is None:
        pred_fn = predict

    model_input = df[model.model_features].copy() # reorder model features according to the order used in training
    meta_cols = ANCHOR_META_COLS[model.anchor]
    model_output = df[meta_cols].copy()
 
    # Drop any row that contains NaN => to work with RF
    # NOTE: mostly when ['height', 'weight', 'body_surface_area'] is missing
    # TODO: impute them instead of dropping

    model_input = model_input.dropna()
    model_output = model_output.loc[model_input.index]

    # Generate prediction probabilities
    model_output["ed_pred_prob"] = pred_fn(model_input, model.model)

    # Compute demographic info
    demog_df = _compute_demographic_info(demographic_info, model_output)

    # Generate binary predictions based on these pre-defined thresholds
    for _, row in thresholds.iterrows():
        assert row["labels"] == "ED_visit"
        alarm_rate = row["alarm_rate"]
        pred_thresh = row["prediction_threshold"]
        model_output[f"ed_pred_alarm_{alarm_rate}"] = (model_output["ed_pred_prob"] > pred_thresh).astype(int)

    # Compute and save SHAP waterfall plots if output_dir is provided
    if output_dir is not None:
        shap_dir = Path(output_dir) / SHAP_SUBDIR
        shap_dir.mkdir(parents=True, exist_ok=True)

        pred_fn_for_shap = partial(predict, models=model.model)
        bg_data = model.orig_x.sample(min(10000, len(model.orig_x)), random_state=42).astype(float)
        explainer = shap.Explainer(pred_fn_for_shap, bg_data, seed=42)

        for row_idx, (idx, row) in enumerate(model_output.iterrows()):
            _save_shap_waterfall(
                model=model,
                model_input=model_input,
                explainer=explainer,
                shap_dir=shap_dir,
                row_idx=row_idx,
                mrn=int(row["mrn"]),
                clinic_date=row["clinic_date"],
                font_scale=dashboard_font_scale,
            )

    return {
        "model_input": model_input,
        "model_output": model_output,
        "demographic_info": demog_df
    }


def get_symp_model_output(df: pd.DataFrame, thresholds: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    # TODO: support trying out multiple different models per target
    # TODO: create a config file that maps targets with the model names
    #       (i.e. Pain: [LGBM_pain.pkl, Mistral_pain.pkl, etc])

    # load the model from disk
    filename = "LGBM_symp.pkl"
    with open(f"{model_dir}/{filename}", "rb") as file:
        model = pickle.load(file)

    # TODO: Ask Noke to provide just the models (no need for rest of the stuff in here)
    # reformat the model object
    symps = ["Pain", "Tired", "Nausea", "Depress", "Anxious", "Drowsy", "Appetite", "WellBeing", "SOB"]
    lgbm_models = {symp: model[("lgbm", f"Label_{symp}_3pt_change")]["best_model"] for symp in symps}
    calib_models = {symp: model[("lgbm", f"Label_{symp}_3pt_change")]["best_ir_model"] for symp in symps}

    # Separate patient id and visit date information
    result = df[["mrn", "treatment_date"]].copy()

    for symp in symps:
        model = lgbm_models[symp]
        calibrator = calib_models[symp]

        # Reorder model features according to the order used in training
        x = df[model.feature_name_]

        # Generate predictions and combine with patient info
        pred_prob = model.predict_proba(x)[:, 1]  # probability of the positive class
        calib_pred_prob = calibrator.transform(pred_prob)
        result[f"{symp}_pred_prob"] = calib_pred_prob

        # Generate binary predictions based on pre-defined thresholds
        thresh = thresholds[f"Label_{symp}_3pt_change"]
        result[f"{symp}_pred"] = (result[f"{symp}_pred_prob"] > thresh).astype(int)

    return result
