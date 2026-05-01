from functools import partial
from pathlib import Path
from typing import TypeVar

import matplotlib
import numpy as np
import pandas as pd
import shap
import textwrap

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from sklearn.base import BaseEstimator

from deployer.loader import Model
from deployer.model_eval.util import clean_feature_name
from make_clinical_dataset.shared.constants import UNIT_MAP

ScikitModel = TypeVar("ScikitModel", bound=BaseEstimator)

SHAP_SUBDIR = "shap_waterfall"
SHAP_LABEL_FONT_SIZE = 8
SHAP_BODY_FONT_SIZE = 8
SHAP_ANNOTATION_FONT_SIZE = 8


def predict(data: pd.DataFrame, models: list[ScikitModel]):
    return np.mean([m.predict_proba(data)[:, 1] for m in models], axis=0)


def wrap_label(label, width=25):
    return "\n".join(textwrap.wrap(label, width=width))


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
            ax.transData
            + plt.matplotlib.transforms.ScaledTranslation(
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


def build_shap_explainer(model: Model) -> shap.Explainer:
    pred_fn_for_shap = partial(predict, models=model.model)
    bg_data = model.orig_x.sample(min(10000, len(model.orig_x)), random_state=42).astype(float)
    return shap.Explainer(pred_fn_for_shap, bg_data, seed=42)


def save_shap_waterfall_plot(
    model: Model,
    model_input: pd.DataFrame,
    explainer: shap.Explainer,
    output_path: str | Path,
    row_idx: int,
    max_display: int = 10,
    font_scale: float = 1.0,
) -> None:
    input_row = model_input.iloc[[row_idx]].astype(float)
    shap_values = explainer(input_row)

    norm_cols = model.prep.norm_cols
    input_row[norm_cols] = model.prep.scaler.inverse_transform(input_row[norm_cols])
    input_data = input_row.iloc[0].tolist()

    unit_map = {feat: unit for unit, feats in UNIT_MAP.items() for feat in feats}
    feature_names = []
    for feat in shap_values.feature_names:
        res = clean_feature_name(feat)
        if feat in unit_map:
            res = f"{res} ({unit_map[feat]})"
        feature_names.append(wrap_label(res, width=30))

    explanation = shap.Explanation(
        values=shap_values.values[0],
        base_values=float(shap_values.base_values[0]),
        data=np.array(input_data),
        feature_names=feature_names,
    )

    with plt.rc_context({"font.size": SHAP_LABEL_FONT_SIZE * font_scale}):
        fig, _ = plt.subplots(figsize=(12, 10))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout(pad=0.5)

    fig = plt.gcf()
    _apply_waterfall_typography(fig, font_scale=font_scale)
    _fix_waterfall_labels(fig)
    fig.subplots_adjust(top=0.90)
    _add_waterfall_annotation_legend(fig, font_scale=font_scale)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close()
    plt.close("all")
