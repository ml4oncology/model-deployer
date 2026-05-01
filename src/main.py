import argparse
import os
import warnings

import pandas as pd
from deployer.dashboard.filter import get_dashboard_keep_mask
from deployer.data_prep.pipeline import build_features, get_data
from deployer.dashboard.generate_dashboard_per_patient import save_dashboard_png
from deployer.loader import Config, Model
from deployer.model_eval.inference import get_model_output
from tqdm import tqdm

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="20240904")
    parser.add_argument("--end-date", type=str, default="20250101")
    parser.add_argument("--model-anchor", type=str, choices=["clinic", "treatment"], default="clinic")
    parser.add_argument("--dashboard-layout", type=str, choices=["portrait", "landscape"], default="portrait")
    parser.add_argument("--dashboard-font-scale", type=float, default=1.0)
    parser.add_argument(
        "--disable-save-dashboard-png",
        action="store_true",
        help="Skip generating dashboard PNG files.",
    )
    parser.add_argument(
        "--subset-dashboard-patients",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate dashboards only for the selected subset of patients. Default is True.",
    )
    parser.add_argument("--run-on-silent-deployment", type=bool, default=False)

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
    anchor = args.model_anchor
    dashboard_layout = args.dashboard_layout
    dashboard_font_scale = args.dashboard_font_scale
    disable_save_dashboard_png = args.disable_save_dashboard_png
    subset_dashboard_patients = args.subset_dashboard_patients
    run_on_silent_deployment = args.run_on_silent_deployment
    output_dir = args.output_dir
    data_dir = args.data_dir
    info_dir = args.info_dir
    model_dir = args.model_dir

    # if run_on_silent_deployment, do not generate dashboard
    if run_on_silent_deployment:
        disable_save_dashboard_png = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = Config(info_dir=info_dir)
    model = Model(model_dir=model_dir, prep_dir=f"{info_dir}/Prep", anchor=anchor, name="ED_visit")
    thresholds = config.thresholds.query(f'model_anchor == "{anchor.title()}-anchored"')

    date_range = pd.date_range(start_date, end_date, freq="d").strftime("%Y%m%d")
    inputs, outputs, meta_data, dashboard_masks = [], [], [], []
    for i, data_pull_date in tqdm(enumerate(date_range)):
        print(f"**** Processing #{i}: {data_pull_date} *****")
        feats = build_features(config, data_dir, data_pull_date, model.anchor)

        if "error" in feats:
            print(feats["error"])
            continue

        data = get_data(config, model, feats, data_pull_date)

        res = get_model_output(
            model,
            data,
            feats['demographic'],
            thresholds,
            pred_fn=None,
        )

        inputs.append(res["model_input"])
        outputs.append(res["model_output"])
        meta_data.append(res["demographic_info"])
        if subset_dashboard_patients:
            dashboard_masks.append(
                get_dashboard_keep_mask(
                    res["model_output"],
                    data_dir,
                    data_pull_date,
                    model.anchor,
                )
            )
        else:
            dashboard_masks.append(pd.Series(1, index=res["model_output"].index, dtype=int))

    out = pd.concat(outputs, ignore_index=True, axis=0)
    out = out.reset_index(drop=True)

    inp = pd.concat(inputs, ignore_index=True, axis=0)
    inp = inp.reset_index(drop=True)
    inp.to_parquet(f"{output_dir}/input_{anchor}.parquet")

    meta = pd.concat(meta_data, ignore_index=True, axis=0)
    meta = meta.reset_index(drop=True)

    mask = pd.concat(dashboard_masks, ignore_index=True, axis=0)
    mask = mask.reset_index(drop=True)

    dashboard_inp = inp.loc[mask == 1].reset_index(drop=True)
    dashboard_meta = meta.loc[mask == 1].reset_index(drop=True)

    out = out.merge(meta[['mrn', 'clinic_date', 'cancer']], on=['mrn', 'clinic_date'], how='left')
    out.to_csv(f"{output_dir}/output_{anchor}.csv", index=False)

    dashboard_out = out.loc[mask == 1].reset_index(drop=True)

    if run_on_silent_deployment:
        # filter out patients from out
        out['clinic_date'] = pd.to_datetime(out['clinic_date'])
        out.sort_values(by=['clinic_date'], ascending=True, inplace=True)
        # for every mrn, next_sched_trt_date, keep rows with the latest clinic_date
        out = out.loc[out.groupby(['mrn', 'next_sched_trt_date'])['clinic_date'].idxmax()]
        out.sort_values(by=['clinic_date'], ascending=True, inplace=True)
        # for every mrn, regimen, keep row with earliest clinic date
        out = out.loc[out.groupby(['mrn', 'regimen'])['clinic_date'].idxmin()]       
        out.to_csv(f"{output_dir}/silent_deployment_output_{anchor}.csv", index=False)

    # Generate dashboard per patient
    if not disable_save_dashboard_png:
        save_dashboard_png(
            model,
            dashboard_inp,
            dashboard_out,
            dashboard_meta,
            output_dir,
            anchor=anchor,
            layout=dashboard_layout,
            font_scale=dashboard_font_scale,
        )
