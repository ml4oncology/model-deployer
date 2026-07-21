# Model Deployer

Silent deployment of AIM2REDUCE models to predict undesirable cancer events in Princess Margaret hospital under the EPIC system.

# Getting started
```bash
git clone https://github.com/ml4oncology/model-deployer
pip install -e ".[dev]"
playwright install chromium

# optional
pre-commit install
mypy --install-types
```

# Please read before running
In the home directory, subdirectories `Models`, `Infos`, and `Data` needs to be present. 
- `Models` contains pickle files of the deployment model and a `model_manifest.yaml` that maps model roles to file names:
  ```yaml
  clinic:
    model: <model_pickle_filename_without_.pkl>
    prep: <prep_pickle_filename_without_.pkl>
    orig_x: <parquet_filename>

  treatment:
    model: <model_pickle_filename_without_.pkl>
    prep: <prep_pickle_filename_without_.pkl>
  ```
  When switching to a new model, place the new files in `Models/` and `Infos/Prep/`, then update the filenames in `model_manifest.yaml`.
- `Infos` contains dictionary mappings as well as a subdirectory `Prep` which contains the config files, data pre-processing modules, and data used to train the deployment model.
- `Data` contains live EHR-pulled data everyday during deployment.

A2R deployment model version 1 was trained using [Preduce v0.1.1](https://github.com/ml4oncology/PredUCE/blob/78e7b064ee5fc91dd913217b67febe4dbab0fa22/notebooks/acu/2.%20Clinic-Centered-Emerg-Pred.ipynb).

# Running the dashboard pipeline
```bash
python src/main.py --start-date 20240904 --end-date 20250804 --model-anchor clinic
python src/monthly_model_eval.py --model-anchor clinic --monthly-pull-date 20250604 --start-date 20240904 --end-date 20250804 --disable-save-dashboard-png
```

Dashboard prerequisite:
- Dashboard generation uses the silent deployment baseline file. To create that baseline, first run 
```bash
python src/main.py --start-date [start_date] --end-date [end_date] --model-anchor clinic --run-on-silent-deployment True
``` 
where `[start_date]` and `[end_date]` are the silent deployment dates.
- If you are not generating that silent deployment baseline first, run `src/main.py` with `--disable-save-dashboard-png` to avoid dashboard generation errors.

Optional dashboard arguments:
- `--dashboard-layout {portrait,landscape}` controls the dashboard image layout. Default is `portrait`.
- `--dashboard-font-scale FLOAT` scales clinician-facing dashboard text and histogram text proportionally. Default is `1.0`.
- `--disable-save-dashboard-png` skips dashboard PNG generation. By default, dashboard PNGs are generated.
- `--subset-dashboard-patients` limits dashboard generation to the selected patient subset. Default is `True`.
- `--no-subset-dashboard-patients` disables subset filtering and generates dashboards for all patients.
- `--run-on-silent-deployment` runs the code on silent deployment patients for baseline histogram in dashboard. Default is `False`.

Example:
```bash
python src/main.py --start-date 20240904 --end-date 20250804 --model-anchor clinic --dashboard-font-scale 1.5
python src/main.py --start-date 20240904 --end-date 20250804 --model-anchor clinic --no-subset-dashboard-patients
```

# Running the evaluation pipeline
```bash
python src/monthly_model_eval.py --start-date <start_date> --end-date <end_date> --monthly-pull-date <monthly_pull_date> --prediction-file-path <path_to_predictions>
```

`<monthly_pull_date>` refers to the monthly chemo file pull date. The monthly chemo file pulls are cumulative. Make sure to choose a date that is 1 month after `<end_date>` if you are predicting the risk of ED visit in 1 month.  

# Docker container

The Docker image bakes in the source code, `Infos/`, and `Models/`. The `Data/` and `Outputs/` directories are mounted when the container runs so that input data and generated files stay outside the image.

## 1. Build the Docker archive

```bash
scripts/build-docker-archive.sh
```

By default, this creates a Linux AMD64 Docker archive tagged with today's date:

```bash
dist/model-deployer_YYYYMMDD_linux-amd64.docker.tar
```

For example:

```bash
dist/model-deployer_20260604_linux-amd64.docker.tar
```

## 2. Load, tag, and push to the container registry

Replace `20260604` with the image date tag you built.

```bash
docker load --input dist/model-deployer_20260604_linux-amd64.docker.tar
docker tag model-deployer:20260604 mira-services.uhn.ca:5000/model-deployer:20260604
docker push mira-services.uhn.ca:5000/model-deployer:20260604
```

## 3. Run the silent deployment baseline with Docker

Run this first because dashboard generation needs the silent deployment baseline file in `Outputs/`.

```bash
scripts/run-silent-deployment.sh mira-services.uhn.ca:5000/model-deployer:20260604
```

## 4. Run the daily dashboard with Docker

Pass the daily pull date in `YYYYMMDD` format.

```bash
scripts/run-today.sh mira-services.uhn.ca:5000/model-deployer:20260604 20260604
```

# Project Organization
```
├── src             <- The source code
├── Data            <- The daily, weekly, and monthly data pulls
├── Models          <- ML models
└── Infos           <- Configuration files (regimen mapping, thresholds, etc)
    └── Prep
```

## Filenames:
AIM2REDUCE_DATA_YYYYMMDD.csv
- patients and their historical records who had a treatment scheduled the following day of
- pulled daily

AIM2REDUCE_DATA_monthly_YYYYMMDD.csv 
- patients and their historical records who had a treatment since March 2024 until the month before the date
- pulled monthly

AIM2REDUCE_DATA_weekly_YYYYMMDD.csv
- patients and their historical records who had a clinic visit on the day of and have treatment scheduled within the next 5 days
- pulled daily

AIM2REDUCE_DATA_weekly_monthly_YYYYMMDD.csv
- patients and their historical records who had a clinic visit since Sep 2024 until the month before the date
- pulled monthly
