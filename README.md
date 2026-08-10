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

# Training setup

## Requirements
The training notebooks require environment variables to be configured. These contain paths to training data that are specific to your local environment.

1. Copy the example environment file:
```bash
cp config.env.example config.env
```

2. Edit `config.env` and fill in the actual paths:
   - `TRAINING_DATA_PATH`: Path to the training dataset (e.g., `clinic_centered_dataset.parquet`)

**Important:** The `config.env` file is gitignored and should never be committed to the repository. Only `config.env.example` is committed as a template.

## Data Processing 
Data used for training the machine learning model is obtained from [make-clinical-dataset](https://github.com/ml4oncology/make-clinical-dataset). The config files for the lookback/lookahead window for relevant features are defined in `make-clinical-dataset` to reduce the risk of inconsistency.  When in doubt how to process retrospective data, check how this data will be obtained during deployment.

In this code base, for training and deployment, the null values for the features `days_since_last_treatment` and `days_since_prev_ED_visit` are replaced by the lookback window for the ED visit, i.e. no event occurrence is indicated by the max value. 

## Model training
The list of features used in model training, accounting for various pulls, is located in `src/training/acu/constants.py`. In choosing the training/testing dates, consider the lookback window of some features and be mindful of covariate shifts due to changing data sources during the retrospective period. For example, if the ED visit lookback window is 1 year, the start date has to be a year after the first occurence of ED visit.

Run `src/training/notebooks/model_deployment_training.ipynb` to train and select the machine learning model for deployment. Save the model `model_name.pkl` to `Models`, save the input data file transformer (normalization, imputation, OHE, etc.) `prep.pkl` to `Infos/Prep`, and save the transformed input across all splits `X.parquet.gzip`, used in the SHAP Kernel Explainer, to `Infos/Prep`.

The notebook also produces the file `src/data_prep/imputation_constants.yaml` which is used to impute `height`, `weight`, `line_of_therapy`, `cycle_number` during the deployment phase.

# Deployment
## Requirements
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
- `Infos` must contain the regimen dictionary mapping and exclusion list `master_regimen_map.csv` as well as a subdirectory `Prep` which contains the config files, data pre-processing modules, and processed input data from the retrospective period. 
- `Data` contains live EHR-pulled data everyday during deployment.

## E-mail alarm pipeline
The e-mail alerts sent to the oncologist contain a dashboard specific to a patient. Dashboard generation uses the silent deployment baseline file. To create that baseline, first run 
```bash
python src/deployment/main.py --start-date [start_date] --end-date [end_date] --model-anchor clinic --run-on-silent-deployment True
``` 
where `[start_date]` (20240904) and `[end_date]` (20260630) are the silent deployment dates.
If you are not generating that silent deployment baseline first, run `src/deployment/main.py` with `--save-dashboard-png False` to avoid dashboard generation errors.

Optional dashboard arguments:
- `--dashboard-layout {portrait,landscape}` controls the dashboard image layout. Default is `portrait`.
- `--dashboard-font-scale FLOAT` scales clinician-facing dashboard text and histogram text proportionally. Default is `1.0`.
- `--save-dashboard-png {True,False}` controls whether dashboard PNG files are generated. Default is `True`.
- `--subset-dashboard-patients {True,False}` controls whether dashboard generation is limited to the selected patient subset. Default is `True`.
- `--run-on-silent-deployment {True,False}` runs the code on silent deployment patients for baseline histogram in dashboard. Default is `False`.

Once `Output/silent_deployment_output_clinic.csv` has been generated, a dashboard for each patient in the deployment period can be produced via
```bash
python src/deployment/main.py --start-date [start_date] --end-date [end_date] --model-anchor clinic --dashboard-font-scale 1.5
```
Use this for the daily operations of the AIM2REDUCE deployment.

Details of the model card in each dashboard are hardcoded in `src/deployment/dashboard/component.py`.

# Model evaluation
During or after the deployment period, the model performance can be prospectively evaluated via
```bash
python src/deployment/monthly_model_eval.py --start-date <start_date> --end-date <end_date> --monthly-pull-date <monthly_pull_date> --prediction-file-path <path_to_predictions>
```
`<monthly_pull_date>` refers to the monthly chemo file pull date. The monthly chemo file pulls are cumulative. Make sure to choose a date that is 2 months after `<end_date>` if you are predicting the risk of ED visit in 1 month.  

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
