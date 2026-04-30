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

# Running the dashboard pipeline
```bash
python src/main.py --start-date 20240904 --end-date 20250804 --model-anchor clinic
python src/monthly_model_eval.py --model-anchor clinic --monthly-pull-date 20250604 --start-date 20240904 --end-date 20250804 --disable-save-dashboard-png
```

Dashboard prerequisite:
- Dashboard generation uses the silent deployment baseline file. To create that baseline, first run `python src/main.py --start-date [start_date] --end-date [end_date] --model-anchor clinic --run-on-silent-deployment True`, where `[start_date]` and `[end_date]` are the silent deployment dates.
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
