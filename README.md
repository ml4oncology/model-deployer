# Model Deployer

Silent deployment of models to predict undesirable cancer events in Princess Margaret hospital under the EPIC system.

# Filenames:
AIM2REDUCE_DATA_YYYYMMDD.csv
- patients and their historical records who had a treatment scheduled the following day of
- pulled daily
AIM2REDUCE_DATA_monthly_YYYYMMDD.csv 
- patients and their historical records who had a treatment since March 2024 until the month before the date
- pulled monthly
AIM2REDUCE_chemo_weekly_20250303.csv
- patients and their historical records who had a clinic visit on the day of and have treatment scheduled within the next 5 days
- pulled weekly
AIM2REDUCE_chemo_weekly_monthly_20250303.csv
- patients and their historical records who had a clinic visit since Sep 2024 until the month before the date
- pulled monthly

# Project Organization
```
├── Codes           <- The source code
├── Data            <- The daily, weekly, and monthly data pulls
├── Models          <- ML models
└── Infos           <- Configuration files (regimen mapping, thresholds, etc)
```