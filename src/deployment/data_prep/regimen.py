import pandas as pd


class RegimenMapper:
    """Single source of truth for regimen mapping and exclusion logic.

    Loads harmonize_regimen_mapping.csv which contains columns:
        - regimen: original regimen name (key)
        - mapped_regimen: harmonized regimen name
        - status: "Remove", "Oral", or NaN/null
    """

    def __init__(self, info_dir: str):
        df = pd.read_csv(f"{info_dir}/master_regimen_map.csv")

        self.master_regimen_map = dict(df[["regimen", "mapped_regimen"]].to_numpy())
        self.master_status_map = dict(df[["regimen", "status"]].to_numpy())

        self.regimens_to_exclude = df.loc[
            df["status"].isin(["Remove", "Oral"]), "regimen"
        ].tolist()

    def clean_regimens(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["regimen"].notnull()].copy()
        df["regimen"] = df["regimen"].str.strip()
        df["regimen"] = df["regimen"].replace(self.master_regimen_map)
        return df
