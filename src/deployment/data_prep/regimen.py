import pandas as pd


def _site_for_mapped_regimen(mapped_regimen) -> str | None:
    if pd.isna(mapped_regimen):
        return None
    name = str(mapped_regimen)
    if name.startswith("GI"):
        return "GI"
    if name.startswith("LU"):
        return "LU"
    return None


class RegimenMapper:
    """Single source of truth for regimen mapping and exclusion logic.

    Loads master_regimen_map.csv which contains columns:
        - regimen: original regimen name (key)
        - mapped_regimen: harmonized regimen name
        - status: "Remove", "Oral", or NaN/null
    """

    def __init__(self, info_dir: str, site: str = "GI"):
        df = pd.read_csv(f"{info_dir}/master_regimen_map.csv")

        self.site = site
        self.master_regimen_map_by_site = {"GI": {}, "LU": {}}
        self.master_status_map_by_site = {"GI": {}, "LU": {}}
        self.regimens_to_exclude_by_site = {"GI": [], "LU": []}

        for _, row in df.iterrows():
            regimen = row["regimen"]
            mapped_regimen = row["mapped_regimen"]
            status = row["status"]
            row_site = _site_for_mapped_regimen(mapped_regimen)
            if row_site is None:
                raise ValueError(
                    f"Unclassified mapped_regimen '{mapped_regimen}' for regimen '{regimen}' in master_regimen_map.csv"
                )
            self.master_regimen_map_by_site[row_site][regimen] = mapped_regimen
            self.master_status_map_by_site[row_site][regimen] = status
            if status in {"Remove", "Oral"}:
                self.regimens_to_exclude_by_site[row_site].append(regimen)

        self.master_regimen_map = self.master_regimen_map_by_site[site]
        self.master_status_map = self.master_status_map_by_site[site]
        self.regimens_to_exclude = self.regimens_to_exclude_by_site[site]

    def clean_regimens(self, df: pd.DataFrame, site: str | None = None) -> pd.DataFrame:
        active_site = site or self.site
        df = df[df["regimen"].notnull()].copy()
        df["regimen"] = df["regimen"].str.strip()
        df["regimen"] = df["regimen"].replace(
            self.master_regimen_map_by_site[active_site]
        )
        return df
