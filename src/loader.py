from pathlib import Path

import pandas as pd
import yaml
from ml_common.util import load_pickle

# Note: this is just temporary so that the saved pickled model
# will run without issues. If we switch to ONNX, we can revisit this.
# If you look at the notebook for training the model, there is an import
# statement "from acu" ... . The import statements below are included
# so that there will be no errors of 'acu' package not installed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "training"))


class Config:
    """Loads configuration files like thresholds, mappings, etc."""

    def __init__(self, info_dir: str):
        self.thresholds = pd.read_excel(f"{info_dir}/ED_Prediction_Threshold.xlsx")
        self.thresholds.columns = self.thresholds.columns.str.lower()

        from deployer.data_prep.regimen import RegimenMapper
        self.regimen_mapper = RegimenMapper(info_dir)
        self.regimens_to_exclude = self.regimen_mapper.regimens_to_exclude

        self.cancer_sites = pd.read_excel(f"{info_dir}/Cancer_Site_List.xlsx")
        self.cancer_site_list = self.cancer_sites["Cancer_Site"].tolist()

        # imputation values
        data_prep_dir = Path(__file__).parent / "data_prep"
        imputation_constants_path = data_prep_dir / "imputation_constants.yaml"

        with open(imputation_constants_path) as file:
            self.imputation_constants = yaml.safe_load(file)

class Model:
    """Loads ML models and pipeline parameters

    #TODO: support multiple models / targets
    #TODO: convert model to ONNX format
    """

    def __init__(self, model_dir: str, prep_dir: str, anchor: str, name: str | None = None):
        self.anchor = anchor
        self.name = name

        data_prep_dir = Path(__file__).parent / "data_prep"
        data_processing_constants = data_prep_dir / "config.yaml"

        with open(data_processing_constants) as file:
            self.prep_cfg = yaml.safe_load(file)

        # Load model file names from manifest
        with open(f"{model_dir}/model_manifest.yaml") as f:
            manifest = yaml.safe_load(f)[anchor]

        # Emergency Department Visit
        self.prep = load_pickle(prep_dir, manifest["prep"])
        self.model = load_pickle(model_dir, manifest["model"])
        if "orig_x" in manifest:
            self.orig_x = pd.read_parquet(f"{prep_dir}/{manifest['orig_x']}")
        self.model_features = self.model[0].feature_names_in_

        # column ordering needs to match
        # TODO: use the scaler, imputer, etc's pre-existing columns in ml-common.prep
        self.prep.norm_cols = self.prep.scaler.feature_names_in_
        self.prep.imp.impute_cols["mean"] = self.prep.imp.imputer["mean"].feature_names_in_ if self.prep.imp.imputer["mean"] is not None else []
        self.prep.imp.impute_cols["most_frequent"] = self.prep.imp.imputer["most_frequent"].feature_names_in_
