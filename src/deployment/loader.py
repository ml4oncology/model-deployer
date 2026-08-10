from pathlib import Path

import pandas as pd
import yaml
from make_clinical_dataset.shared.constants import DEFAULT_CONFIG_PATH
from ml_common.util import load_pickle

from deployer.data_prep.constants import ED_VISIT_COUNT_LOOKBACK_DAYS

# Note: this is just temporary so that the saved pickled model
# will run without issues. If we switch to ONNX, we can revisit this.
# If you look at the notebook for training the model, there is an import
# statement "from acu" ... . The import statements below are included
# so that there will be no errors of 'acu' package not installed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / ".." / "training"))


class Config:
    """Loads configuration files like thresholds, mappings, etc."""

    def __init__(self, info_dir: str):
        self.thresholds = pd.read_excel(f"{info_dir}/ED_Prediction_Threshold.xlsx")
        self.thresholds.columns = self.thresholds.columns.str.lower()

        from deployer.data_prep.regimen import RegimenMapper
        self.regimen_mapper = RegimenMapper(info_dir)
        self.regimens_to_exclude = self.regimen_mapper.regimens_to_exclude

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

        with open(DEFAULT_CONFIG_PATH) as file:
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

        # Inference reorders inputs to fold-0's feature order and scores every fold
        # positionally, so all folds must agree on it.
        for fold_model in self.model[1:]:
            if list(fold_model.feature_names_in_) != list(self.model_features):
                raise ValueError(
                    "Fold models disagree on feature order. Every fold must match "
                    "fold-0's feature_names_in_."
                )

        # Infer which ED prior-visits count feature this model was trained on so the
        # deployment pipeline combines ED visits with the matching lookback window.
        ed_prior_visits_features = [
            feat for feat in self.model_features if feat in ED_VISIT_COUNT_LOOKBACK_DAYS
        ]
        if len(ed_prior_visits_features) != 1:
            raise ValueError(
                f"Expected exactly one ED prior-visits count feature in model features, "
                f"found {ed_prior_visits_features}."
            )
        self.ed_prior_visits_feature = ed_prior_visits_features[0]
        self.ed_visit_lookback_days = ED_VISIT_COUNT_LOOKBACK_DAYS[self.ed_prior_visits_feature]
        ed_lookback_years = self.ed_visit_lookback_days // 365
        deployment_lookback_window = self.prep_cfg["ed_visit_lookback_window_deployment"]
        assert ed_lookback_years == 1 or ed_lookback_years == deployment_lookback_window, (
            f"Model ED lookback of {ed_lookback_years} year(s) must be 1 (deployment uses a hardcoded "
            f"1-year lookback) or match ed_visit_lookback_window_deployment={deployment_lookback_window}"
        )

        # column ordering needs to match
        # TODO: use the scaler, imputer, etc's pre-existing columns in ml-common.prep
        self.prep.norm_cols = self.prep.scaler.feature_names_in_
        self.prep.imp.impute_cols["mean"] = self.prep.imp.imputer["mean"].feature_names_in_ if self.prep.imp.imputer["mean"] is not None else []
        self.prep.imp.impute_cols["most_frequent"] = self.prep.imp.imputer["most_frequent"].feature_names_in_
