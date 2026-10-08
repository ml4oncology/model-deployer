"""
Map an ICD-O-3 topography site (primary_site) to its reporting cancer group.

The mapping data lives in cancer_groups.yaml so it can be updated without
touching code.
"""

from pathlib import Path

import yaml
from ml_common.constants import CANCER_CODE_MAP

DEFAULT_GROUP = "other"

_CONFIG_PATH = Path(__file__).parent / "cancer_groups.yaml"

with open(_CONFIG_PATH) as _file:
    _config = yaml.safe_load(_file)

SITE_TO_GROUP: dict[str, str] = _config["site_to_group"]
CONSIDERED_ICD_CODES: set[str] = set(_config["considered_icd_codes"])

ICD_TO_GROUP: dict[str, str] = {}
for _code in CONSIDERED_ICD_CODES:
    _description = CANCER_CODE_MAP[_code]
    if _description not in SITE_TO_GROUP:
        raise ValueError(
            f"{_code} ({_description!r}) is a considered ICD code but has no entry in site_to_group ({_CONFIG_PATH})"
        )
    ICD_TO_GROUP[_code] = SITE_TO_GROUP[_description]


def to_cancer_group(primary_site) -> str:
    """Return the reporting cancer group for a primary_site value.

    primary_site holds one or more comma-separated 3-character ICD codes.
    Rows with multiple sites, missing values, or codes outside the considered
    set fall back to DEFAULT_GROUP.
    """
    if not isinstance(primary_site, str):
        return DEFAULT_GROUP
    codes = [code.strip() for code in primary_site.split(",") if code.strip()]
    if len(codes) != 1:
        return DEFAULT_GROUP
    return ICD_TO_GROUP.get(codes[0], DEFAULT_GROUP)


def validate_model_site_codes(model_features) -> None:
    """Raise if the model has cancer_site_* codes missing from considered_icd_codes."""
    model_codes = {
        str(feature).removeprefix("cancer_site_")
        for feature in model_features
        if str(feature).startswith("cancer_site_") and str(feature) != "cancer_site_other"
    }
    missing = sorted(model_codes - CONSIDERED_ICD_CODES)
    if missing:
        raise ValueError(f"Model uses cancer site codes {missing} missing from considered_icd_codes in {_CONFIG_PATH}")
