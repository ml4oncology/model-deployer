from ml_common.constants import CANCER_CODE_MAP


# TODO: Just create one large mapping
def clean_feature_name(name: str) -> str:
    if name == "patient_ecog":
        return "Eastern Cooperative Oncology Group (ECOG) Performance Status"

    mapping = {
        "prev": "previous",
        "num_": "number_of_",
        "%_ideal_dose": "percentage_of_ideal_dose",
        "intent": "intent_of_systemic_treatment",
        "cancer_site": "topography_ICD-0-3",
        "morphology": "morphology_ICD-0-3",
        "shortness_of_breath": "dyspnea",
        "tiredness": "fatigue",
        "patient_ecog": "eastern_cooperative_oncology_group_(ECOG)_performance_status",
        "cycle_number": "chemotherapy_cycle",
    }
    for orig, new in mapping.items():
        name = name.replace(orig, new)

    # title the name and replace underscores with space, but don't modify anything inside brackets at the end
    if name.endswith(")") and not name.startswith("regimen"):
        name, extra_info = name.split("(")
        name = "(".join([name.replace("_", " ").title(), extra_info])
    else:
        name = name.replace("_", " ").title()

    # capitalize certain substrings
    for substr in ["Ed V", "Icd", "Other", "Esas", "Ecog"]:
        name = name.replace(substr, substr.upper())
    # lowercase certain substrings
    for substr in [" Of "]:
        name = name.replace(substr, substr.lower())

    if name.startswith("Topography ") or name.startswith("Morphology "):
        # get full cancer description
        code = name.split(" ")[-1]
        if code in CANCER_CODE_MAP:
            name = f"{name}, {CANCER_CODE_MAP[code]}"
    elif name.startswith("ESAS "):
        # add 'score'
        if "Change" in name:
            name = name.replace("Change", "Score Change")
        else:
            name += " Score"

    for prefix in ["Regimen ", "Percentage of Ideal Dose Given "]:
        if name.startswith(prefix):
            # capitalize all regimen / drug names
            name = f"{prefix}{name.split(prefix)[-1].upper()}"

    return name