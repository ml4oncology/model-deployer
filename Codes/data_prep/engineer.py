"""
Module to engineer features
"""
from tqdm import tqdm
import numpy as np
import pandas as pd

###############################################################################
# Treatment
###############################################################################
def get_line_of_therapy(df):
    # identify line of therapy (the nth different palliative intent treatment taken)
    # NOTE: all other intent treatment are given line of therapy of 0. Usually (not always but oh well) once the first
    # palliative treatment appears, the rest of the treatments remain palliative
    new_regimen = (df['first_treatment_date'] != df['first_treatment_date'].shift())
    palliative_intent = df['intent'] == 'PALLIATIVE'
    return (new_regimen & palliative_intent).cumsum()
